"""
render using frames in GS
outpaint with flux
"""

import datetime
import os
import random
import time

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from vistadream.flux.sampling import denoise, get_noise, get_schedule, prepare_fill_empty_prompt, unpack
from vistadream.flux.util import load_ae, load_flow_model
from vistadream.ops.gs.basic import Frame


def get_models(name: str, device: torch.device, offload: bool):
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae


def resize(img: Image.Image, min_mp: float = 0.5, max_mp: float = 2.0) -> Image.Image:
    width, height = img.size
    mp = (width * height) / 1_000_000  # Current megapixels

    if min_mp <= mp <= max_mp:
        # Even if MP is in range, ensure dimensions are multiples of 32
        new_width = int(32 * round(width / 32))
        new_height = int(32 * round(height / 32))
        if new_width != width or new_height != height:
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img

    # Calculate scaling factor
    scale = (min_mp / mp) ** 0.5 if mp < min_mp else (max_mp / mp) ** 0.5

    new_width = int(32 * round(width * scale / 32))
    new_height = int(32 * round(height * scale / 32))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


@torch.no_grad
def get_flux_fill_res(
    tmp_img, tmp_mask, prompt, height, width, num_steps, guidance, model, ae, torch_device, seed, offload
):
    x = get_noise(
        1,
        height,
        width,
        device=torch_device,
        dtype=torch.bfloat16,
        seed=seed,
    )

    if offload:
        ae = ae.to(torch_device)

    inp = prepare_fill_empty_prompt(
        x,
        prompt=prompt,
        ae=ae,
        img_cond_path=tmp_img,
        mask_path=tmp_mask,
    )

    timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=True)

    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    x = unpack(x.float(), height, width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    return x


class Inpaint_Tool:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(self.device)
        self.offload = True
        self._load_model()

    def _load_model(self):
        # Load Flux models for outpainting
        name = "flux-dev-fill"
        self.model, self.ae = get_models(
            name,
            device=self.torch_device,
            offload=self.offload,
        )

    def __call__(self, frame: Frame, outpaint_selections=None, outpaint_extend_times=0.0):
        """
        Perform outpainting on the given frame using Flux model.
        Uses the inpaint mask from the frame to determine areas to outpaint.

        Args:
            frame: Frame object containing RGB image and inpaint mask
            outpaint_selections: Legacy parameter, ignored in Flux implementation
            outpaint_extend_times: Legacy parameter, ignored in Flux implementation
        """
        if outpaint_selections is None:
            outpaint_selections = []

        # Convert frame RGB to PIL Image
        if isinstance(frame.rgb, np.ndarray):
            # Convert from numpy array (0-255) to PIL Image
            rgb_array = (frame.rgb * 255).astype(np.uint8) if frame.rgb.max() <= 1.0 else frame.rgb.astype(np.uint8)
            image = Image.fromarray(rgb_array)
        else:
            image = frame.rgb

        # Ensure image dimensions are multiples of 32 for Flux compatibility
        # image = resize(image)
        width, height = image.size
        print(f"Processing image at resolution: {width}x{height}")

        # Check if we're doing outpainting (expanding image size) or just inpainting
        if len(outpaint_selections) > 0:
            # This is outpainting - we need to expand the image size like lvm_inpaint.py does
            print(f"Performing outpainting with selections: {outpaint_selections}")

            # For now, create a simple expansion (this should be more sophisticated)
            # Expand by outpaint_extend_times factor
            expand_factor = 1.0 + outpaint_extend_times
            original_H, original_W = height, width
            new_H = int(original_H * expand_factor)
            new_W = int(original_W * expand_factor)

            # Ensure dimensions are multiples of 32
            new_H = int(32 * round(new_H / 32))
            new_W = int(32 * round(new_W / 32))

            print(f"Expanding from {original_W}x{original_H} to {new_W}x{new_H}")

            # Create expanded image with black borders
            expanded_image = Image.new("RGB", (new_W, new_H), (0, 0, 0))
            # Calculate where to paste the original image (center it)
            paste_x = (new_W - original_W) // 2
            paste_y = (new_H - original_H) // 2
            expanded_image.paste(image, (paste_x, paste_y))

            # Create mask for outpainting (white where we want to generate, black where we keep original)
            mask = Image.new("L", (new_W, new_H), 255)  # White background (generate everywhere)
            # Paste black rectangle where original image is (keep original)
            mask.paste(0, (paste_x, paste_y, paste_x + original_W, paste_y + original_H))

            # Update dimensions and image
            image = expanded_image
            width, height = new_W, new_H
            print(f"Updated processing resolution: {width}x{height}")

            # Track the expansion for setting the final mask
            expansion_info = {
                "original_H": original_H,
                "original_W": original_W,
                "new_H": new_H,
                "new_W": new_W,
                "paste_x": paste_x,
                "paste_y": paste_y,
            }
        else:
            # This is regular inpainting - use existing mask or create border mask
            expansion_info = None

            if hasattr(frame, "inpaint") and frame.inpaint is not None:
                # Convert boolean mask to PIL Image (white for areas to inpaint, black for areas to keep)
                if isinstance(frame.inpaint, np.ndarray):
                    mask_array = frame.inpaint.astype(np.uint8) * 255
                    original_mask = Image.fromarray(mask_array, mode="L")
                    # Resize mask to match the resized image dimensions
                    mask = original_mask.resize((width, height), Image.Resampling.NEAREST)
                else:
                    # Resize mask to match the resized image dimensions
                    mask = frame.inpaint.resize((width, height), Image.Resampling.NEAREST)
            else:
                # If no mask provided, create a default border mask (10% expansion)
                border_percent = 0.05  # 5% on each side = 10% total expansion
                mask = Image.new("L", (width, height), 255)  # White background
                border_left = int(width * border_percent)
                border_top = int(height * border_percent)
                border_right = width - int(width * border_percent)
                border_bottom = height - int(height * border_percent)
                # Paste black rectangle (keep original area)
                mask.paste(0, (border_left, border_top, border_right, border_bottom))
        print(f"Processing outpainting at resolution: {width}x{height}")

        # Save temporary files for Flux processing (like gradio version)
        output_dir = "./tmp"
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tag = f"{current_time}_{random.randint(10000, 99999)}"
        tmp_img = os.path.join(output_dir, f"{tag}_image.png")
        tmp_mask = os.path.join(output_dir, f"{tag}_mask.png")

        image.save(tmp_img)
        mask.save(tmp_mask)

        # Set outpainting parameters
        seed = getattr(self.cfg.scene.outpaint, "seed", 42)
        prompt = getattr(frame, "prompt", "")  # Use empty prompt if not provided
        if prompt is None:
            prompt = ""  # Ensure prompt is never None
        num_steps = 50
        guidance = 30.0

        print(f"Generating outpainting with seed {seed}")
        print(f"Prompt: '{prompt}'")

        # Perform Flux outpainting (using file paths like gradio version)
        t0 = time.perf_counter()
        x = get_flux_fill_res(
            tmp_img,
            tmp_mask,
            prompt,
            height,
            width,
            num_steps,
            guidance,
            self.model,
            self.ae,
            self.torch_device,
            seed,
            self.offload,
        )
        t1 = time.perf_counter()

        print(f"Outpainting completed in {t1 - t0:.1f}s")

        # Clear GPU memory
        torch.cuda.empty_cache()

        # Process and convert result back to numpy array
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        result_array = (127.5 * (x + 1.0)).cpu().byte().numpy()

        # Update frame with outpainted result
        frame.rgb = result_array
        frame.H, frame.W = result_array.shape[:2]

        # Ensure prompt is set (needed for downstream pipeline)
        if not hasattr(frame, "prompt") or frame.prompt is None:
            frame.prompt = ""  # Set empty string if no prompt

        # Set the inpaint mask for the outpainted frame (following lvm_inpaint.py pattern)
        # The mask should indicate which areas were outpainted (True) vs original (False)
        if expansion_info is not None:
            # For outpainting: create mask like lvm_inpaint.py does
            final_mask_array = np.ones((height, width), dtype=bool)  # All True (outpainted)
            # Set original image area to False (not outpainted)
            paste_x, paste_y = expansion_info["paste_x"], expansion_info["paste_y"]
            original_W, original_H = expansion_info["original_W"], expansion_info["original_H"]
            final_mask_array[paste_y : paste_y + original_H, paste_x : paste_x + original_W] = False
            frame.inpaint = final_mask_array
        else:
            # For regular inpainting: use the mask we created
            frame.inpaint = np.array(mask) > 127

        # Update intrinsics if they exist
        if hasattr(frame, "intrinsic") and frame.intrinsic is not None:
            # Adjust principal point to center of new image
            frame.intrinsic[0, -1] = frame.W / 2.0
            frame.intrinsic[1, -1] = frame.H / 2.0

        print("Outpainting completed and frame updated")
        return frame
