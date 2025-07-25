import datetime
import os
import random
import time

import cv2
import gradio as gr
import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image

from vistadream import misc
from vistadream.flux.sampling import denoise, get_noise, get_schedule, prepare_fill_empty_prompt, unpack
from vistadream.flux.util import load_ae, load_flow_model


def add_border_and_mask(image, zoom_all=1.0, zoom_left=0, zoom_right=0, zoom_up=0, zoom_down=0, overlap=0):
    """Adds a black border around the image with individual side control and mask overlap"""
    orig_width, orig_height = image.size

    # Calculate padding for each side (in pixels)
    left_pad = int(orig_width * zoom_left)
    right_pad = int(orig_width * zoom_right)
    top_pad = int(orig_height * zoom_up)
    bottom_pad = int(orig_height * zoom_down)

    # Calculate overlap in pixels
    overlap_left = int(orig_width * overlap)
    overlap_right = int(orig_width * overlap)
    overlap_top = int(orig_height * overlap)
    overlap_bottom = int(orig_height * overlap)

    # If using the all-sides zoom, add it to each side
    if zoom_all > 1.0:
        extra_each_side = (zoom_all - 1.0) / 2
        left_pad += int(orig_width * extra_each_side)
        right_pad += int(orig_width * extra_each_side)
        top_pad += int(orig_height * extra_each_side)
        bottom_pad += int(orig_height * extra_each_side)

    # Calculate new dimensions (ensure they're multiples of 32)
    new_width = 32 * round((orig_width + left_pad + right_pad) / 32)
    new_height = 32 * round((orig_height + top_pad + bottom_pad) / 32)

    # Create new image with black border
    bordered_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    # Paste original image in position
    paste_x = left_pad
    paste_y = top_pad
    bordered_image.paste(image, (paste_x, paste_y))

    # Create mask (white where the border is, black where the original image was)
    mask = Image.new("L", (new_width, new_height), 255)  # White background
    # Paste black rectangle with overlap adjustment
    mask.paste(
        0,
        (
            paste_x + overlap_left,  # Left edge moves right
            paste_y + overlap_top,  # Top edge moves down
            paste_x + orig_width - overlap_right,  # Right edge moves left
            paste_y + orig_height - overlap_bottom,  # Bottom edge moves up
        ),
    )

    return bordered_image, mask


def get_models(name: str, device: torch.device, offload: bool):
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    # nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    return model, ae  # , nsfw_classifier


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


def get_flux_fill_input(orig_img, mask, mask_cond, vae, torch_device, img_size=512):
    img_cond = orig_img * (1 - mask)
    with torch.no_grad():
        img_cond = vae.encode(img_cond)
    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    img_cond = torch.cat((img_cond, mask_cond), dim=-1)

    bs = img_cond.shape[0]

    bs, c, h, w = orig_img.shape

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    return {"img_ids": img_ids.to(torch_device), "img_cond": img_cond}


def get_visual_learned_conditioning(visual_condition_extractor, alignment_clip, alignment_flant, x, mask):
    with torch.no_grad():
        x = visual_condition_extractor.forward_return_feature(x, mask, decoder_layer=6).detach()
        if torch.any(torch.isnan(x)):
            print("nan found in mae feature")
            x = torch.zeros_like(x)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        clip_fea = alignment_clip(x)  # 8 * 256 * 768
        flant_fea = alignment_flant(x)
    clip_fea = torch.mean(clip_fea, dim=1)
    return clip_fea, flant_fea


def prepare_model(chkpt_dir, arch="mae_vit_base_patch16", random_mask=False, finetune=False, mae_mask_concat=False):
    # build model
    model = misc.get_mae_model(arch, random_mask=random_mask, finetune=finetune, mae_mask_concat=mae_mask_concat)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    print("Load MAE")
    return model


def prepare_data(
    img,
    mask,
    null_mask=False,
    imagenet_mean=np.array([0.485, 0.456, 0.406]),
    imagenet_std=np.array([0.229, 0.224, 0.225]),
):
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = np.array(img) / 255.0
    img = img - imagenet_mean
    img = img / imagenet_std

    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask = np.array(mask)

    x = torch.tensor(img)
    mask = torch.tensor(mask)

    # make it a batch-like
    x = torch.einsum("hwc->chw", x)
    mask = mask.reshape(1, mask.shape[0], mask.shape[1])
    unmasked_img = x * (mask < 0.5)
    return x.float(), mask.float(), unmasked_img.float()


def main(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = True,
    output_dir: str = "output",
):
    torch_device = torch.device(device)

    # Model selection and loading
    name = "flux-dev-fill"

    model, ae = get_models(
        name,
        device=torch_device,
        offload=offload,
    )

    def get_res(image, expansion_percent):
        # For outpainting, we get just the image directly

        # Apply more aggressive resizing to prevent memory issues
        # Limit to 1024x1024 max resolution while preserving aspect ratio
        max_dimension = 1024
        width, height = image.size
        print(f"Original image size: {width}x{height}")

        # Calculate scale factor to fit within max_dimension
        scale = min(max_dimension / width, max_dimension / height)
        if scale < 1.0:
            new_width = int(32 * round(width * scale / 32))
            new_height = int(32 * round(height * scale / 32))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            # Still ensure dimensions are multiples of 32
            image = resize(image)

        width, height = image.size
        print(f"Resized image size: {width}x{height}")

        # Auto-generate outpainting setup: user-controlled border expansion
        border_percent = (
            expansion_percent / 200.0
        )  # Convert percentage to fraction per side (divide by 2 for each side, then by 100 for percentage)
        print(f"Border expansion: {expansion_percent}% total ({border_percent * 100:.1f}% per side)")
        image, mask = add_border_and_mask(
            image,
            zoom_all=1.0,
            zoom_left=border_percent,
            zoom_right=border_percent,
            zoom_up=border_percent,
            zoom_down=border_percent,
            overlap=0,
        )

        width, height = image.size
        print(f"Final outpaint image size: {width}x{height}")

        print(f"Processing image at resolution: {width}x{height}")

        output_dir = "./tmp"
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tag = f"{current_time}_{random.randint(10000, 99999)}"
        tmp_img = os.path.join(output_dir, f"{tag}_image.png")
        tmp_mask = os.path.join(output_dir, f"{tag}_mask.png")

        image.save(tmp_img)
        mask.save(tmp_mask)

        seed = 42

        prompt = ""
        num_steps = 50
        guidance = 30.0
        print(f"Generating outpainting with seed {seed}:\n{prompt}")

        # Outpainting (using the same flux fill model)
        t0 = time.perf_counter()
        x = get_flux_fill_res(
            tmp_img, tmp_mask, prompt, height, width, num_steps, guidance, model, ae, torch_device, seed, offload
        )
        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s")

        # Clear GPU memory before running ASUKA
        torch.cuda.empty_cache()

        # Process and display result
        x = x.clamp(-1, 1)
        # x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        return img

    with gr.Blocks() as demo, gr.Column():
        # Simple image upload for outpainting
        input_image = gr.Image(
            type="pil",
            sources=["upload", "clipboard"],
            label="Upload Image for Outpainting",
        )

        # Slider for border expansion percentage
        expansion_slider = gr.Slider(
            minimum=5,
            maximum=50,
            value=20,
            step=1,
            label="Border Expansion (%)",
            info="Total percentage to expand the image (distributed evenly on all sides)",
        )

        run_btn = gr.Button("Run Outpainting")

        with gr.Row():
            flux_res = gr.Image(label="Outpainted Result")

        run_btn.click(fn=get_res, inputs=[input_image, expansion_slider], outputs=[flux_res])

    demo.launch()


if __name__ == "__main__":
    main()
