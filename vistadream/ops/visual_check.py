from copy import deepcopy

import imageio
import matplotlib
import numpy as np
import torch

from vistadream.ops.gs.basic import Frame, Gaussian_Scene  # Import only the required classes
from vistadream.ops.trajs import _generate_trajectory
from vistadream.ops.utils import visual_pcd  # Add other required imports explicitly


class Check:
    def __init__(self) -> None:
        pass

    def _visual_pcd(self, scene: Gaussian_Scene):
        xyzs, rgbs = [], []
        for i, gf in enumerate(scene.gaussian_frames):
            xyz = gf.xyz.detach().cpu().numpy()
            rgb = torch.sigmoid(gf.rgb).detach().cpu().numpy()
            opacity = gf.opacity.detach().squeeze().cpu().numpy() > 1e-5
            xyzs.append(xyz[opacity])
            rgbs.append(rgb[opacity])
        xyzs = np.concatenate(xyzs, axis=0)
        rgbs = np.concatenate(rgbs, axis=0)
        visual_pcd(xyzs, color=rgbs, normal=True)

    @torch.no_grad()
    def _render_video(self, scene: Gaussian_Scene, save_dir="./", colorize=False):
        # render 5times frames
        nframes = len(scene.frames) * 25
        video_trajs = _generate_trajectory(None, scene, nframes=nframes)
        H, W, intrinsic = scene.frames[0].H, scene.frames[0].W, deepcopy(scene.frames[0].intrinsic)
        if H < W:
            if H > 512:
                ratio = 512 / H
                W, H = int(W * ratio), int(H * ratio)
                intrinsic[0:2] = intrinsic[0:2] * ratio
        else:
            if W > 512:
                ratio = 512 / W
                W, H = int(W * ratio), int(H * ratio)
                intrinsic[0:2] = intrinsic[0:2] * ratio
        # render
        rgbs, dpts = [], []
        print(f"[INFO] rendering final video with {nframes} frames...")
        for pose in video_trajs:
            frame = Frame(H=H, W=W, intrinsic=intrinsic, extrinsic=np.linalg.inv(pose))
            rgb, dpt, alpha = scene._render_RGBD(frame)
            rgb = rgb.detach().float().cpu().numpy()
            dpt = dpt.detach().float().cpu().numpy()
            dpts.append(dpt)
            rgbs.append((rgb * 255).astype(np.uint8))
        rgbs = np.stack(rgbs, axis=0)
        dpts = np.stack(dpts, axis=0)
        valid_dpts = dpts[dpts > 0.0]
        _min = np.percentile(valid_dpts, 1)
        _max = np.percentile(valid_dpts, 99)
        dpts = (dpts - _min) / (_max - _min)
        dpts = dpts.clip(0, 1)

        if colorize:
            cm = matplotlib.colormaps["plasma"]
            dpts_color = cm(dpts, bytes=False)[..., 0:3]
            dpts_color = (dpts_color * 255).astype(np.uint8)
            dpts = dpts_color
        else:
            dpts = dpts[..., None].repeat(3, axis=-1)
            dpts = (dpts * 255).astype(np.uint8)

        imageio.mimwrite(f"{save_dir}video_rgb.mp4", rgbs, fps=20)
        imageio.mimwrite(f"{save_dir}video_dpt.mp4", dpts, fps=20)
