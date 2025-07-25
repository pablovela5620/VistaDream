"""
render using frames in GS
inpaint with fooocus
"""

import os
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from vistadream.ops.utils import *
from vistadream.ops.sky import Sky_Seg_Tool
from vistadream.ops.visual_check import Check
from vistadream.ops.gs.train import GS_Train_Tool
from vistadream.pipe.lvm_inpaint import Inpaint_Tool
from vistadream.pipe.reconstruct import Reconstruct_Tool
from vistadream.ops.trajs import _generate_trajectory
from vistadream.ops.connect import Occlusion_Removal
from vistadream.ops.gs.basic import Frame, Gaussian_Scene
from vistadream.ops.dust3r import Dust3r_Tool

from vistadream.ops.mcs import HackSD_MCS
from vistadream.pipe.refine_mvdps import Refinement_Tool_MCS


class Pipeline:
    def __init__(self, cfg) -> None:
        self.device = "cuda"
        self.cfg = cfg
        self.sky_value = cfg.model.sky.value
        self.sky_segor = Sky_Seg_Tool(cfg)
        self.rgb_inpaintor = Inpaint_Tool(cfg)
        self.dust3r = Dust3r_Tool(cfg)
        self.reconstructor = Reconstruct_Tool(cfg)
        # temp
        self.removalor = Occlusion_Removal()
        self.checkor = Check()

    def _determine_sky(self, out_rgb, out_dpt):
        sky = self.sky_segor(out_rgb)
        valid_dpt = out_dpt[~sky]
        _max = np.percentile(valid_dpt, 95)
        self.sky_value = _max * 1.4

    def _mkdir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def _resize_input(self):
        resize_long_edge = int(self.cfg.scene.input.resize_long_edge)
        print(f"[Preprocess...] Resize the long edge of input images to {resize_long_edge}.")
        for fn in self.files:
            spl = str.rfind(fn, ".")
            backup_fn = fn[:spl] + ".original" + fn[spl:]
            rgb = Image.open(fn)
            rgb.save(backup_fn)  # back up original image
            rgb = np.array(rgb)[:, :, :3] / 255.0
            H, W = rgb.shape[0:2]
            if H > W:
                W = int(W * resize_long_edge / H)
                H = resize_long_edge
            else:
                H = int(H * resize_long_edge / W)
                W = resize_long_edge
            rgb = cv2.resize(rgb, (W, H))
            pic = (rgb * 255.0).clip(0, 255)
            pic_save = Image.fromarray(pic.astype(np.uint8))
            pic_save.save(fn)

    def llava_prompt(self, rgb):
        self.rgb_inpaintor.llava.model.to("cuda")
        prompt = self.rgb_inpaintor.llava(
            rgb,
            prompt="<image>\n USER: Detaily imagine and describe the scene this image taken from? \n ASSISTANT: This image is taken from a scene of ",
        )
        self.rgb_inpaintor.llava.model.to("cpu")
        split = str.rfind(prompt, "ASSISTANT: This image is taken from a scene of ") + len(
            f"ASSISTANT: This image is taken from a scene of "
        )
        self.prompt = prompt[split:]

    def init_scene(self):
        # initial reconstruction
        self.dust3r.to("cuda")
        dust3r_frames = self.dust3r(self.files)
        self.dust3r.to("cpu")
        self._determine_sky(dust3r_frames[0].rgb, dust3r_frames[0].dpt)
        # get prompt
        self.llava_prompt(dust3r_frames[0].rgb)
        # add to gaussian frame
        for i, dust3r_frame in enumerate(dust3r_frames):
            H, W = dust3r_frame.rgb.shape[0:2]
            rgb = np.array(Image.open(self.files[i])) / 255.0
            rgb = rgb[:, :, 0:3]
            rgb = cv2.resize(rgb, (W, H)).clip(0, 1)
            dpt = dust3r_frame.dpt
            edge_mask = edge_filter(dpt, times=0.05)
            # determine sky
            sky = self.sky_segor(rgb)
            dpt[sky] = self.sky_value
            # add
            frame = Frame(
                H=H,
                W=W,
                rgb=rgb,
                dpt=dpt,
                sky=sky,
                intrinsic=dust3r_frame.intrinsic,
                extrinsic=dust3r_frame.extrinsic,
                prompt=self.prompt,
            )
            if i == 0:
                frame.inpaint = (np.ones_like(dust3r_frame.dpt) > 0.1,)
                frame.inpaint_wo_edge = ~edge_mask
            else:
                temp_frame = self.scene._render_for_inpaint(deepcopy(frame))
                frame.inpaint = temp_frame.inpaint
                frame.inpaint_wo_edge = temp_frame.inpaint & (~edge_mask)
            frame.keep = True
            self.scene._add_trainable_frame(frame)
        self.scene = GS_Train_Tool(self.scene, iters=256)()
        self.checkor._render_video(self.scene, save_dir=f"{self.dir}/dust3r.")

    def _generate_traj(self):
        self.dense_trajs = _generate_trajectory(self.cfg, self.scene)

    def _pose_to_frame(self, pose, margin=32):
        extrinsic = np.linalg.inv(pose)
        H = self.scene.frames[0].H + margin
        W = self.scene.frames[0].W + margin
        prompt = self.scene.frames[-1].prompt
        intrinsic = deepcopy(self.scene.frames[0].intrinsic)
        intrinsic[0, -1], intrinsic[1, -1] = W / 2, H / 2
        frame = Frame(H=H, W=W, intrinsic=intrinsic, extrinsic=extrinsic, prompt=prompt)
        frame = self.scene._render_for_inpaint(frame)
        return frame

    def _next_frame(self, margin=32):
        # select the frame with largest holes but less than 60%
        inpaint_area_ratio = []
        for pose in self.dense_trajs:
            temp_frame = self._pose_to_frame(pose, margin)
            inpaint_mask = temp_frame.inpaint
            inpaint_area_ratio.append(np.mean(inpaint_mask))
        inpaint_area_ratio = np.array(inpaint_area_ratio)
        inpaint_area_ratio[inpaint_area_ratio > 0.6] = 0.0
        # remove adjacency frames
        for s in self.select_frames:
            inpaint_area_ratio[s] = 0.0
            if s - 1 > -1:
                inpaint_area_ratio[s - 1] = 0.0
            if s + 1 < len(self.dense_trajs):
                inpaint_area_ratio[s + 1] = 0.0
        # select the largest ones
        select = np.argmax(inpaint_area_ratio)
        if inpaint_area_ratio[select] < 0.0001:
            return None
        self.select_frames.append(select)
        pose = self.dense_trajs[select]
        frame = self._pose_to_frame(pose, margin)
        return frame

    def _inpaint_next_frame(self, margin=32):
        frame = self._next_frame(margin)
        if frame is None:
            return None
        # inpaint rgb
        frame = self.rgb_inpaintor(frame)
        # inpaint dpt
        connect_dpt, metric_dpt, _, edge_msk = self.reconstructor._Guide_ProDpt_(
            frame.rgb, frame.intrinsic, frame.dpt, ~frame.inpaint
        )
        frame.dpt = connect_dpt
        frame = self.removalor(self.scene, frame)
        sky = self.sky_segor(frame.rgb)
        frame.sky = sky
        frame.dpt[sky] = self.sky_value
        frame.inpaint = (frame.inpaint) & (~sky)
        frame.inpaint_wo_edge = (frame.inpaint) & (~edge_msk)
        # determine target depth and normal
        frame.ideal_dpt = metric_dpt
        # temp visualization
        save_pic(frame.rgb, self.coarse_interval_rgb_fn)
        self.scene._add_trainable_frame(frame)
        return 0

    def _coarse_scene(self):
        self.init_scene()
        self._generate_traj()
        self.select_frames = []
        for i in range(self.n_sample + 2 - len(self.files)):
            print(f"Processing {i + len(self.files)}/{self.n_sample} frame...")
            sign = self._inpaint_next_frame()
            if sign is None:
                break
            self.scene = GS_Train_Tool(self.scene, iters=self.opt_iters_per_frame)(self.scene.frames)

    def _MCS_Refinement(self):
        refiner = HackSD_MCS(
            device="cuda",
            use_lcm=True,
            denoise_steps=self.mcs_iterations,
            sd_ckpt=self.cfg.model.optimize.sd,
            lcm_ckpt=self.cfg.model.optimize.lcm,
        )
        self.MVDPS = Refinement_Tool_MCS(
            self.scene,
            device="cuda",
            refiner=refiner,
            traj_type=self.traj_type,
            n_view=self.mcs_n_view,
            rect_w=self.mcs_rect_w,
            pre_blur=self.pre_blur,
            n_gsopt_iters=self.mcs_gsopt_per_frame,
        )
        self.scene = self.MVDPS(temp_rgb_fn=self.refine_interval_rgb_fn)
        refiner.to("cpu")

    def __call__(self, files):
        self.dir = files[0][: str.rfind(files[0], "/")]
        self.files = files
        # temp_interval_image
        self.coarse_interval_rgb_fn = f"{self.dir}/temp.coarse.interval.png"
        self.refine_interval_rgb_fn = f"{self.dir}/temp.refine.interval.png"
        # coarse
        self._resize_input()
        self.scene = Gaussian_Scene(self.cfg)
        # for trajectory genearation
        self.n_sample = self.cfg.scene.traj.n_sample
        self.traj_type = self.cfg.scene.traj.traj_type
        self.scene.traj_type = self.cfg.scene.traj.traj_type
        # for scene generation
        self.opt_iters_per_frame = self.cfg.scene.gaussian.opt_iters_per_frame
        # for scene refinement
        self.pre_blur = self.cfg.scene.mcs.blur
        self.mcs_n_view = self.cfg.scene.mcs.n_view
        self.mcs_rect_w = self.cfg.scene.mcs.rect_w
        self.mcs_iterations = self.cfg.scene.mcs.steps
        self.mcs_gsopt_per_frame = self.cfg.scene.mcs.gsopt_iters
        # # coarse scene
        self._coarse_scene()
        self.dust3r = None
        self.sky_segor = None
        self.rgb_inpaintor = None
        self.reconstructor = None
        torch.cuda.empty_cache()
        # refinement
        self._MCS_Refinement()
        torch.save(self.scene, f"{self.dir}/scene.pth")
        self.checkor._render_video(self.scene, save_dir=f"{self.dir}/refine.")
