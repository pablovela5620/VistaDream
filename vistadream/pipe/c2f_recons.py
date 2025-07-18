"""
render using frames in GS
inpaint with fooocus
"""

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from PIL.ImageFile import ImageFile
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole

from vistadream.ops.connect import Occlusion_Removal
from vistadream.ops.gs.basic import Frame, Gaussian_Scene
from vistadream.ops.gs.train import GS_Train_Tool
from vistadream.ops.mcs import HackSD_MCS
from vistadream.ops.sky import Sky_Seg_Tool
from vistadream.ops.trajs import _generate_trajectory
from vistadream.ops.utils import save_pic, save_ply
from vistadream.ops.visual_check import Check
from vistadream.pipe.lvm_inpaint import Inpaint_Tool

# from vistadream.pipe.flux_inpaint import Inpaint_Tool
from vistadream.pipe.reconstruct import Reconstruct_Tool
from vistadream.pipe.refine_mvdps import Refinement_Tool_MCS


def log_frame(frame: Frame, cam_log_path: Path) -> None:
    """
    Logs a Frame object to Rerun.

    Args:
        frame (Frame): The Frame object to log.
        log_path (str): The path in Rerun where the frame will be logged.
    """
    pinhole_log_path = cam_log_path / "pinhole"
    rr.log(
        f"{pinhole_log_path}/rgb",
        rr.Image(frame.rgb, color_model=rr.ColorModel.RGB),
    )
    depth = deepcopy(frame.dpt)
    mask = frame.inpaint_wo_edge
    if mask is not None:
        depth[~mask] = 0.0

    rr.log(f"{pinhole_log_path}/dpt", rr.DepthImage(depth))
    intri = Intrinsics(
        fl_x=frame.intrinsic[0, 0],
        fl_y=frame.intrinsic[1, 1],
        cx=frame.intrinsic[0, 2],
        cy=frame.intrinsic[1, 2],
        camera_conventions="RDF",
        width=frame.W,
        height=frame.H,
    )
    extri = Extrinsics(
        world_R_cam=frame.extrinsic[:3, :3],
        world_t_cam=frame.extrinsic[:3, 3],
    )
    pinhole_param = PinholeParameters(
        name=cam_log_path.name,
        intrinsics=intri,
        extrinsics=extri,
    )

    log_pinhole(
        camera=pinhole_param,
        cam_log_path=f"{cam_log_path}",
        static=False,
        image_plane_distance=0.05,
    )


@dataclass
class PipelineConfig:
    rr_config: RerunTyroConfig
    image_path: Path = Path("data/sd_readingroom/color.png")


def create_blueprint(parent_log_path: Path, num_frames: int) -> rrb.Blueprint:
    """
    Creates a Rerun blueprint for the pipeline.

    Returns:
        rrb.Blueprint: The created blueprint.
    """
    # generate baseline view for each frame
    frame_view = rrb.Vertical(
        contents=[
            rrb.Horizontal(
                contents=[
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_{i}/pinhole/rgb",
                        contents=[
                            "+ $origin/**",
                        ],
                        name="RGB",
                    ),
                    # rrb.Spatial2DView(
                    #     origin=f"{parent_log_path}/camera_{i}/pinhole/confidence",
                    #     contents=[
                    #         "+ $origin/**",
                    #     ],
                    #     name="Mask",
                    # ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_{i}/pinhole/dpt",
                        contents=[
                            "+ $origin/**",
                        ],
                        name="Depth",
                    ),
                    # rrb.Spatial2DView(
                    #     origin=f"{parent_log_path}/camera_{i}/pinhole/confidence",
                    #     contents=[
                    #         "+ $origin/**",
                    #     ],
                    #     name="RGB + Inpaint",
                    # ),
                ]
            )
            # show at most 4 cameras
            for i in range(num_frames)
        ],
        name="2D Predictions",
    )
    blueprint = rrb.Blueprint(
        rrb.Tabs(
            frame_view,
            rrb.Spatial3DView(origin=parent_log_path, name="3D View"),
            active_tab=0,
        ),
        collapse_panels=True,
    )
    return blueprint


class Pipeline:
    def __init__(self, cfg) -> None:
        self.device = "cuda"
        self.cfg = cfg
        self.sky_value = cfg.model.sky.value
        self.sky_segor = Sky_Seg_Tool(cfg)
        self.rgb_inpaintor = Inpaint_Tool(cfg)
        self.reconstructor = Reconstruct_Tool(cfg)
        # temp
        self.removalor = Occlusion_Removal()
        self.checkor = Check()
        self.intri: Intrinsics | None = None
        self.parent_log_path: Path = Path("world")
        # blueprint: rrb.Blueprint = create_blueprint(self.parent_log_path, num_frames=2)
        # rr.send_blueprint(blueprint)

    def _mkdir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def _resize_input(self, fn):
        """
        Resizes the input image so that its longest edge matches the configured size,
        saves a backup of the original image, and overwrites the original file with the resized image.
        Ensures that the final dimensions are multiples of 32.

        Args:
            fn (str): The file path of the image to be resized.

        Side Effects:
            - Saves a backup of the original image with ".original" appended before the file extension.
            - Overwrites the original image file with the resized version.
            - Prints a message indicating the resize operation.

        Notes:
            - The resize dimension is determined by `self.cfg.scene.input.resize_long_edge`.
            - Only the first three channels (RGB) of the image are used.
            - Final dimensions are rounded to the nearest multiple of 32 for compatibility.
        """
        resize_long_edge: int = int(self.cfg.scene.input.resize_long_edge)
        print(f"[Preprocess...] Resize the long edge of input image to {resize_long_edge}.")
        spl: int = str.rfind(fn, ".")
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

    def _determine_sky(self, out_rgb, out_dpt):
        """
        Determines the sky region in an RGB image and computes a representative sky depth value.

        This method uses the `sky_segor` function to segment the sky from the provided RGB image (`out_rgb`).
        It then selects the depth values (`out_dpt`) that do not correspond to the sky region.
        The 95th percentile of these non-sky depth values is calculated and scaled by 1.2 to set `self.sky_value`.

        Args:
            out_rgb (np.ndarray): The input RGB image used for sky segmentation.
            out_dpt (np.ndarray): The corresponding depth map.

        Side Effects:
            Sets `self.sky_value` to a scaled value based on the non-sky depth distribution.
        """
        sky = self.sky_segor(out_rgb)
        valid_dpt = out_dpt[~sky]
        _max = np.percentile(valid_dpt, 95)
        self.sky_value = _max * 1.2

    def _initialization(self, rgb):
        """
        Initializes the reconstruction pipeline with the given RGB image.

        This method performs the following steps:
        1. Logs the input RGB image.
        2. Applies outpainting to the input image and logs the result.
        3. Estimates camera intrinsics and depth maps for both the input and outpainted images.
        4. Determines sky regions in both images and updates their depth maps accordingly.
        5. Splits the outpainted frame into input and outpainted areas, and prepares corresponding Frame objects.
        6. Computes and assigns edge masks, inpaint masks, and ideal depth maps for both frames.
        7. Saves a visualization of the outpainted RGB image.
        8. Adds both the input and outpainted frames to the training scene and initializes the training tool.

        Args:
            rgb (np.ndarray): Input RGB image as a NumPy array of shape (H, W, 3).

        Side Effects:
            - Updates the scene with new trainable frames.
            - Logs images for visualization.
            - Saves temporary visualizations to disk.
        """
        rgb: UInt8[np.ndarray, "h w 3"] = np.array(rgb)[:, :, :3]
        # conduct outpainting on rgb and change cu,cv
        outpaint_frame: Frame = self.rgb_inpaintor(
            Frame(rgb=rgb),
            outpaint_selections=self.outpaint_selections,
            outpaint_extend_times=self.outpaint_extend_times,
        )
        # conduct reconstruction on outpaint results
        _, intrinsic, _ = self.reconstructor._ProDpt_(rgb)  # estimate focal on input view
        metric_dpt, intrinsic, edge_msk = self.reconstructor._ProDpt_(outpaint_frame.rgb)
        self._determine_sky(outpaint_frame.rgb, metric_dpt)
        outpaint_frame.intrinsic = deepcopy(intrinsic)
        # split to input and outpaint areas
        input_frame = Frame(H=rgb.shape[0], W=rgb.shape[1], rgb=rgb, intrinsic=deepcopy(intrinsic), extrinsic=np.eye(4))
        input_frame.intrinsic[0, -1] = input_frame.W / 2.0
        input_frame.intrinsic[1, -1] = input_frame.H / 2.0
        self.intri = Intrinsics(
            fl_x=input_frame.intrinsic[0, 0],
            fl_y=input_frame.intrinsic[1, 1],
            cx=input_frame.intrinsic[0, -1],
            cy=input_frame.intrinsic[1, -1],
            width=rgb.shape[1],
            height=rgb.shape[0],
            camera_conventions="RDF",
        )
        # others
        input_area = ~outpaint_frame.inpaint
        input_edg = edge_msk[input_area].reshape(input_frame.H, input_frame.W)
        input_dpt = metric_dpt[input_area].reshape(input_frame.H, input_frame.W)
        sky = self.sky_segor(input_frame.rgb)
        input_frame.sky = sky
        input_dpt[sky] = self.sky_value
        input_frame.dpt = input_dpt
        input_frame.inpaint = np.ones_like(input_edg, bool)
        input_frame.inpaint_wo_edge = ~input_edg
        input_frame.ideal_dpt = deepcopy(input_dpt)
        input_frame.prompt = outpaint_frame.prompt
        log_frame(input_frame, self.parent_log_path / "camera_0")
        # outpaint frame
        sky = self.sky_segor(outpaint_frame.rgb)
        outpaint_frame.sky = sky
        metric_dpt[sky] = self.sky_value
        outpaint_frame.dpt = metric_dpt
        outpaint_frame.ideal_dpt = deepcopy(metric_dpt)
        outpaint_frame.inpaint = outpaint_frame.inpaint
        outpaint_frame.inpaint_wo_edge = (outpaint_frame.inpaint) & (~edge_msk)
        log_frame(outpaint_frame, self.parent_log_path / "camera_1")
        # temp visualization
        save_pic(outpaint_frame.rgb, self.coarse_interval_rgb_fn)
        # add init frame
        input_frame.keep = True
        outpaint_frame.keep = True
        self.scene._add_trainable_frame(input_frame, require_grad=True)
        self.scene._add_trainable_frame(outpaint_frame, require_grad=True)
        self.scene = GS_Train_Tool(self.scene, iters=100)(self.scene.frames)

    def _generate_traj(self) -> None:
        self.dense_trajs: Float[np.ndarray, "n_cams 4 4"] = _generate_trajectory(self.cfg, self.scene)
        # log trajectory to rerun
        for idx, world_T_cam in enumerate(self.dense_trajs):
            extrinsic = Extrinsics(
                cam_R_world=world_T_cam[:3, :3],
                cam_t_world=world_T_cam[:3, 3],
            )

            pinhole_param = PinholeParameters(
                name=f"cam_{idx}",
                extrinsics=extrinsic,
                intrinsics=self.intri,
            )
            log_pinhole(
                camera=pinhole_param,
                cam_log_path=f"{self.parent_log_path}/cam_{idx}",
                static=False,
                image_plane_distance=0.05,
            )

    def _pose_to_frame(self, pose, margin=32):
        """
        Converts a given camera pose into a Frame object with adjusted dimensions and intrinsic parameters.

        Args:
            pose (np.ndarray): The camera pose matrix (typically a 4x4 transformation matrix).
            margin (int, optional): The number of pixels to add as margin to the frame's height and width. Defaults to 32.

        Returns:
            Frame: A new Frame object with updated height, width, intrinsic, and extrinsic parameters, rendered for inpainting.

        Notes:
            - The extrinsic matrix is computed as the inverse of the input pose.
            - The intrinsic matrix is deep-copied and its principal point is shifted to the center of the new frame size.
            - The prompt from the last frame in the scene is used.
            - The frame is rendered for inpainting using the scene's internal method.
        """
        extrinsic = np.linalg.inv(pose)
        H = self.scene.frames[0].H + margin
        W = self.scene.frames[0].W + margin
        prompt = self.scene.frames[-1].prompt or ""  # Ensure prompt is never None
        intrinsic = deepcopy(self.scene.frames[0].intrinsic)
        intrinsic[0, -1], intrinsic[1, -1] = W / 2, H / 2
        frame = Frame(H=H, W=W, intrinsic=intrinsic, extrinsic=extrinsic, prompt=prompt)
        frame: Frame = self.scene._render_for_inpaint(frame)
        return frame

    def _next_frame(self, margin: int = 32):
        """
        Selects and returns the next frame for processing based on inpainting area criteria.

        This method evaluates all candidate frames (poses) and computes the ratio of pixels
        requiring inpainting for each. It excludes frames where the inpainting area exceeds 60%,
        as well as frames adjacent to those already selected. Among the remaining candidates,
        it selects the frame with the largest inpainting area ratio. If no suitable frame is found
        (i.e., all ratios are below 0.001), the method returns None.

        Args:
            margin (int, optional): The margin to use when converting a pose to a frame. Defaults to 32.

        Returns:
            Frame or None: The selected frame object with the largest inpainting area ratio,
            or None if no suitable frame is found.
        """
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
        select = np.argmin(inpaint_area_ratio)
        if inpaint_area_ratio[select] < 0.001:
            return None
        self.select_frames.append(select)
        pose = self.dense_trajs[select]
        frame = self._pose_to_frame(pose, margin)
        return frame

    def _inpaint_next_frame(self, margin: int = 32):
        """
        Inpaints the next frame in the sequence by filling missing RGB and depth (dpt) information, updating frame attributes, and preparing it for further processing.

        Args:
            margin (int, optional): The margin size to use when selecting the next frame. Defaults to 32.

        Returns:
            int or None: Returns 0 if the frame was successfully inpainted and added to the scene, or None if there are no more frames to process.

        Process Overview:
            1. Retrieves the next frame with the specified margin.
            2. Applies RGB inpainting to the frame.
            3. Inpaints the depth map using a guided reconstruction method.
            4. Applies additional frame processing (e.g., removal of unwanted regions).
            5. Segments the sky region and updates the depth map accordingly.
            6. Updates inpainting masks for the frame.
            7. Saves a temporary visualization of the inpainted RGB image.
            8. Sets the ideal depth for the frame and adds it to the scene for training.
        """
        frame: Frame | None = self._next_frame(margin)
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
        frame.inpaint = frame.inpaint  # & (~sky)
        frame.inpaint_wo_edge = (frame.inpaint) & (~edge_msk)
        # temp visualization
        save_pic(frame.rgb, self.coarse_interval_rgb_fn)
        log_frame(frame, self.parent_log_path / f"cam_{len(self.scene.frames)}")
        # determine target depth and normal
        frame.ideal_dpt = metric_dpt
        self.scene._add_trainable_frame(frame)
        return 0

    def _coarse_scene(self, rgb: ImageFile):
        self._initialization(rgb)
        self._generate_traj()
        self.select_frames = []
        for i in range(self.n_sample - 2):
            print(f"Processing {i + 2}/{self.n_sample} frame...")
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

    def __call__(self):
        # intialize rerun
        rr.log(f"{self.parent_log_path}", rr.ViewCoordinates.RDF, static=True)
        rgb_fn: str = self.cfg.scene.input.rgb
        dir = rgb_fn[: str.rfind(rgb_fn, "/")]
        # temp_interval_image
        self.coarse_interval_rgb_fn: str = f"{dir}/temp.coarse.interval.png"
        self.refine_interval_rgb_fn: str = f"{dir}/temp.refine.interval.png"
        # coarse
        self.scene = Gaussian_Scene(self.cfg)
        # for trajectory genearation
        self.n_sample: int = self.cfg.scene.traj.n_sample
        self.traj_type: Literal["spiral"] = self.cfg.scene.traj.traj_type
        self.scene.traj_type: Literal["spiral"] = self.cfg.scene.traj.traj_type
        # for scene generation
        self.opt_iters_per_frame: int = self.cfg.scene.gaussian.opt_iters_per_frame
        self.outpaint_selections = self.cfg.scene.outpaint.outpaint_selections
        self.outpaint_extend_times: float = self.cfg.scene.outpaint.outpaint_extend_times
        # for scene refinement
        self.pre_blur: bool = self.cfg.scene.mcs.blur
        self.mcs_n_view: int = self.cfg.scene.mcs.n_view
        self.mcs_rect_w: float = self.cfg.scene.mcs.rect_w
        self.mcs_iterations: int = self.cfg.scene.mcs.steps
        self.mcs_gsopt_per_frame: int = self.cfg.scene.mcs.gsopt_iters
        # coarse scene
        self._resize_input(rgb_fn)
        rgb: ImageFile = Image.open(rgb_fn)
        self._coarse_scene(rgb)
        # for more vram
        self.sky_segor = None
        self.reconstructor = None
        self.rgb_inpaintor = None
        torch.cuda.empty_cache()
        # refinement
        self._MCS_Refinement()
        torch.save(self.scene, f"{dir}/scene.pth")
        self.checkor._render_video(self.scene, save_dir=f"{dir}/")
        # generate final scene
        scene = torch.load(f"{dir}/scene.pth", weights_only=False)
        save_ply(scene, f"{dir}/gf.ply")
