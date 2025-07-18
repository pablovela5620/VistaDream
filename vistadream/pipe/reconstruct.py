"""
Dust3R reconstrucion
GeoWizard Estimation
Smooth Projection
"""

import torch
from jaxtyping import Bool, Float
from numpy import ndarray

from vistadream.ops.connect import Smooth_Connect_Tool
from vistadream.ops.depth_pro import Depth_Pro_Tool
from vistadream.ops.utils import edge_filter


class Reconstruct_Tool:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._load_model()
        self.connector = Smooth_Connect_Tool()

    def _load_model(self):
        self.pro_dpt = Depth_Pro_Tool(ckpt=self.cfg.model.mde.dpt_pro.ckpt, device="cpu")

    def _ProDpt_(
        self, rgb, intrinsic=None
    ) -> tuple[Float[ndarray, "h w"], Float[ndarray, "3 3"], Bool[ndarray, "h w"]]:
        # conduct reconstruction
        print("Pro_dpt[1/3] Move Pro_dpt.model to GPU...")
        self.pro_dpt.to("cuda")
        print("Pro_dpt[2/3] Pro_dpt Estimation...")
        f_px = intrinsic[0, 0] if intrinsic is not None else None
        metric_dpt, intrinsic = self.pro_dpt(rgb, f_px)
        print("Pro_dpt[3/3] Move Pro_dpt.model to CPU...")
        self.pro_dpt.to("cpu")
        torch.cuda.empty_cache()
        edge_mask = edge_filter(metric_dpt, times=0.05)
        return metric_dpt, intrinsic, edge_mask

    def _Guide_ProDpt_(self, rgb, intrinsic=None, refer_dpt=None, refer_msk=None):
        """
        Performs depth reconstruction using the ProDPT model and aligns the estimated depth with a reference depth map.

        Args:
            rgb (torch.Tensor): The input RGB image tensor.
            intrinsic (torch.Tensor, optional): The camera intrinsic matrix. If provided, the focal length is extracted from intrinsic[0, 0].
            refer_dpt (torch.Tensor, optional): The reference depth map to which the estimated depth will be aligned.
            refer_msk (torch.Tensor, optional): The reference mask indicating valid regions in the reference depth map.

        Returns:
            tuple:
                - metric_dpt_connect (torch.Tensor): The aligned depth map after affine transformation.
                - metric_dpt (torch.Tensor): The raw depth map estimated by the ProDPT model.
                - intrinsic (torch.Tensor): The (possibly updated) intrinsic matrix.
                - edge_mask (torch.Tensor): The edge mask computed from the aligned depth map.
        """
        # conduct reconstruction
        print("Pro_dpt[1/3] Move Pro_dpt.model to GPU...")
        self.pro_dpt.to("cuda")
        print("Pro_dpt[2/3] Pro_dpt Estimation...")
        f_px = intrinsic[0, 0] if intrinsic is not None else None
        metric_dpt, intrinsic = self.pro_dpt(rgb, f_px=f_px)
        metric_dpt_connect = self.connector._affine_dpt_to_GS(refer_dpt, metric_dpt, ~refer_msk)
        print("Pro_dpt[3/3] Move Pro_dpt.model to CPU...")
        self.pro_dpt.to("cpu")
        torch.cuda.empty_cache()
        edge_mask = edge_filter(metric_dpt_connect, times=0.05)
        return metric_dpt_connect, metric_dpt, intrinsic, edge_mask

    # ------------- TODO: Metricv2 + Guide-GeoWizard ------------------ #
