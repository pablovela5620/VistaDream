import torch
from ops.utils import save_ply

scene = torch.load(f"data/test/scene.pth", weights_only=False)
save_ply(scene, "gf.ply")
