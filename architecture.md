# VistaDream Architecture

VistaDream is a training-free framework for 3D scene reconstruction from single-view or sparse-view images using a two-stage coarse-to-fine pipeline with Gaussian Splatting and diffusion-based inpainting.

## Big-Picture Architecture

### Two-Stage Pipeline Flow
```
Stage 1 (Coarse): Input → Outpaint → Depth → Iterative Warp-and-Inpaint → Global 3D Scaffold
Stage 2 (Refine): Multi-view Consistency Sampling (MCS) → View-Consistent Gaussian Splats
```

**Single-view path**: `demo.py` → `c2f_recons.py` → outpainting → trajectory generation → iterative inpainting → MCS refinement
**Multi-view path**: `demo_sparse.py` → `sparse_recons.py` → Dust3r initialization → trajectory generation → iterative inpainting → MCS refinement

### Core Components
- **`vistadream/pipe/`**: Pipeline implementations (`c2f_recons.py`, `sparse_recons.py`, `lvm_inpaint.py`, `refine_mvdps.py`)
- **`vistadream/ops/gs/`**: Gaussian Splatting core (`basic.py` for Frame/Scene, `train.py` for optimization)
- **`vistadream/ops/`**: Operations (`trajs/` for trajectories, `mcs.py` for consistency, `connect.py` for occlusion removal)
- **`tools/`**: External model integrations (Fooocus, Depth-Pro, OneFormer, Dust3r, LLaVA)

## Developer Workflows

### Environment & Running
```bash
# CRITICAL: Always use pixi environment for correct CUDA dependencies
pixi shell               # Enter environment first
python demo.py           # Single-view reconstruction

# OR use pixi run for individual commands
pixi run python demo.py
pixi run python demo_sparse.py

# Download all model weights first
bash download_weights.sh
```

### Project Setup
```bash
# Install MSDA operations for OneFormer
pixi run install-msda

# Configuration files location
vistadream/pipe/cfgs/basic.yaml        # Single-view config
vistadream/pipe/cfgs/basic_sparse.yaml # Multi-view config
```

### Build & Test
```bash
# Check scene quality early
ls data/{scene_name}/temp.coarse.interval.png   # Monitor coarse generation
ls data/{scene_name}/dust3r.video_rgb.mp4       # Verify sparse-view initialization

# Output verification
ls data/{scene_name}/scene.pth      # Final Gaussian scene
ls data/{scene_name}/gf.ply         # Visualize in SuperSplat
ls data/{scene_name}/video_rgb.mp4  # RGB rendering video
```

## Project-Specific Conventions

### Data Structures & Memory Management
```python
# Frame: Single-view representation with auto-normalization
frame = Frame(rgb=rgb)  # Auto-converts 255→0-1 range, PIL→numpy
frame.inpaint_wo_edge   # Excludes edge regions from optimization
frame.keep = True       # Mark for supervision frames

# Gaussian_Frame: 3D representation with deactivated parameters
gf.rgbs_deact = torch.logit     # Stable optimization in logit space
gf.scales_deact = torch.log     # Log space for scales
gf._paint_filter()              # Only inpainted pixels become Gaussians

# Memory management pattern (models are large)
model.to("cpu")
torch.cuda.empty_cache()
```

### Sky & Edge Handling
```python
# Sky regions computed and assigned special depth value
sky_value = np.percentile(valid_dpt, 95) * 1.2
frame.dpt[sky] = sky_value

# Edge masking prevents artifacts
frame.inpaint_wo_edge = frame.inpaint & (~edge_mask)
```

### Gaussian Splatting Integration
```python
# Coordinate transformation: depth → 3D world coordinates
xyz = dpt2xyz(frame.dpt, frame.intrinsic)
xyz = transform_points(xyz, inv_extrinsic)

# Rendering with gsplat
rgb, dpt, alpha = gs.rendering.rasterization(
    means=xyz, scales=scale, quats=rotation,
    render_mode="RGB+ED"  # RGB + depth + alpha
)
```

### Configuration & Trajectory System
```python
# Load and override config
from vistadream.pipe.cfgs import load_cfg
cfg = load_cfg("vistadream/pipe/cfgs/basic.yaml")
cfg.scene.input.rgb = "path/to/image.jpg"

# Key parameters for tuning quality
cfg.scene.outpaint.outpaint_extend_times  # 0.3-0.6 outpaint ratio
cfg.scene.traj.n_sample                   # Number of inpainting iterations
cfg.scene.mcs.steps                       # MCS refinement steps (8-15)
cfg.scene.traj.traj_type                  # "spiral" single-view, "interp" sparse-view
```

## External Integrations

### Model Dependencies & Usage
- **Fooocus**: Diffusion inpainting (`tools/Fooocus/` → `vistadream.ops.fooocus`)
- **Depth-Pro**: Monocular depth estimation (`tools/DepthPro/` → depth maps)
- **OneFormer**: Sky segmentation (requires `pixi run install-msda`)
- **Dust3r**: Multi-view stereo for sparse initialization
- **LLaVA**: Image captioning for diffusion prompts
- **gsplat**: High-performance Gaussian Splatting rendering

### File System & Outputs
```bash
# Model checkpoints pattern
tools/{ModelName}/checkpoints/

# Scene output structure  
data/{scene_name}/
├── scene.pth                    # Final Gaussian scene (torch.load compatible)
├── gf.ply                      # Gaussian splats for SuperSplat viewer
├── video_rgb.mp4               # RGB rendering video
├── temp.coarse.interval.png    # Monitor coarse generation
└── temp.refine.interval.png    # Monitor refinement

# Temporary processing files
tmp/YYYYMMDD_HHMMSS_*_image.png
tmp/YYYYMMDD_HHMMSS_*_mask.png
```

### Debugging & Quality Control
```python
# Early termination checks
if distortion_detected:
    # Change seed and retry
    cfg.scene.outpaint.seed = new_seed

# Rerun logging for real-time visualization
import rerun as rr
rr.log(f"{path}/rgb", rr.Image(frame.rgb))
rr.log(f"{path}/dpt", rr.DepthImage(frame.dpt))

# PLY export for external visualization
from vistadream.ops.utils import save_ply
scene = torch.load('data/scene_name/scene.pth')
save_ply(scene, 'output.ply')  # View in SuperSplat
```

### GPU & Dimension Requirements
- **GPU Memory**: Large models require aggressive memory management
- **Image Dimensions**: Must be multiples of 32 for proper processing
- **CUDA**: Requires CUDA 12+ with specific architecture support (see `pyproject.toml`)

---

*For detailed parameter tuning and troubleshooting, see `vistadream/pipe/cfgs/INSTRUCT.md`*
