# VistaDream AI Coding Instructions

VistaDream is a training-free framework for 3D scene reconstruction from single-view or sparse-view images using a two-stage coarse-to-fine pipeline with Gaussian Splatting and diffusion-based inpainting.

## Architecture Overview

### Two-Stage Pipeline Flow
VistaDream follows a **coarse-to-fine reconstruction** approach:

1. **Stage 1 - Coarse Scene**: Build global 3D scaffold through outpainting → depth estimation → iterative warp-and-inpaint
2. **Stage 2 - Refinement**: Multi-view Consistency Sampling (MCS) using diffusion models for view consistency

**Pipeline Execution Paths:**
- **Single-view**: `demo.py` → `c2f_recons.py` → outpainting → trajectory generation → iterative inpainting → MCS refinement
- **Multi-view**: `demo_sparse.py` → `sparse_recons.py` → Dust3r initialization → trajectory generation → iterative inpainting → MCS refinement

### Core Components

- **`vistadream/pipe/`**: Main pipeline implementations
  - `c2f_recons.py`: Single-view coarse-to-fine reconstruction pipeline
  - `sparse_recons.py`: Multi-view sparse reconstruction pipeline using Dust3r
  - `lvm_inpaint.py`: Fooocus-based inpainting
  - `refine_mvdps.py`: Multi-view diffusion prior sampling for refinement

- **`vistadream/ops/`**: Core operations and models
  - `gs/`: Gaussian Splatting implementation (`basic.py` for Frame/Scene, `train.py` for optimization)
  - `trajs/`: Camera trajectory generation
  - `mcs.py`: Multi-view consistency sampling
  - `connect.py`: Occlusion removal and depth connection
  - `utils.py`: Utilities including PLY export (`save_ply()`)

### Key Data Structures

- **`Frame`**: Represents a single view with RGB, depth, masks, camera parameters
  - `rgb`: H×W×3 image (0-1 range) - auto-normalized from 255 range
  - `dpt`: H×W depth map (metric depth values)
  - `inpaint`: Boolean mask for areas to inpaint
  - `inpaint_wo_edge`: Inpaint mask excluding edge regions
  - `sky`: Boolean mask for sky regions (set to `sky_value` depth)
  - `intrinsic`/`extrinsic`: 3×3 and 4×4 camera matrices
  - `prompt`: Text description for diffusion models
  - `keep`: Boolean flag for supervision frames

- **`Gaussian_Frame`**: 3D Gaussian representation of a Frame
  - `xyz`, `rgb`, `scale`, `opacity`, `rotation`: Gaussian Splatting parameters
  - Uses deactivated parameters (logit/log space) for optimization
  - `_paint_filter()`: Filters to only painted pixels for efficiency

- **`Gaussian_Scene`**: Contains multiple frames and their Gaussian representations
  - `frames[]`: List of Frame objects for supervision
  - `gaussian_frames[]`: List of Gaussian_Frame objects for training
  - `_render_RGBD()`: Core rendering function using gsplat
  - `_render_for_inpaint()`: Renders frame and generates inpaint mask

## Critical Workflows

### Running Demos
**⚠️ CRITICAL**: Always use pixi environment for correct dependencies:
```bash
# Option 1: Enter pixi shell first (recommended)
pixi shell
python demo.py          # Single-view reconstruction
python demo_sparse.py   # Multi-view reconstruction

# Option 2: Use pixi run for individual commands
pixi run python demo.py
pixi run python demo_sparse.py

# Download all required model weights FIRST
bash download_weights.sh

# Install required MSDA operations for OneFormer
pixi run install-msda
```

### Environment Setup & Dependencies
```bash
# CUDA requirements (see pyproject.toml)
# - CUDA 12+ with specific architecture support
# - Set TORCH_CUDA_ARCH_LIST="12.0" for RTX 5090 compatibility

# GPU memory requirements are significant due to multiple large models:
# - Fooocus (diffusion inpainting)
# - Depth-Pro (monocular depth estimation)  
# - LLaVA (image captioning)
# - OneFormer (sky segmentation)
# - Dust3r (multi-view stereo for sparse inputs)
```

### Configuration System
- **Single-view config**: `vistadream/pipe/cfgs/basic.yaml`
- **Multi-view config**: `vistadream/pipe/cfgs/basic_sparse.yaml`

**Key parameters for quality tuning:**
```yaml
scene:
  outpaint:
    outpaint_extend_times: 0.3-0.6    # Outpainting ratio
  traj:
    n_sample: 10                      # Number of inpainting iterations  
    traj_type: "spiral"               # "spiral" single-view, "interp" sparse-view
  mcs:
    steps: 8-15                       # MCS refinement steps
    n_view: 8                         # Simultaneous viewpoints in MCS
```

### Output Structure & Verification
```bash
data/{scene_name}/
├── scene.pth                    # Final Gaussian scene (torch.load compatible)
├── gf.ply                      # Gaussian splats (visualize in SuperSplat)
├── video_rgb.mp4               # RGB rendering video
├── video_dpt.mp4               # Depth rendering video
├── temp.coarse.interval.png    # Monitor coarse generation progress
├── temp.refine.interval.png    # Monitor refinement progress
└── dust3r.video_rgb.mp4        # Early check for sparse-view (stop if fails)

# Temporary processing files with timestamps
tmp/YYYYMMDD_HHMMSS_*_image.png
tmp/YYYYMMDD_HHMMSS_*_mask.png
```

## Project-Specific Patterns

### Frame Processing Workflow
- **Sky Handling**: Sky regions automatically detected and assigned `sky_value` depth (95th percentile × 1.2)
- **Edge Masking**: `inpaint_wo_edge` excludes edge regions from optimization to avoid artifacts
- **Frame Supervision**: Original input frames marked with `frame.keep = True` for loss computation
- **Automatic Normalization**: RGB values auto-converted from 0-255 to 0-1 range in Frame constructor

### Gaussian Splatting Integration
- **Deactivated Parameters**: Gaussians use logit/log space (`rgbs_deact`, `scales_deact`, `opacity_deact`) for stable optimization
- **Pixel Filtering**: Only inpainted pixels become Gaussians via `_paint_filter()` method
- **Coordinate Transformation**: Depth maps converted to world coordinates via `dpt2xyz()` + extrinsics
- **Rendering Modes**: `RGB+ED` mode returns RGB + depth + alpha channels using gsplat

### Memory Management & Model Lifecycle
```python
# Models are offloaded to CPU between uses due to GPU memory constraints
model.to("cpu")
torch.cuda.empty_cache()

# Aggressive memory management pattern throughout pipeline
self.sky_segor = None
self.reconstructor = None  
self.rgb_inpaintor = None
torch.cuda.empty_cache()
```

### Trajectory Generation & Camera Movement
- Camera trajectories defined in `vistadream/ops/trajs/`
- **Spiral trajectories**: Most common for single-view scenarios
- **"interp" trajectories**: Recommended for sparse-view scenarios  
- Trajectory bounds controlled by `near_percentage`, `far_percentage`, forward/backward ratios

### Inpainting & Diffusion Integration
```python
# Fooocus-based diffusion inpainting with outpainting support
outpaint_frame = self.rgb_inpaintor(
    Frame(rgb=rgb),
    outpaint_selections=['Left','Right','Top','Bottom'],
    outpaint_extend_times=0.3  # Creates expanded canvas with original centered
)

# Multi-view Consistency Sampling (MCS) for refinement
refiner = HackSD_MCS(use_lcm=True, denoise_steps=10)
self.scene = Refinement_Tool_MCS(refiner, n_view=8)(self.scene)
```

### Critical Debugging Patterns
```python
# Early termination monitoring
monitor_file = f"{dir}/temp.coarse.interval.png"  # Check for distortion
if severe_distortion_detected:
    cfg.scene.outpaint.seed = new_seed  # Change seed and retry

# Sparse view validation  
early_check = f"{dir}/dust3r.video_rgb.mp4"  # Stop if Dust3r fails

# Rerun logging for real-time visualization debugging
import rerun as rr
rr.log(f"{path}/rgb", rr.Image(frame.rgb, color_model=rr.ColorModel.RGB))
rr.log(f"{path}/dpt", rr.DepthImage(frame.dpt))

# Prompt handling ensures no None values
prompt = frame.prompt or ""  # Fallback to empty string
```

### Integration Points & External Dependencies

### Model Dependencies & Tool Chain
- **Dust3r**: Multi-view stereo for sparse input initialization (`tools/Dust3r/`)
- **Depth-Pro**: Monocular depth estimation (`tools/DepthPro/`)
- **OneFormer**: Sky segmentation (requires `pixi run install-msda`)
- **LLaVA**: Image captioning for diffusion prompts (auto-downloaded from HuggingFace)
- **Fooocus**: Diffusion-based inpainting (`tools/Fooocus/`)
- **gsplat**: High-performance Gaussian Splatting rendering (GitHub install)

### File System Conventions
```bash
# Model checkpoints pattern
tools/{ModelName}/checkpoints/

# Temporary processing files with timestamp naming
./tmp/YYYYMMDD_HHMMSS_*_image.png
./tmp/YYYYMMDD_HHMMSS_*_mask.png

# Scene output directory structure  
data/{scene_name}/
├── scene.pth                    # Final Gaussian scene (torch.load compatible)
├── gf.ply                      # Gaussian splats for SuperSplat viewer
├── video_rgb.mp4 & video_dpt.mp4 # Rendering videos
└── temp.*.interval.png         # Progress monitoring images
```

### Configuration Loading & Override Pattern
```python
from vistadream.pipe.cfgs import load_cfg
cfg = load_cfg("vistadream/pipe/cfgs/basic.yaml")
cfg.scene.input.rgb = "path/to/image.jpg"  # Override input path

# Key config sections:
# cfg.model.* - Model checkpoints and paths
# cfg.scene.input.* - Input processing settings  
# cfg.scene.outpaint.* - Outpainting parameters
# cfg.scene.traj.* - Trajectory generation settings
# cfg.scene.mcs.* - Multi-view consistency sampling
```

### GPU & System Requirements
```bash
# CUDA 12+ with architecture-specific compilation
export TORCH_CUDA_ARCH_LIST="12.0"  # For RTX 5090 compatibility

# Image dimension requirements
# Input images must have dimensions that are multiples of 32

# GPU memory considerations
# Multiple large models require ~16GB+ VRAM for comfortable operation
# Models are loaded/unloaded dynamically to manage memory
```

### PLY Export & Visualization Workflow
```python
# Standard PLY export for external visualization
import torch
from vistadream.ops.utils import save_ply

# Load and export scene
scene = torch.load('data/scene_name/scene.pth', weights_only=False)
save_ply(scene, 'output.ply')

# View in SuperSplat: https://playcanvas.com/supersplat/editor
# Alternative: Use Open3D or other point cloud viewers
```

When working with VistaDream:
1. **Always use pixi environment**: Either `pixi shell` then run commands, or prefix with `pixi run`
2. **Image requirements**: Ensure dimensions are multiples of 32 for proper processing
3. **Monitor GPU memory**: Use provided memory management patterns - models are large
4. **Choose appropriate trajectories**: "spiral" for single-view, "interp" for sparse-view inputs
5. **Early termination checks**: Monitor intermediate results and stop if quality degrades
6. **Scene persistence**: Final outputs saved as `.pth` files loadable with `torch.load()`
7. **Quality tuning**: Adjust `outpaint_extend_times`, `n_sample`, and `mcs.steps` for better results

## Code Style Notes

- Heavy use of `jaxtyping` annotations - respect the documented array shapes  
- Uses `strict=True` in `zip()` calls for safety
- Frame constructor automatically handles PIL/numpy conversion and normalization
- Gaussian parameters stored in deactivated (logit/log) space for stable optimization
- Sky regions handled specially with computed `sky_value` based on depth percentiles

---

*Feedback welcome—open an issue or PR for documentation improvements*
