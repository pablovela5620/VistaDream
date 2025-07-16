## Instruction for using VistaDream

### 1. Key Hyperparameters
Here, we provide an explanation of some key parameters in ```pipe/cfgs/basic.yaml``` to facilitate parameter adjustments. For more details, please refer to our paper.

- ```scene.outpaint.outpaint_extend_times``` 
  - This parameter controls the outpaint ratio of the image when constructing the global scaffold. A larger value will result in smoother scene boundaries, but it may also introduce distortion. A recommended range is between 0.3 and 0.6.
  
- ```scene.traj```
    - ```.n_sample```: This parameter controls the number of warp-and-inpaint iterations. The more iterations, the higher the scene integrity (fewer holes). In most cases, a value of 10 is sufficient.
    - ```.far_percentage``` / ```.traj_forward_ratio``` / ```.traj_backward_ratio``` 
      - These parameters control the range of the camera's spiral trajectory (also the final scene) in ```ops/trajs```. Directly reconstruct a quite large scene might cause distortions.
      - ```far_percentage``` controls the scale of the trajectory range. For large-scale scenes (especially those involving the sky or large windows), we recommend reducing this value. An example is in [this issue](https://github.com/WHU-USI3DV/VistaDream/issues/3).
      - ```traj_forward_ratio``` and ```traj_backward_ratio``` control the forward and backward range of the camera, respectively.

- ```scene.mcs```
  - ```.steps``` means the MCS refine steps. We suggest a value between 8 and 15.
  - ```.n_view``` means the number of viewpoints optimized simultaneously in MCS. On a RTX4090 (24GB), 8 is feasible.
  - ```.rect_w``` determines the MCS control strength. We suggest 0.3-0.8.

### 2. Allow Early Check

During the execution of VistaDream, intermediate results are also retained. In the "warp-and-inpaint" coarse Gaussian Field generation stage, the inpainted result from each novel inpainting, ```temp.coarse.interval.png```, will be saved to the folder of the input image. If severe distortion issues occur, it is recommended to terminate the process early and change the ```scene.outpaint.seed``` in ```pipe/cfgs/basic.yaml``` for re-generating.

### 3. Allow Trajectory Design

You can follow [here](../../ops/trajs/TRAJECTORY.MD) to design your own camera trajectory.

### 4. For sparse view generation

You can try sparse-view reconstruction of VistaDream. 
For sparse view, we use [Dust3r](https://github.com/naver/dust3r) to first reconstruct the input views and conduct generation on it (w/o zoom-out painting).
We will output a ```dust3r.video_rgb.mp4``` at very beginning for early check. Please stop if the dust3r results already fail.
We might try [Fast3r](https://github.com/facebookresearch/fast3r) or [FLARE](https://github.com/ant-research/FLARE) in future for better initialization.
We highly recommand use ```interp``` trajectory at ```scene.traj.traj_type``` of ```pipe/cfgs/basic_sparse.yaml```