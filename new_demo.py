from vistadream.pipe.cfgs import load_cfg
from vistadream.pipe.c2f_recons import Pipeline

cfg = load_cfg(f"vistadream/pipe/cfgs/basic.yaml")
cfg.scene.input.rgb = "data/test/132771460_p0.jpg"
vistadream = Pipeline(cfg)
vistadream()
