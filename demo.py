from pipe.cfgs import load_cfg
from pipe.c2f_recons import Pipeline

cfg = load_cfg(f"pipe/cfgs/basic.yaml")
cfg.scene.input.rgb = "data/test/533957_p0.jpg"
vistadream = Pipeline(cfg)
vistadream()
