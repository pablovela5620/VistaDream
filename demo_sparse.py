from glob import glob
from vistadream.pipe.cfgs import load_cfg
from vistadream.pipe.sparse_recons import Pipeline

base = f"data/bedroom"
images = glob(f"{base}/*.png")

cfg = load_cfg("vistadream/pipe/cfgs/basic_sparse.yaml")
vistadream = Pipeline(cfg)
vistadream(images)
