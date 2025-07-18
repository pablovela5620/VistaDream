import shutil
from pathlib import Path

import tyro

from vistadream.pipe.c2f_recons import Pipeline, PipelineConfig
from vistadream.pipe.cfgs import load_cfg

if __name__ == "__main__":
    pipe_config = tyro.cli(PipelineConfig)
    # create a new directory and copy over image
    data_dir = Path(f"data/{pipe_config.image_path.stem}_dir")
    print(data_dir)
    new_image_path = data_dir / pipe_config.image_path.name
    data_dir.mkdir(parents=True, exist_ok=True)
    # copy image (only if it doesn't exist)
    if not new_image_path.exists():
        shutil.copy2(pipe_config.image_path, new_image_path)
    else:
        print(f"Image already exists at {new_image_path}, skipping copy")
    cfg = load_cfg("vistadream/pipe/cfgs/basic.yaml")
    cfg.scene.input.rgb = str(new_image_path)
    vistadream = Pipeline(cfg)
    vistadream()
