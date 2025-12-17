import importlib
import os
import sys

test_level = os.getenv("HAT_SP_TEST_LEVEL")
model_type = os.getenv("HAT_SP_MODEL_TYPE", "fisheye_multitask")
training_stage = os.getenv("HAT_TRAINING_STEP", "float")

config_dict = {
    "fisheye_multitask": "projects/superparking/app/fisheye/multitask.py",
}
cfg_path = config_dict[model_type]

cfg_dir = os.path.dirname(cfg_path)
cfg_module = os.path.basename(cfg_path).split(".")[0]

sys.path.append(cfg_dir)

globals().update(importlib.import_module(cfg_module).__dict__)

sys.path.pop()


assert test_level in ("commit", "daily", "local")

stage_trainer = eval(f"{training_stage}_trainer")
if test_level == "daily":
    # ------------------- Add dump callback ------------------------
    # dump internal data for check.
    dump_callback = dict(
        type="DumpData",
        save_interval=5,
        name_prefix=f"test_{model_type}_pipeline",
        output_dir=f"dump_res/train_dump/{model_type}",
    )

    stage_trainer["callbacks"].append(dump_callback)
