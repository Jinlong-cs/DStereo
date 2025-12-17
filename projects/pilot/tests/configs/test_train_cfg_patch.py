import importlib
import os
import sys

import yaml

_model_type = os.getenv("HAT_PILOT_MODEL_TYPE")
with open(f"{os.path.dirname(__file__)}/../../model_meta.yaml", "r") as r:
    model_meta = yaml.safe_load(r)
cfg_path = model_meta[_model_type]["train"]["entry"]
cfg_dir = os.path.dirname(cfg_path)
cfg_module = os.path.basename(cfg_path).split(".")[0]
test_level = os.getenv("HAT_PILOT_TEST_LEVEL")
training_stage = os.getenv(
    "HAT_TRAINING_STEP", model_meta[_model_type]["train"]["stages"][0]
)

sys.path.insert(0, cfg_dir)

globals().update(importlib.import_module(cfg_module).__dict__)

sys.path.pop(0)


assert test_level in ("commit", "daily")

if test_level == "daily":
    # ------------------- Add dump callback ------------------------
    # dump internal data for check.
    dump_callback = dict(
        type="DumpData",
        save_interval=5,
        name_prefix=f"test_{_model_type}_{training_stage}",
        output_dir=f"dump_res/train_dump/{_model_type}",
    )

    stage_trainer = eval(f"{training_stage}_trainer")
    stage_trainer["callbacks"].append(dump_callback)

else:
    # ----------------- modify training step -----------------------
    # Initialization phase only
    # also filter out AIDI-related callbacks
    stage_trainer = eval(f"{training_stage}_trainer")
    stage_trainer["callbacks"] = [
        callback
        for callback in stage_trainer["callbacks"]
        if "aidi" not in callback["type"].lower()
    ]
    stage_trainer["num_steps"] = 0
