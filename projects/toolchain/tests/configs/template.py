import importlib
import os
import sys

cfg_path = os.getenv("HAT_CFG_PATH")
assert cfg_path is not None

cfg_dir = os.path.dirname(cfg_path)
cfg_module = os.path.basename(cfg_path).split(".")[0]

sys.path.append(cfg_dir)

globals().update(importlib.import_module(cfg_module).__dict__)

sys.path.pop()

# for CPU training
device_ids = None
# fix seed
seed = 1234

# reduce batch_size for CPU training
data_loader = eval("data_loader")
data_loader["batch_size"] = 2
data_loader["num_workers"] = 0
val_data_loader = eval("val_data_loader")
val_data_loader["batch_size"] = 1
val_data_loader["num_workers"] = 0
if "motr" in cfg_module:
    data_loader["sampler"] = None

dump_callback = dict(
    type="DumpData",
    output_dir="./tmp_dump_data",
    dump_batch_data=True,
    dump_model_outs=True,
    dump_model_grad=True,
    dump_model_param=True,
    save_interval=1,
    name_prefix=f"float_{cfg_module}_cmp",
)
float_trainer = eval("float_trainer")
float_trainer["type"] = "Trainer"
float_trainer["num_epochs"] = None
float_trainer["num_steps"] = 2  # just 2 steps
float_trainer["stop_by"] = "step"
float_trainer["callbacks"] = [dump_callback]
if "sync_bn" in float_trainer.keys():
    float_trainer.pop("sync_bn")

val_dump_callback = dict(
    type="DumpData",
    output_dir="./tmp_dump_data",
    dump_batch_data=True,
    dump_model_outs=True,
    save_interval=1,
    name_prefix=f"float_val_{cfg_module}_cmp",
)

float_predictor = eval("float_predictor")
float_predictor["num_epochs"] = None
float_predictor["num_steps"] = 1  # just 1 steps
float_predictor["stop_by"] = "step"
float_predictor["callbacks"] = [val_dump_callback]
