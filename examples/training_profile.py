from examples.classification.resnet18 import *

gpu_affinity = "socket_unique_contiguous"

float_trainer.pop("num_epochs")

float_trainer["num_steps"] = 10
float_trainer["stop_by"] = "step"
float_trainer["profiler"] = dict(
    type="PyTorchProfiler",
)

for callback in float_trainer["callbacks"]:
    if isinstance(callback, dict):
        callback_type = callback.get("type", None)
        if callback_type == "CosLrUpdater":
            callback["max_steps"] = 10
