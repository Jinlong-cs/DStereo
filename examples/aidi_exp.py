from examples.classification.resnet18 import *

aidi_exp_model_callback = dict(
    type="AIDIExpModel",
    model_name=task_name,
    task_type="classification",
    save_model="best",
    platforms=["J5"],
)

float_trainer["callbacks"].append(aidi_exp_model_callback)
