from copy import deepcopy
from importlib import import_module

from projects.pilot.configs.bev_7v_temporal.common import (
    inputs,
    opt_inputs,
    tasks,
)

march = "bayes"

task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]


def get_model(mode):
    converted_decoders = None

    return dict(
        type="MultitaskGraphModel",
        inputs=inputs[mode],
        opt_inputs=opt_inputs[mode],
        task_inputs={T.task_name: T.inputs[mode] for T in TASK_CONFIGS},
        task_modules={T.task_name: T.get_model(mode) for T in TASK_CONFIGS},
        funnel_modules=converted_decoders,
        flatten_outputs="val" not in mode,
        lazy_forward=False,
        force_cpu_init=True,  # currently init on gpu for multitask will cause cuda oom
        force_eval_init="train" not in mode,
    )


model = deepcopy(get_model("train"))
val_model = deepcopy(get_model("val"))
deploy_model = deepcopy(get_model("deploy"))
