from collections import OrderedDict
from copy import deepcopy
from importlib import import_module

import torch
from common import input_size, mask_in_bpu, tasks, val_decoders
from horizon_plugin_pytorch.quantization import March

march = March.BAYES

# -------------------------- task --------------------------
task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

# -------------------------- data --------------------------
inputs = dict(
    img=torch.zeros((1, 3, *input_size)),
)

opt_inputs = dict(
    train=dict(),
    val=dict(),
    test=dict(),
)

traced_inputs = dict(
    img=torch.zeros((1, 3, *input_size)),
)

if mask_in_bpu:
    inputs.update(
        mask_width=torch.ones((1, 1, 1, input_size[0])),
        mask_height=torch.ones((1, 1, 1, input_size[1])),
    )
    traced_inputs.update(
        mask_width=torch.ones((1, 1, 1, input_size[0])),
        mask_height=torch.ones((1, 1, 1, input_size[1])),
    )


# -------------------------- model --------------------------
def get_model(mode):
    converted_decoders = OrderedDict(
        {
            (tuple(task_names), group): decoder
            for group, (task_names, decoder) in val_decoders.items()
        }
    )
    return dict(
        type="MultitaskGraphModel",
        inputs=inputs,
        opt_inputs=opt_inputs[mode],
        task_inputs={T.task_name: T.inputs[mode] for T in TASK_CONFIGS},
        task_modules={T.task_name: T.get_model(mode) for T in TASK_CONFIGS},
        funnel_modules=converted_decoders if "val" in mode else None,
        flatten_outputs="val" not in mode,
        lazy_forward=False,
    )


model = get_model("train")
val_model = deepcopy(get_model("val"))
test_model = deepcopy(get_model("test"))
