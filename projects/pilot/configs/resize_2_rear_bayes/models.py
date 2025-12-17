from collections import OrderedDict
from copy import deepcopy
from importlib import import_module

import torch
from common import (
    default_calib,
    default_distCoeffs,
    input_hw,
    is_with_pe,
    pe_config,
    tasks,
    val_decoders,
    with_cam_standiardization,
)

march = "bayes"  # "bernoulli2"

task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

inputs = dict(img=torch.zeros((1, 3, *input_hw)))

opt_inputs = dict(
    train=dict(),
    val=dict(calib=default_calib, distCoeffs=default_distCoeffs),
    test=dict(),
)

traced_inputs = dict(
    img=torch.randn((1, 3, input_hw[0], input_hw[1])),
)

if is_with_pe:
    map_shape = (1, pe_config["pe_c"], pe_config["pe_h"], pe_config["pe_w"])
    coordinate_map = torch.randn(map_shape)
    inputs.update(coordinate_map=coordinate_map)

    traced_inputs.update(coordinate_map=coordinate_map)

if with_cam_standiardization:
    uv_map = torch.zeros(1, input_hw[0], input_hw[1], 2)
    opt_inputs["train"].update(uv_map=None)
    opt_inputs["val"].update(uv_map=None)
    opt_inputs["test"].update(uv_map=uv_map)
    traced_inputs.update(uv_map=uv_map)


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
