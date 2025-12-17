from collections import OrderedDict
from copy import deepcopy

import torch
from common import (
    feat_channels,
    input_size,
    int_infer_stage,
    is_int_infer,
    rpn_out_strides,
    val_decoders,
)
from task_schedule import TASK_CONFIGS

march = "bernoulli2"

inputs = dict(img=torch.zeros((1, 3, *input_size)))

opt_inputs = dict(
    train=dict(),
    val=dict(),
    test=dict(),
)

traced_inputs = dict(
    img=torch.randn((1, 3, input_size[0], input_size[1])),
)

traced_inputs_stage_two = {
    "feats_input": [
        torch.randn(
            1, feat_channels, input_size[0] // stride, input_size[1] // stride
        )
        for stride in rpn_out_strides
    ],
    "person_rois_input": [torch.randn(1, 4)],
    "vehicle_rois_input": [torch.randn(1, 4)],
    "rear_rois_input": [torch.randn(1, 4)],
}


def get_model(mode):
    converted_decoders = OrderedDict(
        {
            (tuple(task_names), group): decoder
            for group, (task_names, decoder) in val_decoders.items()
        }
    )
    return dict(
        type="MultitaskGraphModel",
        inputs=traced_inputs_stage_two
        if (is_int_infer and int_infer_stage == "stage_two")
        else inputs,
        opt_inputs=opt_inputs[mode],
        task_inputs={T.task_name: T.get_inputs(mode) for T in TASK_CONFIGS},
        task_modules={T.task_name: T.get_model(mode) for T in TASK_CONFIGS},
        funnel_modules=converted_decoders if "val" in mode else None,
        flatten_outputs=True,
        lazy_forward=False,
    )


model = get_model("train")
val_model = deepcopy(get_model("val"))
test_model = deepcopy(get_model("test"))
