# Copyright (c) Horizon Robotics. All rights reserved.
from init_path import assert_func, get_data
from torch import Tensor

from hat.data.transforms.gesture.label_transform import (
    ActionLabelMap,
    ActionLabelToTensor,
)


def test_label_map():
    labelmap_params = {"convert_label_map": {"9": 8, "10": 8}}
    alm = ActionLabelMap(**labelmap_params)

    # case 1
    data = {"act_label": 9}
    data = alm(data)
    assert data["act_label"] == 8

    # case 2
    data = {"act_label": 4}
    data = alm(data)
    assert data["act_label"] == 4


def test_label_totensor():
    input_data, gt_data = get_data(
        "label_totensor_input.pkl",
        "label_totensor_output.pkl",
    )
    label_to_tensor_params = {
        "seq_len": 16,
        "use_weight_box_vis_ratio": True,
    }
    altt = ActionLabelToTensor(**label_to_tensor_params)
    pred_data = altt(input_data)

    assert_func(pred_data, gt_data)

    assert type(pred_data["act_label"]) == Tensor
    assert type(pred_data["seq_label"]) == Tensor
    assert type(pred_data["label_weight"]) == Tensor
