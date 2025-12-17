# Copyright (c) Horizon Robotics. All rights reserved.

import pytest

from hat.models.task_modules.ipm_seg import IPMSegTarget
from tests.utils import gen_fake_seg_data


@pytest.mark.parametrize(
    ["strides"],
    [
        pytest.param([1, 1, 1]),
    ],
)
def test_seg_target(strides):
    w, h = 896, 896
    feats, label = gen_fake_seg_data(
        w, h, strides=strides, num_classes=32, n=1, label_name="gt"
    )
    seg_target = IPMSegTarget(label_name="gt")
    loss_inputs = seg_target(label, feats)
    for loss_input in loss_inputs:
        assert len(loss_input["pred"]) == len(loss_input["target"])
        assert (
            loss_input["target"][0].shape[-2:]
            == loss_input["pred"][0].shape[-2:]
        )
