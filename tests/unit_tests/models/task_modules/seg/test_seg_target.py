# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.task_modules.seg import SegTarget
from tests.utils import gen_fake_torch_randint_data


@pytest.mark.parametrize(
    ["shape", "label_name"],
    [
        pytest.param((1, 300, 300), "gt_seg"),
        pytest.param((2, 400, 400, 1), "gt_seg_xx"),
    ],
)
def test_seg_target(shape, label_name):
    gt_seg = gen_fake_torch_randint_data(shape)
    label = {label_name: gt_seg}
    N = label[label_name].shape[0]
    pred_list = [
        [torch.randn((N, 15, 150, 150))],
        [torch.randn((N, 15, 100, 100)), torch.randn((N, 15, 200, 200))],
    ]
    seg_target = SegTarget(label_name=label_name)
    for pred in pred_list:
        loss_inputs = seg_target(label, pred)
        assert len(loss_inputs) == len(pred)
        for loss_input in loss_inputs:
            assert loss_input["pred"].shape[2:] == shape[1:3]
