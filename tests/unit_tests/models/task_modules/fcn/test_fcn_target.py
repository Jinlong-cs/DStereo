# Copyright (c) Horizon Robotics. All rights reserved.

import pytest

from hat.models.task_modules.fcn import FCNTarget
from tests.utils import gen_fake_torch_randn_data


@pytest.mark.parametrize(
    ["num_classes"],
    [
        pytest.param(16),
        pytest.param(12),
        pytest.param(1),
    ],
)
def test_fcn_target(num_classes):
    w, h = 512, 1024
    label_shape = (2, w, h)
    pred_shape = (2, num_classes, h // 32, w // 32)
    pred = gen_fake_torch_randn_data(pred_shape)
    label = gen_fake_torch_randn_data(label_shape)
    fcn_target = FCNTarget(
        num_classes=num_classes,
    )

    out = fcn_target(label, pred)
    assert out["pred"].size(1) == num_classes
    assert out["pred"].size(2) == label.shape[1]
    assert out["pred"].size(3) == label.shape[2]
