# Copyright (c) Horizon Robotics. All rights reserved.

import pytest

from hat.models.task_modules.fcn import FCNHead
from tests.unit_tests.models.base import qat_test, qtensor_test
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    ["num_classes", "feat_channels", "num_convs"],
    [
        pytest.param(16, 64, 2),
        pytest.param(12, 32, 1),
        pytest.param(1, 64, 3),
    ],
)
def test_fcn_head(num_classes, feat_channels, num_convs):
    w, h = 512, 1024
    strides = [32]
    feats, _, _ = gen_fake_feats(w, h, strides, n=2, feat_channels=128)
    fcn_head = FCNHead(
        input_index=0,
        in_channels=128,
        feat_channels=feat_channels,
        num_classes=num_classes,
        num_convs=num_convs,
    )

    out = fcn_head(feats)
    assert out.size(1) == num_classes
    assert out.size(2) == h // 32
    assert out.size(3) == w // 32
    # test qat
    feats_qat = qtensor_test(feats)
    qat_test(fcn_head, feats_qat, with_quantized=False)
