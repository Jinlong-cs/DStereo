# Copyright (c) Horizon Robotics. All rights reserved.
import pytest

from hat.models.task_modules.deeplab import Deeplabv3plusHead
from tests.unit_tests.models.base import qat_test, qtensor_test
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    ["num_classes", "feat_channels", "dilations", "num_repeats"],
    [
        pytest.param(16, 64, [1, 2, 4, 8], [1, 1, 1, 1]),
        pytest.param(12, 32, [1, 2, 4, 4], [1, 1, 1, 2]),
        pytest.param(1, 64, [1, 4, 8, 16], [1, 1, 1, 1]),
    ],
)
def test_deeplab_head(num_classes, feat_channels, dilations, num_repeats):
    w, h = 512, 1024
    strides = [8, 32]
    channels_strides = [4, 1]
    feats, _, _ = gen_fake_feats(
        w,
        h,
        strides=strides,
        n=2,
        feat_channels=128,
        channels_strides=channels_strides,
    )
    deeplab_head = Deeplabv3plusHead(
        in_channels=128,
        c1_index=-2,
        c1_in_channels=32,
        num_classes=num_classes,
        feat_channels=feat_channels,
        dilations=dilations,
        num_repeats=num_repeats,
    )

    out = deeplab_head(feats)
    assert out.size(1) == num_classes
    assert out.size(2) == h // 8
    assert out.size(3) == w // 8
    # test qat
    feats_qat = qtensor_test(feats)
    qat_test(deeplab_head, feats_qat, with_quantized=False)
