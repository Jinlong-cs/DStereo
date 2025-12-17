# Copyright (c) Horizon Robotics. All rights reserved.
import pytest

from hat.models.task_modules.centernet import CenterNetHead
from tests.unit_tests.models.base import qat_test, qtensor_test
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "num_classes",
        "strides2levels",
        "in_strides",
        "feat_channels",
        "conv_type",
        "width",
        "height",
    ],
    [
        pytest.param(
            6, {4: 0, 8: 1, 16: 2, 32: 3, 64: 4}, 4, 32, "sep_conv", 704, 576
        ),
        pytest.param(
            1, {4: 0, 8: 1, 16: 2, 32: 3, 64: 4}, 8, 32, "sep_conv", 704, 576
        ),
    ],
)
def test_centernet_head(
    num_classes,
    strides2levels,
    in_strides,
    feat_channels,
    conv_type,
    width,
    height,
):
    channels_strides = None
    feats, _, _ = gen_fake_feats(
        width,
        height,
        strides=[4, 8, 16, 32, 64],
        n=1,
        feat_channels=feat_channels,
        channels_strides=channels_strides,
    )
    centernet_head = CenterNetHead(
        num_classes=num_classes,
        strides2levels=strides2levels,
        in_strides=in_strides,
        feat_channels=feat_channels,
        conv_type=conv_type,
    )

    out = centernet_head(feats)
    assert out["heatmap_pred"].size(1) == num_classes
    assert out["heatmap_pred"].size(2) == height // in_strides
    assert out["heatmap_pred"].size(3) == width // in_strides
    assert out["wh_pred"].size(1) == 2
    assert out["wh_pred"].size(2) == height // in_strides
    assert out["wh_pred"].size(3) == width // in_strides
    assert out["offset_pred"].size(1) == 2
    assert out["offset_pred"].size(2) == height // in_strides
    assert out["offset_pred"].size(3) == width // in_strides
    assert out["mp_heat_pred"].size(1) == num_classes
    assert out["mp_heat_pred"].size(2) == height // in_strides
    assert out["mp_heat_pred"].size(3) == width // in_strides
    # test qat
    feats_qat = qtensor_test(feats)
    qat_test(centernet_head, feats_qat, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
