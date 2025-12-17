# Copyright (c) Horizon Robotics. All rights reserved.


import pytest

from hat.models.task_modules.seg import FRCNNSegHead
from tests.unit_tests.models.base import qat_test, qtensor_test
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "num_classes",
        "feat_channels",
        "in_strides",
        "in_channel",
        "out_strides",
        "group_base",
    ],
    [
        pytest.param(
            16,
            16,
            [4, 8, 16, 32, 64],
            [16, 32, 64, 128, 128],
            [1, 2, 4],
            4,
        ),
        pytest.param(1, 16, [4], [16], [4], 4),
        pytest.param(12, 32, [8, 16, 64, 128], [32, 128, 128, 128], [8], 4),
        pytest.param(
            16,
            16,
            [4, 8, 16, 32, 64],
            [16, 32, 64, 128, 128],
            [1, 2, 4],
            8,
        ),
        pytest.param(
            10,
            32,
            [8, 16, 32, 64],
            [32, 64, 128, 128],
            [8],
            8,
        ),
    ],
)
def test_seg_head(
    num_classes, feat_channels, in_strides, in_channel, out_strides, group_base
):
    # generate fake feats
    w, h = 640, 640
    feats, stride2channels, stride2hw = gen_fake_feats(
        w, h, in_strides, feat_channels=feat_channels
    )

    seg_head = FRCNNSegHead(
        group_base=group_base,
        in_strides=in_strides,
        in_channels=in_channel,
        out_strides=out_strides,
        out_channels=[num_classes] * len(out_strides),
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
        argmax_output=False,
        dequant_output=False,
        with_extra_conv=False,
    )

    # do forward and check somethings
    outs = seg_head(feats)
    assert len(outs) == len(out_strides)
    for ii, out in enumerate(outs):
        if out_strides[ii] == 1:
            assert out.shape[2:] == (640, 640)
        elif out_strides[ii] == 2:
            assert out.shape[2:] == (320, 320)
        else:
            assert out.shape[2:] == stride2hw[out_strides[ii]]
        assert out.size(1) == num_classes
    # test qat
    feats_qat = [qtensor_test(feat) for feat in feats]
    qat_test(seg_head, feats_qat, with_quantized=False)
