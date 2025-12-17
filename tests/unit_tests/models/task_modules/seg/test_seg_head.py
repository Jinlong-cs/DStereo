# Copyright (c) Horizon Robotics. All rights reserved.

import pytest

from hat.models.task_modules.seg import SegHead
from tests.unit_tests.models.base import qat_test, qtensor_test
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "num_classes",
        "feat_channels",
        "stacked_convs",
        "output_with_bn",
        "argmax_output",
        "output_conf",
    ],
    [
        pytest.param(16, 64, 4, True, False, False),
        pytest.param(16, 64, 4, True, True, False),
        pytest.param(12, 32, 4, False, False, False),
        pytest.param(12, 32, 4, False, True, False),
        pytest.param(1, 64, 2, True, False, False),
        pytest.param(1, 64, 2, True, True, True),
    ],
)
def test_seg_head(
    num_classes,
    feat_channels,
    stacked_convs,
    output_with_bn,
    argmax_output,
    output_conf,
):
    in_strides_list = [
        [4],
        [4, 8],
        [4, 8, 16, 32, 64],
        [4, 8, 16, 32, 64],
        [4, 8, 16, 32, 64],
    ]
    out_strides_list = [[4], [4], [2, 4, 8, 16, 32, 64], [8], [2]]
    upscale_list = [False, False, True, False, True]
    upscale_stride_list = [4, 4, 4, 4, 4]
    for in_strides, out_strides, upscale, upscale_stride in zip(
        in_strides_list, out_strides_list, upscale_list, upscale_stride_list
    ):
        # generate fake feats
        w, h = 640, 640
        feats, stride2channels, stride2hw = gen_fake_feats(w, h, in_strides)
        seg_head = SegHead(
            num_classes,
            in_strides,
            out_strides,
            stride2channels,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs,
            argmax_output=argmax_output,
            dequant_output=False if argmax_output else True,
            upscale=upscale,
            upscale_stride=upscale_stride,
            output_with_bn=output_with_bn,
            output_conf=output_conf,
        )

        assert len(seg_head.seg_convs) == len(out_strides)
        assert (
            len(seg_head.seg_convs["stride" + str(out_strides[0])])
            == stacked_convs
        )
        assert len(seg_head.output_convs) == len(out_strides)
        # do forward and check sometings
        assert len(feats) == len(in_strides)
        outs = seg_head(feats)
        assert len(outs) == len(out_strides)
        if upscale:
            assert outs[0].shape[2:] == (
                h * 2 // upscale_stride,
                w * 2 // upscale_stride,
            )
            outs.pop(0)
            out_strides.pop(0)
        for ii, out in enumerate(outs):
            assert out.shape[2:] == stride2hw[out_strides[ii]]
            if argmax_output and output_conf:
                assert out.size(1) == 2
            elif argmax_output:
                assert out.size(1) == 1
            else:
                assert out.size(1) == num_classes
        # test qat
        feats_qat = [qtensor_test(feat) for feat in feats]
        qat_test(seg_head, feats_qat, with_quantized=False)
