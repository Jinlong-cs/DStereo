import math
from typing import List

import horizon_plugin_pytorch as horizon
import pytest
import torch

from hat.models.task_modules.semanticfpn import (
    Interpolate_C,
    SemanticFPNplusHead,
)
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "upscale",
        "pixel_shuffle_factor",
        "input_feat_num",
        "feat_channels",
        "out_stride",
    ],
    [
        pytest.param(False, -1, 3, 256, 2),
        pytest.param(True, -1, 3, 256, 2),
        pytest.param(False, -1, 5, 256, 2),
        pytest.param(False, -1, 3, 128, 2),
        pytest.param(False, -1, 3, 256, 1),
        pytest.param(False, 2, 3, 256, None),
    ],
)
def test_semanticfpn_head(
    upscale,
    pixel_shuffle_factor,
    input_feat_num,
    feat_channels,
    out_stride,
):
    cls_num = 5
    batch_size = 2
    in_strides = [2 ** i * 4 for i in range(input_feat_num)]
    resized_h = 512
    resized_w = 1024

    semanticfpn_head = SemanticFPNplusHead(
        num_classes=cls_num,
        in_strides=in_strides,
        in_channels=[feat_channels] * input_feat_num,
        out_stride=out_stride,
        pixel_shuffle_factor=pixel_shuffle_factor,
        upscale=upscale,
    )

    paded_h, paded_w = (
        math.ceil(resized_h / 128) * 128,
        math.ceil(resized_w / 128) * 128,
    )
    fake_neck_feat, _, _ = gen_fake_feats(
        paded_w, paded_h, [4, 8, 16, 32, 64], batch_size, feat_channels
    )

    pred_seg = semanticfpn_head(fake_neck_feat)

    if out_stride is None:
        out_stride = in_strides[0]
        if upscale:
            out_stride //= 2
        if pixel_shuffle_factor > 1:
            out_stride //= pixel_shuffle_factor

    output_h = paded_h // out_stride
    output_w = paded_w // out_stride
    pred_seg_shape = pred_seg[0].shape

    assert isinstance(pred_seg, List)
    assert len(pred_seg) == 1
    assert pred_seg_shape[0] == batch_size
    assert pred_seg_shape[1] == cls_num
    assert pred_seg_shape[2] == output_h
    assert pred_seg_shape[3] == output_w


@pytest.mark.parametrize(
    [
        "size",
        "scale_factor",
        "align_corners",
        "recompute_scale_factor",
    ],
    [
        pytest.param((4, 4), None, True, False),
        pytest.param((4, 4), None, False, False),
        pytest.param(None, (2, 2), True, True),
        pytest.param(None, (2, 2), False, True),
    ],
)
def test_Interpolate_C(
    size, scale_factor, align_corners, recompute_scale_factor
):
    data = torch.randn((1, 1, 2, 2))
    interpolate = Interpolate_C(
        size=size,
        scale_factor=scale_factor,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    horizon_interpolate = horizon.nn.Interpolate(
        size=size,
        scale_factor=scale_factor,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    output1 = interpolate(data)
    output2 = horizon_interpolate(data)
    assert (output1 != output2).sum() == 0
