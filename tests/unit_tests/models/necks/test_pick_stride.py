# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.necks.pick_stride import PickStrideNeck


@pytest.mark.parametrize(
    [
        "in_strides",
        "in_channels",
        "out_strides",
    ],
    [
        pytest.param([8, 16, 32], [64, 128, 256], [8, 16, 32]),
    ],
)
def test_pick_stride_neck(in_strides, in_channels, out_strides):
    fix_channel = PickStrideNeck(
        in_strides=in_strides,
        out_strides=out_strides,
    )
    input_size = 512
    x = [
        torch.randn(1, in_ch, input_size // in_st, input_size // in_st)
        for (in_ch, in_st) in zip(in_channels, in_strides)
    ]
    outputs = fix_channel(x)
    assert isinstance(outputs, list) and len(outputs) == len(out_strides)
    for (i, output) in enumerate(outputs):
        assert isinstance(output, torch.Tensor)
        assert len(output.shape) == 4
        assert output.shape[-1] == input_size // out_strides[i]
