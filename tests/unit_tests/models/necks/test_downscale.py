# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.necks.downscale import ConvDownscaleNeck
from tests.unit_tests.models.base import qat_test, qtensor_test


@pytest.mark.parametrize(
    [
        "in_strides",
        "in_channels",
        "out_strides",
        "out_channels",
    ],
    [
        pytest.param([8, 16, 32], [64, 128, 256], [16, 32, 64], [64, 64, 64]),
    ],
)
def test_ConvDownscaleNeck(in_strides, in_channels, out_strides, out_channels):
    downscale = ConvDownscaleNeck(
        in_strides=in_strides,
        in_channels=in_channels,
        out_strides=out_strides,
        out_channels=out_channels,
        bn_kwargs={},
    )
    input_size = 512
    x = [
        torch.randn(1, in_ch, input_size // in_st, input_size // in_st)
        for (in_ch, in_st) in zip(in_channels, in_strides)
    ]
    outputs = downscale(x)
    assert isinstance(outputs, list) and len(outputs) == len(out_strides)
    for (i, output) in enumerate(outputs):
        assert isinstance(output, torch.Tensor)
        assert len(output.shape) == 4
        assert output.shape[-1] == input_size // out_strides[i]
        assert output.shape[1] == out_channels[i]

    x = qtensor_test(x)
    qat_test(downscale, x, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
