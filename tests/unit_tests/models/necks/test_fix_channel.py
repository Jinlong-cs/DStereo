# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.necks.fix_channel import FixChannelNeck, VargFixChannelNeck
from tests.unit_tests.models.base import qat_test, qtensor_test


def test_fcn_error():
    # Test len(in_strides) == len(in_channels)
    model = None
    with pytest.raises(AssertionError):
        model = FixChannelNeck(
            in_strides=[8, 16, 32],
            in_channels=[128, 256],
            out_strides=[8, 16, 32],
            out_channel=64,
        )

    # Test in_strides must be in [2, 4, 8, 16, 32, 64, 128, 256]
    with pytest.raises(AssertionError):
        model = FixChannelNeck(
            in_strides=[64, 128, 256, 512],
            in_channels=[256, 256, 256, 256],
            out_strides=[64, 128, 256],
            out_channel=64,
        )

    # Test in_strides must be continuous and in ascending order
    with pytest.raises(AssertionError):
        model = FixChannelNeck(
            in_strides=[8, 16, 32, 128],
            in_channels=[64, 128, 256, 256],
            out_strides=[32, 64, 128],
            out_channel=64,
        )

    # Test len(set(out_strides)) == len(out_strides)
    with pytest.raises(AssertionError):
        model = FixChannelNeck(
            in_strides=[8, 16, 32],
            in_channels=[64, 128, 256],
            out_strides=[8, 8, 32],
            out_channel=64,
        )

    # Test all stride of output stride must be in input stride
    with pytest.raises(AssertionError):
        model = FixChannelNeck(
            in_strides=[8, 16, 32],
            in_channels=[64, 128, 256],
            out_strides=[8, 16, 32, 64],
            out_channel=64,
        )
    # Test out_strides must be continuous and in ascending order
    with pytest.raises(AssertionError):
        model = FixChannelNeck(
            in_strides=[8, 16, 32],
            in_channels=[64, 128, 256],
            out_strides=[8, 16, 32, 128],
            out_channel=64,
        )

    assert model is None


@pytest.mark.parametrize(
    [
        "in_strides",
        "in_channels",
        "out_strides",
        "out_channel",
    ],
    [
        pytest.param([8, 16, 32], [64, 128, 256], [8, 16, 32], 64),
    ],
)
def test_FixChannelNeck(in_strides, in_channels, out_strides, out_channel):
    fix_channel = FixChannelNeck(
        in_strides=in_strides,
        in_channels=in_channels,
        out_strides=out_strides,
        out_channel=out_channel,
        bn_kwargs={},
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
        assert output.shape[1] == out_channel

    x = qtensor_test(x)
    qat_test(fix_channel, x, with_quantized=False)


@pytest.mark.parametrize(
    [
        "in_strides",
        "in_channels",
        "out_strides",
        "out_channel",
        "num_conv_per_stride",
        "varg_block_type",
        "group_base",
    ],
    [
        pytest.param(
            [8, 16, 32],
            [64, 128, 256],
            [8, 16, 32],
            64,
            1,
            "BasicVarGBlock",
            8,
        ),
        pytest.param(
            [8, 16, 32], [64, 128, 256], [16], 64, 2, "BasicMixVarGEBlock", 16
        ),
    ],
)
def test_VargFixChannelNeck(
    in_strides,
    in_channels,
    out_strides,
    out_channel,
    num_conv_per_stride,
    varg_block_type,
    group_base,
):
    fix_channel = VargFixChannelNeck(
        in_strides=in_strides,
        in_channels=in_channels,
        out_strides=out_strides,
        out_channel=out_channel,
        num_conv_per_stride=num_conv_per_stride,
        varg_block_type=varg_block_type,
        group_base=group_base,
        bn_kwargs={},
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
        assert output.shape[1] == out_channel

    x = qtensor_test(x)
    qat_test(fix_channel, x, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
