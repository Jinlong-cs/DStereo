# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.models.necks.ufpn import UFPN
from tests.unit_tests.models.base import qat_test, qtensor_test

group_base = 4


def test_fpn_error():
    # Test len(in_strides) == len(in_channels) == len(out_channels)
    model = None
    with pytest.raises(AssertionError):
        model = UFPN(
            in_strides=[8, 16, 32],
            in_channels=[64, 128, 256],
            out_channels=[32, 64, 128, 256],
            group_base=group_base,
            bn_kwargs={},
        )
    assert model is None

    model = None
    with pytest.raises(AssertionError):
        model = UFPN(
            in_strides=[8, 16, 32],
            in_channels=[64, 128, 256],
            out_channels=[64, 128, 256],
            group_base=group_base,
            bn_kwargs={},
            bottom_proj_blocks=[
                MixVarGENetConfig(
                    in_channels=64,
                    out_channels=128,
                    head_op="mixvarge_f2",
                    stack_ops=[
                        "mixvarge_f2",
                        "mixvarge_f2",
                    ],
                    stack_factor=1,
                    stride=2,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
            ],
        )
    assert model is None


@pytest.mark.parametrize(
    [
        "in_strides",
        "in_channels",
        "out_channels",
        "ds_blocks",
        "bottom_proj_blocks",
        "up_proj_blocks",
        "fusion_kernel_size",
    ],
    [
        pytest.param(
            [8, 16, 32], [64, 128, 256], [64, 128, 256], None, None, None, 3
        ),
        pytest.param(
            [8, 16, 32],
            [64, 128, 256],
            [64, 128, 256],
            [
                MixVarGENetConfig(
                    in_channels=64,
                    out_channels=128,
                    head_op="mixvarge_f2",
                    stack_ops=[
                        "mixvarge_f2",
                        "mixvarge_f2",
                    ],
                    stack_factor=1,
                    stride=2,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
                MixVarGENetConfig(
                    in_channels=128,
                    out_channels=256,
                    head_op="mixvarge_f2",
                    stack_ops=[
                        "mixvarge_f2",
                        "mixvarge_f2",
                    ],
                    stack_factor=1,
                    stride=2,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
            ],
            None,
            None,
            5,
        ),
        pytest.param(
            [8, 16, 32],
            [64, 128, 256],
            [64, 128, 256],
            None,
            [
                MixVarGENetConfig(
                    in_channels=64,
                    out_channels=128,
                    head_op="mixvarge_f2",
                    stack_ops=[
                        "mixvarge_f2",
                        "mixvarge_f2",
                    ],
                    stack_factor=1,
                    stride=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
                MixVarGENetConfig(
                    in_channels=128,
                    out_channels=256,
                    head_op="mixvarge_f2",
                    stack_ops=[
                        "mixvarge_f2",
                        "mixvarge_f2",
                    ],
                    stack_factor=1,
                    stride=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
            ],
            None,
            3,
        ),
        pytest.param(
            [8, 16, 32, 64],
            [16, 32, 64, 128],
            [16, 32, 64, 128],
            [
                MixVarGENetConfig(
                    in_channels=16,
                    out_channels=32,
                    head_op="mixvarge_f2",
                    stack_ops=[
                        "mixvarge_f2",
                        "mixvarge_f2",
                    ],
                    stack_factor=1,
                    stride=2,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=64,
                    head_op="mixvarge_f2",
                    stack_ops=["mixvarge_f2", "mixvarge_f2"],
                    stack_factor=1,
                    stride=2,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
                MixVarGENetConfig(
                    in_channels=64,
                    out_channels=128,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                    stack_factor=1,
                    stride=2,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
            ],
            [
                MixVarGENetConfig(
                    in_channels=16,
                    out_channels=16,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=[
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                    ],
                    stack_factor=1,
                    stride=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=32,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=[
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                    ],
                    stack_factor=1,
                    stride=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
                MixVarGENetConfig(
                    in_channels=64,
                    out_channels=64,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=[
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                    ],
                    stack_factor=1,
                    stride=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
            ],
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=16,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=[
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                    ],
                    stack_factor=1,
                    stride=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
                MixVarGENetConfig(
                    in_channels=64,
                    out_channels=32,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=[
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                    ],
                    stack_factor=1,
                    stride=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
                MixVarGENetConfig(
                    in_channels=128,
                    out_channels=64,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=[
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                    ],
                    stack_factor=1,
                    stride=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                ),
            ],
            5,
        ),
    ],
)
def test_UFPN(
    in_strides,
    in_channels,
    out_channels,
    ds_blocks,
    bottom_proj_blocks,
    up_proj_blocks,
    fusion_kernel_size,
):
    ufpn = UFPN(
        in_strides=in_strides,
        in_channels=in_channels,
        out_channels=out_channels,
        bn_kwargs={},
        group_base=group_base,
        ds_blocks=ds_blocks,
        bottom_proj_blocks=bottom_proj_blocks,
        up_proj_blocks=up_proj_blocks,
        fusion_kernel_size=fusion_kernel_size,
    )
    input_size = 512
    inputs = [
        torch.randn(1, in_ch, input_size // in_st, input_size // in_st)
        for (in_ch, in_st) in zip(in_channels, in_strides)
    ]
    outputs = ufpn(inputs)
    assert isinstance(outputs, list) and len(outputs) == len(out_channels)
    for (i, (output, input)) in enumerate(zip(outputs, inputs)):
        assert isinstance(output, torch.Tensor)
        assert len(output.shape) == 4
        assert output.shape[-2::] == input.shape[-2::]
        assert output.shape[1] == out_channels[i]

    inputs = qtensor_test(inputs)
    qat_test(ufpn, inputs, with_quantized=False)
