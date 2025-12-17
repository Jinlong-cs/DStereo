# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import pytest
import torch
from horizon_plugin_pytorch import qat_mode

from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.models.base_modules.basic_mixvargenet_module import (
    BasicMixVarGEBlock,
    MixVarGEBlock,
)


@pytest.mark.parametrize(
    ["config_i"],
    [
        pytest.param([[32, 32], "mixvarge_f2", [], 0, 1]),
        pytest.param(
            [[32, 32], "mixvarge_f4", ["mixvarge_f4", "mixvarge_f4"], 2, 2]
        ),
        pytest.param(
            [[32, 64], "mixvarge_f4", ["mixvarge_f4", "mixvarge_f4"], 2, 2]
        ),
        pytest.param(
            [
                [64, 96],
                "mixvarge_f2_gb16",
                [
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                ],
                6,
                2,
            ]
        ),
        pytest.param(
            [
                [96, 160],
                "mixvarge_f2_gb16",
                ["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                2,
                2,
            ]
        ),
    ],
)
def test_mixvargenet_block_without_fusion(config_i):
    data_shape = [1, config_i[0][0], 32, 32]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    bvg_module = MixVarGEBlock(
        in_ch=config_i[0][0],
        block_ch=config_i[0][1],
        head_op=config_i[1],
        stack_ops=config_i[2],
        stride=config_i[-1],
        bias=True,
        bn_kwargs={},
    )
    output, downsampled = bvg_module(fake_data)
    assert output is not None
    assert len(downsampled) == 0


@pytest.mark.parametrize(
    ["config_i"],
    [
        pytest.param([[32, 32], "mixvarge_f2", [], 0, 1, [], 0]),
        pytest.param(
            [
                [32, 32],
                "mixvarge_f4",
                ["mixvarge_f4", "mixvarge_f4"],
                2,
                2,
                [],
                2,
            ]
        ),
        pytest.param(
            [
                [32, 64],
                "mixvarge_f4",
                ["mixvarge_f4", "mixvarge_f4"],
                2,
                2,
                [32],
                2,
            ]
        ),
        pytest.param(
            [
                [64, 96],
                "mixvarge_f2_gb16",
                [
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                    "mixvarge_f2_gb16",
                ],
                6,
                2,
                [32, 64],
                2,
            ]
        ),
        pytest.param(
            [
                [96, 160],
                "mixvarge_f2_gb16",
                ["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                2,
                2,
                [64, 96],
                0,
            ]
        ),
    ],
)
def test_mixvargenet_block_with_fusion(config_i):
    data_shape = [1, config_i[0][0], 64, 64]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )

    if len(config_i[5]) > 0:
        fake_data = [fake_data]
        for input_channel in config_i[5]:
            fusion_input_shape = [
                1,
                input_channel,
                64 // config_i[4],
                64 // config_i[4],
            ]
            fake_data.append(
                torch.tensor(
                    np.random.random(size=fusion_input_shape),
                    dtype=torch.float,
                )
            )

    bvg_module = MixVarGEBlock(
        in_ch=config_i[0][0],
        block_ch=config_i[0][1],
        head_op=config_i[1],
        stack_ops=config_i[2],
        stride=config_i[4],
        bias=True,
        fusion_channels=config_i[5],
        downsample_num=config_i[6],
        bn_kwargs={},
    )
    output, downsampled = bvg_module(fake_data)
    assert output is not None
    assert len(downsampled) == config_i[6]


@pytest.mark.parametrize(
    ["ic", "oc", "s", "c1k", "c2k", "c1g", "c2g"],
    [
        pytest.param(16, 32, 1, {3: 1, 1: 1}, {5: 2, 3: 3, 1: 1}, 8, None),
        pytest.param(16, 32, 2, {3: 2, 1: 1}, {5: 2}, 8, None),
        pytest.param(32, 16, 2, {3: 1, 1: 1}, {5: 2, 3: 3, 1: 1}, 4, 16),
    ],
)
def test_rep_basicmixvarg(ic, oc, s, c1k, c2k, c1g, c2g):
    from horizon_plugin_pytorch import qat_mode

    qat_mode.set_qat_mode(qat_mode.QATMode.WithBN)

    block = BasicMixVarGEBlock(
        in_channels=ic,
        out_channels=oc,
        stride=s,
        conv1_kernel_size=c1k,
        conv2_kernel_size=c2k,
        conv1_group_base=c1g,
        conv2_group_base=c2g,
        bias=True,
        bn_kwargs={},
        use_reparam=True,
    )
    x = torch.randn((2, ic, 32, 64), dtype=torch.float32)
    y = block(x)
    out_shape = (2, oc, 32 // s, 64 // s)
    assert y.shape == out_shape
    block.conv[0].create_merge_conv()
    block.conv[1].create_merge_conv()
    assert isinstance(block.conv[0].merge_conv, torch.nn.Conv2d)
    assert isinstance(block.conv[1].merge_conv, torch.nn.Conv2d)
    y_merge = block(x)
    assert (y - y_merge).abs().sum() == 0.0

    block.fuse_model()
    y_fuse = block(x)
    assert (y - y_fuse).abs().sum() == 0.0
    qat_mode.set_qat_mode(qat_mode.QATMode.FuseBN)


@pytest.mark.parametrize(
    ["cfg"],
    [
        pytest.param(
            MixVarGENetConfig(
                in_channels=32,
                out_channels=32,
                head_op="repmixvarge_f2",
                stack_ops=[],
                stack_factor=2,
                stride=1,
            )
        ),
        pytest.param(
            MixVarGENetConfig(
                in_channels=32,
                out_channels=32,
                head_op="repmixvarge_k3k3_f4",
                stack_ops=["repmixvarge_k3k3_f4", "repmixvarge_k3k3_f4"],
                stack_factor=1,
                stride=2,
            )
        ),
    ],
)
def test_rep_mixvarg_block(cfg):
    qat_mode.set_qat_mode(qat_mode.QATMode.WithBN)

    block = MixVarGEBlock(
        in_ch=cfg.in_channels,
        block_ch=cfg.out_channels,
        head_op=cfg.head_op,
        stack_ops=cfg.stack_ops,
        stride=cfg.stride,
        bias=True,
        bn_kwargs={},
        stack_factor=cfg.stack_factor,
        fusion_channels=[],
        downsample_num=0,
        output_downsample=False,
    )
    assert isinstance(block, torch.nn.Module)

    x = torch.randn((2, cfg.in_channels, 16, 32), dtype=torch.float32)
    y = block(x)
    out_shape = (2, cfg.out_channels, 16 // cfg.stride, 32 // cfg.stride)
    assert y.shape == out_shape
    block.fuse_model()
    y_fuse = block(x)
    assert (y - y_fuse).abs().sum() == 0.0
    qat_mode.set_qat_mode(qat_mode.QATMode.FuseBN)
