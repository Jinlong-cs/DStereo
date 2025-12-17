import copy

import pytest
import torch
from torch import nn

from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


def test_stereonet_task():

    loss_weights = [0.3, 0.3, 0.5, 0.5, 1.0]
    maxdisp = 192
    use_bn = True
    bias = False
    bn_kwargs = {}
    refine_levels = 4

    out_channels = [32, 32, 64, 128, 128, 16]

    config = dict(
        type="StereoNet",
        backbone=dict(
            type="StereoNetNeck",
            out_channels=out_channels,
            use_bn=use_bn,
            bias=bias,
            bn_kwargs=bn_kwargs,
            act_type=nn.ReLU(),
        ),
        head=dict(
            type="StereoNetHead",
            maxdisp=maxdisp,
            bn_kwargs=bn_kwargs,
            refine_levels=refine_levels,
        ),
        post_process=dict(
            type="StereoNetPostProcess",
            maxdisp=maxdisp,
        ),
        loss=dict(type="SmoothL1Loss"),
        loss_weights=loss_weights,
    )

    stereonet_model = build_from_registry(config)
    qat_stereonet_model = copy.deepcopy(stereonet_model)

    x = dict(
        img=torch.randn((1, 6, 256, 512), dtype=torch.float),
        gt_disp=torch.randn((1, 256, 512), dtype=torch.float32),
    )

    stereonet_model(x)

    qat_test(qat_stereonet_model, x, with_quantized=False)


def test_stereonetplus_task():

    loss_weights = [1 / 3, 2 / 3]
    maxdisp = 192
    bn_kwargs = {}
    refine_levels = 3

    model = dict(
        type="StereoNetPlus",
        backbone=dict(
            type="MixVarGENet",
            net_config=[
                [
                    MixVarGENetConfig(
                        in_channels=32,
                        out_channels=32,
                        head_op="mixvarge_f2",
                        stack_ops=[],
                        stride=1,
                        stack_factor=1,
                        fusion_strides=[],
                        extra_downsample_num=0,
                    )
                ],
                [
                    MixVarGENetConfig(
                        in_channels=32,
                        out_channels=32,
                        head_op="mixvarge_f4",
                        stack_ops=["mixvarge_f4", "mixvarge_f4"],
                        stride=2,
                        stack_factor=1,
                        fusion_strides=[],
                        extra_downsample_num=0,
                    )
                ],
                [
                    MixVarGENetConfig(
                        in_channels=32,
                        out_channels=64,
                        head_op="mixvarge_f4",
                        stack_ops=["mixvarge_f4", "mixvarge_f4"],
                        stride=2,
                        stack_factor=1,
                        fusion_strides=[],
                        extra_downsample_num=0,
                    )
                ],
                [
                    MixVarGENetConfig(
                        in_channels=64,
                        out_channels=96,
                        head_op="mixvarge_f2_gb16",
                        stack_ops=[
                            "mixvarge_f2_gb16",
                            "mixvarge_f2_gb16",
                            "mixvarge_f2_gb16",
                            "mixvarge_f2_gb16",
                            "mixvarge_f2_gb16",
                            "mixvarge_f2_gb16",
                        ],
                        stride=2,
                        stack_factor=1,
                        fusion_strides=[],
                        extra_downsample_num=0,
                    )
                ],
                [
                    MixVarGENetConfig(
                        in_channels=96,
                        out_channels=160,
                        head_op="mixvarge_f2_gb16",
                        stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                        stride=2,
                        stack_factor=1,
                        fusion_strides=[],
                        extra_downsample_num=0,
                    )
                ],
            ],
            disable_quanti_input=True,
            input_channels=3,
            input_sequence_length=1,
            num_classes=1000,
            bn_kwargs=bn_kwargs,
            include_top=False,
            bias=True,
            output_list=[0, 1, 2, 3, 4],
        ),
        neck=dict(
            type="FPN",
            in_strides=[8, 16, 32],
            in_channels=[64, 96, 160],
            out_strides=[8, 16, 32],
            out_channels=[16, 16, 16],
        ),
        head=dict(
            type="StereoNetHeadPlus",
            maxdisp=maxdisp,
            bn_kwargs=bn_kwargs,
            refine_levels=refine_levels,
        ),
        post_process=dict(
            type="StereoNetPostProcessPlus",
            maxdisp=maxdisp,
        ),
        loss=dict(type="SmoothL1Loss"),
        loss_weights=loss_weights,
    )

    stereonet_model = build_from_registry(model)
    qat_stereonet_model = copy.deepcopy(stereonet_model)

    x = dict(
        img=torch.randn((2, 3, 256, 512), dtype=torch.float),
        gt_disp=torch.randn((1, 256, 512), dtype=torch.float32),
    )

    stereonet_model(x)

    qat_test(qat_stereonet_model, x, with_quantized=False)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
