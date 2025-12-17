# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import Interpolate

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["GroupFPN"]


@OBJECT_REGISTRY.register
class GroupFPN(nn.Module):
    """FPN use depthwise convolution.

    This FPN module is from GluonFace.

    Args:
        feature_dim: Feature dimensions of mid layer.
        in_channels: Input feature map channels.
            Channels of stride [4, 8, 16, 32].
        min_output_stride: Min output stride of FPN.
        use_group: Whether use group conv in 3*3 conv.
        group_base: Group base of conv.
        bn_kwargs: Params of bn layer.
    """

    def __init__(
        self,
        feature_dim,
        in_channels: List[int],
        min_output_stride: int = 4,
        use_group: bool = True,
        group_base: int = 8,
        bn_kwargs: Optional[Dict[str, float]] = None,
    ):
        super(GroupFPN, self).__init__()
        assert min_output_stride in [4, 8, 16, 32]
        if bn_kwargs is None:
            bn_kwargs = {}
        self.min_output_stride = min_output_stride
        self.in_channels = in_channels
        self.p5_1x1 = ConvModule2d(
            in_channels=in_channels[3],
            out_channels=feature_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            norm_layer=nn.BatchNorm2d(feature_dim, **bn_kwargs),
        )

        if self.min_output_stride < 32:
            self.p5_upscale = Interpolate(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
            self.p4_1x1 = ConvModule2d(
                in_channels=in_channels[2],
                out_channels=feature_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                norm_layer=nn.BatchNorm2d(feature_dim, **bn_kwargs),
            )
            self.conv_add1 = nn.quantized.FloatFunctional()
            self.relu1 = nn.ReLU(inplace=True)
            self.p4_3x3 = ConvModule2d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=int(feature_dim / group_base) if use_group else 1,
                norm_layer=nn.BatchNorm2d(feature_dim, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            )

        if self.min_output_stride < 16:
            self.p4_upscale = Interpolate(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
            self.p3_1x1 = ConvModule2d(
                in_channels=in_channels[1],
                out_channels=feature_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                norm_layer=nn.BatchNorm2d(feature_dim, **bn_kwargs),
            )
            self.conv_add2 = nn.quantized.FloatFunctional()
            self.relu2 = nn.ReLU(inplace=True)
            self.p3_3x3 = ConvModule2d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=int(feature_dim / group_base) if use_group else 1,
                norm_layer=nn.BatchNorm2d(feature_dim, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            )
        if self.min_output_stride < 8:
            self.p3_upscale = Interpolate(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
            self.p2_1x1 = ConvModule2d(
                in_channels=in_channels[0],
                out_channels=feature_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                norm_layer=nn.BatchNorm2d(feature_dim, **bn_kwargs),
            )
            self.conv_add3 = nn.quantized.FloatFunctional()
            self.relu3 = nn.ReLU(inplace=True)
            self.p2_3x3 = ConvModule2d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=int(feature_dim / group_base) if use_group else 1,
                norm_layer=nn.BatchNorm2d(feature_dim, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(self.in_channels) >= 4
        if len(features) == 4:
            s2 = features[0]  # stride4
            s3 = features[1]  # stride8
            s4 = features[2]  # stride16
            s5 = features[3]  # stride32
        elif len(features) == 5:
            s2 = features[1]  # stride4
            s3 = features[2]  # stride8
            s4 = features[3]  # stride16
            s5 = features[4]  # stride32
        else:
            raise Exception("please check backbone output.")

        # p5, stride32
        p5_1x1 = self.p5_1x1(s5)

        # p4, stride16
        if self.min_output_stride < 32:
            p5_upscale = self.p5_upscale(p5_1x1)
            s4 = self.p4_1x1(s4)
            p4 = self.conv_add1.add(s4, p5_upscale)
            p4_3x3 = self.p4_3x3(self.relu1(p4))
        else:
            p4_3x3 = None

        # p3, stride8
        if self.min_output_stride < 16:
            p4_upscale = self.p4_upscale(p4_3x3)
            s3 = self.p3_1x1(s3)
            p3 = self.conv_add2.add(s3, p4_upscale)
            p3_3x3 = self.p3_3x3(self.relu2(p3))
        else:
            p3_3x3 = None

        # p2, stride4
        if self.min_output_stride < 8:
            p3_upscale = self.p3_upscale(p3_3x3)
            s2 = self.p2_1x1(s2)
            p2 = self.conv_add3.add(s2, p3_upscale)
            p2_3x3 = self.p2_3x3(self.relu3(p2))
        else:
            p2_3x3 = None

        return [p2_3x3, p3_3x3, p4_3x3, p5_1x1]

    def fuse_model(self):
        from horizon_plugin_pytorch import quantization

        self.p5_1x1.fuse_model()
        torch.quantization.fuse_modules(
            self,
            [
                "p4_1x1.0",  # conv
                "p4_1x1.1",  # bn
                "conv_add1",  # add
                "relu1",  # relu
            ],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )
        self.p4_3x3.fuse_model()
        torch.quantization.fuse_modules(
            self,
            [
                "p3_1x1.0",  # conv
                "p3_1x1.1",  # bn
                "conv_add2",  # add
                "relu2",  # relu
            ],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )
        self.p3_3x3.fuse_model()
        torch.quantization.fuse_modules(
            self,
            [
                "p2_1x1.0",  # conv
                "p2_1x1.1",  # bn
                "conv_add3",  # add
                "relu3",  # relu
            ],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )
        self.p2_3x3.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
