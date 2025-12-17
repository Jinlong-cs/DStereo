# Copyright (c) Horizon Robotics. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY


class Conv(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size,
        stride,
        conv_layer=nn.Conv2d,
        bias=False,
        **kwargs
    ):
        super(Conv, self).__init__()
        padding = kwargs.get("padding", kernel_size // 2)  # dafault same size

        self.conv = conv_layer(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size,
        stride=1,
        conv_layer=nn.Conv2d,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
        **kwargs
    ):
        super(ConvBlock, self).__init__()
        padding = kwargs.get("padding", kernel_size // 2)  # dafault same size

        self.conv = Conv(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            conv_layer=conv_layer,
        )

        self.norm = norm_layer(planes)
        self.act = act_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.block1 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.block2 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = out + identity
        out = self.act(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(channels // r, channels, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.attn(x) * x


class ASPPBlock(nn.Module):
    """The basic structure of ASPP.

    See Atrous Spatial Pyramid Pooling
    <https://arxiv.org/pdf/1606.00915v2.pdf> for detail.

    Args:
        in_channels: input feature channel number.
        use_se: flag of whether to use SE-block.

    """

    def __init__(self, in_channels, use_se=False):

        super(ASPPBlock, self).__init__()

        self.pre_conv = BasicBlock(in_channels)
        self.conv1x1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            padding=0,
        )
        self.weight = nn.Parameter(torch.randn(in_channels, in_channels, 3, 3))
        self.post_conv = ConvBlock(
            in_channels * 6, in_channels, kernel_size=1, stride=1
        )
        self.use_se = use_se
        self.se = SEBlock(channels=in_channels * 6, r=32)

    def forward(self, x):
        x = self.pre_conv(x)
        branch1x1 = self.conv1x1(x)
        branch1 = F.conv2d(
            x, self.weight, stride=1, bias=None, padding=1, dilation=1
        )
        branch6 = F.conv2d(
            x, self.weight, stride=1, bias=None, padding=6, dilation=6
        )
        branch12 = F.conv2d(
            x, self.weight, stride=1, bias=None, padding=12, dilation=12
        )
        branch18 = F.conv2d(
            x, self.weight, stride=1, bias=None, padding=18, dilation=18
        )
        if self.use_se:
            x = self.post_conv(
                self.se(
                    torch.cat(
                        (x, branch1x1, branch1, branch6, branch12, branch18),
                        dim=1,
                    )
                )
            )
        else:
            x = self.post_conv(
                torch.cat(
                    (x, branch1x1, branch1, branch6, branch12, branch18), dim=1
                )
            )
        return x


@OBJECT_REGISTRY.register_module
class ASPPNeck(nn.Module):
    """Neck based on ASPP Module.

    Args:
        in_channels: input feature channel number.
        block_nums: number of ASPP blocks.
        use_se: flag of whether to use SE-block.

    """

    def __init__(self, in_channels=256, block_nums=3, use_se=False):
        super(ASPPNeck, self).__init__()
        self.asppneck = nn.Sequential(
            *[ASPPBlock(in_channels, use_se=use_se) for _ in range(block_nums)]
        )

    def forward(self, x):
        x = self.asppneck(x)
        return x, None
