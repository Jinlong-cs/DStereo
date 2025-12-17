# Copyright (c) Horizon Robotics, All rights reserved.

import math

import torch
from torch import nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class AdaptiveLookAheadConv(nn.Module):
    def __init__(
        self,
        bn_kwargs: dict,
        num_cached_frames=15,
        num_channels=512,
        channels_factor=1.0,
        num_cached_padding=0,
        bias: bool = True,
        disable_quanti_input: bool = False,
        use_stride_2_conv=True,
    ):

        super(AdaptiveLookAheadConv, self).__init__()
        self.disable_quanti_input = disable_quanti_input
        stride_size = 2 if use_stride_2_conv else 1
        kernels = []
        strides = []
        paddings = []
        lefted_size = num_cached_frames
        layers = []
        while lefted_size >= 3:
            kernels.append(3)
            strides.append(stride_size)
            paddings.append(num_cached_padding)
            lefted_size = math.ceil(
                (lefted_size + 2 * num_cached_padding - 2) / stride_size
            )
        if lefted_size > 1:
            kernels.append(lefted_size)
            strides.append(1)
            paddings.append(0)
        for i, (k, s, p) in enumerate(zip(kernels, strides, paddings)):
            if i < len(kernels):
                layers.append(
                    ConvModule2d(
                        num_channels,
                        num_channels,
                        (1, k),
                        (1, s),
                        (0, p),
                        bias=bias,
                        norm_layer=nn.BatchNorm2d(num_channels, **bn_kwargs),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
            self.out_channels = num_channels
            num_channels = int(num_channels * channels_factor)
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        return x
