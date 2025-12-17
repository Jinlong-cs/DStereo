# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from hat.models.base_modules.conv_module import ConvModule2d

__all__ = ["HMBackboneModel"]


class HMBackboneModel(nn.Module):
    """A modularized MobileNetV2 which any part of it could be used alone.

    The channels of linearbottlenecks can also be controlled.
    """

    def __init__(
        self,
        name: str,
        alpha: float,
        feat_size: int,
        bn_kwargs: Dict,
        use_bias: bool,
        use_dw_relu: bool,
        in_channel_list: Tuple,
        out_channel_list: Tuple,
        start_point=0,
        end_point=4,
        expand_t=6,
    ):
        super(HMBackboneModel, self).__init__()

        # fea_names = ["feat0", "feat1", "feat2", "feat3", "feat4", "feat5"]
        # self._fea_names = fea_names
        self.features = nn.Sequential()

        # 1/4 for layer2, 1/8 for layer3, 1/16 for layer4, 1/32 for layer5
        in_channel_list_default = [
            [int(x * alpha) for x in [32, 32]],
            [int(x * alpha) for x in [32, 32, 32]],
            [int(x * alpha) for x in [32] + [64] * 4 + [96] * 2],
            [int(x * alpha) for x in [96] + [160] * 3],
            [int(x * alpha) for x in [320] * 3],
            [int(x * alpha) for x in [320] * 2],
        ]
        out_channel_list_default = [
            [int(x * alpha) for x in [32, 32]],
            [int(x * alpha) for x in [32, 32, 32]],
            [int(x * alpha) for x in [64] * 4 + [96] * 3],
            [int(x * alpha) for x in [160] * 3] + [320],
            [int(x * alpha) for x in [320] * 3],
            [int(x * alpha) for x in [320] * 2],
        ]

        if in_channel_list is None:
            in_channel_list = in_channel_list_default
        if out_channel_list is None:
            out_channel_list = out_channel_list_default

        for idx, i in enumerate(range(start_point, end_point)):
            in_channel = in_channel_list[i]
            out_channel = out_channel_list[i]
            assert len(in_channel) == len(out_channel)
            if idx != 0:
                feat_size = out_channel_list[i - 1][-1]

            feature = nn.Sequential()
            for block_id, (in_c, out_c) in enumerate(
                zip(in_channel, out_channel)
            ):
                strides = (2, 2) if block_id == 0 else (1, 1)
                if block_id != 0:
                    feat_size = out_channel[block_id - 1]
                block = LinearBottleneck(
                    feat_size=feat_size,
                    in_channels=in_c,
                    out_channels=out_c,
                    expand_t=expand_t,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    strides=strides,
                    use_bias=use_bias,
                    use_dw_relu=use_dw_relu,
                    bn_kwargs=bn_kwargs,
                    name=f"{name}_stage{i}_block{block_id}",
                )
                feature.add_module(
                    name=f"{name}_stage{i}_block{block_id}", module=block
                )
            self.features.add_module(name=f"{name}_stage{i}", module=feature)

    def forward(self, x):
        x = self.features(x)
        return x

    def fuse_model(self):
        for m in self.features:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


class LinearBottleneck(nn.Module):
    """Return Linear Bottleneck Block for building MobilenetV2.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of out channels.
        expand_t: Layer expansion ratio.
        kernel_size: Kernel used in convolution.
        padding: Pad used in convolution.
        strides: Strides used in convolution.
        bn_kwargs: Kwargs for bn block.
        use_bias: Whether to use bias in convolution.
        use_dw_relu: True means relu is used in depthwise.
    """

    def __init__(
        self,
        name: str,
        feat_size: int,
        in_channels: List,
        out_channels: int,
        expand_t: int,
        kernel_size: Tuple,
        padding: Tuple,
        strides: Tuple,
        bn_kwargs: Dict,
        use_bias=True,
        use_dw_relu=False,
    ):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = False
        if strides == (1, 1) and in_channels == out_channels:
            self.use_shortcut = True

        self.shortcut_add = nn.quantized.FloatFunctional()
        self.features = nn.Sequential()
        self.features.add_module(
            name=f"{name}_conv0",
            module=ConvModule2d(
                in_channels=feat_size,
                out_channels=in_channels * expand_t,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=1,
                bias=use_bias,
                norm_layer=nn.BatchNorm2d(in_channels * expand_t, **bn_kwargs),
                act_layer=nn.ReLU(),
            ),
        )
        self.features.add_module(
            name=f"{name}_conv1",
            module=ConvModule2d(
                in_channels=in_channels * expand_t,
                out_channels=in_channels * expand_t,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                groups=in_channels * expand_t,
                bias=use_bias,
                norm_layer=nn.BatchNorm2d(in_channels * expand_t, **bn_kwargs),
                act_layer=nn.ReLU() if use_dw_relu else None,
            ),
        )
        self.features.add_module(
            name=f"{name}_conv2",
            module=ConvModule2d(
                in_channels=in_channels * expand_t,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=1,
                bias=use_bias,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                act_layer=None,
            ),
        )

    def forward(self, x):
        out = self.features[1](self.features[0](x))
        if self.use_shortcut:
            out = self.shortcut_add.add(self.features[2](out), x)
        else:
            out = self.features[2](out)
        return out

    def fuse_model(self):
        # for qat mappings
        from horizon_plugin_pytorch import quantization

        getattr(self.features, "0").fuse_model()
        getattr(self.features, "1").fuse_model()
        if self.use_shortcut:
            torch.quantization.fuse_modules(
                self,
                ["conv.2.0", "conv.2.1", "skip_add", "act"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )
        else:
            torch.quantization.fuse_modules(
                self,
                ["conv.2.0", "conv.2.1", "act"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
