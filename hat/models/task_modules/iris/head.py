# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Optional, Tuple, Union

import torch.nn as nn

from hat.models.base_modules.conv_module import ConvModule2d


class IrisSingleBranchHead(nn.Module):
    """Single branch head for Iris visibility.

    Args:
        bn_kwargs: Dict for BN layer.
        alpha: Alpha for mobilenetv2.
        bias: Whether to use bias in module.
        classifier_num: Num classes of output layer.
        start_stage: Which stage to start.
        end_stage: Which stage to end.
        use_pool: Whether to use pool in the out layer.
        in_chls: A list of channels for every module to input.
        out_chls: A list of channels for every module to output.
        pre_channels: Multiple to in_ch.
    """

    def __init__(
        self,
        bn_kwargs: dict,
        alpha: float = 1.0,
        bias: bool = True,
        classifier_num: int = 2,
        start_stage: int = 0,
        end_stage: int = 4,
        use_pool: bool = True,
        in_chls: Optional[list] = None,
        out_chls: Optional[list] = None,
        pre_channels: int = 4,
    ):
        super(IrisSingleBranchHead, self).__init__()
        self.alpha = alpha
        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.start_stage = start_stage
        self.end_stage = end_stage
        self.use_pool = use_pool

        in_chls_defalut = [
            [32] * 2,
            [32] * 3,
            [32] + [64] * 4 + [96] * 2,
            [96] + [160] * 3,
            [320] * 3,
            [320] * 2,
        ]
        out_chls_defalut = [
            [32] * 2,
            [32] * 3,
            [64] * 4 + [96] * 3,
            [160] * 3 + [320],
            [320] * 3,
            [320] * 2,
        ]
        in_chls = in_chls_defalut if in_chls is None else in_chls
        out_chls = out_chls_defalut if out_chls is None else out_chls

        self.feature = []
        expand_t = 6
        for index in range(start_stage, end_stage):
            in_channel = in_chls[index]
            out_channel = out_chls[index]
            assert len(in_channel) == len(
                out_channel
            ), "length of in_ch mush equal to length of out_ch"
            feature = []
            for block_id, (in_c, out_c) in enumerate(
                zip(in_channel, out_channel)
            ):
                strides = (2, 2) if block_id == 0 else (1, 1)
                feature.append(
                    LinearBottleneck(
                        in_ch=in_c,
                        out_ch=out_c,
                        expand_t=expand_t,
                        bn_kwargs=bn_kwargs,
                        kernel_size=(3, 1),
                        padding=(1, 0),
                        strides=strides,
                        bias=bias,
                        pre_channels=pre_channels,
                    )
                )
                pre_channels = 1
                feature.append(
                    LinearBottleneck(
                        in_ch=out_c,
                        out_ch=out_c,
                        expand_t=expand_t,
                        bn_kwargs=bn_kwargs,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        strides=strides,
                        bias=bias,
                        pre_channels=pre_channels,
                    )
                )
            self.feature.append(nn.Sequential(*feature))
        self.feature = nn.Sequential(*self.feature)

        out_layer = []
        if use_pool:
            out_layer.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        out_layer.append(
            ConvModule2d(
                in_channels=out_c,
                out_channels=classifier_num,
                kernel_size=1,
                stride=2,
                padding=0,
            )
        )
        self.output = nn.Sequential(*out_layer)

    def forward(self, x):
        out = self.feature(x)
        out = self.output(out)
        return out

    def fuse_model(self):
        for module in self.feature:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.output[-1].qconfig = qconfig_manager.get_default_qat_out_qconfig()


class LinearBottleneck(nn.Module):
    """A module of inverted residual.

    Kernel_size, padding can be specify,
    which is different of InvertedResidual.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of out channels.
        expand_t: Layer expansion ratio.
        bn_kwargs: Kwargs for bn block.
        kernel_size: Kernel used in convolution.
        padding: Pad used in convolution.
        strides: Strides used in convolution.
        bias: Whether to use bias in convolution.
        pre_channels: Multiple to in_ch.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_t: int,
        bn_kwargs: dict,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]],
        bias: bool = True,
        pre_channels: int = 1,
    ):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = strides == (1, 1) and in_ch == out_ch
        mid_channels = int(in_ch * expand_t)
        self.conv = nn.Sequential(
            ConvModule2d(
                in_channels=pre_channels * in_ch,
                out_channels=mid_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(mid_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                groups=mid_channels,
                bias=bias,
                norm_layer=nn.BatchNorm2d(mid_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=mid_channels,
                out_channels=out_ch,
                kernel_size=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(out_ch, **bn_kwargs),
            ),
        )
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        data = x
        if self.use_shortcut:
            out = self.skip_add.add(self.conv(x), data)
        else:
            out = self.conv(x)
        return out

    def fuse_model(self):
        # for qat mappings
        for m in self.conv[:-1]:
            if hasattr(m, "fuse_model"):
                m.fuse_model()
