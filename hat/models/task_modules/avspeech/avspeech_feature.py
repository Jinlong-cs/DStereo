# Copyright (c) Horizon Robotics, All rights reserved.

from torch import nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "VideoInterFrameEncoder",
    "CocktailFCLayer",
]


@OBJECT_REGISTRY.register
class VideoInterFrameEncoder(nn.Module):
    """VideoInterFrameEncoder.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        last_layer_with_relu: Whether the last layer uses relu.
        dropout_rate: Dropout rate.
        bn_kwargs: Kwargs of bn layer.
        num_blocks: Number of blocks.
    """

    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        last_layer_with_relu=True,
        dropout_rate=0,
        num_blocks=3,
        bn_kwargs=dict,
    ):
        super(VideoInterFrameEncoder, self).__init__()
        blocks = []
        kernel_size = (1, 3)
        padding = (0, 1)
        for _ in range(num_blocks - 1):
            norm_layer = nn.BatchNorm2d(num_features=256, **bn_kwargs)
            act_layer = nn.ReLU()
            blocks.append(
                ConvModule2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    padding=padding,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )
            blocks.append(nn.Dropout(p=dropout_rate))
            in_channels = out_channels
            norm_layer = nn.BatchNorm2d(num_features=256, **bn_kwargs)
            act_layer = nn.ReLU() if last_layer_with_relu else None
            blocks.append(
                ConvModule2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    padding=padding,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )
            blocks.append(nn.Dropout(p=dropout_rate))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


@OBJECT_REGISTRY.register
class FrequencyOnChannelAudioFeatureExtractor(nn.Module):
    """FrequencyOnChannelAudioFeatureExtractor.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        encoder_kernel_3: Specify the kernel size of the encoder as 3.
        last_layer_with_relu: Whether the last layer uses relu.
        dropout_rate: Dropout rate.
        bn_kwargs: Kwargs of bn layer.
    """

    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        encoder_kernel_3=False,
        last_layer_with_relu=True,
        dropout_rate=0,
        bn_kwargs=dict,
    ):
        super(FrequencyOnChannelAudioFeatureExtractor, self).__init__()
        blocks = []
        if encoder_kernel_3:
            kernel_size = (1, 3)
            padding = (0, 1)
        else:
            kernel_size = (1, 1)
            padding = (0, 0)

        norm_layer = nn.BatchNorm2d(num_features=out_channels, **bn_kwargs)
        act_layer = nn.ReLU()
        blocks.append(
            ConvModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=(1, 1),
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        )
        blocks.append(nn.Dropout(p=dropout_rate))
        blocks.append(
            ConvModule2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
                norm_layer=nn.BatchNorm2d(
                    num_features=out_channels, **bn_kwargs
                ),
                act_layer=nn.ReLU(),
            )
        )
        blocks.append(nn.Dropout(p=dropout_rate))
        blocks.append(
            ConvModule2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
                norm_layer=nn.BatchNorm2d(
                    num_features=out_channels, **bn_kwargs
                ),
                act_layer=nn.ReLU() if last_layer_with_relu else None,
            )
        )
        blocks.append(nn.Dropout(p=dropout_rate))

        self.conv_block = nn.Sequential(*blocks)

    def forward(self, ax):
        xout = self.conv_block(ax)
        return xout


@OBJECT_REGISTRY.register
class CocktailFCLayer(nn.Module):
    """CocktailFCLayer.

    Args:
        in_channels: Input channels.
        num_classes: Number of classes.
        bias: Whether to use bias.
    """

    def __init__(self, in_channels, num_classes, bias):
        super(CocktailFCLayer, self).__init__()
        blocks = []
        blocks.append(
            ConvModule2d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            )
        )
        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        xout = self.conv_block(x)
        return xout
