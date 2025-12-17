# Copyright (c) Horizon Robotics, All rights reserved.


import torch
from torch import nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["VideoAudioDefaultFusion", "AddFusion"]


@OBJECT_REGISTRY.register
class VideoAudioDefaultFusion(nn.Module):
    """VideoAudioDefaultFusion.

    Args:
        fusion_block: Audio and visual feature fusion module.
        lookahead_conv_net: Time series feature extraction network.
    """

    def __init__(
        self,
        fusion_block: nn.Module = None,
        lookahead_conv_net: nn.Module = None,
    ):
        super(VideoAudioDefaultFusion, self).__init__()
        self.fusion_block = fusion_block
        self._va_lookahead_conv = lookahead_conv_net

    def forward(self, ax, vx):
        x = self.fusion_block(ax, vx)
        return self._va_lookahead_conv(x)


@OBJECT_REGISTRY.register
class AddFusion(nn.Module):
    """AddFusion.

    Args:
        bn_kwargs: Kwargs of bn layer.
        v_in_channels: Number of input channels for visual features.
        a_in_channels: Number of input channels for audio features.
        a_v_out_channels: Number of out channels for visual and audio features.
        fusion_out_channels: Number of out channels for fusion features.
        bias: Whether to use bias.
    """

    def __init__(
        self,
        bn_kwargs: dict,
        v_in_channels: int,
        a_in_channels: int,
        a_v_out_channels: int,
        fusion_out_channels: int,
        bias: bool = True,
    ):
        super(AddFusion, self).__init__()
        self.bn_kwargs = bn_kwargs
        self.v_in_channels = v_in_channels
        self.a_in_channels = a_in_channels
        self.a_v_out_channels = a_v_out_channels
        self.fusion_out_channels = fusion_out_channels
        self.bias = bias
        self.mod = torch.nn.quantized.FloatFunctional()
        self.ffn1_1 = ConvModule2d(
            self.a_in_channels,
            self.a_v_out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            bias=self.bias,
            norm_layer=nn.BatchNorm2d(self.a_v_out_channels, **self.bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )
        self.ffn1_2 = ConvModule2d(
            self.v_in_channels,
            self.a_v_out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            bias=self.bias,
            norm_layer=nn.BatchNorm2d(self.a_v_out_channels, **self.bn_kwargs),
            act_layer=nn.ReLU(inplace=True),
        )
        self.ffn2 = ConvModule2d(
            self.a_v_out_channels,
            self.fusion_out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            bias=self.bias,
            norm_layer=nn.BatchNorm2d(
                self.fusion_out_channels, **self.bn_kwargs
            ),
            act_layer=nn.ReLU(inplace=True),
        )

    def forward(self, ax: torch.Tensor, vx: torch.Tensor) -> torch.Tensor:
        ax = self.ffn1_1(ax)
        vx = self.ffn1_2(vx)
        # avx = torch.cat((ax, vx), dim=1)
        avx = self.mod.add(ax, vx)
        xout = self.ffn2(avx)
        return xout


class SimpleMixBlock(nn.Module):
    """Simple Mix Block.

    Args:
        mode (str): Mix mode, "add" or "cat".
        kwargs (dict): Other arguments for the operation.
    """

    MODES = ["add", "cat"]

    def __init__(self, mode: str = "add", **kwargs):
        super().__init__()
        assert mode in self.MODES, f"Unknown Mix Mode: {mode}"
        self.mode = mode
        self.mix_op = nn.quantized.FloatFunctional()
        self.kwargs = kwargs

    def forward(self, fea1, fea2):
        if self.mode == "add":
            return self.mix_op.add(fea1, fea2, **self.kwargs)
        elif self.mode == "cat":
            return self.mix_op.cat((fea1, fea2), **self.kwargs)
        else:
            return None


class CatFusion(nn.Module):
    """Concatenate two features and apply a 1x1 conv.

    Args:
        idim (int): Input feature dimension.
        odim (int): Output feature dimension.
    """

    def __init__(self, idim, odim) -> None:
        super().__init__()
        self.conv = ConvModule2d(idim * 2, odim, 1, bias=True)
        self.mix_op = nn.quantized.FloatFunctional()

    def forward(self, x1, x2):
        x = self.mix_op.cat((x1, x2), dim=1)
        x = self.conv(x)
        return x

    def fuse_model(self):
        self.conv.fuse_model()
