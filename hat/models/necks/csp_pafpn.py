from typing import List

import torch
import torch.nn as nn

from hat.models.base_modules.basic_cspdarknet_module import CSPLayer
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.task_modules.semanticfpn.semanticfpn_head import Interpolate_C
from hat.registry import OBJECT_REGISTRY

__all__ = ["CSPPAFPN"]


@OBJECT_REGISTRY.register
class CSPPAFPN(nn.Module):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels: Number of input channels per scale.
        out_channels: Number of output channels (used at each scale)
        num_outs: Number of output strides.
        num_csp_blocks: Number of bottlenecks in CSPLayer.
        use_depthwise: Whether to depthwise separable convolution in blocks.
        act: Activation layer.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        num_csp_blocks: int = 3,
        use_depthwise: bool = False,
        act: str = "silu",
        use_native_op=False,
    ):
        super(CSPPAFPN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.extra_levels = self.num_outs - len(self.in_channels)
        assert self.extra_levels >= 0, (
            f"num_outs should be larger than the length of in_channels, "
            f"but got {self.num_outs}."
        )

        # build top-down blocks
        if use_native_op:
            self.upsample = Interpolate_C(
                scale_factor=(2, 2),
                align_corners=False,
                recompute_scale_factor=True,
            )
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule2d(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    stride=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(in_channels[idx - 1]),
                    act_layer=nn.SiLU(inplace=True),
                )
            )
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    n=num_csp_blocks,
                    depthwise=use_depthwise,
                    act=act,
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                ConvModule2d(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(in_channels[idx]),
                    act_layer=nn.SiLU(inplace=True),
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    n=num_csp_blocks,
                    depthwise=use_depthwise,
                    act=act,
                )
            )

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule2d(
                    in_channels[i],
                    out_channels,
                    1,
                    stride=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(out_channels),
                    act_layer=nn.SiLU(inplace=True),
                )
            )

        self.extra_convs = nn.ModuleList()
        for _ in range(self.extra_levels):
            self.extra_convs.append(
                ConvModule2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(out_channels),
                    act_layer=nn.SiLU(inplace=True),
                )
            )

    def forward(self, inputs):
        """Forward features.

        Args:
            inputs (list[tensor]): Input tensors

        Returns (list[tensor]): Output tensors

        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh
            )
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1)
            )
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        for extra_conv in self.extra_convs:
            outs.append(extra_conv(outs[-1]))
        return outs

    def fuse_model(self):
        for m in self.reduce_layers:
            m.fuse_model()
        for m in self.top_down_blocks:
            m.fuse_model()
        for m in self.downsamples:
            m.fuse_model()
        for m in self.bottom_up_blocks:
            m.fuse_model()
        for m in self.out_convs:
            m.fuse_model()
        for m in self.extra_convs:
            m.fuse_model()
