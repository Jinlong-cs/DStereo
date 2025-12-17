# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from hat.models.base_modules.basic_mixvargenet_module import BasicMixVarGEBlock
from hat.models.base_modules.basic_vargnet_module import BasicVarGBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.weight_init import normal_init
from hat.registry import OBJECT_REGISTRY

__all__ = ["FixChannelNeck", "VargFixChannelNeck"]


@OBJECT_REGISTRY.register
class FixChannelNeck(nn.Module):
    """Fix channel neck.

    There will be a 1x1 conv following each output feature map,
    so that each final output has fix_out_channel channels.

    Args:
        in_strides: strides of each input feature map
        in_channels: channels of each input feature map,
            the length of in_channels should be equal to in_strides
        out_strides: strides of each output feature map,
            should be a subset of in_strides, and continuous (any
            subsequence of 2, 4, 8, 16, 32, 64 ...). The largest
            stride in in_strides and out_strides should be equal
        out_channel: channels of each output feature maps
            the length of out_channels should be equal to out_strides
        bn_kwargs: Dict for Bn layer. No Bn layer if
            bn_kwargs=None
    """

    def __init__(
        self,
        in_strides: List[int],
        in_channels: List[int],
        out_strides: List[int],
        out_channel: int,
        bn_kwargs: Optional[Dict] = None,
    ):

        super(FixChannelNeck, self).__init__()
        self._valid_strides = [2, 4, 8, 16, 32, 64, 128, 256]
        self.bn_kwargs = bn_kwargs

        # in_strides check
        assert len(in_strides) == len(in_channels)
        for s in in_strides:
            assert s in self._valid_strides

        min_idx = self._valid_strides.index(in_strides[0])
        max_idx = self._valid_strides.index(in_strides[-1])

        assert tuple(in_strides) == tuple(
            self._valid_strides[min_idx : max_idx + 1]
        ), "Input strides must be continuous and in ascending order"
        self.in_strides = in_strides

        # out_strides check

        assert len(set(out_strides)) == len(out_strides)

        out_indices = []
        for s in out_strides:
            assert s in in_strides
            out_indices.append(in_strides.index(s))

        assert tuple(out_indices) == tuple(
            sorted(out_indices)
        ), "Out strides must be in ascending order"

        self.conv1x1 = nn.ModuleDict()
        for idx, s in zip(out_indices, out_strides):
            self.conv1x1[f"stride_{s}"] = ConvModule2d(
                in_channels=in_channels[idx],
                out_channels=out_channel,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=True,
                norm_layer=None
                if bn_kwargs is None
                else nn.BatchNorm2d(out_channel, **bn_kwargs),
            )
        self.out_strides = out_strides
        self.out_indices = out_indices

    def _init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01, bias=0)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:

        assert len(features) == len(self.in_strides)

        outputs = []
        for idx, s in zip(self.out_indices, self.out_strides):
            outputs.append(self.conv1x1[f"stride_{s}"](features[idx]))

        return outputs

    def fuse_model(self):
        for module in self.conv1x1.values():
            module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class VargFixChannelNeck(nn.Module):
    """Fix channel neck with varg block.

    There will be some varg block layer following each output
    feature map, so that each final output has fix_out_channel
    channels.

    Args:
        in_strides: strides of each input feature map
        in_channels: channels of each input feature map,
            the length of in_channels should be equal to in_strides
        out_strides: strides of each output feature map,
            should be a subset of in_strides, and continuous (any
            subsequence of 2, 4, 8, 16, 32, 64 ...). The largest
            stride in in_strides and out_strides should be equal
        out_channel: channels of each output feature maps
            the length of out_channels should be equal to out_strides.
        bn_kwargs: Dict for Bn layer.
        num_conv_per_stride: number basic vargblock per out stride.
            Default is 2.
        varg_block_type: Choice varg block type from
            ["BasicVarGBlock", "BasicMixVarGEBlock"],
            Default is "BasicMixVarGEBlock".
        group_base: group base for vaBasicVarGBlock.
            Default is 16.
    """

    def __init__(
        self,
        in_strides: List[int],
        in_channels: List[int],
        out_strides: List[int],
        out_channel: int,
        bn_kwargs: Dict,
        num_conv_per_stride: int = 2,
        varg_block_type: str = "BasicMixVarGEBlock",
        group_base: int = 16,
    ):

        super(VargFixChannelNeck, self).__init__()
        self._valid_strides = [2, 4, 8, 16, 32, 64, 128, 256]
        self.bn_kwargs = bn_kwargs

        # in_strides check
        assert len(in_strides) == len(in_channels)
        for s in in_strides:
            assert s in self._valid_strides

        min_idx = self._valid_strides.index(in_strides[0])
        max_idx = self._valid_strides.index(in_strides[-1])

        assert tuple(in_strides) == tuple(
            self._valid_strides[min_idx : max_idx + 1]
        ), "Input strides must be continuous and in ascending order"
        self.in_strides = in_strides

        # out_strides check

        assert len(set(out_strides)) == len(out_strides)

        out_indices = []
        for s in out_strides:
            assert s in in_strides
            out_indices.append(in_strides.index(s))

        assert tuple(out_indices) == tuple(
            sorted(out_indices)
        ), "Out strides must be in ascending order"

        self.conv_blocks = nn.ModuleDict()
        for idx, s in zip(out_indices, out_strides):
            layers = []
            for i in range(num_conv_per_stride):
                if varg_block_type == "BasicVarGBlock":
                    layers.append(
                        BasicVarGBlock(
                            in_channels=in_channels[idx]
                            if i == 0
                            else out_channel,
                            mid_channels=out_channel,
                            out_channels=out_channel,
                            stride=1,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            factor=1,
                            group_base=group_base,
                            merge_branch=True,
                            bn_kwargs=bn_kwargs,
                        )
                    )
                elif varg_block_type == "BasicMixVarGEBlock":
                    layers.append(
                        BasicMixVarGEBlock(
                            in_channels=in_channels[idx]
                            if i == 0
                            else out_channel,
                            out_channels=out_channel,
                            stride=1,
                            bias=True,
                            bn_kwargs=bn_kwargs,
                            factor=1,
                            conv1_group_base=group_base,
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"varg_block_type {varg_block_type} not support now,"
                        "please choice from\
                            [BasicVarGBlock, BasicMixVargNetBlock]"
                    )
            self.conv_blocks[f"stride_{s}"] = nn.Sequential(*layers)
        self.out_strides = out_strides
        self.out_indices = out_indices

    def _init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01, bias=0)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:

        assert len(features) == len(self.in_strides)

        outputs = []
        for idx, s in zip(self.out_indices, self.out_strides):
            outputs.append(self.conv_blocks[f"stride_{s}"](features[idx]))

        return outputs

    def fuse_model(self):
        for mod in self.conv_blocks.values():
            if isinstance(mod, nn.Sequential):
                for m in mod:
                    if hasattr(m, "fuse_model"):
                        m.fuse_model()
            elif hasattr(mod, "fuse_model"):
                mod.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
