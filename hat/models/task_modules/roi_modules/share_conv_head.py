from typing import Dict

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import Interpolate

from hat.models.base_modules.basic_vargnet_module import BasicVarGBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "ShareConv",
]


@OBJECT_REGISTRY.register
class ShareConv(nn.Module):
    """Share conv module.

    Currently it is mainly used in halo full image detection multi-task.

    Args:
        in_num_filter: Input feature dim.
        group_base: Group base used in VargBlock.
        num_conv: The number of conv. Defaults to 4.
        num_filter: Num filter used in mid layer. Defaults to 128.
        shared_conv_method: Conv method. Defaults to "BasicVargBlockDownUp".
            Support ["Conv", "BasicVargBlock", "BasicVargBlockDownUp"] now.
        bn_kwargs: Batch norm parameters. Defaults to None.
    """

    def __init__(
        self,
        in_num_filter: int,
        group_base: int,
        num_conv: int = 4,
        num_filter: int = 128,
        shared_conv_method: str = "BasicVargBlockDownUp",
        bn_kwargs: Dict = None,
    ):
        super().__init__()
        assert shared_conv_method in [
            "Conv",
            "BasicVargBlock",
            "BasicVargBlockDownUp",
        ]
        if bn_kwargs is None:
            bn_kwargs = {}
        self.shared_conv_method = shared_conv_method
        if shared_conv_method == "BasicVargBlockDownUp":
            num_conv_1 = num_conv // 2
            num_conv_2 = num_conv - num_conv_1
            self.shared_conv_1 = []
            self.shared_down_up = []
            self.shared_conv_2 = []
            # conv 1
            for i in range(num_conv_1):  # noqa B007
                self.shared_conv_1.append(
                    BasicVarGBlock(
                        in_channels=in_num_filter if i == 0 else num_filter,
                        mid_channels=num_filter,
                        out_channels=num_filter,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        factor=1,
                        group_base=group_base,
                        bn_kwargs=bn_kwargs,
                    )
                )
            # down up
            self.shared_down_up.append(
                BasicVarGBlock(
                    in_channels=in_num_filter
                    if num_conv_1 == 0
                    else num_filter,
                    mid_channels=num_filter,
                    out_channels=num_filter,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    factor=1,
                    group_base=group_base,
                    bn_kwargs=bn_kwargs,
                    merge_branch=True,
                )
            )
            self.shared_down_up.append(
                BasicVarGBlock(
                    in_channels=num_filter,
                    mid_channels=num_filter,
                    out_channels=num_filter,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    factor=1,
                    group_base=group_base,
                    bn_kwargs=bn_kwargs,
                )
            )
            self.shared_down_up.append(
                Interpolate(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=True,
                )
            )
            self.shared_conv1x1 = ConvModule2d(
                in_channels=in_num_filter if num_conv_1 == 0 else num_filter,
                out_channels=num_filter,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                norm_layer=nn.BatchNorm2d(num_filter, **bn_kwargs),
            )
            # conv 2
            for i in range(num_conv_2):  # noqa B007
                self.shared_conv_2.append(
                    BasicVarGBlock(
                        in_channels=num_filter,
                        mid_channels=num_filter,
                        out_channels=num_filter,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        factor=1,
                        group_base=group_base,
                        bn_kwargs=bn_kwargs,
                    )
                )
            self.shared_conv_1 = nn.Sequential(*self.shared_conv_1)
            self.shared_down_up = nn.Sequential(*self.shared_down_up)
            self.shared_conv_2 = nn.Sequential(*self.shared_conv_2)
            self.conv_add = nn.quantized.FloatFunctional()
            self.relu = nn.ReLU(inplace=True)
        elif shared_conv_method == "BasicVargBlock":
            self.shared_conv = []
            for i in range(num_conv):  # noqa B007
                self.shared_conv.append(
                    BasicVarGBlock(
                        in_channels=in_num_filter if i == 0 else num_filter,
                        mid_channels=num_filter,
                        out_channels=num_filter,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        factor=1,
                        group_base=group_base,
                        bn_kwargs=bn_kwargs,
                    )
                )
            self.shared_conv = nn.Sequential(*self.shared_conv)
        elif shared_conv_method == "Conv":
            self.shared_conv = []
            for i in range(num_conv):  # noqa B007
                self.shared_conv.append(
                    ConvModule2d(
                        in_channels=in_num_filter if i == 0 else num_filter,
                        out_channels=num_filter,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        norm_layer=nn.BatchNorm2d(num_filter, **bn_kwargs),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
            self.shared_conv = nn.Sequential(*self.shared_conv)
        else:
            raise NotImplementedError()

    def forward(self, x):
        if self.shared_conv_method == "BasicVargBlockDownUp":
            x1 = self.shared_conv_1(x)
            x2 = self.shared_down_up(x1)
            x3 = self.shared_conv1x1(x1)
            x3 = self.conv_add.add(x3, x2)
            x3 = self.relu(x3)
            return self.shared_conv_2(x3)
        else:
            return self.shared_conv(x)

    def fuse_model(self):
        from horizon_plugin_pytorch import quantization

        if self.shared_conv_method == "BasicVargBlockDownUp":
            torch.quantization.fuse_modules(
                self,
                [
                    "shared_conv1x1.0",  # conv
                    "shared_conv1x1.1",  # bn
                    "conv_add",  # add
                    "relu",  # relu
                ],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )
            for m in self.shared_conv_1:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()
            for m in self.shared_conv_2:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()
            for m in self.shared_down_up:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()
        else:
            for m in self.shared_conv:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()
