from collections import OrderedDict
from typing import Dict, Optional

import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_vargnet_module import BasicVarGBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import SeparableConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["PersonPostionHead"]


@OBJECT_REGISTRY.register
class PersonPostionHead(nn.Module):
    """Person Position Head.

    Args:
        head_add_conv: Whether add extra conv layer on head.
        dms_position_num_classes: Class number of DMS person position.
        oms_position_num_classes: Class number of OMS person position.
        num_conv: Conv layer number in head. Defaults to 0.
        num_filter: Channel number in mid layer in head. Defaults to 256.
        in_num_filter: Channel number of input feature map. Defaults to 128.
        feature_size: Size of feature map. Defaults to 8.
        bn_kwargs: Bn layer parameters. Defaults to None.
    """

    def __init__(
        self,
        head_add_conv: bool,
        dms_position_num_classes: int,
        oms_position_num_classes: int,
        num_conv: int = 0,
        num_filter: int = 256,
        in_num_filter: int = 128,
        feature_size: int = 8,
        bn_kwargs: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        self.dequant = DeQuantStub()
        position_output = []
        for i in range(num_conv):
            position_output.append(
                BasicVarGBlock(
                    in_channels=in_num_filter if i == 0 else num_filter,
                    mid_channels=num_filter,
                    out_channels=num_filter,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    factor=1,
                    group_base=16,
                    bn_kwargs=bn_kwargs,
                )
            )

        times_downsampling = int(feature_size / 8)
        for i in range(times_downsampling):  # noqa B007
            position_output.append(
                BasicVarGBlock(
                    in_channels=num_filter if num_conv > 0 else in_num_filter,
                    mid_channels=num_filter,
                    out_channels=num_filter,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    factor=1,
                    group_base=16,
                    bn_kwargs=bn_kwargs,
                )
            )
        position_output.append(
            SeparableConvModule2d(
                in_channels=num_filter,
                out_channels=num_filter,
                kernel_size=(4, 4),
                stride=(1, 1),
                padding=(0, 0),
                bias=True,
                dw_norm_layer=nn.BatchNorm2d(num_filter, **bn_kwargs),
                pw_norm_layer=nn.BatchNorm2d(num_filter, **bn_kwargs),
            )
        )
        self.position_output = nn.Sequential(*position_output)

        dms_head = []
        if head_add_conv:
            dms_head.append(
                ConvModule2d(
                    in_channels=num_filter,
                    out_channels=num_filter,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    norm_layer=nn.BatchNorm2d(num_filter, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                )
            )
        dms_head.append(
            ConvModule2d(
                in_channels=num_filter,
                out_channels=dms_position_num_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                norm_layer=nn.BatchNorm2d(
                    dms_position_num_classes, **bn_kwargs
                ),
            )
        )
        oms_head = []
        if head_add_conv:
            oms_head.append(
                ConvModule2d(
                    in_channels=num_filter,
                    out_channels=num_filter,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    norm_layer=nn.BatchNorm2d(num_filter, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                )
            )
        oms_head.append(
            ConvModule2d(
                in_channels=num_filter,
                out_channels=oms_position_num_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                norm_layer=nn.BatchNorm2d(
                    oms_position_num_classes, **bn_kwargs
                ),
            )
        )
        self.dms_head = nn.Sequential(*dms_head)
        self.oms_head = nn.Sequential(*oms_head)

    def forward(self, x):
        position_output = self.position_output(x)
        dms_output = self.dms_head(position_output)
        oms_output = self.oms_head(position_output)
        dms_output = self.dequant(dms_output)
        oms_output = self.dequant(oms_output)
        output = OrderedDict(pred_oms=oms_output, pred_dms=dms_output)
        return output

    def fuse_model(self):
        for module in self.position_output:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
        for module in self.dms_head:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
        for module in self.oms_head:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.oms_head[
            -1
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
        self.dms_head[
            -1
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
