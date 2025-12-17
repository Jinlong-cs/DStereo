from collections import OrderedDict
from typing import Dict, Tuple

import torch.nn as nn
from horizon_plugin_pytorch.nn import Interpolate
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_vargnet_module import BasicVarGBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import (
    SeparableGroupConvModule2d,
)
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class HumanPoseHead(nn.Module):
    """Human landmark head.

    Generate Human landmark output(B, NUM_LDMK * 3, W, H).
    (B, :NUM_LDMK, W, H) represent heatmap,
    (B, NUM_LDMK:NUM_LDMK * 3, W, H) represent offset map.
    This head generation method uses the GRMI. Which is suggested
    by the following paper.
    'Towards Accurate Multi-person Pose Estimation in the Wild'
    https://arxiv.org/abs/1701.01779v2

    Args:
        input_shape: Shape of input feature map.
        num_ldmk: Number of landmark.
        num_conv: Number of conv layer used in head.
        group_base: Number of channels in one group.
        head_conv_method: Method used in conv layer.
            Support "BasicVarGBlock", "SeparableConv", "Conv".
        bn_kwargs: BatchNorm layer's params.
        num_filter: Number of filter used for the input, mid and output layers.
        Except for the predict layer output.
        target_shape: Output shape(h, w). Defaults to None.
            If specified, interpolation is performed on featuremap.
        in_num_filter: Number of input channels. Defaults to -1.
    """

    def __init__(
        self,
        num_ldmk: int,
        num_conv: int,
        group_base: int,
        head_conv_method: str,
        bn_kwargs: Dict,
        num_filter: int,
        target_shape: Tuple[int] = None,
        in_num_filter: int = -1,
    ):
        super().__init__()
        if target_shape is not None:
            assert len(target_shape) == 2
        head = []
        in_num_filter = num_filter if in_num_filter == -1 else in_num_filter
        self.num_ldmk = num_ldmk

        self.dequant = DeQuantStub()
        self.target_shape = target_shape

        if target_shape is not None:
            self.need_upscale = True
            self.interpolate = Interpolate(
                (target_shape[0], target_shape[1]),
                align_corners=False,
            )
        else:
            self.need_upscale = False

        for i in range(num_conv):
            if head_conv_method == "BasicVarGBlock" and group_base > 0:
                head.append(
                    BasicVarGBlock(
                        in_channels=num_filter if i > 0 else in_num_filter,
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

            elif head_conv_method == "SeparableConv" and group_base > 0:
                head.append(
                    SeparableGroupConvModule2d(
                        in_channels=num_filter if i > 0 else in_num_filter,
                        out_channels=num_filter,
                        bias=True,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        factor=1,
                        group_base=group_base,
                        pw_act_layer=nn.ReLU(inplace=True),
                        dw_act_layer=nn.ReLU(inplace=True),
                        pw_norm_layer=nn.BatchNorm2d(
                            num_filter if i > 0 else in_num_filter, **bn_kwargs
                        )
                        if bn_kwargs
                        else None,
                        dw_norm_layer=nn.BatchNorm2d(num_filter, **bn_kwargs)
                        if bn_kwargs
                        else None,
                    )
                )
            elif head_conv_method == "Conv":
                head.append(
                    ConvModule2d(
                        in_channels=num_filter if i > 0 else in_num_filter,
                        out_channels=num_filter,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        norm_layer=nn.BatchNorm2d(num_filter, **bn_kwargs)
                        if bn_kwargs
                        else None,
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
            else:
                raise NotImplementedError()
        self.head = nn.Sequential(*head)

        self.out_conv = ConvModule2d(
            in_channels=num_filter,
            out_channels=num_ldmk * 3,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        if len(self.head) != 0:
            x = self.head(x)

        if self.need_upscale:
            x = self.interpolate(x)

        pred = self.dequant(self.out_conv(x))
        output = OrderedDict(ldmk_pred=pred)
        return output

    def fuse_model(self):
        for module in self.head:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
        self.out_conv.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.head.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.out_conv.qconfig = qconfig_manager.get_default_qat_out_qconfig()
