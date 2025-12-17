import logging
from typing import Dict, List, Tuple, Union

import horizon_plugin_pytorch.nn as hnn
import torch
from torch import nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import SeparableConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "FPEM",
    "FPEM_FFM",
]


logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class FPEM(nn.Module):
    """
    Eye ldmk FPEM neck module for eye status. \
        Note: FPEM(Feature Pyramid Enhancement Module).

    Args:
        bn_kwargs: Dict BN layer.
        in_channels: In channels of the module.
        act_type: NonLinear type.
        inplace: Whether to perform inplace.
    """

    def __init__(
        self,
        bn_kwargs: Dict,
        act_type: str = "relu",
        in_channels: int = 128,
        inplace: bool = True,
    ):
        super(FPEM, self).__init__()
        if act_type == "relu":
            act_layer = nn.ReLU
        else:
            raise TypeError(f"not support {act_type} act type")

        self.up_add1 = UpsampleSeparableConvModule2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            up_stride=2,
            pw_norm_layer=nn.BatchNorm2d(
                in_channels,
                **bn_kwargs,
            ),
            pw_act_layer=act_layer(inplace=inplace),
        )
        self.up_add2 = UpsampleSeparableConvModule2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            up_stride=2,
            padding=1,
            pw_norm_layer=nn.BatchNorm2d(
                in_channels,
                **bn_kwargs,
            ),
            pw_act_layer=act_layer(inplace=inplace),
        )
        self.up_add3 = UpsampleSeparableConvModule2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            pw_norm_layer=nn.BatchNorm2d(
                in_channels,
                **bn_kwargs,
            ),
            pw_act_layer=act_layer(inplace=inplace),
        )
        self.down_add1 = UpsampleSeparableConvModule2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            pw_norm_layer=nn.BatchNorm2d(
                in_channels,
                **bn_kwargs,
            ),
            pw_act_layer=act_layer(inplace=inplace),
        )
        self.down_add2 = UpsampleSeparableConvModule2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            pw_norm_layer=nn.BatchNorm2d(
                in_channels,
                **bn_kwargs,
            ),
            pw_act_layer=act_layer(inplace=inplace),
        )
        self.down_add3 = UpsampleSeparableConvModule2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            pw_norm_layer=nn.BatchNorm2d(
                in_channels,
                **bn_kwargs,
            ),
            pw_act_layer=act_layer(inplace=inplace),
        )

    def forward(
        self,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        c5: torch.Tensor,
    ):
        # up stage
        c4 = self.up_add1(c5, c4)
        c3 = self.up_add2(c4, c3)
        c2 = self.up_add3(c3, c2)

        # down stage
        c3 = self.down_add1(c3, c2)
        c4 = self.down_add2(c4, c3)
        c5 = self.down_add3(c5, c4)
        return c2, c3, c4, c5


class UpsampleSeparableConvModule2d(nn.Module):
    """
    Depthwise sparable convolution module.

    Args:
        in_channels: Same as nn.Conv2d.
        out_channels: Same as nn.Conv2d.
        kernel_size: Same as nn.Conv2d.
        stride: Same as nn.Conv2d.
        padding: Same as nn.Conv2d.
        dilation: Same as nn.Conv2d.
        groups: Same as nn.Conv2d.
        bias: Same as nn.Conv2d.
        padding_mode: Same as nn.Conv2d.
        dw_norm_layer: Normalization layer in dw conv.
        dw_act_layer: Activation layer in dw conv.
        pw_norm_layer: Normalization layer in pw conv.
        pw_act_layer: Activation layer in pw conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        up_stride: Union[int, Tuple[int, int]] = 2,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dw_norm_layer: Union[None, nn.Module] = None,
        dw_act_layer: Union[None, nn.Module] = None,
        pw_norm_layer: Union[None, nn.Module] = None,
        pw_act_layer: Union[None, nn.Module] = None,
    ):
        super(UpsampleSeparableConvModule2d, self).__init__()
        self.up = hnn.Interpolate(
            scale_factor=up_stride, recompute_scale_factor=True
        )
        self.conv = SeparableConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            dw_norm_layer=dw_norm_layer,
            dw_act_layer=dw_act_layer,
            pw_norm_layer=pw_norm_layer,
            pw_act_layer=pw_act_layer,
        )
        self.add = nn.quantized.FloatFunctional()

    def forward(self, x, y):
        x = self.up(x)
        x = self.add.add(x, y)
        x = self.conv(x)
        return x


@OBJECT_REGISTRY.register
class FPEM_FFM(nn.Module):
    """
    Eye ldmk FPEM_FFM neck module. \
        Note: FFM(Feature Fusion Module).

    Args:
        bn_kwargs: Dict for BN layer.
        in_channels_list: In channel list of modules.
        out_channels: Out channels of modules.
        fpem_repeat: Module repeat times.
        act_type: NonLinear type.
        inplace: Whether to perform inplace.
        kernel_size: Upsample kernel size.
    """

    def __init__(
        self,
        bn_kwargs: Dict,
        in_channels_list: List[int],
        out_channels: int = 128,
        fpem_repeat: int = 1,
        act_type: str = "relu",
        inplace: bool = True,
        kernel_size: int = 1,
    ):
        assert len(in_channels_list) >= 4, (
            f"in channels list must larger equal than 4, "
            f"but get {len(in_channels_list)}"
        )
        assert kernel_size == 1, "only support kernel size == 1"

        super().__init__()

        if act_type == "relu":
            act_layer = nn.ReLU
        else:
            raise TypeError(f"not support {act_type} act type")

        self.fpem_repeat = fpem_repeat
        # reduce layers
        self.reduce_conv_c2 = ConvModule2d(
            in_channels=in_channels_list[0],
            out_channels=out_channels,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            act_layer=act_layer(inplace=inplace),
        )

        self.reduce_conv_c3 = ConvModule2d(
            in_channels=in_channels_list[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            act_layer=act_layer(inplace=inplace),
        )
        self.reduce_conv_c4 = ConvModule2d(
            in_channels=in_channels_list[2],
            out_channels=out_channels,
            kernel_size=kernel_size,
            norm_layer=nn.BatchNorm2d(
                out_channels,
                **bn_kwargs,
            ),
            act_layer=act_layer(inplace=inplace),
        )
        self.reduce_conv_c5 = ConvModule2d(
            in_channels=in_channels_list[3],
            out_channels=out_channels,
            kernel_size=kernel_size,
            norm_layer=nn.BatchNorm2d(
                out_channels,
                **bn_kwargs,
            ),
            act_layer=act_layer(inplace=inplace),
        )
        self.fpems = nn.ModuleList()
        for _ in range(fpem_repeat):
            self.fpems.append(
                FPEM(
                    bn_kwargs=bn_kwargs,
                    act_type=act_type,
                    in_channels=out_channels,
                    inplace=inplace,
                )
            )
        self.out_conv = ConvModule2d(
            in_channels=out_channels * 4,
            out_channels=16,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True,
        )

        self.up_c5 = hnn.Interpolate(
            scale_factor=8, recompute_scale_factor=True
        )

        self.up_c4 = hnn.Interpolate(
            scale_factor=4, recompute_scale_factor=True
        )
        self.up_c3 = hnn.Interpolate(
            scale_factor=2, recompute_scale_factor=True
        )
        self.merge_cat = nn.quantized.FloatFunctional()
        self.add_c2_list = None
        self.add_c3_list = None
        self.add_c4_list = None
        self.add_c5_list = None

        if fpem_repeat > 1:
            self.add_c2_list = [nn.quantized.FloatFunctional()] * fpem_repeat
            self.add_c3_list = [nn.quantized.FloatFunctional()] * fpem_repeat
            self.add_c4_list = [nn.quantized.FloatFunctional()] * fpem_repeat
            self.add_c5_list = [nn.quantized.FloatFunctional()] * fpem_repeat

    def forward(self, x):

        c2, c3, c4, c5 = x
        # reduce channel
        c2 = self.reduce_conv_c2(c2)
        c3 = self.reduce_conv_c3(c3)
        c4 = self.reduce_conv_c4(c4)
        c5 = self.reduce_conv_c5(c5)

        # FPEM
        for i, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm = self.add_c2_list[i - 1].add(c2_ffm, c2)
                c3_ffm = self.add_c3_list[i - 1].add(c3_ffm, c3)
                c4_ffm = self.add_c4_list[i - 1].add(c4_ffm, c4)
                c5_ffm = self.add_c5_list[i - 1].add(c5_ffm, c5)

        # FFM
        c5 = self.up_c5(c5_ffm)
        c4 = self.up_c4(c4_ffm)
        c3 = self.up_c3(c3_ffm)
        Fy = self.merge_cat.cat([c2_ffm, c3, c4, c5], dim=1)
        y = self.out_conv(Fy)
        return y
