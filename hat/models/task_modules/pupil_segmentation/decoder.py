from typing import Dict, Optional

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import Interpolate

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY
from .utils import ConvBlock, get_sizes

__all__ = ["PupilSegDecoder"]


class PupilSegUpBlock(nn.Module):
    """The block of encoder.

    Args:
        skip_c: Num of channels for feature of encoder.
        in_c: Num of channels for input.
        out_c: Num of channels for output.
        up_stride: Coefficient of upsampling.
        bn_kwargs: Dict for BN layer.
    """

    def __init__(
        self,
        skip_c: int,
        in_c: int,
        out_c: int,
        up_stride: int,
        bn_kwargs: Optional[Dict] = None,
    ):
        super(PupilSegUpBlock, self).__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        self.conv1 = nn.Sequential(
            ConvModule2d(
                in_channels=skip_c + in_c,
                out_channels=out_c,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=nn.BatchNorm2d(out_c, **bn_kwargs),
                # act_layer=nn.ReLU(inplace=True),
                act_layer=nn.LeakyReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d(out_c, **bn_kwargs),
                # act_layer=nn.ReLU(inplace=True),
                act_layer=nn.LeakyReLU(inplace=True),
            ),
        )
        self.conv2 = nn.Sequential(
            ConvModule2d(
                in_channels=skip_c + in_c + out_c,
                out_channels=out_c,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=nn.BatchNorm2d(out_c, **bn_kwargs),
                # act_layer=nn.ReLU(inplace=True),
                act_layer=nn.LeakyReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d(out_c, **bn_kwargs),
                # act_layer=nn.ReLU(inplace=True),
                act_layer=nn.LeakyReLU(inplace=True),
            ),
        )
        self.cat_op = nn.quantized.FloatFunctional()
        self.up_scale = Interpolate(
            scale_factor=up_stride,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True,
        )

    def forward(self, prev_feature_map: torch.Tensor, x: torch.Tensor):
        x = self.up_scale(x)
        x = self.cat_op.cat((x, prev_feature_map), dim=1)
        x1 = self.conv1(x)
        x2 = self.cat_op.cat((x, x1), dim=1)
        out = self.conv2(x2)
        return out

    def fuse_model(self):
        for module in [self.conv1, self.conv2]:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class PupilSegDecoder(nn.Module):
    """The block of encoder.

    Args:
        chz: Channels for the module.
        out_c: Num of channels for output.
        growth: The coefficient of channel expansion.
        include_top: Whether to include output layer.
    """

    def __init__(
        self, chz: int, out_c: int, growth: float, include_top: bool = True
    ):
        super(PupilSegDecoder, self).__init__()
        sizes = get_sizes(chz, growth)
        skip_size = sizes["dec"]["skip"]
        op_size = sizes["dec"]["op"]
        ip_size = sizes["dec"]["ip"]

        self.up_block4 = PupilSegUpBlock(
            int(skip_size[0]), int(ip_size[0]), int(op_size[0]), 2
        )
        self.up_block3 = PupilSegUpBlock(
            int(skip_size[1]), int(ip_size[1]), int(op_size[1]), 2
        )
        self.up_block2 = PupilSegUpBlock(
            int(skip_size[2]), int(ip_size[2]), int(op_size[2]), 2
        )
        self.up_block1 = PupilSegUpBlock(
            int(skip_size[3]), int(ip_size[3]), int(op_size[3]), 2
        )
        self.final = ConvBlock(chz, chz, out_c, is_top=include_top)

    def forward(self, skip4, skip3, skip2, skip1, x):
        x = self.up_block4(skip4, x)
        x = self.up_block3(skip3, x)
        x = self.up_block2(skip2, x)
        x = self.up_block1(skip1, x)
        o = self.final(x)
        return o

    def fuse_model(self):
        for m in [
            self.up_block4,
            self.up_block3,
            self.up_block2,
            self.up_block1,
            self.final,
        ]:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if hasattr(self.final, "set_qconfig"):
            self.final.set_qconfig()
