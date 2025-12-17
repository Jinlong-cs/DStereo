from typing import Dict, Optional

import torch.nn as nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY
from .utils import ConvBlock, get_sizes

__all__ = ["PupilSegEncoder"]


class TransitionDown(nn.Module):
    """The downsampler block of encoder.

    Args:
        in_c: Num of channels for input.
        out_c: Num of channels for output.
        down_size: Coefficient of downsampling.
        bn_kwargs: Dict for BN layer.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        down_size: int,
        bn_kwargs: Optional[Dict] = None,
    ):
        super(TransitionDown, self).__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        if down_size == 0:
            self.layer = nn.Sequential(
                ConvModule2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(out_c, **bn_kwargs),
                    # act_layer=nn.ReLU(inplace=True),
                    act_layer=nn.LeakyReLU(inplace=True),
                )
            )
        else:
            self.layer = nn.Sequential(
                ConvModule2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(out_c, **bn_kwargs),
                    # act_layer=nn.ReLU(inplace=True),
                    act_layer=nn.LeakyReLU(inplace=True),
                ),
                nn.AvgPool2d(kernel_size=down_size),
            )

    def forward(self, x):
        x = self.layer(x)
        return x

    def fuse_model(self):
        for m in self.layer:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


class PupilSegDownBlock(nn.Module):
    """The block of encoder.

    Args:
        in_c: Num of channels for input.
        inter_c: Num of channels for  feat.
        op_c: Num of channels for output.
        down_size: Coefficient of downsampling.
        bn_kwargs: Dict for BN layer.
    """

    def __init__(
        self,
        in_c: int,
        inter_c: int,
        op_c: int,
        down_size: int,
        bn_kwargs: Optional[Dict] = None,
    ):
        super(PupilSegDownBlock, self).__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        self.cat_op = nn.quantized.FloatFunctional()
        self.conv1 = ConvModule2d(
            in_channels=in_c,
            out_channels=inter_c,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_layer=nn.BatchNorm2d(inter_c, **bn_kwargs),
            # act_layer=nn.ReLU(inplace=True),
            act_layer=nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            ConvModule2d(
                in_channels=in_c + inter_c,
                out_channels=inter_c,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=nn.BatchNorm2d(inter_c, **bn_kwargs),
                # act_layer=nn.ReLU(inplace=True),
                act_layer=nn.LeakyReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=inter_c,
                out_channels=inter_c,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d(inter_c, **bn_kwargs),
                # act_layer=nn.ReLU(inplace=True),
                act_layer=nn.LeakyReLU(inplace=True),
            ),
        )
        self.conv3 = nn.Sequential(
            ConvModule2d(
                in_channels=in_c + 2 * inter_c,
                out_channels=inter_c,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=nn.BatchNorm2d(inter_c, **bn_kwargs),
                # act_layer=nn.ReLU(inplace=True),
                act_layer=nn.LeakyReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=inter_c,
                out_channels=inter_c,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d(inter_c, **bn_kwargs),
                # act_layer=nn.ReLU(inplace=True),
                act_layer=nn.LeakyReLU(inplace=True),
            ),
        )
        self.trans_down = TransitionDown(inter_c + in_c, op_c, down_size)

    def forward(self, x):
        x1 = self.conv1(x)
        x21 = self.cat_op.cat((x, x1), dim=1)
        x22 = self.conv2(x21)
        x31 = self.cat_op.cat((x21, x22), dim=1)
        out = self.conv3(x31)
        out = self.cat_op.cat((out, x), dim=1)
        return out, self.trans_down(out)

    def fuse_model(self):
        for module in [
            [self.conv1],
            self.conv2,
            self.conv3,
            [self.trans_down],
        ]:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class PupilSegEncoder(nn.Module):
    """Encoder for the task of pupil segmentation.

    Args:
        in_c: Channels of each input feature map.
        chz: Channels for the module.
        growth: The coefficient of channel expansion.
    """

    def __init__(
        self,
        in_c: int = 1,
        chz: int = 32,
        growth: float = 1.5,
    ):
        super(PupilSegEncoder, self).__init__()
        sizes = get_sizes(chz, growth)
        interSize = sizes["enc"]["inter"]
        opSize = sizes["enc"]["op"]
        ipSize = sizes["enc"]["ip"]

        self.head = ConvBlock(in_c=in_c, inter_c=chz, out_c=chz)
        self.down_block1 = PupilSegDownBlock(
            in_c=int(ipSize[0]),
            inter_c=int(interSize[0]),
            op_c=int(opSize[0]),
            down_size=2,
        )
        self.down_block2 = PupilSegDownBlock(
            in_c=int(ipSize[1]),
            inter_c=int(interSize[1]),
            op_c=int(opSize[1]),
            down_size=2,
        )
        self.down_block3 = PupilSegDownBlock(
            in_c=int(ipSize[2]),
            inter_c=int(interSize[2]),
            op_c=int(opSize[2]),
            down_size=2,
        )
        self.down_block4 = PupilSegDownBlock(
            in_c=int(ipSize[3]),
            inter_c=int(interSize[3]),
            op_c=int(opSize[3]),
            down_size=2,
        )
        self.bottleneck = PupilSegDownBlock(
            in_c=int(opSize[3]),
            inter_c=int(interSize[3]),
            op_c=int(opSize[3]),
            down_size=0,
        )

    def forward(self, x):
        x = self.head(x)  # chz
        skip_1, x = self.down_block1(x)  # chz
        skip_2, x = self.down_block2(x)  # 2 chz
        skip_3, x = self.down_block3(x)  # 4 chz
        skip_4, x = self.down_block4(x)  # 8 chz
        _, x = self.bottleneck(x)
        return skip_4, skip_3, skip_2, skip_1, x

    def fuse_model(self):
        for m in [
            self.head,
            self.down_block1,
            self.down_block2,
            self.down_block3,
            self.down_block4,
            self.bottleneck,
        ]:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
