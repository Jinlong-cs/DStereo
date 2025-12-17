from typing import Dict, Optional

import numpy as np
import torch.nn as nn

from hat.models.base_modules.conv_module import ConvModule2d

__all__ = ["get_sizes", "ConvBlock"]


def get_sizes(chz, growth, blks=4):
    # This function does not calculate the size requirements for head and tail

    # Encoder sizes
    sizes = {
        "enc": {"inter": [], "ip": [], "op": []},
        "dec": {"skip": [], "ip": [], "op": []},
    }
    sizes["enc"]["inter"] = np.array([chz * (i + 1) for i in range(0, blks)])
    sizes["enc"]["op"] = np.array(
        [np.int(growth * chz * (i + 1)) for i in range(0, blks)]
    )
    sizes["enc"]["ip"] = np.array(
        [chz] + [np.int(growth * chz * (i + 1)) for i in range(0, blks - 1)]
    )

    # Decoder sizes
    sizes["dec"]["skip"] = (
        sizes["enc"]["ip"][::-1] + sizes["enc"]["inter"][::-1]
    )
    sizes["dec"]["ip"] = sizes["enc"]["op"][::-1]  # + sizes['dec']['skip']
    sizes["dec"]["op"] = np.append(sizes["enc"]["op"][::-1][1:], chz)
    return sizes


class ConvBlock(nn.Module):
    """The block of pupil segmentation.

    Args:
        in_c: Num of channels for input.
        inter_c: Num of channels for  feat.
        out_c: Num of channels for output.
        is_top: Is it the top block of the model? \
            The last layer of the block will not be quantified.
        bn_kwargs: Dict for BN layer.
    """

    def __init__(
        self,
        in_c: int,
        inter_c: int,
        out_c: int,
        is_top: bool = False,
        bn_kwargs: Optional[Dict] = None,
    ):
        super(ConvBlock, self).__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        self.is_top = is_top
        if self.is_top:
            self.layer = nn.Sequential(
                ConvModule2d(
                    in_channels=in_c,
                    out_channels=inter_c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(inter_c, **bn_kwargs),
                    # act_layer=nn.ReLU(inplace=True),
                    act_layer=nn.LeakyReLU(inplace=True),
                ),
                ConvModule2d(
                    in_channels=inter_c,
                    out_channels=out_c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_layer=None,
                    act_layer=None,
                ),
            )
        else:
            self.layer = nn.Sequential(
                ConvModule2d(
                    in_channels=in_c,
                    out_channels=inter_c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(inter_c, **bn_kwargs),
                    # act_layer=nn.ReLU(inplace=True),
                    act_layer=nn.LeakyReLU(inplace=True),
                ),
                ConvModule2d(
                    in_channels=inter_c,
                    out_channels=out_c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(out_c, **bn_kwargs),
                    # act_layer=nn.ReLU(inplace=True),
                    act_layer=nn.LeakyReLU(inplace=True),
                ),
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
        if self.is_top:
            getattr(
                self.layer, "1"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()
