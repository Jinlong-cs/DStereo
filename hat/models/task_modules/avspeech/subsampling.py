# Copyright (c) Horizon Robotics, All rights reserved.

import logging
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from hat.models.base_modules.conv_module import ConvModule2d


class BaseSubsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
        )
        self.out = nn.Conv2d(odim * (((idim - 1) // 2 - 1) // 2), odim, 1, 1)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def fuse_model(self):
        try:
            from horizon_plugin_pytorch import quantization

            fuser_func = quantization.fuse_known_modules
        except Warning:
            logging.warning(
                "Please install horizon_plugin_pytorch first, otherwise use"
                "pytorch official quantification"
            )
            from torch.quantization.fuse_modules import fuse_known_modules

            fuser_func = fuse_known_modules
        torch.quantization.fuse_modules(
            self,
            [["conv.0", "conv.1"]],
            inplace=True,
            fuser_func=fuser_func,
        )
        torch.quantization.fuse_modules(
            self,
            [["conv.2", "conv.3"]],
            inplace=True,
            fuser_func=fuser_func,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, 1, time, idim).
            masks (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, odim, 1, time'),
                where time' = time // 4.

        """
        x = self.conv(x)
        b, c, t, f = x.size()
        x = x.transpose(2, 3).contiguous().view(b, c * f, 1, t)
        x = self.dropout(self.out(x))

        return x

    @staticmethod
    def subsample_mask(mask: torch.Tensor) -> torch.Tensor:
        """Subsample mask.

        Args:
            mask: Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        return mask[:, :, 2::2][:, :, 2::2]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
        )
        self.out = nn.Conv2d(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim, 1, 1
        )
        self.dropout = nn.Dropout(dropout_rate, inplace=True)
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, 1, time, idim).
            masks (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            torch.Tensor: positional encoding
        """
        x = self.conv(x)
        b, c, t, f = x.size()
        x = x.permute(0, 1, 3, 2).contiguous().reshape(b, c * f, 1, t)
        x = self.dropout(self.out(x))

        return x

    @staticmethod
    def subsample_mask(mask: torch.Tensor) -> torch.Tensor:
        """Subsample mask.

        Args:
            mask: Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.

        """
        return mask[:, :, 2::2][:, :, 2::2][:, :, 2::2]

    def fuse_model(self):
        try:
            from horizon_plugin_pytorch import quantization

            fuser_func = quantization.fuse_known_modules
        except Warning:
            logging.warning(
                "Please install horizon_plugin_pytorch first, otherwise use "
                "pytorch official quantification"
            )
            from torch.quantization.fuse_modules import fuse_known_modules

            fuser_func = fuse_known_modules
        torch.quantization.fuse_modules(
            self,
            [["conv.0", "conv.1"]],
            inplace=True,
            fuser_func=fuser_func,
        )
        torch.quantization.fuse_modules(
            self,
            [["conv.2", "conv.3"]],
            inplace=True,
            fuser_func=fuser_func,
        )
        torch.quantization.fuse_modules(
            self,
            [["conv.4", "conv.5"]],
            inplace=True,
            fuser_func=fuser_func,
        )


class CausalSubsampling2(nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        bn_kwargs: Optional[Mapping] = {},  # noqa: B006
    ):
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.stride = [2, 1]
        self.lorder = 5

        self.conv = nn.Sequential(
            ConvModule2d(
                in_channels=idim,
                out_channels=odim,
                kernel_size=(1, 3),
                stride=self.stride[0],
                padding=0,
                norm_layer=nn.BatchNorm2d(odim, **bn_kwargs),
                act_layer=nn.ReLU(),
            ),
            ConvModule2d(
                in_channels=odim,
                out_channels=odim,
                kernel_size=(1, 3),
                stride=self.stride[1],
                padding=0,
                norm_layer=nn.BatchNorm2d(odim, **bn_kwargs),
                act_layer=nn.ReLU(),
            ),
        )
        self.pad = nn.ConstantPad2d((self.lorder, 0, 0, 0), 0.0)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def trace(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def fuse_model(self):
        for mod in self.conv:
            mod.fuse_model()

    @staticmethod
    def subsample_mask(mask: torch.Tensor) -> torch.Tensor:
        """Subsample mask.

        Args:
            mask: Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.

        """
        mask = F.pad(mask, (5, 0), mode="constant", value=True)
        mask = mask[:, :, 2::2][:, :, 2::1]
        return mask
