# Copyright (c) Horizon Robotics. All rights reserved.

import horizon_plugin_pytorch as horizon
import horizon_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.task_modules.traj_pred.base_modules.traj_qconfig import (
    qint8_qconfig,
    qint16_qconfig,
)

__all__ = [
    "LinearLN",
    "Embedding",
    "MLP",
]


class LinearLN(nn.Module):
    """A linear layer with layernorm for 4-D tensor.

    Note: this module is inefficient, if the user does not use layernorm,
    please use torch.nn.Linear instead.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        use_norm: bool = False,
        use_relu: bool = True,
        in_dim: int = 2,
        out_dim: int = 2,
    ):
        """Initialize method.

        Args:
            in_channels: the input channels of the sub graph layer.
            out_channels: the output channels of the sub graph layer.
            kernel_size: the kernal size.
            stride: the stride.
            padding: the padding.
            use_norm: whether to use layer norm.
            use_relu: whether to use relu.
            in_dim: the dimension of the input tensor.
            out_dim: the dimension of the output tensor.
        """
        super(LinearLN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_relu = use_relu
        self.use_norm = use_norm
        assert in_dim in [2, 4], "The input dim should be either 2 or 4."
        assert out_dim in [2, 4], "The output dim should be either 2 or 4."
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        if self.use_norm:
            self.norm = horizon.nn.LayerNorm((out_channels, 1, 1), dim=1)
        if self.use_relu:
            self.relu = nn.ReLU()

    def forward(self, data: torch.Tensor):
        """Forward.

        Args:
            data ([batch_size, in_channels, H, W]): the input features.

         Returns:
            out ([batch_size, out_channels, H, W]): the results.
        """
        batch_size = data.shape[0]
        if self.in_dim == 2:
            data = data.reshape([batch_size, -1, 1, 1])
        out = self.conv(data)
        if self.use_norm:
            out = self.norm(out)
        if self.use_relu:
            out = self.relu(out)
        if self.out_dim == 2:
            out = out.reshape([batch_size, -1])
        return out

    def fuse_model(self):
        if not self.use_norm and self.use_relu:
            torch.quantization.fuse_modules(
                self,
                ["conv", "relu"],
                inplace=True,
                fuser_func=horizon.quantization.fuse_known_modules,
            )


class Embedding(nn.Module):
    """Return embed as nn.Parameter."""

    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape))
        nn.init.uniform_(self.weight)
        self.q_stub = QuantStub()

    def forward(self):
        return self.q_stub(self.weight)

    def set_qconfig(self):
        self.q_stub.qconfig = qint16_qconfig()


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        H_dim,
        W_dim,
        p_drop=0.0,
        hidden_dim=None,
        residual=False,
    ):
        super(MLP, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        layer2_dim = hidden_dim
        if residual:
            layer2_dim = hidden_dim + input_dim

        self.residual = residual
        self.layer1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.layer2 = nn.Conv2d(
            in_channels=layer2_dim,
            out_channels=output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.dropout1 = nn.Dropout(p=p_drop)
        self.dropout2 = nn.Dropout(p=p_drop)

        self.norm = hnn.LayerNorm([hidden_dim, H_dim, W_dim], dim=1)

        self.relu = nn.ReLU()

        self.residual_cat_op = FF()

    def forward(self, x):
        out = self.layer1(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout1(out)
        if self.residual:
            out = self.layer2(self.residual_cat_op.cat([out, x], dim=1))
        else:
            out = self.layer2(out)

        out = self.dropout2(out)
        return out

    def set_qconfig(self):
        self.qconfig = qint16_qconfig()
        self.norm.qconfig = qint8_qconfig()
