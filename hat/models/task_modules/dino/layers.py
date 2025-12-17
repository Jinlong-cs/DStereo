import math

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import QuantStub

from hat.utils.model_helpers import fx_wrap


def inverse_sigmoid(x, eps=1e-3):
    """Inverse function for sigmoid activation function.

    Note: It might face numberical issues with fp16 small eps.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class GetSinPosEmbed(nn.Module):
    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: int = 10000,
    ):
        super(GetSinPosEmbed, self).__init__()
        self.sin1 = horizon.nn.Sin()
        self.cos1 = horizon.nn.Cos()
        self.mul1 = FloatFunctional()
        self.sin_cat1 = FloatFunctional()
        self.pos_cat = FloatFunctional()
        self.scale_quant = QuantStub(scale=None)
        self.scale = 2 * math.pi
        self.dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        self.dim_t = temperature ** (
            2 * torch.div(self.dim_t, 2, rounding_mode="floor") / num_pos_feats
        )
        self.sin_x_mult = self.scale / self.dim_t

    def sine_func(self, x: torch.Tensor, sin_x_mult):
        sin_x = self.mul1.mul(x, self.scale_quant(sin_x_mult.to(x.device)))
        sin_cos_tuple = (
            self.sin1(sin_x[:, :, 0::2]).unsqueeze(3),
            self.cos1(sin_x[:, :, 1::2]).unsqueeze(3),
        )
        sin_x = self.sin_cat1.cat(sin_cos_tuple, dim=3).flatten(2)
        return sin_x

    @fx_wrap()
    def forward(
        self,
        pos_tensor: torch.Tensor,
        exchange_xy: bool = False,
    ) -> torch.Tensor:
        pos_res = []
        i = 0
        for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1):
            i += 1
            pos_res.append(self.sine_func(x, self.sin_x_mult))

        if exchange_xy:
            pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
        pos_res = self.pos_cat.cat(pos_res, dim=2)
        return pos_res
