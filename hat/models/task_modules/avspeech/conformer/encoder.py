# Copyright (c) Horizon Robotics, All rights reserved.

from copy import deepcopy
from typing import Tuple

import horizon_plugin_pytorch as hopp
import torch
from torch import nn

from hat.models.base_modules.basic_conformer_module import (
    CausalConvolutionModule,
    MultiHeadAttention,
    PositionwiseFeedForward,
)


class ConformerLayer(nn.Module):
    """ConformerLayer.

    Args:
        n_feat: 基础特征维度
        dropout_rate: Conformer单元结构的 dropout 的比例. 默认是 0.1.
        activation: 使用的激活层, 默认是 nn.SiLU()
        ff_hidden_dim: feedforward层的隐藏层维度, 默认是 2048.
        ff_dropout_rate: feedforward 中 dropout 的比例, 默认是 0.1.
        attn_head: 自注意力模块的头数，默认是 4.
        attn_dropout_rate: 自注意力模块的 dropout 的比例, 默认是 0.1.
        attn_scores_fill_value: 自注意力模块的份数矩阵填充值，默认是 -inf.
        conv_kernel_size: conv module 的 kernel_size 大小, 默认是 7.
        conv_bias: conv module 是否使用 bias，默认是 True.
    """

    def __init__(
        self,
        n_feat: int = 256,
        dropout_rate: float = 0.1,
        activation: nn.Module = nn.SiLU(),  # noqa: B008
        ff_hidden_dim: int = 2048,
        ff_dropout_rate: float = 0.1,
        attn_head: int = 4,
        attn_dropout_rate: float = 0.1,
        attn_scores_fill_value: float = -float("inf"),
        conv_kernel_size: int = 7,
        conv_bias: bool = True,
    ):
        super().__init__()
        # Conformer Module 的四个主要模块
        self.feed_forward_macaron = PositionwiseFeedForward(
            idim=n_feat,
            hidden_dim=ff_hidden_dim,
            activation=deepcopy(activation),
            dropout_rate=ff_dropout_rate,
        )

        self.self_attn = MultiHeadAttention(
            n_head=attn_head,
            n_feat=n_feat,
            dropout_rate=attn_dropout_rate,
            scores_fill_value=attn_scores_fill_value,
        )

        self.conv_module = CausalConvolutionModule(
            in_channels=n_feat,
            kernel_size=conv_kernel_size,
            activation=deepcopy(activation),
            bias=conv_bias,
        )

        self.feed_forward = PositionwiseFeedForward(
            idim=n_feat,
            hidden_dim=ff_hidden_dim,
            activation=deepcopy(activation),
            dropout_rate=ff_dropout_rate,
        )

        # 各种辅助
        self.ff_scale = 0.5
        # macaron feed forward
        self.macaron_add = hopp.nn.quantized.FloatFunctional()
        self.macaron_mul = hopp.nn.quantized.FloatFunctional()
        # attention module
        self.attn_add = hopp.nn.quantized.FloatFunctional()
        # convolution module
        self.conv_add = hopp.nn.quantized.FloatFunctional()
        # feed forward
        self.ff_add = hopp.nn.quantized.FloatFunctional()
        self.ff_mul = hopp.nn.quantized.FloatFunctional()
        self.dropout = nn.Dropout(dropout_rate, inplace=True)
        # # layer normalized
        self.norm_macaron = hopp.nn.layer_norm.LayerNorm(
            (n_feat, -1, -1), dim=1
        )
        self.norm_mha = hopp.nn.layer_norm.LayerNorm((n_feat, -1, -1), dim=1)
        self.norm_conv = hopp.nn.layer_norm.LayerNorm((n_feat, -1, -1), dim=1)
        self.norm_ff = hopp.nn.layer_norm.LayerNorm((n_feat, -1, -1), dim=1)
        self.norm_final = hopp.nn.layer_norm.LayerNorm((n_feat, -1, -1), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        mask_attn: torch.Tensor = torch.zeros((0, 0, 0, 0)),  # noqa: B008
        mask_pad: torch.Tensor = torch.zeros((0, 0, 0, 0)),  # noqa: B008
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),  # noqa: B008
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),  # noqa: B008
    ) -> torch.Tensor:
        # macaron feed forward module.
        residual = x  # (#b, size, 1, chunk_size)
        x = self.norm_macaron(x)
        x = self.feed_forward_macaron(x)
        x = self.dropout(x)
        x = self.macaron_mul.mul_scalar(x, self.ff_scale)
        x = self.macaron_add.add(x, residual)

        # multi-head self-attention module
        residual = x
        x = self.norm_mha(x)
        x, new_att_cache = self.self_attn(x, x, x, mask_attn, att_cache)
        x = self.dropout(x)
        x = self.attn_add.add(x, residual)

        # convolution module
        residual = x
        x = self.norm_conv(x)
        x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
        x = self.dropout(x)
        x = self.conv_add.add(x, residual)

        # feed forward module
        residual = x
        x = self.norm_ff(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.ff_mul.mul_scalar(x, self.ff_scale)
        x = self.ff_add.add(x, residual)
        x = self.norm_final(x)
        return x, mask_attn, new_att_cache, new_cnn_cache

    def trace(self, x, cnn_cache) -> Tuple[torch.Tensor, torch.Tensor]:
        # macaron feed forward module.
        residual = x  # (#b, size, 1, chunk_size)
        x = self.norm_macaron(x)
        x = self.feed_forward_macaron(x)
        x = self.macaron_mul.mul_scalar(x, self.ff_scale)
        x = self.macaron_add.add(x, residual)

        # multi-head self-attention module
        residual = x
        x = self.norm_mha(x)
        x = self.self_attn.trace(x, x, x)
        x = self.dropout(x)
        x = self.attn_add.add(x, residual)

        # convolution module
        residual = x
        x = self.norm_conv(x)
        x, new_cache = self.conv_module.trace(x, cnn_cache)
        x = self.dropout(x)
        x = self.conv_add.add(x, residual)

        # feed forward module
        residual = x
        x = self.norm_ff(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.ff_mul.mul_scalar(x, self.ff_scale)
        x = self.ff_add.add(x, residual)
        x = self.norm_final(x)
        return x, new_cache

    def fuse_model(self):
        self.self_attn.fuse_model()
        self.feed_forward.fuse_model()
        self.feed_forward_macaron.fuse_model()
        self.conv_module.fuse_model()


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        embed: nn.Module = nn.Identity(),  # noqa: B008
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        cnn_inner_channel: int = 512,
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        activation: nn.Module = nn.ReLU(),  # noqa: B008
        cnn_module_kernel: int = 15,
        final_norm: str = "layer_norm",
        attn_scores_fill_value: float = -float("inf"),
    ):
        super().__init__()
        self.n_feat = output_size
        self.embed = embed
        self.encoders = nn.ModuleList(
            ConformerLayer(
                n_feat=output_size,
                dropout_rate=dropout_rate,
                activation=activation,
                ff_hidden_dim=linear_units,
                ff_dropout_rate=dropout_rate,
                attn_head=attention_heads,
                attn_dropout_rate=attention_dropout_rate,
                attn_scores_fill_value=attn_scores_fill_value,
                conv_kernel_size=cnn_module_kernel,
                conv_bias=True,
            )
            for _ in range(num_blocks)
        )

        if final_norm is None:
            self.final_norm = torch.nn.Identity()
        elif final_norm == "layer_norm":
            self.final_norm = hopp.nn.layer_norm.LayerNorm(
                (output_size, -1, -1),
                eps=1e-12,
                dim=1,
            )
        else:
            raise ValueError("unknown norm_type: " + final_norm)

    def forward(
        self,
        x,
        chunk_mask,
        pad_mask,
        att_caches=None,
        cnn_caches=None,
    ):
        if att_caches is None:
            att_caches = [
                torch.zeros((0, 0, 0)) for _ in range(self.num_layers)
            ]
        if cnn_caches is None:
            cnn_caches = [
                torch.zeros((0, 0, 0)) for _ in range(self.num_layers)
            ]

        x = self.embed(x)

        new_att_caches = []
        new_cnn_caches = []
        for layer, att_cache, cnn_cache in zip(
            self.encoders, att_caches, cnn_caches
        ):
            x, chunk_mask, new_att_cache, new_cnn_cache = layer(
                x, chunk_mask, pad_mask, att_cache, cnn_cache
            )
            new_att_caches.append(new_att_cache)
            new_cnn_caches.append(new_cnn_cache)

        x = self.final_norm(x)

        return x, pad_mask, new_att_caches, new_cnn_caches

    def fuse_model(self):
        for layer in self.layers:
            layer.fuse_model()

    @property
    def num_layers(self):
        return len(self.encoders)

    @property
    def output_size(self):
        return self.n_feat
