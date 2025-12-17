# Copyright (c) Horizon Robotics, All rights reserved.
from typing import Optional, Tuple

import horizon_plugin_pytorch as hopp
import torch
from torch import nn
from typeguard import check_argument_types

from hat.data.transforms.avspeech.mask import make_pad_mask, subsequent_mask
from hat.models.base_modules.basic_conformer_module import (
    MultiHeadAttention,
    PositionwiseFeedForward,
)
from hat.models.base_modules.conv_module import ConvModule2d
from hat.utils import qconfig_manager
from ..embedding import PositionalEncoding
from ..utils import get_activation


class TransformerDecoderLayer(nn.Module):
    """一个Transformer Decoder层.

    端到端结构中, 解码器部分transformer结构的实现.

    Args:
        self_attn: target的mask自注意力模块.
        src_attn: target作为q, encoder output作为k,v的注意力模块.
        feed_forward: feed forward模块.
        dropout_rate: dropout 的比例.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        src_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float,
        output_size: int = 256,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.self_add = hopp.nn.quantized.FloatFunctional()
        self.src_add = hopp.nn.quantized.FloatFunctional()
        self.ff_add = hopp.nn.quantized.FloatFunctional()
        self.dropout = nn.Dropout(dropout_rate, inplace=True)
        self.norm1 = hopp.nn.layer_norm.LayerNorm((output_size, -1, -1), dim=1)
        self.norm2 = hopp.nn.layer_norm.LayerNorm((output_size, -1, -1), dim=1)
        self.norm3 = hopp.nn.layer_norm.LayerNorm((output_size, -1, -1), dim=1)

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        训练过程中的前向过程.

        Args:
            tgt: <sos> 开头的标签表征输入.或者上一层解码器的输出.
                 训练过程中跟 mask 配合模拟自回归解码的过程.
            tgt_mask: 输入标签的 mask. 在训练过程中是一个下三角矩阵.
            memory: Encoder 部分的输出.
            memory_mask: Encoder 输出的 padding mask.
                         对数据padding的部分进行掩盖.

        Returns:
            一层的解码结果
        """
        residual = tgt
        tgt = self.norm1(tgt)
        # self attention
        x = self.self_attn(tgt, tgt, tgt, tgt_mask)[0]
        x = self.dropout(x)
        x = self.self_add.add(x, residual)

        # cross attnention
        residual = x
        x = self.norm2(x)
        x = self.src_attn(x, memory, memory, memory_mask)[0]
        x = self.dropout(x)
        x = self.src_add.add(x, residual)

        # feed forward
        residual = x
        self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.ff_add.add(x, residual)

        return x

    def trace(self, tgt, memory):
        residual = tgt
        tgt = self.norm1(tgt)
        # self attention
        x = self.self_attn.trace(tgt, tgt, tgt)
        x = self.dropout(x)
        x = self.self_add.add(x, residual)

        # cross attention
        residual = x
        x = self.norm2(x)
        x = self.src_attn.trace(x, memory, memory)
        x = self.dropout(x)
        x = self.src_add.add(x, residual)

        # feed forward
        residual = x
        self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.ff_add.add(x, residual)

        return x

    def fuse_model(self):
        self.self_attn.fuse_model()
        self.src_attn.fuse_model()
        self.feed_forward.fuse_model()


class TransformerDecoder(nn.Module):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.1,
        src_attention_dropout_rate: float = 0.1,
        input_layer: str = "embed",
        activation_type: str = "relu",
        tgt_quant_scale: float = 1.0 / 128,
        mem_quant_scale: float = 1.0 / 128,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(
                f"found: {input_layer}, only 'embed' is supported"
            )
        self.after_norm = hopp.nn.layer_norm.LayerNorm(
            (encoder_output_size, -1, -1), eps=1e-12, dim=1
        )
        self.output_layer = ConvModule2d(
            encoder_output_size, vocab_size, 1, 1, 0
        )
        activation = get_activation(activation_type)
        self.decoders = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    self_attn=MultiHeadAttention(
                        n_head=attention_heads,
                        n_feat=encoder_output_size,
                        dropout_rate=self_attention_dropout_rate,
                    ),
                    src_attn=MultiHeadAttention(
                        n_head=attention_heads,
                        n_feat=encoder_output_size,
                        dropout_rate=src_attention_dropout_rate,
                    ),
                    feed_forward=PositionwiseFeedForward(
                        idim=encoder_output_size,
                        hidden_dim=linear_units,
                        activation=activation,
                        dropout_rate=positional_dropout_rate,
                    ),
                    dropout_rate=dropout_rate,
                    output_size=encoder_output_size,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.quant1 = hopp.quantization.QuantStub(scale=tgt_quant_scale)
        self.quant3 = hopp.quantization.QuantStub(scale=mem_quant_scale)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        x: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_x: torch.Tensor = torch.empty(0),  # noqa B008
        r_ys_in_pad: Optional[torch.Tensor] = None,
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            memory: encoded memory, float32  (batch, time, feat)
            memory_mask: encoder memory mask, (batch, 1, time)
            ys_in_pad: padded input token ids, int64 (batch, length)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, length,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with
                bidirectional decoder
                olens: (batch, )
        """
        # attention mask
        B, L = ys_in_pad.size()
        tgt_mask = ~make_pad_mask(ys_in_lens, L).unsqueeze(1)
        tgt_mask = tgt_mask.to(ys_in_pad.device)  # (B, 1, L)
        m = subsequent_mask(
            tgt_mask.size(-1), device=tgt_mask.device
        ).unsqueeze(0)
        tgt_mask = tgt_mask & m  # (B, L, L)
        olens = tgt_mask.sum(1)

        # decoder layers, 4D dataflow
        x = x.transpose(1, 2).contiguous().unsqueeze(2)  # (B, D, 1, L)
        tgt_mask = (
            tgt_mask.unsqueeze(1).float().expand(B, 1, L, L)
        )  # int -> float, (B, 1, L, L)
        memory_mask = (
            memory_mask.unsqueeze(1).float().expand(B, 1, 1, -1)
        )  # int -> float, (B, 1, 1, T)
        for layer in self.decoders:
            x = layer(x, tgt_mask, memory, memory_mask)

        # final norm and projection
        # x = x.squeeze(2).transpose(1, 2).contiguous()  # (B, L, D)
        x = self.after_norm(x)
        # x = x.transpose(1, 2).contiguous().unsqueeze(2)  # (B, D, 1, L)
        x = self.output_layer(x)  # (B, vocab_size, 1, L)
        x = x.squeeze(2).transpose(1, 2).contiguous()  # (B, L, vocab_size)
        return x, torch.tensor(0.0), olens

    def trace(self, x, memory):
        x = self.quant1(x)
        memory = self.quant3(memory)
        for layer in self.decoder_layers:
            x = layer.trace(x, memory)
        x = self.after_norm(x)
        x = self.out_conv(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for layer in self.decoder_layers:
            layer.fuse_model()
        self.out_conv.fuse_model()

    def set_qconfig(self):
        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.out_conv.qconfig = qconfig_manager.get_default_qat_out_qconfig()

    def infer(self, mode: bool = True):
        if mode:
            setattr(self, "forward_before", self.forward)  # noqa: B010
            self.forward = self.trace
        else:
            self.forward = getattr(self, "forward_before", self.forward)

    def forward_embedding(self, ys_in_pad):
        x, _ = self.embed(ys_in_pad)
        return x


class BiTransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        r_num_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        activation_type: str = "relu",
    ):
        assert check_argument_types()
        super().__init__()
        self.left_decoder = TransformerDecoder(
            vocab_size,
            encoder_output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            self_attention_dropout_rate,
            src_attention_dropout_rate,
            input_layer,
            activation_type,
        )

        self.right_decoder = TransformerDecoder(
            vocab_size,
            encoder_output_size,
            attention_heads,
            linear_units,
            r_num_blocks,
            dropout_rate,
            positional_dropout_rate,
            self_attention_dropout_rate,
            src_attention_dropout_rate,
            input_layer,
            activation_type,
        )

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        x: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_x: torch.Tensor = torch.empty(0),  # noqa B008
        r_ys_in_pad: Optional[torch.Tensor] = None,
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            memory: encoded memory, float32  (batch, time, feat)
            memory_mask: encoder memory mask, (batch, 1, time)
            ys_in_pad: padded input token ids, int64 (batch, length)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, length),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, length,
                    vocab_size) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, length, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        l_x, _, olens = self.left_decoder(
            memory, memory_mask, x, ys_in_pad, ys_in_lens
        )
        r_x, _, olens = self.right_decoder(
            memory, memory_mask, r_x, r_ys_in_pad, ys_in_lens
        )
        return l_x, r_x, olens

    def fuse_model(self):
        self.left_decoder.fuse_model()
        self.right_decoder.fuse_model()

    def seg_qconfig(self):
        pass

    def forward_embedding(self, ys_in_pad, r_ys_in_pad):
        x = self.left_decoder.forward_embedding(ys_in_pad)
        r_x = self.right_decoder.forward_embedding(r_ys_in_pad)
        return x, r_x
