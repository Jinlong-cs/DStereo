# Copyright (c) Horizon Robotics, All rights reserved.

import logging
from typing import List, Optional, Tuple

import horizon_plugin_pytorch as hopp
import torch
from torch import nn
from typeguard import check_argument_types

from hat.data.transforms.avspeech.mask import make_pad_mask, subsequent_mask
from hat.models.base_modules.basic_fusionformer_module import (
    MultiHeadAttention,
    PositionwiseFeedForward,
)
from hat.models.task_modules.avspeech.embedding import PositionalEncoding
from ..utils import get_activation

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


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Inter-attention module instance.
            `MultiHeadAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

        # 辅助算子
        self.self_add = hopp.nn.quantized.FloatFunctional()
        self.src_add = hopp.nn.quantized.FloatFunctional()
        self.ff_add = hopp.nn.quantized.FloatFunctional()

    def fuse_model(self):
        self.self_attn.fuse_model()
        self.src_attn.fuse_model()
        self.feed_forward.fuse_model()

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),  # noqa: B008
    ) -> torch.Tensor:
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, size, 1, time_out).
            tgt_mask (torch.Tensor): Mask for input tensor
                (#batch, 1, time_out, time_out).
            memory (torch.Tensor): Encoded memory
                (#batch, size, 1, time_in).
            memory_mask (torch.Tensor): Encoded memory mask
                (#batch, 1, 1, time_in).
            cache (torch.Tensor): cached tensors.
                (#batch, time_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, size, 1, t_out).

        """
        residual = tgt

        if cache.size(0) == 0:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            # assert cache.shape == (
            #     tgt.shape[0],
            #     self.size,
            #     1,
            #     tgt.shape[3] - 1,
            # ), "{cache.shape} == {(tgt.shape[0], size, 1, tgt.shape[1] - 1)}"
            tgt_q = tgt[:, :, :, -1:]
            residual = residual[:, :, :, -1:]
            tgt_q_mask = tgt_mask[:, :, -1:, :]
        x = self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0])
        x = self.self_add.add(x, residual)

        residual = x
        x = self.dropout(self.src_attn(x, memory, memory, memory_mask)[0])

        x = self.src_add.add(x, residual)

        residual = x
        x = self.dropout(self.feed_forward(x))
        x = self.ff_add.add(x, residual)

        if cache.size(0) > 0:
            x = torch.cat([cache, x], dim=3)

        return x


class TransformerCnnDecoder(torch.nn.Module):
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
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        activation_type: str = "relu",
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(
                f"found: {input_layer}, only 'embed' is supported"
            )

        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-12)
        self.output_layer = torch.nn.Conv2d(attention_dim, vocab_size, 1, 1, 0)
        self.num_blocks = num_blocks
        activation = get_activation(activation_type)
        self.decoders = torch.nn.ModuleList(
            [
                DecoderLayer(
                    attention_dim,
                    MultiHeadAttention(
                        attention_heads,
                        attention_dim,
                        self_attention_dropout_rate,
                    ),
                    MultiHeadAttention(
                        attention_heads,
                        attention_dim,
                        src_attention_dropout_rate,
                    ),
                    PositionwiseFeedForward(
                        attention_dim, linear_units, dropout_rate, activation
                    ),
                    dropout_rate,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def fuse_model(self):
        for layer in self.decoders:
            layer.fuse_model()

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
                torch.tensor(0.0), in order to unify api with bidirectiona
                    decoder
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
        x = x.squeeze(2).transpose(1, 2).contiguous()  # (B, L, D)
        x = self.after_norm(x)
        x = x.transpose(1, 2).contiguous().unsqueeze(2)  # (B, D, 1, L)
        x = self.output_layer(x)  # (B, vocab_size, 1, L)
        x = x.squeeze(2).transpose(1, 2).contiguous()  # (B, L, vocab_size)
        return x, torch.tensor(0.0), olens

    def forward_embedding(self, ys_in_pad):
        # positional encoding
        x, _ = self.embed(ys_in_pad)  # (B, L, D)
        return x

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

            This is only used for decoding.

        Args:
            memory: encoded memory, float32  (batch, time, feat)
            memory_mask: encoded memory mask, (batch, 1, time)
            tgt: input token ids, int64 (batch, length)
            tgt_mask: input token mask,  (batch, length)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, length-1, size)

        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        # positional encoding
        x, _ = self.embed(tgt)

        # decoder layers, 4D dataflow
        x = x.transpose(1, 2).contiguous().unsqueeze(2)  # (B, D, 1, L)
        B, _, _, L = x.size()
        memory = (
            memory.transpose(1, 2).contiguous().unsqueeze(2)
        )  # (B, D, 1, T)
        tgt_mask = (
            tgt_mask.unsqueeze(1).float().expand(B, 1, L, L)
        )  # int -> float, (B, 1, L, L)
        memory_mask = (
            memory_mask.unsqueeze(1).float().expand(B, 1, 1, -1)
        )  # int -> float, (B, 1, 1, T)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = torch.zeros((0, 0, 0, 0))
            else:
                c = cache[i]
            x = decoder(x, tgt_mask, memory, memory_mask, cache=c)
            new_cache.append(x)

        # final norm and projection
        x = x.squeeze(2).transpose(1, 2).contiguous()
        y = self.after_norm(x)
        y = y.transpose(1, 2).contiguous().unsqueeze(2)  # (B, D, 1, L)
        y = torch.log_softmax(self.output_layer(y[:, :, :, -1:]), dim=1)
        y = y.squeeze(2).transpose(1, 2).contiguous().squeeze(1)  # (B, v_size)
        return y, new_cache


class BiTransformerCnnDecoder(torch.nn.Module):
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
        self.left_decoder = TransformerCnnDecoder(
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

        self.right_decoder = TransformerCnnDecoder(
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

    def fuse_modules(self):
        self.left_decoder.fuse_modules()
        self.right_decoder.fuse_modules()

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        x: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_x: torch.Tensor,
        r_ys_in_pad: torch.Tensor,
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
            memory,
            memory_mask,
            x,
            ys_in_pad,
            ys_in_lens,
            r_x,
        )
        if reverse_weight > 0.0:
            r_x, _, olens = self.right_decoder(
                memory, memory_mask, r_x, r_ys_in_pad, ys_in_lens, r_x
            )
        return l_x, r_x, olens

    def forward_embedding(self, ys_in_pad, r_ys_in_pad):
        x = self.left_decoder.forward_embedding(ys_in_pad)
        r_x = self.right_decoder.forward_embedding(r_ys_in_pad)
        return x, r_x

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

            This is only used for decoding.

        Args:
            memory: encoded memory, float32  (batch, time, feat)
            memory_mask: encoded memory mask, (batch, 1, time)
            tgt: input token ids, int64 (batch, length)
            tgt_mask: input token mask,  (batch, length)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, length-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        return self.left_decoder.forward_one_step(
            memory, memory_mask, tgt, tgt_mask, cache
        )
