# Copyright (c) Horizon Robotics, All rights reserved.

import logging
from typing import Optional, Sequence, Tuple

import horizon_plugin_pytorch as hopp
import torch
from torch import nn

from hat.models.base_modules.basic_fusionformer_module import (
    ConvolutionModule,
    MultiHeadAttention,
    PositionwiseFeedForward,
)

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


class FusionformerEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): attention module instance.
            `MultiHeadAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance. `PositionwiseFeedForward` instance can
             be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,  # noqa: B008
        feed_forward_macaron: nn.Module,
        conv_module: nn.Module,
        dropout_rate: float = 0.1,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.dropout = nn.Dropout(dropout_rate, inplace=True)
        self.ff_scale = 0.5
        # 各种辅助
        # macaron feed forward add and mul
        self.macaron_add = hopp.nn.quantized.FloatFunctional()
        self.macaron_mul = hopp.nn.quantized.FloatFunctional()
        # attention module add
        self.attn_add = hopp.nn.quantized.FloatFunctional()
        # convolution module add
        self.conv_add = hopp.nn.quantized.FloatFunctional()
        # feed forward add and mul
        self.ff_add = hopp.nn.quantized.FloatFunctional()
        self.ff_mul = hopp.nn.quantized.FloatFunctional()

    def fuse_model(self):
        self.self_attn.fuse_model()
        self.feed_forward.fuse_model()
        self.feed_forward_macaron.fuse_model()
        self.conv_module.fuse_model()

    def forward(
        self,
        x: torch.Tensor,
        mask_attn: torch.Tensor = torch.zeros((0, 0, 0, 0)),  # noqa: B008
        mask_pad: torch.Tensor = torch.zeros((0, 0, 0, 0)),  # noqa: B008
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),  # noqa: B008
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),  # noqa: B008
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, size, 1, time)
            mask_attn (torch.Tensor): Mask tensor for attn (#b, 1, time，time).
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，1, time)
            cnn_cache (torch.Tensor): Cache tensor for cnn_module,
                (1, size, 1, cache_time1)
            att_cache (torch.Tensor): Cache tensor for KEY & VALUE,
                (1, head, cache_time2, d_k * 2), head * d_k == size.

        Returns:
            torch.Tensor: Output tensor (#batch, size, 1, time).
            torch.Tensor: Cnn cache tensor (1, size, 1, cache_t1).
            torch.Tensor: Att cache tensor (1, head, cache_t2 + time, d_k * 2).

        """
        # macaron feed forward module.
        residual = x  # (#b, size, 1, chunk_size)
        x = self.dropout(self.feed_forward_macaron(x))
        x = self.macaron_mul.mul_scalar(x, self.ff_scale)
        x = self.macaron_add.add(x, residual)

        # multi-headed self-attention module.
        residual = x  # (#b, size, 1, chunk_size)
        x_att, new_att_cache = self.self_attn(x, x, x, mask_attn, att_cache)
        x = self.attn_add.add(self.dropout(x_att), residual)

        # convolution module.
        residual = x  # (#b, size, 1, chunk_size)
        x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
        x = self.conv_add.add(self.dropout(x), residual)

        # feed forward module
        residual = x  # (#b, size, 1, chunk_size)
        x = self.feed_forward(x)
        x = self.ff_mul.mul_scalar(self.dropout(x), self.ff_scale)
        x = x = self.ff_add.add(x, residual)

        return x, mask_attn, new_att_cache, new_cnn_cache


class FusionformerEncoder(nn.Module):
    """Conformer encoder module. Fully-convolutional version."""

    def __init__(
        self,
        embed: nn.Module,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        cnn_inner_channel: int = 512,
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        activation: nn.Module = nn.ReLU(),  # noqa: B008
        cnn_module_kernel: int = 7,
        causal: bool = False,
        final_norm: str = "layer_norm",
    ):
        """Construct ConformerEncoder.

        Args:
            embed (torch.nn.Embedding): Input embedding module
            output_size (int): Dimension of output feature
            attention_heads (int): The number of heads of multi head attention
            linear_units (int): The number of units of feedforward
            cnn_inner_channel (int): The number of channles of CNN module
            num_blocks (int): The number of encoder blocks
            dropout_rate (float): Dropout rate
            attention_dropout_rate (float): Dropout rate in attention
            activation (torch.nn.Module): Activation function
            cnn_module_kernel (int): Kernerl size of convolution module
            causal (bool): Whether to use causal convolution or not
            final_norm (str): normalization type for final output
        """
        super().__init__()
        self.embed = embed
        self.n_feat = output_size

        # self-attention module definition
        encoder_attn_layer = MultiHeadAttention
        encoder_attn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )

        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (
            output_size,
            cnn_inner_channel,
            cnn_module_kernel,
            activation,
            causal,
            True,
        )

        self.encoders = torch.nn.ModuleList(
            [
                FusionformerEncoderLayer(
                    output_size,
                    encoder_attn_layer(*encoder_attn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    convolution_layer(*convolution_layer_args),
                    dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

        if final_norm == "layer_norm":
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
        for _, (layer, att_cache, cnn_cache) in enumerate(
            zip(self.encoders, att_caches, cnn_caches)
        ):
            x, chunk_mask, new_att_cache, new_cnn_cache = layer(
                x, chunk_mask, pad_mask, att_cache, cnn_cache
            )
            new_att_caches.append(new_att_cache)
            new_cnn_caches.append(new_cnn_cache)

        x = self.final_norm(x)

        return x, pad_mask, new_att_caches, new_cnn_caches

    @property
    def output_size(self):
        return self.n_feat

    @property
    def num_layers(self):
        return len(self.encoders)

    def fuse_model(self):
        self.embed.fuse_modules()
        for layer in self.encoders:
            layer.fuse_modules()


class FusionEncoder(torch.nn.Module):
    """Conformer encoder module. Fully-convolutional version."""

    def __init__(
        self,
        audio_encoder: nn.Module,
        video_encoder: nn.Module,
        fusion_block: nn.Module,
        av_encoder: nn.Module,
        norm_before_fusion: bool = False,
    ):
        # assert check_argument_types()
        super().__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.fusion_block = fusion_block
        self.av_encoder = av_encoder

        self.norm_before_fusion = norm_before_fusion
        if self.norm_before_fusion:
            self.norm_a = hopp.nn.layer_norm.LayerNorm(
                (self.audio_encoder.output_size, -1, -1),
                eps=1e-12,
                dim=1,
            )
            self.norm_v = hopp.nn.layer_norm.LayerNorm(
                (self.video_encoder.output_size, -1, -1),
                eps=1e-12,
                dim=1,
            )

    def forward(
        self,
        afea: torch.Tensor,
        vfea: Optional[torch.Tensor],
        chunk_mask: Optional[torch.Tensor],
        pad_mask: Optional[torch.Tensor],
        att_caches: Optional[Sequence[torch.Tensor]] = None,
        cnn_caches: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: not used.
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        if att_caches is None:
            att_caches = [None for _ in range(self.num_layers)]
        if cnn_caches is None:
            cnn_caches = [None for _ in range(self.num_layers)]

        # forward audio encoder
        _beg = 0
        _end = _beg + len(self.audio_encoder.encoders)

        (
            afea,
            # chunk_mask,
            apad_mask,
            ao_att_caches,
            ao_cnn_caches,
        ) = self.audio_encoder(
            afea,
            chunk_mask,
            pad_mask,
            att_caches[_beg:_end],
            cnn_caches[_beg:_end],
        )

        # forward video encoder
        _beg = _end
        _end = _beg + len(self.video_encoder.encoders)
        (
            vfea,
            # chunk_mask,
            vpad_mask,
            vo_att_caches,
            vo_cnn_caches,
        ) = self.video_encoder(
            vfea,
            chunk_mask,
            pad_mask,
            att_caches[_beg:_end],
            cnn_caches[_beg:_end],
        )

        # forward fusio block
        if self.norm_before_fusion:
            afea = self.norm_a(afea)
            vfea = self.norm_v(vfea)
        avfea = self.fusion_block(afea, vfea)

        # forward av_encoder
        _beg = _end
        _end = _beg + len(self.av_encoder.encoders)
        (
            avfea,
            # chunk_mask,
            avpad_mask,
            av_att_caches,
            av_cnn_caches,
        ) = self.av_encoder(
            avfea,
            chunk_mask,
            pad_mask,
            att_caches[_beg:_end],
            cnn_caches[_beg:_end],
        )

        att_caches = [*ao_att_caches, *vo_att_caches, *av_att_caches]
        cnn_caches = [*ao_cnn_caches, *vo_cnn_caches, *av_cnn_caches]
        return avfea, pad_mask, att_caches, cnn_caches

    def fuse_model(self):
        self.audio_encoder.fuse_model()
        self.video_encoder.fuse_model()
        self.fusion_block.fuse_model()
        self.av_encoder.fuse_model()

    @property
    def num_layers(self):
        return (
            self.audio_encoder.num_layers
            + self.video_encoder.num_layers
            + self.av_encoder.num_layers
        )

    @property
    def output_size(self):
        return self.av_encoder.output_size
