# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from torch import Tensor, nn

from hat.models.task_modules.bevformer.utils import constant_init, xavier_init
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = [
    "HorizonSpatialCrossAttention",
    "HorizonMSDeformableAttention3D",
    "HorizonTemporalSelfAttention",
    "HorizonMSDeformableAttention",
]


@OBJECT_REGISTRY.register
class HorizonSpatialCrossAttention(nn.Module):
    """The basic structure of HorizonSpatialCrossAttention.

    Args:
        deformable_attention: Deformabel attention module.
        embed_dims: The embedding dimension of Attention.
        num_cams: The num of camera.
        dropout: Probability of an element to be zeroed.
        view_num: The num of view the inputs for mean op.
    """

    def __init__(
        self,
        deformable_attention: nn.Module,
        embed_dims: int = 256,
        num_cams: int = 6,
        dropout: int = 0.1,
        view_num: int = 8,
    ):
        super(HorizonSpatialCrossAttention, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = deformable_attention
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.mean = FF()
        self.add_res = FF()
        self.add_query_pos = FF()

        self.view_num = view_num
        self.queries_mean_pad = nn.Conv2d(
            self.num_cams * self.view_num,
            self.view_num,
            1,
            bias=False,
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize for parameters of module."""

        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

        # # init queries_mean_pad weight
        queries_mean_pad_weight = torch.zeros(
            self.queries_mean_pad.weight.size(),
            dtype=self.queries_mean_pad.weight.dtype,
        )
        for i in range(self.view_num):
            for j in range(self.num_cams):
                queries_mean_pad_weight[i, j * self.view_num + i] = (
                    1 / self.num_cams
                )

        self.queries_mean_pad.weight = torch.nn.Parameter(
            queries_mean_pad_weight, requires_grad=False
        )

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor] = None):
        """Add pos embed."""
        return tensor if pos is None else self.add_query_pos.add(tensor, pos)

    @fx_wrap()
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        reference_points_cam: Tensor = None,
        spatial_shapes: Tensor = None,
        query_pos: Tensor = None,
    ) -> Tensor:
        """Forward HorizonSpatialCrossAttention."""
        inp_residual = query
        query = self.with_pos_embed(query, query_pos)

        bs, max_len, _ = query.size()
        D = reference_points_cam.size(3)

        queries_rebatch = query.unsqueeze(1).repeat(1, self.num_cams, 1, 1)

        _, l, _, _ = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims
        )
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims
        )

        reference_points_rebatch = reference_points_cam.permute(1, 0, 2, 3, 4)
        reference_points_rebatch = reference_points_rebatch.reshape(
            bs * self.num_cams, max_len, D, 2
        )

        queries_out = self.deformable_attention(
            query=queries_rebatch.reshape(
                bs * self.num_cams, max_len, self.embed_dims
            ),
            value=value,
            reference_points=reference_points_rebatch,
            spatial_shapes=spatial_shapes,
        )

        queries_out = queries_out.reshape(
            bs, self.num_cams * self.view_num, -1, max_len
        )
        queries1 = (
            self.queries_mean_pad(queries_out).flatten(1, 2).permute(0, 2, 1)
        )
        queries = queries1
        queries = self.output_proj(queries)
        queries = self.add_res.add(self.dropout(queries), inp_residual)
        return queries

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        int16_module = [
            self.output_proj,
            self.add_res,
        ]
        for m in int16_module:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )
        if hasattr(self.deformable_attention, "set_qconfig"):
            self.deformable_attention.set_qconfig()

    def fix_weight_qscale(self) -> None:
        """Fix the qscale of conv weight when calibration or qat stage."""

        self.queries_mean_pad.weight_fake_quant.disable_observer()

        weight_scale = torch.ones(
            self.queries_mean_pad.weight.shape[0],
            device=self.queries_mean_pad.weight.device,
        )

        weight_scale[...] = (1.0 / self.num_cams) / 128.0
        self.queries_mean_pad.weight_fake_quant.set_qparams(weight_scale)


@OBJECT_REGISTRY.register
class HorizonMSDeformableAttention3D(nn.Module):
    """The basic structure of HorizonMSDeformableAttention3D.

    Args:
        embed_dims: The embedding dimension of Attention.
        num_heads: Parallel attention heads.
        num_levels: The num of featuremap.
        num_points: The num points for each head sample.
        view_gird_in: The num of view the input for gridsample.
        view_gird_out: The num of view the output for gridsample.
        feats_size: The Size of featmaps.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        view_gird_in: int = 32,
        view_gird_out: int = 1,
        feats_size: Sequence[Sequence[int]] = ((28, 16),),
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )

        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.softmax = nn.Softmax(-1)
        self.cat_sampling = FF()
        self.mul_attention = FF()
        self.offset_ref_cat = FF()
        self.norm_offsets = nn.Linear(
            self.num_levels * self.num_heads * self.num_points * 2,
            self.num_levels * self.num_heads * self.num_points * 2,
            bias=False,
        )
        self.sum_ref_offset = nn.Linear(
            self.num_levels * self.num_heads * self.num_points * 2 * 2,
            self.num_levels * self.num_heads * self.num_points * 2,
            bias=False,
        )

        self.norm_locations = nn.Linear(
            self.num_levels * self.num_heads * self.num_points * 2,
            self.num_levels * self.num_heads * self.num_points * 2,
        )

        self.feats_size = feats_size
        self.view_gird_in = view_gird_in
        self.view_gird_out = view_gird_out

        self.sum_out = nn.Linear(
            self.view_gird_out * self.num_points * self.num_levels,
            self.view_gird_out,
            bias=False,
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize for parameters of module."""
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)

        # init offsetnorm weight
        norm_offsets_weight = torch.zeros(
            self.norm_offsets.weight.size(),
            dtype=self.norm_offsets.weight.dtype,
        )

        for i in range(self.num_heads):
            for j in range(self.num_levels):
                x_scale = 1 / self.feats_size[j][0]
                y_scale = 1 / self.feats_size[j][1]
                for k in range(self.num_points):
                    offset_point = (
                        (i * self.num_levels + j) * self.num_points + k
                    ) * 2
                    norm_offsets_weight[offset_point, offset_point] = x_scale
                    norm_offsets_weight[
                        offset_point + 1, offset_point + 1
                    ] = y_scale
        self.norm_offsets.weight = torch.nn.Parameter(
            norm_offsets_weight, requires_grad=False
        )

        # init_sum_ref_offset weight
        sum_ref_offset_weight = torch.zeros(
            self.sum_ref_offset.weight.size(),
            dtype=self.sum_ref_offset.weight.dtype,
        )
        all_num_points = self.num_levels * self.num_heads * self.num_points * 2
        for i in range(all_num_points):
            sum_ref_offset_weight[i, i] = 1
            sum_ref_offset_weight[i, i + all_num_points] = 1
        self.sum_ref_offset.weight = torch.nn.Parameter(
            sum_ref_offset_weight, requires_grad=False
        )

        # init_norm_locations weight
        norm_locations_weight = torch.zeros(
            self.norm_locations.weight.size(),
            dtype=self.norm_locations.weight.dtype,
        )
        norm_locations_bias = torch.zeros(
            self.norm_locations.bias.size(),
            dtype=self.norm_locations.bias.dtype,
        )
        for i in range(all_num_points):
            norm_locations_weight[i, i] = 2
            norm_locations_bias[i] = -1
        self.norm_locations.weight = torch.nn.Parameter(
            norm_locations_weight, requires_grad=False
        )
        self.norm_locations.bias = torch.nn.Parameter(
            norm_locations_bias, requires_grad=False
        )

        # init sumout weight
        sumout_weight = torch.zeros(
            self.sum_out.weight.size(), dtype=self.sum_out.weight.dtype
        )

        for i in range(self.view_gird_out):
            sumout_weight[
                i,
                i
                * self.num_levels
                * self.num_points : (i + 1)
                * self.num_levels
                * self.num_points,
            ] = 1

        self.sum_out.weight = torch.nn.Parameter(
            sumout_weight, requires_grad=False
        )

    def multi_scale_deformable_attn_pytorch(
        self,
        value: Tensor,
        spatial_shapes: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        """Get the multi scale deformable attention."""

        bs, _, num_heads, embed_dims = value.shape

        split_size = tuple(
            [int(H_.item()) * int(W_.item()) for W_, H_ in spatial_shapes]
        )

        value_list = value.split(split_size, dim=1)

        sampling_value_list = []
        sampling_grids = torch.split(sampling_locations, 1, dim=-2)

        for level, (W_, H_) in enumerate(spatial_shapes):
            value_l_ = (
                value_list[level]
                .flatten(2)
                .transpose(1, 2)
                .reshape(
                    bs * num_heads, embed_dims, int(H_.item()), int(W_.item())
                )
            )
            sampling_grid_l_ = sampling_grids[level].squeeze(-2).flatten(0, 1)
            sampling_grid_l_ = sampling_grid_l_.reshape(
                bs * num_heads, -1, self.view_gird_in, 2
            )
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_l_ = sampling_value_l_.reshape(
                bs * num_heads,
                embed_dims,
                -1,
                self.view_gird_out,
                self.num_points,
            )
            sampling_value_list.append(sampling_value_l_)

        sampling_value_list_outs = self.cat_sampling.cat(
            sampling_value_list, dim=-1
        )
        sampling_value_list_outs = sampling_value_list_outs.reshape(
            bs * num_heads,
            embed_dims,
            -1,
            self.view_gird_out * self.num_levels * self.num_points,
        )
        sampling_value_list_outs_tmp = self.mul_attention.mul(
            sampling_value_list_outs, attention_weights
        )

        sampling_value_list_outs_tmp = self.sum_out(
            sampling_value_list_outs_tmp
        ).reshape(bs, self.num_heads * embed_dims, -1)

        return sampling_value_list_outs_tmp

    @fx_wrap()
    def forward(
        self,
        query: Tensor,
        value: Tensor = None,
        reference_points: Tensor = None,
        spatial_shapes: Tensor = None,
    ) -> Tensor:
        """Forward HorizonSpatialCrossAttention."""

        if value is None:
            value = query

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.reshape(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        attention_weights = self.attention_weights(query)

        attention_weights = attention_weights.reshape(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        # bs*head, num_levels*num_points, num_query
        attention_weights = self.softmax(attention_weights)

        attention_weights = (
            attention_weights.reshape(
                bs,
                -1,
                self.view_gird_out,
                self.num_heads,
                self.num_levels * self.num_points,
            )
            .permute(0, 3, 1, 2, 4)
            .flatten(0, 1)
            .flatten(-2)
            .unsqueeze(1)
        )

        sampling_locations = self.get_sampling_locations(
            reference_points, sampling_offsets
        )

        output = self.multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        return output

    def get_sampling_locations(
        self, reference_points: Tensor, sampling_offsets: Tensor
    ) -> Tensor:
        """Get the sampling locations."""
        # reference_points: bs, num_query, num_Z_anchors,  2
        # sampling_offsets: bs, num_query, num_heads, num_levels, num_points, 2

        if reference_points.shape[-1] == 2:
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            repeat_num_points = (
                self.num_heads
                * self.num_levels
                * (self.num_points // num_Z_anchors)
            )

            reference_points = reference_points.repeat(
                1, 1, repeat_num_points, 1
            ).flatten(-2)
            # bs, num_query, (num_heads, num_levels, num_points,  2)

            sampling_offsets = sampling_offsets.flatten(-4)
            # bs, num_query, (num_heads, num_levels, num_points,  2)

            normed_offsets = self.norm_offsets(sampling_offsets)
            # bs, num_query, (num_heads, num_levels, num_points,  2)

            offset_ref_concat = self.offset_ref_cat.cat(
                [normed_offsets, reference_points], dim=-1
            )
            # bs, num_query, 2 * (num_heads, num_levels, num_points,  2)

            offset_ref_add = self.sum_ref_offset(offset_ref_concat)
            # bs, num_query, (num_heads, num_levels, num_points,  2)

            loacations = self.norm_locations(offset_ref_add)  # *2-1
            # bs, num_query, (num_heads, num_levels, num_points,  2)

            loacations = loacations.reshape(
                bs,
                num_query,
                self.num_heads,
                self.num_levels,
                self.num_points,
                xy,
            ).permute(0, 2, 1, 4, 3, 5)

            # bs, num_heads, num_query, num_points, num_levels, xy
            return loacations
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2, but get {reference_points.shape[-1]} instead."
            )

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        int16_module = [
            self.sampling_offsets,
            self.norm_offsets,
            self.offset_ref_cat,
            self.sum_ref_offset,
            self.norm_locations,
        ]
        for m in int16_module:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )

    def fix_weight_qscale(self) -> None:
        """Fix the qscale of conv weight when calibration or qat stage."""

        self.norm_offsets.weight_fake_quant.disable_observer()
        weight_scale_norm_offsets = torch.ones(
            self.norm_offsets.weight.shape[0],
            device=self.norm_offsets.weight.device,
        )
        flat_list = [item for sublist in self.feats_size for item in sublist]
        min_shape = min(flat_list)
        weight_scale_norm_offsets[...] = (1.0 / min_shape) / 128.0
        self.norm_offsets.weight_fake_quant.set_qparams(
            weight_scale_norm_offsets
        )

        self.sum_ref_offset.weight_fake_quant.disable_observer()
        weight_scale_sum_ref_offset = torch.ones(
            self.sum_ref_offset.weight.shape[0],
            device=self.sum_ref_offset.weight.device,
        )
        weight_scale_sum_ref_offset[...] = 1.0 / 128
        self.sum_ref_offset.weight_fake_quant.set_qparams(
            weight_scale_sum_ref_offset
        )

        self.norm_locations.weight_fake_quant.disable_observer()
        weight_scale_norm_locations = torch.ones(
            self.norm_locations.weight.shape[0],
            device=self.norm_locations.weight.device,
        )
        weight_scale_norm_locations[...] = (2.0) / 128.0

        self.norm_locations.weight_fake_quant.set_qparams(
            weight_scale_norm_locations
        )

        self.sum_out.weight_fake_quant.disable_observer()
        weight_scale_sum_out = torch.ones(
            self.sum_out.weight.shape[0], device=self.sum_out.weight.device
        )
        weight_scale_sum_out[...] = (1.0) / 128.0
        self.sum_out.weight_fake_quant.set_qparams(weight_scale_sum_out)


@OBJECT_REGISTRY.register
class HorizonTemporalSelfAttention(nn.Module):
    """The basic structure of HorizonTemporalSelfAttention.

    Args:
        embed_dims: The embedding dimension of Attention.
        num_heads: Parallel attention heads.
        num_levels: The num of featuremap.
        num_points: The num points for each head sample.
        num_bev_queue: The num queue for temporal fusion.
        dropout: Probability of an element to be zeroed.
        view_gird_in: The num of view the input for gridsample.
        view_gird_out: The num of view the output for gridsample.
        view_num: The num of view the inputs for mean op.
        feats_size: The Size of featmaps.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        num_bev_queue: int = 2,
        dropout: float = 0.1,
        view_gird_in: int = 32,
        view_gird_out: int = 1,
        view_num: int = 1,
        feats_size: Sequence[Sequence[int]] = ((128, 128),),
    ) -> None:

        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        self.dropout = nn.Dropout(dropout)

        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dims * self.num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2,
        )
        self.attention_weights = nn.Linear(
            embed_dims * self.num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points,
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.cat_query_pos = FF()
        self.cat_query_value = FF()
        self.softmax = nn.Softmax(-1)
        self.cat_sampling = FF()
        self.mul_attention = FF()
        self.add_res = FF()
        self.offset_ref_cat = FF()
        self.add_pos = nn.Linear(
            self.embed_dims * 2,
            self.embed_dims,
            bias=False,
        )
        self.feats_size = feats_size
        self.norm_offsets = nn.Linear(
            self.num_levels * self.num_heads * self.num_points * 2,
            self.num_levels * self.num_heads * self.num_points * 2,
            bias=False,
        )

        self.sum_ref_offset = nn.Linear(
            self.num_levels * self.num_heads * self.num_points * 2 * 2,
            self.num_levels * self.num_heads * self.num_points * 2,
            bias=False,
        )

        self.norm_locations = nn.Linear(
            self.num_levels * self.num_heads * self.num_points * 2,
            self.num_levels * self.num_heads * self.num_points * 2,
        )
        self.view_gird_out = view_gird_out
        self.view_gird_in = view_gird_in

        self.sum_out = nn.Linear(
            self.view_gird_out * self.num_points * self.num_levels,
            self.view_gird_out,
            bias=False,
        )
        self.view_num = view_num

        self.queries_mean_pad = nn.Conv2d(
            self.num_bev_queue * self.view_num,
            self.view_num,
            1,
            bias=False,
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize for parameters of module."""
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(
                1, self.num_levels * self.num_bev_queue, self.num_points, 1
            )
        )

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

        # init_add_pos

        add_pos_weight = torch.zeros(
            self.add_pos.weight.size(), dtype=self.add_pos.weight.dtype
        )

        for i in range(self.embed_dims):
            add_pos_weight[i, i] = 1
            add_pos_weight[i, i + self.embed_dims] = 1
        self.add_pos.weight = torch.nn.Parameter(
            add_pos_weight, requires_grad=False
        )

        # init offsetnorm weight
        norm_offsets_weight = torch.zeros(
            self.norm_offsets.weight.size(),
            dtype=self.norm_offsets.weight.dtype,
        )

        for i in range(self.num_heads):
            for j in range(self.num_levels):
                x_scale = 1 / self.feats_size[j][0]
                y_scale = 1 / self.feats_size[j][1]
                for k in range(self.num_points):
                    offset_point = (
                        (i * self.num_levels + j) * self.num_points + k
                    ) * 2
                    norm_offsets_weight[offset_point, offset_point] = x_scale
                    norm_offsets_weight[
                        offset_point + 1, offset_point + 1
                    ] = y_scale
        self.norm_offsets.weight = torch.nn.Parameter(
            norm_offsets_weight, requires_grad=False
        )

        # init_sum_ref_offset weight
        sum_ref_offset_weight = torch.zeros(
            self.sum_ref_offset.weight.size(),
            dtype=self.sum_ref_offset.weight.dtype,
        )
        all_num_points = self.num_levels * self.num_heads * self.num_points * 2
        for i in range(all_num_points):
            sum_ref_offset_weight[i, i] = 1
            sum_ref_offset_weight[i, i + all_num_points] = 1
        self.sum_ref_offset.weight = torch.nn.Parameter(
            sum_ref_offset_weight, requires_grad=False
        )

        # init_norm_locations weight
        norm_locations_weight = torch.zeros(
            self.norm_locations.weight.size(),
            dtype=self.norm_locations.weight.dtype,
        )
        norm_locations_bias = torch.zeros(
            self.norm_locations.bias.size(),
            dtype=self.norm_locations.bias.dtype,
        )
        for i in range(all_num_points):
            norm_locations_weight[i, i] = 2
            norm_locations_bias[i] = -1
        self.norm_locations.weight = torch.nn.Parameter(
            norm_locations_weight, requires_grad=False
        )
        self.norm_locations.bias = torch.nn.Parameter(
            norm_locations_bias, requires_grad=False
        )

        # init sumout weight
        sumout_weight = torch.zeros(
            self.sum_out.weight.size(), dtype=self.sum_out.weight.dtype
        )

        for i in range(self.view_gird_out):
            start_idx = i * self.num_levels * self.num_points
            end_idx = (i + 1) * self.num_levels * self.num_points
            sumout_weight[i, start_idx:end_idx] = 1

        self.sum_out.weight = torch.nn.Parameter(
            sumout_weight, requires_grad=False
        )

        # # init queries_mean_pad weight
        queries_mean_weight = torch.zeros(
            self.queries_mean_pad.weight.size(),
            dtype=self.queries_mean_pad.weight.dtype,
        )
        for i in range(self.view_num):
            for j in range(self.num_bev_queue):
                queries_mean_weight[i, j * self.view_num + i] = (
                    1 / self.num_bev_queue
                )

        self.queries_mean_pad.weight = torch.nn.Parameter(
            queries_mean_weight, requires_grad=False
        )

    def multi_scale_deformable_attn_pytorch(
        self,
        value: Tensor,
        spatial_shapes: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        """Get the multi scale deformable attention."""
        bs, _, num_heads, embed_dims = value.shape

        split_size = tuple(
            [int(H_.item()) * int(W_.item()) for W_, H_ in spatial_shapes]
        )

        value_list = value.split(split_size, dim=1)
        sampling_grids = torch.split(sampling_locations, 1, dim=-2)

        sampling_value_list = []
        for level, (W_, H_) in enumerate(spatial_shapes):

            value_l_ = (
                value_list[level]
                .flatten(2)
                .transpose(1, 2)
                .reshape(
                    bs * num_heads, embed_dims, int(H_.item()), int(W_.item())
                )
            )

            sampling_grid_l_ = sampling_grids[level].squeeze(-2)
            sampling_grid_l_ = sampling_grid_l_.reshape(
                bs * num_heads, -1, self.view_gird_in, 2
            )
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_l_ = sampling_value_l_.reshape(
                bs * num_heads,
                embed_dims,
                -1,
                self.view_gird_out,
                self.num_points,
            )
            sampling_value_list.append(sampling_value_l_)

        sampling_value_list_outs = self.cat_sampling.cat(
            sampling_value_list, dim=-1
        )
        sampling_value_list_outs = sampling_value_list_outs.flatten(-2)
        sampling_value_outs = self.mul_attention.mul(
            sampling_value_list_outs, attention_weights
        )
        sampling_value_outs = self.sum_out(sampling_value_outs).reshape(
            bs, self.num_heads * embed_dims, -1
        )
        return sampling_value_outs

    @fx_wrap()
    def forward(
        self,
        query: Tensor,
        value: Tensor = None,
        query_pos: Tensor = None,
        reference_points: Tensor = None,
        spatial_shapes: Tensor = None,
    ) -> Tensor:
        """Forward HorizonTemporalSelfAttention."""
        identity = query
        if query_pos is not None:
            query = self.cat_query_pos.cat([query, query_pos], dim=-1)
            query = self.add_pos(query)

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape

        value = value.reshape(bs, self.num_bev_queue, num_value, -1)
        prev_value = torch.split(value, (self.num_bev_queue - 1), dim=1)
        prev_value = prev_value[0].flatten(0, 1)
        query = self.cat_query_value.cat([prev_value, query], -1)

        value = self.value_proj(value)
        value = value.reshape(
            bs * self.num_bev_queue, num_value, self.num_heads, -1
        )

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
            2,
        )

        attention_weights = self.attention_weights(query).reshape(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels * self.num_points,
        )

        attention_weights = self.softmax(attention_weights)

        attention_weights = (
            attention_weights.permute(0, 3, 2, 1, 4)
            .reshape(
                bs * self.num_bev_queue,
                self.num_heads,
                -1,
                self.view_gird_out * self.num_levels * self.num_points,
            )
            .flatten(0, 1)
            .unsqueeze(1)
            .contiguous()
        )

        sampling_offsets = sampling_offsets.permute(
            0, 3, 1, 2, 4, 5, 6
        ).reshape(
            bs * self.num_bev_queue,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )

        sampling_locations = (
            self.get_sampling_locations(reference_points, sampling_offsets)
            .flatten(0, 1)
            .contiguous()
        )
        output = self.multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        output = output.reshape(
            bs, self.num_bev_queue * self.view_num, -1, num_query
        )
        output = self.queries_mean_pad(output).flatten(1, 2).permute(0, 2, 1)
        output = self.output_proj(output)
        output = self.add_res.add(self.dropout(output), identity)

        return output

    def get_sampling_locations(
        self, reference_points: Tensor, sampling_offsets: Tensor
    ) -> Tensor:
        """Get the sampling locations."""

        if reference_points.shape[-1] == 2:
            bs, num_query, _, _ = reference_points.shape
            sampling_offsets = sampling_offsets.flatten(-4)
            normed_offsets = self.norm_offsets(sampling_offsets)
            reference_points = reference_points.repeat(
                1, 1, self.num_heads, self.num_points
            ).flatten(-2)
            offset_ref_concat = self.offset_ref_cat.cat(
                [normed_offsets, reference_points], dim=-1
            )
            offset_ref_add = self.sum_ref_offset(offset_ref_concat)
            loacations = self.norm_locations(offset_ref_add)
            loacations = loacations.reshape(
                bs,
                num_query,
                self.num_heads,
                self.num_levels,
                self.num_points,
                2,
            ).permute(0, 2, 1, 4, 3, 5)

            return loacations
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2, but get {reference_points.shape[-1]} instead."
            )

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        int16_module = [
            self.norm_offsets,
            self.offset_ref_cat,
            self.sampling_offsets,
            self.output_proj,
            self.sum_ref_offset,
            self.norm_locations,
            self.add_res,
        ]
        for m in int16_module:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )

    def fix_weight_qscale(self) -> None:
        """Fix the qscale of conv weight when calibration or qat stage."""

        self.norm_offsets.weight_fake_quant.disable_observer()
        weight_scale_norm_offsets = torch.ones(
            self.norm_offsets.weight.shape[0],
            device=self.norm_offsets.weight.device,
        )
        flat_list = [item for sublist in self.feats_size for item in sublist]
        min_shape = min(flat_list)
        weight_scale_norm_offsets[...] = (1.0 / min_shape) / 128.0
        self.norm_offsets.weight_fake_quant.set_qparams(
            weight_scale_norm_offsets
        )
        self.sum_ref_offset.weight_fake_quant.disable_observer()
        weight_scale_sum_ref_offset = torch.ones(
            self.sum_ref_offset.weight.shape[0],
            device=self.sum_ref_offset.weight.device,
        )
        weight_scale_sum_ref_offset[...] = 1.0 / 128
        self.sum_ref_offset.weight_fake_quant.set_qparams(
            weight_scale_sum_ref_offset
        )
        self.norm_locations.weight_fake_quant.disable_observer()
        weight_scale_norm_locations = torch.ones(
            self.norm_locations.weight.shape[0],
            device=self.norm_locations.weight.device,
        )
        weight_scale_norm_locations[...] = (2.0) / 128.0
        self.norm_locations.weight_fake_quant.set_qparams(
            weight_scale_norm_locations
        )

        self.sum_out.weight_fake_quant.disable_observer()
        weight_scale_sum_out = torch.ones(
            self.sum_out.weight.shape[0], device=self.sum_out.weight.device
        )
        weight_scale_sum_out[...] = (1.0) / 128.0
        self.sum_out.weight_fake_quant.set_qparams(weight_scale_sum_out)

        self.add_pos.weight_fake_quant.disable_observer()
        weight_scale_add_pos = torch.ones(
            self.add_pos.weight.shape[0], device=self.add_pos.weight.device
        )
        weight_scale_add_pos[...] = (1.0) / 128.0
        self.add_pos.weight_fake_quant.set_qparams(weight_scale_add_pos)

        self.queries_mean_pad.weight_fake_quant.disable_observer()
        weight_scale_queries_mean_pad = torch.ones(
            self.queries_mean_pad.weight.shape[0],
            device=self.queries_mean_pad.weight.device,
        )
        weight_scale_queries_mean_pad[...] = (1.0 / self.num_bev_queue) / 128.0
        self.queries_mean_pad.weight_fake_quant.set_qparams(
            weight_scale_queries_mean_pad
        )


@OBJECT_REGISTRY.register
class HorizonMSDeformableAttention(nn.Module):
    """The basic structure of HorizonMSDeformableAttention.

    Args:
        embed_dims: The embedding dimension of Attention.
        num_heads: Parallel attention heads.
        num_levels: The num of featuremap.
        num_points: The num points for each head sample.
        dropout: Probability of an element to be zeroed.
        batch_first: Wheather the first dim is batch.
        view_gird_in: The num of view the input for gridsample.
        view_gird_out: The num of view the output for gridsample.
        view_num: The num of view the inputs for mean op.
        feats_size: The Size of featmaps.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        batch_first: bool = False,
        view_gird_in: int = 1,
        view_gird_out: int = 1,
        feats_size: Sequence[Sequence[int]] = ((128, 128),),
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.cat_sampling = FF()
        self.mul_attention = FF()
        self.add_res = FF()

        self.cat_query_pos = FF()
        self.offset_ref_cat = FF()

        self.add_pos = nn.Linear(
            self.embed_dims * 2,
            self.embed_dims,
            bias=False,
        )
        self.norm_offsets = nn.Linear(
            self.num_levels * self.num_heads * self.num_points * 2,
            self.num_levels * self.num_heads * self.num_points * 2,
            bias=False,
        )
        self.sum_ref_offset = nn.Linear(
            self.num_levels * self.num_heads * self.num_points * 2 * 2,
            self.num_levels * self.num_heads * self.num_points * 2,
            bias=False,
        )
        self.norm_locations = nn.Linear(
            self.num_levels * self.num_heads * self.num_points * 2,
            self.num_levels * self.num_heads * self.num_points * 2,
        )
        self.feats_size = feats_size
        self.view_gird_out = view_gird_out
        self.view_gird_in = view_gird_in

        self.sum_out = nn.Linear(
            self.view_gird_out * self.num_points * self.num_levels,
            self.view_gird_out,
            bias=False,
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize for parameters of module."""
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

        # init_add_pos

        add_pos_weight = torch.zeros(
            self.add_pos.weight.size(), dtype=self.add_pos.weight.dtype
        )

        for i in range(self.embed_dims):
            add_pos_weight[i, i] = 1
            add_pos_weight[i, i + self.embed_dims] = 1
        self.add_pos.weight = torch.nn.Parameter(
            add_pos_weight, requires_grad=False
        )

        # init offsetnorm weight
        norm_offsets_weight = torch.zeros(
            self.norm_offsets.weight.size(),
            dtype=self.norm_offsets.weight.dtype,
        )

        for i in range(self.num_heads):
            for j in range(self.num_levels):
                x_scale = 1 / self.feats_size[j][0]
                y_scale = 1 / self.feats_size[j][1]
                for k in range(self.num_points):
                    offset_point = (
                        (i * self.num_levels + j) * self.num_points + k
                    ) * 2
                    norm_offsets_weight[offset_point, offset_point] = x_scale
                    norm_offsets_weight[
                        offset_point + 1, offset_point + 1
                    ] = y_scale
        self.norm_offsets.weight = torch.nn.Parameter(
            norm_offsets_weight, requires_grad=False
        )
        # init_sum_ref_offset weight
        sum_ref_offset_weight = torch.zeros(
            self.sum_ref_offset.weight.size(),
            dtype=self.sum_ref_offset.weight.dtype,
        )

        all_num_points = self.num_levels * self.num_heads * self.num_points * 2
        for i in range(all_num_points):
            sum_ref_offset_weight[i, i] = 1
            sum_ref_offset_weight[i, i + all_num_points] = 1
        self.sum_ref_offset.weight = torch.nn.Parameter(
            sum_ref_offset_weight, requires_grad=False
        )
        # init_norm_locations weight
        norm_locations_weight = torch.zeros(
            self.norm_locations.weight.size(),
            dtype=self.norm_locations.weight.dtype,
        )
        norm_locations_bias = torch.zeros(
            self.norm_locations.bias.size(),
            dtype=self.norm_locations.bias.dtype,
        )
        for i in range(all_num_points):
            norm_locations_weight[i, i] = 2
            norm_locations_bias[i] = -1
        self.norm_locations.weight = torch.nn.Parameter(
            norm_locations_weight, requires_grad=False
        )
        self.norm_locations.bias = torch.nn.Parameter(
            norm_locations_bias, requires_grad=False
        )
        # init sumout weight
        sumout_weight = torch.zeros(
            self.sum_out.weight.size(), dtype=self.sum_out.weight.dtype
        )

        for i in range(self.view_gird_out):
            start_idx = i * self.num_levels * self.num_points
            end_idx = (i + 1) * self.num_levels * self.num_points
            sumout_weight[i, start_idx:end_idx] = 1

        self.sum_out.weight = torch.nn.Parameter(
            sumout_weight, requires_grad=False
        )

    @fx_wrap()
    def forward(
        self,
        query: Tensor,
        value: Tensor = None,
        query_pos: Tensor = None,
        reference_points: Tensor = None,
        spatial_shapes: Tensor = None,
    ) -> Tensor:
        """Forward HorizonMSDeformableAttention."""
        if value is None:
            value = query
        identity = query
        if query_pos is not None:
            query = self.cat_query_pos.cat([query, query_pos], dim=-1)
            query = self.add_pos(query)
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        value = self.value_proj(value)
        value = value.reshape(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        attention_weights = self.attention_weights(query).reshape(
            bs,
            num_query,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        attention_weights = self.softmax(attention_weights)
        attention_weights = (
            attention_weights.permute(0, 2, 1, 3)
            .reshape(
                bs,
                self.num_heads,
                -1,
                self.view_gird_out * self.num_levels * self.num_points,
            )
            .flatten(0, 1)
            .unsqueeze(1)
            .contiguous()
        )

        sampling_locations = self.get_sampling_locations(
            reference_points, sampling_offsets
        )
        output = self.multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = output.permute(0, 2, 1)
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        output = self.add_res.add(self.dropout(output), identity)
        return output

    def get_sampling_locations(
        self,
        reference_points: Tensor,
        sampling_offsets: Tensor,
    ) -> Tensor:
        """Get the sampling locations."""

        if reference_points.shape[-1] == 2:
            bs, num_query, _, _ = reference_points.shape
            normed_offsets = self.norm_offsets(sampling_offsets)
            reference_points = reference_points.repeat(
                1, 1, self.num_heads, self.num_points
            ).flatten(-2)
            offset_ref_concat = self.offset_ref_cat.cat(
                [normed_offsets, reference_points], dim=-1
            )
            offset_ref_add = self.sum_ref_offset(offset_ref_concat)
            loacations = self.norm_locations(offset_ref_add)
            loacations = loacations.reshape(
                bs,
                num_query,
                self.num_heads,
                self.num_levels,
                self.num_points,
                2,
            )
            loacations = loacations.reshape(
                bs,
                num_query,
                self.num_heads,
                self.num_levels,
                self.num_points,
                2,
            ).permute(0, 2, 1, 4, 3, 5)

        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        return loacations

    def multi_scale_deformable_attn_pytorch(
        self,
        value: Tensor,
        spatial_shapes: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        """Get the multi scale deformable attention."""
        bs, _, num_heads, embed_dims = value.shape
        split_size = tuple(
            [int(H_.item()) * int(W_.item()) for W_, H_ in spatial_shapes]
        )
        value_list = value.split(split_size, dim=1)
        sampling_grids = torch.split(sampling_locations, 1, dim=-2)
        sampling_value_list = []
        for level, (W_, H_) in enumerate(spatial_shapes):
            value_l_ = (
                value_list[level]
                .flatten(2)
                .transpose(1, 2)
                .reshape(
                    bs * num_heads, embed_dims, int(H_.item()), int(W_.item())
                )
            )

            sampling_grid_l_ = sampling_grids[level].squeeze(-2)
            sampling_grid_l_ = sampling_grid_l_.reshape(
                bs * num_heads, -1, self.view_gird_in, 2
            )
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            sampling_value_l_ = sampling_value_l_.reshape(
                bs * num_heads,
                embed_dims,
                -1,
                self.view_gird_out,
                self.num_points,
            )
            sampling_value_list.append(sampling_value_l_)

        sampling_value_list_outs = self.cat_sampling.cat(
            sampling_value_list, dim=-1
        )
        sampling_value_list_outs = sampling_value_list_outs.flatten(-2)
        sampling_value_outs = self.mul_attention.mul(
            sampling_value_list_outs, attention_weights
        )
        sampling_value_outs = self.sum_out(sampling_value_outs).reshape(
            bs, self.num_heads * embed_dims, -1
        )
        return sampling_value_outs

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        int16_module = [
            self.sampling_offsets,
            self.norm_offsets,
            self.offset_ref_cat,
            self.sum_ref_offset,
            self.norm_locations,
            self.add_res,
            self.output_proj,
        ]
        for m in int16_module:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )

    def fix_weight_qscale(self) -> None:
        """Fix the qscale of conv weight when calibration or qat stage."""

        self.norm_offsets.weight_fake_quant.disable_observer()
        weight_scale_norm_offsets = torch.ones(
            self.norm_offsets.weight.shape[0],
            device=self.norm_offsets.weight.device,
        )
        flat_list = [item for sublist in self.feats_size for item in sublist]
        min_shape = min(flat_list)
        weight_scale_norm_offsets[...] = (1.0 / min_shape) / 128.0
        self.norm_offsets.weight_fake_quant.set_qparams(
            weight_scale_norm_offsets
        )

        self.sum_ref_offset.weight_fake_quant.disable_observer()
        weight_scale_sum_ref_offset = torch.ones(
            self.sum_ref_offset.weight.shape[0],
            device=self.sum_ref_offset.weight.device,
        )
        weight_scale_sum_ref_offset[...] = 1.0 / 128
        self.sum_ref_offset.weight_fake_quant.set_qparams(
            weight_scale_sum_ref_offset
        )

        self.norm_locations.weight_fake_quant.disable_observer()
        weight_scale_norm_locations = torch.ones(
            self.norm_locations.weight.shape[0],
            device=self.norm_locations.weight.device,
        )
        weight_scale_norm_locations[...] = (2.0) / 128.0
        self.norm_locations.weight_fake_quant.set_qparams(
            weight_scale_norm_locations
        )

        self.sum_out.weight_fake_quant.disable_observer()
        weight_scale_sum_out = torch.ones(
            self.sum_out.weight.shape[0], device=self.sum_out.weight.device
        )
        weight_scale_sum_out[...] = (1.0) / 128.0
        self.sum_out.weight_fake_quant.set_qparams(weight_scale_sum_out)

        self.add_pos.weight_fake_quant.disable_observer()
        weight_scale_add_pos = torch.ones(
            self.add_pos.weight.shape[0], device=self.add_pos.weight.device
        )
        weight_scale_add_pos[...] = (1.0) / 128.0
        self.add_pos.weight_fake_quant.set_qparams(weight_scale_add_pos)
