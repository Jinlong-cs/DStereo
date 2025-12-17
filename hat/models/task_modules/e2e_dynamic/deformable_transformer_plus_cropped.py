#! this file is tmp implement file, will be refactor, TODO: xiangyu.li, wenming.meng
from typing import Sequence

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import (
    FixedScaleObserver,
    get_default_qat_qconfig,
)

from hat.models.task_modules.e2e_dynamic.e2e_dynamic_util import (
    FFN,
    INV_SIGMOID_MAX,
    QINT16_MAX,
    SIGMOID_MAX,
    Linear,
    MultiHeadCrossAttention,
    SingleHeadSelfAttention,
    _get_clones,
    qint8_qconfig,
)
from hat.registry import OBJECT_REGISTRY

__all__ = ["CroppedDeformableTransformer"]


@OBJECT_REGISTRY.register
class CroppedDeformableTransformer(nn.Module):
    """Cropped deformable transformer removing the encoder.

    Args:
        feat_shape: Multi level feature shape.
        embedding_dim: Dimension of the query embedding.
        num_head: Num of head in the MHSA.
        num_queries: Num of queries in the decoder module, same
            as num_det_queries when detection task, otherwise
            is the sum of num_det_queries and num_track_queries.
        num_det_queries: Num of queries for detection.
        num_decoder_layers: Num of layers in the decoder module.
        feedforward_dim: Dimension of the intermediate embedding
            in the FFN module.
        dropout_ratio: Drop ratio for dropout OP.
        return_intermediate_dec: Whether the outputs contains
            all the output from each decoder layer.
        extra_track_attn: Whether use extra self attention on track queries.
    """

    def __init__(
        self,
        feat_shape: Sequence,
        embedding_dim: int,
        num_head: int,
        num_queries: int,
        num_det_queries: int,
        num_decoder_layers: int,
        feedforward_dim: int,
        dropout_ratio: float,
        return_intermediate_dec: bool = False,
        extra_track_attn: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_head = num_head

        decoder_layer = DeformableTransformerDecoderLayer(
            feat_shape=feat_shape,
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            dropout_ratio=dropout_ratio,
            num_head=num_head,
            extra_track_attn=extra_track_attn,
            num_queries=num_queries,
            num_det_queries=num_det_queries,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            return_intermediate_dec,
        )

        self.reference_points_xy = Linear(embedding_dim, 2)
        self.reference_points_wh = Linear(embedding_dim, 2)
        self.ref_pts_cat = FF()
        self.reference_points_sigmoid = nn.Sigmoid()

    def forward(
        self,
        srcs,
        query_embed=None,
        ref_pts=None,
        xy_range=None,
        active_mask_for_fix_all_query=None,
    ):
        assert query_embed is not None

        # prepare input for decoder
        feat_channel = srcs[0].size(1)
        query_embed, tgt = torch.split(
            query_embed, feat_channel, dim=3
        )  # query是512, 拆成256, 256
        if ref_pts is None:
            cur_xy = self.reference_points_xy(query_embed)
            cur_wh = self.reference_points_wh(query_embed)
            reference_points_xy = self.reference_points_sigmoid(
                self.ref_pts_cat.cat([cur_xy, cur_wh], dim=3)
            )
        else:
            reference_points_init = ref_pts
            reference_points_xy = self.reference_points_sigmoid(
                ref_pts
            )  # ref_pts 需要先过一个sigmoid->[0,1],因为是多尺度的

        # decoder
        (
            inter_embedding,
            inter_references_xy,
            inter_references_wh,
        ) = self.decoder(
            tgt,
            reference_points_xy,
            reference_points_init,
            srcs,
            query_embed,
            xy_range,
            active_mask_for_fix_all_query,
        )

        return (
            inter_embedding,
            inter_references_xy,
            inter_references_wh,
        )

    def fuse_model(self):
        self.decoder.fuse_model()

    def set_qconfig(self):
        self.reference_points_sigmoid.qconfig = get_default_qat_qconfig(
            dtype="qint16",
            activation_qkwargs={
                "observer": FixedScaleObserver,
                "scale": SIGMOID_MAX / QINT16_MAX,
            },
        )
        self.ref_pts_cat.qconfig = get_default_qat_qconfig(
            dtype="qint16",
            activation_qkwargs={
                "observer": FixedScaleObserver,
                "scale": INV_SIGMOID_MAX / QINT16_MAX,
            },
        )
        self.decoder.set_qconfig()


class DeformableTransformerDecoderLayer(nn.Module):
    """Deformable Transformer decoder layer.

    Args:
        feat_shape: Multi level feature shape.
        embedding_dim: Dimension of the query embedding.
        feedforward_dim: Dimension of the intermediate embedding
            in the FFN module.
        dropout_ratio: Drop ratio for dropout OP.
        num_head: Num of head in the MHSA.
        extra_track_attn: Whether use self-attn on the extra track queries.
        num_queries: Num of queries in the decoder module.
    """

    def __init__(
        self,
        feat_shape: Sequence,
        embedding_dim: int,
        feedforward_dim: int,
        dropout_ratio: float,
        num_head: int,
        extra_track_attn: bool,
        num_queries: int,
        num_det_queries: int,
    ):
        super().__init__()

        self.num_head = num_head
        self.num_det_queries = num_det_queries

        # cross attention
        self.cross_attn = MultiHeadCrossAttention(
            embedding_dim,
            num_head=num_head,
            num_queries=num_queries,
            feat_shape=feat_shape,
        )
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.self_attn = SingleHeadSelfAttention(embedding_dim)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # ffn
        self.ffn = FFN(embedding_dim, feedforward_dim, dropout_ratio)
        self.norm3 = nn.LayerNorm(embedding_dim)

        # update track query_embed
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            self.update_attn = SingleHeadSelfAttention(embedding_dim)
            self.dropout5 = nn.Dropout(dropout_ratio)
            self.norm4 = nn.LayerNorm(embedding_dim)
            self.extra_track_attn_cat = FF()
            self.ta_input_add = FF()

        self.ca_dp_add = FF()
        self.sa_dp_add1 = FF()
        self.sa_dp_add2 = FF()
        self.ca_input_add = FF()
        self.add = FF()

    def _forward_self_attn(
        self, tgt, query_pos, active_mask_for_fix_all_query=None
    ):
        if self.extra_track_attn:
            tgt = self._forward_track_attn(
                tgt, query_pos, active_mask_for_fix_all_query
            )

        q = k = self.sa_dp_add1.add(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, active_mask_for_fix_all_query)
        tgt = self.sa_dp_add2.add(tgt, self.dropout2(tgt2))
        tgt = self.norm2(tgt)
        return tgt

    def _forward_track_attn(
        self, tgt, query_pos, active_mask_for_fix_all_query=None
    ):
        tgt_tmp = tgt
        query_pos_tmp = query_pos
        q = self.ta_input_add.add(tgt_tmp, query_pos_tmp)
        if q.shape[2] > self.num_det_queries:
            q_tmp2 = q[:, :, self.num_det_queries :, :]
            tgt_tmp1, tgt_tmp2 = torch.split(
                tgt,
                (self.num_det_queries, tgt.shape[2] - self.num_det_queries),
                dim=2,
            )

            active_mask_for_fix_track_query = active_mask_for_fix_all_query[
                :, :, self.num_det_queries :, :
            ]

            tgt2 = self.update_attn(
                q_tmp2, q_tmp2, tgt_tmp2, active_mask_for_fix_track_query
            )
            tgt2 = self.add.add(tgt_tmp2, self.dropout5(tgt2))
            tgt2 = self.norm4(tgt2)
            tgt = self.extra_track_attn_cat.cat([tgt_tmp1, tgt2], dim=2)

        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        xy_range=None,
        active_mask_for_fix_all_query=None,
    ):
        # cross attention
        tgt2 = self.cross_attn(
            self.ca_input_add.add(tgt, query_pos),
            src,
            reference_points,
            xy_range,
        )
        tgt = self.ca_dp_add.add(
            tgt, self.dropout1(tgt2.unsqueeze(0))
        )  # int16和int8 add？
        tgt = self.norm1(tgt)
        # self attention
        tgt = self._forward_self_attn(
            tgt, query_pos, active_mask_for_fix_all_query
        )
        # ffn
        tgt = self.ffn(tgt, tgt)
        tgt = self.norm3(tgt)

        return tgt

    def fuse_model(self):
        self.cross_attn.fuse_model()
        self.self_attn.fuse_model()
        self.ffn.fuse_model()
        if self.extra_track_attn:
            self.update_attn.fuse_model()

    def set_qconfig(self):
        self.cross_attn.set_qconfig()
        self.ffn.set_qconfig()
        self.norm1.qconfig = qint8_qconfig()
        self.norm2.qconfig = qint8_qconfig()
        self.norm3.qconfig = qint8_qconfig()
        self.norm4.qconfig = qint8_qconfig()


class DeformableTransformerDecoder(nn.Module):
    """Transformer decoder consisting of multiple layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class.
        num_layers: the number of sub-decoder-layers in the decoder.
        return_intermediate: whether to return intermediate results.
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        return_intermediate: bool,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed_xy = None
        self.bbox_embed_wh = None
        self.class_embed = None
        self.new_reference_points_sigmoids = _get_clones(
            nn.Sigmoid(), num_layers - 1
        )
        self.reference_points_add = nn.ModuleList()
        for _ in range(num_layers):
            self.reference_points_add.append(FF())

    def forward(
        self,
        tgt,
        reference_points,
        reference_points_init,
        src,
        query_pos=None,
        xy_range=None,
        active_mask_for_fix_all_query=None,
    ):
        inter_embedding = tgt

        intermediate_output = []
        intermediate_reference_points_xy = []
        intermediate_reference_points_wh = []
        for lid, layer in enumerate(self.layers):
            inter_embedding = layer(
                inter_embedding,
                query_pos,
                reference_points,
                src,
                xy_range,
                active_mask_for_fix_all_query,
            )

            # hack implementation for iterative bounding box refinement
            # decouple xy and wh
            tmp_xy = self.bbox_embed_xy[lid](
                inter_embedding
            )  # 对于xywh在逆sigmoid空间的增量
            new_reference_points_xy = self.reference_points_add[lid].add(
                tmp_xy, reference_points_init
            )
            new_reference_points_wh = self.bbox_embed_wh[lid](inter_embedding)

            if lid < len(self.layers) - 1:
                new_reference_points_xy = self.new_reference_points_sigmoids[
                    lid
                ](new_reference_points_xy)

            if self.return_intermediate:
                intermediate_output.append(inter_embedding)
                intermediate_reference_points_xy.append(
                    new_reference_points_xy
                )
                intermediate_reference_points_wh.append(
                    new_reference_points_wh
                )

        if self.return_intermediate:  # 每个layer输出放在list中，不用cat
            return (
                intermediate_output,
                intermediate_reference_points_xy,
                intermediate_reference_points_wh,
            )

        return (
            inter_embedding,
            new_reference_points_xy,
            new_reference_points_wh,
        )

    def fuse_model(self):
        for layer in self.layers:
            layer.fuse_model()

    def set_qconfig(self):
        for layer in self.layers:
            layer.set_qconfig()
        for layer in self.bbox_embed_xy:
            layer.set_qconfig()
        for layer in self.bbox_embed_wh:
            layer.set_qconfig()
        for layer in self.new_reference_points_sigmoids:
            layer.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": SIGMOID_MAX / QINT16_MAX,
                },
            )
        for layer in self.reference_points_add:
            layer.qconfig = get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": INV_SIGMOID_MAX / QINT16_MAX,
                },
            )
