import torch
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from torch import nn

from hat.models.task_modules.e2e_dynamic.e2e_dynamic import (
    qint8_qconfig,
    qint16_qconfig,
)
from hat.models.task_modules.e2e_dynamic.e2e_dynamic_util import (
    FFN,
    SingleHeadSelfAttention,
)
from hat.registry import OBJECT_REGISTRY

__all__ = ["QuerySpatialInteractionModule"]


@OBJECT_REGISTRY.register
class QuerySpatialInteractionModule(nn.Module):
    """Query Interaction Module.

    The module is used to update query embedding between two frames.
    Reference: https://arxiv.org/abs/2105.03247.

    Args:
        update_query_pos: whether to update query_pos.
        merger_dropout: dropout ratio.
        dim_in: input dimension.
        hidden_dim: hidden dimension of FFN.
        dim_out: output dimension.
        replace_identity_with_tgt: whether to replace tgt with identity.
    """

    def __init__(
        self,
        update_query_pos: bool,
        dropout_ratio: float,
        dim_in: int,
        hidden_dim: int,
        dim_out: int,
        replace_identity_with_tgt: bool,
    ):

        super().__init__()
        self.update_query_pos = update_query_pos
        self.dropout_ratio = dropout_ratio
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.replace_identity_with_tgt = replace_identity_with_tgt
        self._build_layers()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_layers(self):
        self.pos_out_add = FF()
        self.dropout1_add = FF()
        self.self_attn = SingleHeadSelfAttention(self.dim_in)
        self.ffn1 = FFN(self.dim_in, self.hidden_dim, self.dropout_ratio)
        if self.update_query_pos:
            self.ffn2 = FFN(self.dim_in, self.hidden_dim, self.dropout_ratio)
            self.norm_pos = nn.LayerNorm(self.dim_in)

        self.ffn3 = FFN(self.dim_in, self.hidden_dim, self.dropout_ratio)

        self.norm_feat = nn.LayerNorm(self.dim_in)

        self.norm1 = nn.LayerNorm(self.dim_in)
        self.norm2 = nn.LayerNorm(self.dim_in)

        self.dropout1 = nn.Dropout(self.dropout_ratio)
        self.dropout2 = nn.Dropout(self.dropout_ratio)
        self.query_pos_cat = FF()

    def forward(self, query_embed, output_embedding, active_mask):
        dim = query_embed.shape[-1]

        query_pos, query_feat = torch.split(
            query_embed,
            dim // 2,
            dim=3,
        )

        # self-attention
        q = k = self.pos_out_add.add(query_pos, output_embedding)
        tgt = output_embedding
        tgt2 = self.self_attn(q, k, tgt, active_mask)
        tgt = self.dropout1_add.add(tgt, self.dropout1(tgt2))
        tgt = self.norm1(tgt)
        tgt = self.ffn1(tgt, tgt)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            if self.replace_identity_with_tgt:
                query_pos = self.ffn2(tgt, tgt)
            else:
                query_pos = self.ffn2(tgt, query_pos)
            query_pos = self.norm_pos(query_pos)

        if self.replace_identity_with_tgt:
            query_feat = self.ffn3(tgt, tgt)
        else:
            query_feat = self.ffn3(tgt, query_feat)
        query_feat = self.norm_feat(query_feat)

        update_query_embed = self.query_pos_cat.cat(
            [query_pos, query_feat], dim=-1
        )
        return update_query_embed

    def fuse_model(self):
        self.self_attn.fuse_model()
        self.ffn1.fuse_model()
        if self.update_query_pos:
            self.ffn2.fuse_model()
        self.ffn3.fuse_model()

    def set_qconfig(self):
        import horizon_plugin_pytorch as horizon  # noqa

        self.qconfig = qint16_qconfig()
        self.ffn1.set_qconfig()
        self.ffn3.set_qconfig()
        self.norm1.qconfig = qint8_qconfig()
        self.norm2.qconfig = qint8_qconfig()
        if self.update_query_pos:
            self.ffn2.set_qconfig()
            self.norm_pos.qconfig = qint8_qconfig()
        self.norm_feat.qconfig = qint8_qconfig()
        self.query_pos_cat.qconfig = qint8_qconfig()
