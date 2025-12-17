import torch
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from torch import nn
from torch.quantization import DeQuantStub, QuantStub

from hat.models.task_modules.e2e_dynamic.e2e_dynamic import (
    qat_out_qconfig,
    qint8_qconfig,
    qint16_qconfig,
)
from hat.models.task_modules.e2e_dynamic.e2e_dynamic_util import (
    FFN,
    MultiHeadMulTemporalAttention,
    PositionEmbeddingLearned,
    SingleHeadSelfAttention,
)
from hat.registry import OBJECT_REGISTRY

__all__ = ["MemoryBankModule"]


@OBJECT_REGISTRY.register
class MemoryBankModule(nn.Module):
    """Memory Bank Module to save history embedding.

    Args:
        memory_bank_len: Length of history embeddings.
        memory_bank_with_self_attn: Whether to use self_attn module.
        memory_bank_with_temp_attn: Whether to use temporal_attn module.
        num_heads: Number of head in temporal_attn module.
        dim_in: The dimension of the embedding input into the MHSA.
        hidden_dim: The dimension of the intermediate embedding.
        dim_out: The dimension of the output embedding.
        out_track_score: Whether output track_score.
        calibration_model: Whether is calibration step.
    """

    def __init__(
        self,
        memory_bank_len: int,
        memory_bank_with_self_attn: bool,
        memory_bank_with_temp_attn: bool,
        num_heads: int,
        dim_in: int,
        hidden_dim: int,
        dim_out: int,
        out_track_score: bool = True,
        calibration_model: bool = False,
    ):
        super().__init__()
        self.memory_bank_len = memory_bank_len
        self.memory_bank_with_self_attn = memory_bank_with_self_attn
        self.memory_bank_with_temp_attn = memory_bank_with_temp_attn
        self.num_heads = num_heads
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.out_track_score = out_track_score
        self.build_layers()
        self.reset_parameters()

        self.emb_odo_add = FF()
        self.emb_fps_queue_add = FF()
        self.prev_emb_odo_add = FF()
        self.emb_add = FF()

        self.query_spatial_add = FF()
        self.emb_spatial_add = FF()

        self.calibration_model = calibration_model
        self.calib_tmp_quant = QuantStub(None)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def build_layers(self):
        if self.memory_bank_with_temp_attn:
            self.temporal_attn = MultiHeadMulTemporalAttention(
                self.dim_in, self.num_heads, self.dim_in // self.num_heads
            )
            self.temporal_ffn = FFN(self.dim_in, self.hidden_dim, dropout=0)
            self.temporal_norm1 = nn.LayerNorm(self.dim_in)
            self.temporal_norm2 = nn.LayerNorm(self.dim_in)
            self.temporal_pos_embedding = PositionEmbeddingLearned(
                (self.memory_bank_len, self.dim_in)
            )
        if self.memory_bank_with_self_attn:
            self.spatial_attn = SingleHeadSelfAttention(self.dim_in)
            self.spatial_ffn = FFN(self.dim_in, self.hidden_dim, dropout=0)
            self.spatial_norm1 = nn.LayerNorm(self.dim_in)
            self.spatial_norm2 = nn.LayerNorm(self.dim_in)

        if self.out_track_score:
            self.track_cls = nn.Linear(self.dim_in, 1)
            self.track_score_dquant = DeQuantStub()
        self.odo_proj = nn.Linear(4, self.dim_in)
        self.fps_queue_proj = nn.Linear(1, self.dim_in)

    def _forward_spatial_attn(self, query_pos, embed, active_mask=None):
        k = q = self.query_spatial_add.add(embed, query_pos)
        v = embed
        embed2 = self.spatial_attn(q, k, v, active_mask)
        embed = self.spatial_norm1(self.emb_spatial_add.add(embed, embed2))
        embed = self.spatial_ffn(embed, embed)

        embed = self.spatial_norm2(embed)
        return embed

    def _forward_temporal_attn(
        self,
        embed,
        mem_bank,
        mem_padding_mask,
        odo_input,
        fps_queue_input,
    ):
        odo_embed = self.odo_proj(odo_input)
        fps_queue_embed = self.fps_queue_proj(fps_queue_input)
        embed = self.emb_odo_add.add(embed, odo_embed[:, -1:, :, :])
        embed = self.emb_fps_queue_add.add(
            embed, fps_queue_embed[:, -1:, :, :]
        )

        prev_embed = mem_bank  # track_instances["mem_bank"]
        prev_embed = self.prev_emb_odo_add.add(
            prev_embed, odo_embed[:, :-1, :, :]
        )
        prev_embed = self.emb_fps_queue_add.add(
            prev_embed, fps_queue_embed[:, :-1, :, :]
        )

        prev_embed_with_pos = self.temporal_pos_embedding(prev_embed)

        key_padding_mask = mem_padding_mask.permute(1, 0, 3, 2)
        if self.calibration_model and key_padding_mask.dequantize().max() <= 0:
            embed2 = torch.zeros_like(
                embed.dequantize(), device=key_padding_mask.device
            )
            embed2 = self.calib_tmp_quant(embed2)
        else:
            embed2 = self.temporal_attn(
                embed,
                prev_embed_with_pos.permute(1, 0, 2, 3),
                prev_embed.permute(1, 0, 2, 3),
                key_padding_mask,
            )

        embed = self.temporal_norm1(self.emb_add.add(embed, embed2))
        embed = self.temporal_ffn(embed, embed)
        embed = self.temporal_norm2(embed)
        return embed

    def forward(
        self,
        query_pos,
        output_embedding,
        mem_bank,
        mem_padding_mask,
        odo_input,
        fps_queue_input,
        active_mask=None,
    ):
        if self.memory_bank_with_temp_attn:
            temporal_embedding = self._forward_temporal_attn(
                output_embedding,
                mem_bank,
                mem_padding_mask,
                odo_input,
                fps_queue_input,
            )

        if self.memory_bank_with_self_attn:
            output_embedding = self._forward_spatial_attn(
                query_pos, temporal_embedding, active_mask
            )
        else:
            output_embedding = temporal_embedding

        if self.out_track_score:
            track_scores = self.track_cls(output_embedding)
            track_scores = self.track_score_dquant(track_scores)  # .squeeze()
        else:
            track_scores = None
        return output_embedding, temporal_embedding, track_scores

    def fuse_model(self):
        if self.memory_bank_with_temp_attn:
            self.temporal_attn.fuse_model()
            self.temporal_ffn.fuse_model()
        if self.memory_bank_with_self_attn:
            self.spatial_attn.fuse_model()
            self.spatial_ffn.fuse_model()

    def set_qconfig(self):
        import horizon_plugin_pytorch as horizon  # noqa

        self.qconfig = qint16_qconfig()
        self.temporal_ffn.set_qconfig()
        self.spatial_ffn.set_qconfig()
        self.temporal_norm1.qconfig = qint8_qconfig()
        self.temporal_norm2.qconfig = qint8_qconfig()
        self.query_spatial_add.qconfig = qint8_qconfig()
        self.spatial_norm1.qconfig = qint8_qconfig()
        self.spatial_norm2.qconfig = qint8_qconfig()
        self.track_cls.qconfig = qat_out_qconfig()
