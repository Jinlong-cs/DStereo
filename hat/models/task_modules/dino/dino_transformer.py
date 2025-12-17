import copy
import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn import MultiScaleDeformableAttention
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.base_modules.attention import MultiheadAttention
from hat.models.base_modules.mlp_module import FFN, MLP
from hat.models.task_modules.deform_detr.layers import (
    BaseTransformerLayer,
    TransformerLayerSequence,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager
from hat.utils.model_helpers import fx_wrap
from .layers import GetSinPosEmbed

__all__ = [
    "DINOTransformerDecoder",
    "DINOTransformer",
]


@OBJECT_REGISTRY.register
class DINOTransformerDecoder(TransformerLayerSequence):
    """Transformer decoder the DINO.

    It incorporates multi-head and multi-scale deformable attention mechanisms.

    Args:
        embed_dim: Dimension of input embeddings.
        num_heads: Number of attention heads.
        feedforward_dim: Dimension of the feedforward network.
        attn_dropout: Dropout rate for attention layers.
        ffn_dropout: Dropout rate for feedforward network.
        num_layers: Number of decoder layers.
        return_intermediate: Whether to return intermediate outputs.
        num_feature_levels: Number of feature levels for multi-scale attention.
        look_forward_twice: Whether use look forward twice trick,
                if True, intermediate reference points not detach.
        with_box_refine: Enables bounding box refinement.
        as_two_stage: Enable two stage detection mode.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
        num_classes: int = 80,
        look_forward_twice: bool = True,
        with_box_refine: bool = True,
        as_two_stage: bool = True,
    ):
        super(DINOTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                embed_dim=embed_dim,
                attn=[
                    # "multi_head_attn",
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    # "deform_attn",
                    MultiScaleDeformableAttention(
                        embed_dims=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=(
                    "self_attn",
                    "norm",
                    "cross_attn",
                    "norm",
                    "ffn",
                    "norm",
                ),
            ),
            num_layers=num_layers,
        )
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.return_intermediate = return_intermediate

        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)
        if self.with_box_refine:
            self.ref_mul = nn.ModuleList()
            self.ref_add = nn.ModuleList()
            self.sigmoid1 = nn.ModuleList()
            for _ in range(len(self.layers)):
                self.ref_mul.append(FloatFunctional())
                self.ref_add.append(FloatFunctional())
                self.sigmoid1.append(nn.Sigmoid())
        else:
            self.ref_mul = FloatFunctional()
            self.sigmoid1 = nn.Sigmoid()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.look_forward_twice = look_forward_twice
        self.get_sine_pos_embed = GetSinPosEmbed()
        self._build_head()

    def _build_head(self):
        # initialize weights
        class_embed = nn.Linear(self.embed_dim, self.num_classes)
        bbox_embed = MLP(self.embed_dim, self.embed_dim, 4, 3)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        # if two-stage, the last class_embed and bbox_embed is for region
        # proposal generation
        num_pred = (
            (self.num_layers + 1) if self.as_two_stage else self.num_layers
        )
        self.num_pred = num_pred
        if self.with_box_refine:
            self.class_embed = nn.ModuleList(
                [copy.deepcopy(class_embed) for i in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList(
                [copy.deepcopy(bbox_embed) for i in range(num_pred)]
            )
            nn.init.constant_(
                self.bbox_embed[0].layers[-1].bias.data[2:], -2.0
            )
        else:
            nn.init.constant_(bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList(
                [bbox_embed for _ in range(num_pred)]
            )
        if self.as_two_stage:
            # two-stage
            # hack implementation for two-stage
            for bbox_embed_layer in self.bbox_embed:
                nn.init.constant_(
                    bbox_embed_layer.layers[-1].bias.data[2:], 0.0
                )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points_unact: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,  # nlvl, 2
        valid_ratio: Optional[torch.Tensor] = None,
    ):
        output = query
        bs, num_queries, _ = output.size()
        if not self.with_box_refine:
            reference_points = self.sigmoid1(reference_points_unact)
            reference_points_input = self.ref_mul.mul(
                reference_points[:, :, None], valid_ratio
            )
            query_sine_embed = self.get_sine_pos_embed(
                reference_points_input[:, :, :1, :].squeeze(2)
            )
            query_pos = self.ref_point_head(query_sine_embed)

        intermediate = []
        intermediate_tmp = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if self.with_box_refine:
                reference_points = self.sigmoid1[layer_idx](
                    reference_points_unact
                )
                reference_points_input = self.ref_mul[layer_idx].mul(
                    reference_points[:, :, None], valid_ratio
                )

                query_sine_embed = self.get_sine_pos_embed(
                    reference_points_input[:, :, :1, :].squeeze(2)
                )
                query_pos = self.ref_point_head(query_sine_embed)

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,  # nlvl, 2
            )

            if self.with_box_refine:
                tmp = self.bbox_embed[layer_idx](output)
                new_reference_points_unact = self.ref_add[layer_idx].add(
                    tmp, reference_points_unact
                )
                reference_points_unact = new_reference_points_unact.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_tmp.append(tmp)
                if self.look_forward_twice and self.with_box_refine:
                    intermediate_reference_points.append(
                        new_reference_points_unact
                    )
                else:
                    intermediate_reference_points.append(
                        reference_points_unact
                    )

        if self.return_intermediate:
            return (
                intermediate,
                intermediate_tmp,
                intermediate_reference_points,
            )

        return output, tmp, reference_points

    def set_qconfig(self):
        modules_list = [
            self.sigmoid1,
            self.ref_mul,
        ]
        if self.with_box_refine:
            modules_list.append(self.ref_add)
        for module in modules_list:
            module.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
        for layer in self.layers:
            module_list = [
                layer.attentions[0].identity_add,
                layer.attentions[1],
                layer.ffns[0].add_identity_op,
                layer.ffns[0].layers[-2],
            ]
            for module in module_list:
                module.qconfig = qconfig_manager.get_qconfig(
                    activation_qat_qkwargs={"dtype": qint16},
                    activation_calibration_qkwargs={"dtype": qint16},
                )


@OBJECT_REGISTRY.register
class DINOTransformer(nn.Module):
    """Transformer module for DINO.

    Args:
        encoder: DINO encoder module.
        decoder: DINO decoder module.
        num_feature_levels: number of feature levels. Default 4.
        num_proposals: number of proposals in decoder transformer.
        learnt_init_query: if not use two-stage mode,
            whether use learnt init query embeding.
        as_two_stage: whether to use two-stage transformer.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_feature_levels: int = 4,
        num_proposals: int = 900,
        learnt_init_query: bool = True,
        as_two_stage: bool = True,
    ):
        super(DINOTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.num_proposals = num_proposals
        self.as_two_stage = as_two_stage

        self.embed_dim = self.encoder.embed_dim

        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dim)
        )

        self.learnt_init_query = learnt_init_query
        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.num_proposals, self.embed_dim)
            self.tgt_embed_quant = QuantStub()
        if not self.as_two_stage:
            self.query_pos_embed = nn.Embedding(
                self.num_proposals, self.embed_dim
            )
            self.reference_points = nn.Linear(self.embed_dim, 4)
        else:
            self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
            self.enc_output_norm = nn.LayerNorm(self.embed_dim)

        self.feat_cat = FloatFunctional()
        self.add_proposal = FloatFunctional()

        self.query_embed_cat = FloatFunctional()
        self.target_query_cat = FloatFunctional()

        self.lvl_pos_embed_quant = QuantStub()

        self.query_embed_quant = QuantStub()
        self.query_label_emb_quant = QuantStub()
        self.ref_enc_quant = QuantStub(scale=1 / 32767.0)
        self.ref4_quant = QuantStub(scale=0.0002)
        self.valid_quant = QuantStub()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                if hasattr(m, "init_weights"):
                    m.init_weights()
                elif hasattr(m, "_reset_parameters"):
                    m._reset_parameters()
        nn.init.normal_(self.level_embeds)

    @fx_wrap()
    def gen_encoder_output_proposals(
        self,
        memory_padding_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        reference_points: torch.Tensor,
    ):
        N = memory_padding_mask.shape[0]
        proposals = []
        wh_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            wh = (
                torch.ones([N, H * W, 2]).to(reference_points.device)
                * 0.05
                * (2.0 ** lvl)
            )
            wh_list.append(wh)
        wh = torch.cat(wh_list, dim=1)
        proposals = torch.cat([reference_points, wh], dim=-1)

        output_proposals = proposals
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)

        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if self.training:
            output_proposals = output_proposals.masked_fill(
                memory_padding_mask.unsqueeze(-1), 12  # float("inf")
            )
            output_proposals = output_proposals.masked_fill(
                ~output_proposals_valid, 12  # float("inf")
            )

        return output_proposals, output_proposals_valid

    @fx_wrap()
    def gen_encoder_output_memory(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        output_proposals_valid: torch.Tensor,
    ):
        output_memory = memory
        if self.training:
            output_memory = output_memory.masked_fill(
                memory_padding_mask.unsqueeze(-1), float(0)
            )
            output_memory = output_memory.masked_fill(
                ~output_proposals_valid, float(0)
            )
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory

    @fx_wrap()
    def get_reference_points(
        self, spatial_shapes: torch.Tensor, valid_ratios: torch.Tensor
    ):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes: The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios: The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        """
        reference_points_list = []
        device = spatial_shapes.device
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.arange(0.5, H, 1, dtype=torch.float32, device=device),
                torch.arange(0.5, W, 1, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H
            )
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W
            )
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(
            reference_points_list, 1
        )  # (bs, num_keys, 2)
        multi_level_reference_points = (
            reference_points[:, :, None] * valid_ratios[:, None]
        )
        # (bs, num_keys, 1, 2) * (bs, 1, num_levels, 2)
        return reference_points, multi_level_reference_points

    @fx_wrap()
    def get_valid_ratio(self, mask: torch.Tensor):
        """Get the valid ratios of feature maps of all levels."""
        bs, H, W = mask.shape
        if self.training:
            valid_H = torch.sum(~mask[:, :, 0], 1)
            valid_W = torch.sum(~mask[:, 0, :], 1)
            valid_ratio_h = valid_H.float() / H
            valid_ratio_w = valid_W.float() / W
            valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        else:  # valid ratio is constant for inference
            valid_ratio = torch.ones([bs, 2]).float().to(mask.device)

        return valid_ratio

    @fx_wrap()
    def get_spatial_shape(
        self, spatial_shapes: torch.Tensor, feat_flatten: torch.Tensor
    ):
        return torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )

    @fx_wrap()
    def before_encoder(
        self,
        multi_level_feats: List[torch.Tensor],
        multi_level_masks: List[torch.Tensor],
        multi_level_pos_embeds: List[torch.Tensor],
    ):
        """Prepare encoder inputs.

        This method flattens and transposes the input features and embeddings,
        calculates valid ratios, and generates reference points for each level.
        It's used to prepare the data for the deformable attention mechanism in
        the encoder.

        Args:
            multi_level_feats: List of feature maps from different levels, each
                has shape (feat_h, feat_w)
            multi_level_masks: List of masks corresponding to the feature maps.
            multi_level_pos_embeds: List of positional embeddings for each
                    feature map level.
        """
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for lvl in range(self.num_feature_levels):
            feat = multi_level_feats[lvl]
            mask = multi_level_masks[lvl]
            pos_embed = multi_level_pos_embeds[lvl]

            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            valid_ratio = self.get_valid_ratio(mask)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)

            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
            valid_ratios.append(valid_ratio)
        feat_flatten = self.feat_cat.cat(feat_flatten, 1)

        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        spatial_shapes = self.get_spatial_shape(spatial_shapes, feat_flatten)

        # constant value for inference  (bs, num_levels, 2)
        valid_ratios = torch.stack(valid_ratios, 1)

        cat_valid_ratio = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        (
            reference_points,
            multi_level_reference_points,
        ) = self.get_reference_points(spatial_shapes, valid_ratios)

        multi_level_reference_points = self.ref_enc_quant(
            multi_level_reference_points
        )
        lvl_pos_embed_flatten = self.lvl_pos_embed_quant(lvl_pos_embed_flatten)
        return (
            feat_flatten,
            lvl_pos_embed_flatten,
            mask_flatten,
            spatial_shapes,
            valid_ratios,
            cat_valid_ratio,
            reference_points,
            multi_level_reference_points,
        )

    @fx_wrap()
    def before_decoder(
        self,
        memory: torch.Tensor,
        mask_flatten: torch.Tensor,
        spatial_shapes: torch.Tensor,
        reference_points: torch.Tensor,
        query_embed: torch.Tensor,
    ):
        """Process inputs before feeding them to the decoder.

        Args:
            memory: The encoded memory tensor from the encoder with
                shape (bs, feat_len, embed_dim).
            mask_flatten: Flattened mask tensor with shape (bs, feat_len).
            spatial_shapes: The spatial shapes of feature maps.
            reference_points: 2d Reference points for deformable attention
            query_embed: Tensor containing query embeddings.
        """
        bs = memory.shape[0]
        if self.as_two_stage:  # same as deform detr two_stage version
            (
                reference_point_4,
                output_proposals_valid,
            ) = self.gen_encoder_output_proposals(
                mask_flatten, spatial_shapes, reference_points
            )

            output_memory = self.gen_encoder_output_memory(
                memory, mask_flatten, output_proposals_valid
            )  # output_memory: bs, num_tokens, c

            enc_outputs_class = self.decoder.class_embed[-1](output_memory)
            enc_outputs_coord_unact = self.add_proposal.add(
                self.decoder.bbox_embed[-1](output_memory),
                self.ref4_quant(reference_point_4),
            )  # unsigmoided.

            topk = self.num_proposals
            max_cls = torch.max(enc_outputs_class, dim=2, keepdim=True)[0]
            enc_classes, topk_proposals = torch.topk(max_cls, topk, dim=1)

            # extract region proposal boxes
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact,
                1,
                topk_proposals.repeat(1, 1, 4),
            )  # unsigmoided.
            reference_points_unact = topk_coords_unact.detach()
            enc_state = torch.gather(
                output_memory,
                1,
                topk_proposals.repeat(1, 1, output_memory.shape[-1]),
            )
            query_pos = None
        else:
            query_pos = self.query_pos_embed.weight[None].repeat(bs, 1, 1)
            reference_points_unact = self.reference_points(query_pos)
            reference_points_unact = self.ref4_quant(reference_points_unact)
            enc_state = None
            topk_coords_unact = None

        if self.learnt_init_query or not self.as_two_stage:
            target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
            target = self.tgt_embed_quant(target)
        else:
            target = enc_state.detach()

        # dn training related
        if query_embed[1] is not None:
            query_embed_bbox = self.query_embed_quant(query_embed[1])
            reference_points_unact = self.query_embed_cat.cat(
                [query_embed_bbox, reference_points_unact], 1
            )

        if query_embed[0] is not None:
            query_label_embed = self.query_label_emb_quant(query_embed[0])
            target = self.target_query_cat.cat([query_label_embed, target], 1)

        return (
            target,
            memory,
            reference_points_unact,
            enc_state,
            topk_coords_unact,
        )

    def forward(
        self,
        multi_level_feats: List[torch.Tensor],
        multi_level_masks: List[torch.Tensor],
        multi_level_pos_embeds: List[torch.Tensor],
        query_embed: torch.Tensor,
        attn_masks: List[torch.Tensor],
    ):
        bs, c, h, w = multi_level_feats[0].shape
        (
            feat_flatten,
            lvl_pos_embed_flatten,
            mask_flatten,
            spatial_shapes,
            valid_ratios,
            decoder_valid_ratio,
            reference_points,
            multi_level_reference_points,
        ) = self.before_encoder(
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
        )

        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=multi_level_reference_points,
            # bs, num_token, num_level, 2
        )

        (
            target,
            memory,
            init_reference_unact,
            enc_states,
            topk_coords_unact,
        ) = self.before_decoder(
            memory, mask_flatten, spatial_shapes, reference_points, query_embed
        )
        inter_states, inter_bboxes, inter_references_unact = self.decoder(
            query=target,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=None,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points_unact=init_reference_unact,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            valid_ratio=self.valid_quant(decoder_valid_ratio),
            attn_masks=attn_masks,
        )

        return (
            inter_states,
            inter_bboxes,
            init_reference_unact,
            inter_references_unact,
            enc_states,
            topk_coords_unact,
        )

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        modules_list = [
            self.ref_enc_quant,
            self.ref4_quant,
            self.valid_quant,
            self.add_proposal,
            self.enc_output,
        ]
        for module in modules_list:
            module.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
        for module in [self.encoder, self.decoder]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
