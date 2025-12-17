# mypy: ignore-errors

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.core.nus_box3d_utils import inverse_sigmoid
from hat.models.base_modules.transformer_attentions import (
    ObjectDetr3DCrossAtten,
)
from hat.models.base_modules.transformer_bricks import TransformerLayerSequence
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages
from .stage2nd_bev_head import BevStage2nd

try:
    from mmcv.version import __version__ as mm_version

    if mm_version >= "2.0.0":
        from mmengine.model.weight_init import bias_init_with_prob
    else:
        from mmcv.cnn import bias_init_with_prob
except ImportError:
    bias_init_with_prob = None


@OBJECT_REGISTRY.register
class DeformableHead(TransformerLayerSequence):
    """DeformableHead like DeformableDetr.

    Args:
        pc_range: Range of point cloud, same as vcs_range.
        bev_h: Bev feat height.
        bev_w: Bev feat width.
        num_reg_fcs: Layers' number of regression fcs. Default: 2.
        num_classes: Number of classes for classification.
        with_box_refine: Whether to refine the reference points
            in the decoder. Defaults to False.
        code_index: Sequence shows the meaning of per box
             code dims.
        stage2nd_cfg: Config of MVT4D 2 stage head.
        transformerlayers: Modules of TransformerLayerSequence.
        num_layers: Number of TransformerLayers.
    """

    def __init__(
        self,
        pc_range: List[float],
        num_query: int = 900,
        num_reg_fcs: int = 2,
        num_classes: int = 10,
        bev_h: int = 100,
        bev_w: int = 100,
        with_box_refine: bool = True,
        stage2nd_cfg: Dict[str, Any] = None,
        code_index: Sequence[str] = (
            "CX",
            "CY",
            "W",
            "L",
            "CZ",
            "H",
            "SIN_YAW",
            "COS_YAW",
            "VX",
            "VY",
        ),
        transformerlayers: Optional[nn.Module] = None,
        num_layers: int = 6,
    ):

        super(DeformableHead, self).__init__(
            transformerlayers=transformerlayers, num_layers=num_layers
        )

        self.stage2nd_cfg = stage2nd_cfg
        self.with_box_refine = with_box_refine

        self.cls_out_channels = num_classes

        self.code_size = len(code_index)
        self.code_index = {v: i for i, v in enumerate(code_index)}

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_query = num_query
        self.num_classes = num_classes
        self.num_reg_fcs = num_reg_fcs

        self.pc_range = pc_range

        self.reference_points = nn.Linear(self.embed_dims, 3)

        if self.stage2nd_cfg is not None:
            self.stage2nd = BevStage2nd(
                embed_dims=self.embed_dims,
                code_index=code_index,
                **self.stage2nd_cfg,
            )

        self._init_layers()
        self.init_weights()
        self.init_bev_shapes_buffers()

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module: nn.Module, N: int) -> nn.Module:
            return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

        num_pred: int = self.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)

        self.object_query_embedding = nn.Embedding(
            self.num_query, self.embed_dims * 2
        )

    @require_packages("mmcv")
    def init_weights(self) -> None:
        """Initialize weights of the DeformDETR head."""
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:  # type: ignore
            nn.init.constant_(m[-1].bias, bias_init)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, ObjectDetr3DCrossAtten):
                try:
                    m.init_weights()
                except AttributeError:
                    m.init_weight()

        nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        nn.init.constant_(self.reference_points.bias.data, 0.0)

    def init_bev_shapes_buffers(self) -> None:

        bev_feat_shapes = [(self.bev_h, self.bev_w)]
        bev_feat_shapes = torch.as_tensor(
            bev_feat_shapes,
            dtype=torch.long,
        )
        bev_feat_level_start_index = torch.cat(
            (
                bev_feat_shapes.new_zeros((1,)),
                bev_feat_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        self.register_buffer(
            "bev_feat_shapes",
            bev_feat_shapes,
            persistent=False,
        )
        self.register_buffer(
            "bev_feat_level_start_index",
            bev_feat_level_start_index,
            persistent=False,
        )

    def decode_with_layers(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor = None,
        reg_branches: nn.ModuleList = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode function with TransformerLayers.

        Args:
            query: Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points: The reference
                points of offset.
            reg_branch: Used for refining the regression
                results. Only would be passed when
                with_box_refine is True, otherwise would
                be passed a `None`.
        Returns:
            output & reference_points: Results with shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2]

            output = layer(
                output,
                reference_points=reference_points_input,
                **kwargs,
            )

            if reg_branches is not None:
                output = output.permute(1, 0, 2)
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = (
                    tmp[
                        ...,
                        [
                            self.code_index["CX"],
                            self.code_index["CY"],
                            self.code_index["CZ"],
                        ],
                    ]
                    + inverse_sigmoid(reference_points)
                )
                new_reference_points = new_reference_points.sigmoid()
                reference_points = (
                    new_reference_points.detach()
                )  # lookforward twice

                output = output.permute(1, 0, 2)

            intermediate.append(output)
            intermediate_reference_points.append(new_reference_points)

        return torch.stack(intermediate), torch.stack(
            intermediate_reference_points
        )

    def decode_object_query(
        self,
        bev_feat: torch.Tensor,
        object_query_embeds: torch.Tensor,
        reg_branches: nn.ModuleList,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode object query for 3D detector head.

        Args:
            bev_feat: Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            object_query_embed: The query embedding for decoder,
                with shape [num_query, c].
            reg_branches : Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. Output has shape \
                      (num_dec_layers, bs, num_query, embed_dims),
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
        """
        assert object_query_embeds is not None

        bs = bev_feat.shape[1]

        query_pos, query = torch.split(
            object_query_embeds, self.embed_dims, dim=1
        )
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points.clone()

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decode_with_layers(
            query=query,
            key=None,
            value=bev_feat,
            query_pos=query_pos,
            reference_points=reference_points.detach(),
            bev_feat_shapes=self.bev_feat_shapes,
            bev_feat_level_start_index=self.bev_feat_level_start_index,
            reg_branches=reg_branches,
            **kwargs,
        )

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out

    @autocast(enabled=False)
    def forward(
        self,
        cur_bev_feat: torch.Tensor,
        img_metas: Dict[str, Any],
        pre_img_metas: Dict[str, Any],
        mlvl_feats: torch.Tensor = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Forward function.

        Args:
            cur_bev_feat: Bev Feat of current frame. shape (B, C, H, W).
            img_metas: image metas including calibs.
            mlvl_feats: Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores: Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds: Sigmoid outputs from the regression \
                head with normalized coordinate format \
                (cx, cy, w, l, cz, h, yaw_sin, yaw_cos, vx, vy). \
                Shape [nb_dec, bs, num_query, 10].
        """

        object_query_embeds = self.object_query_embedding.weight

        hs, init_reference, inter_references = self.decode_object_query(
            cur_bev_feat,
            object_query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])  # type: ignore
            tmp = self.reg_branches[lvl](hs[lvl])  # type: ignore

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[
                ..., [self.code_index["CX"], self.code_index["CY"]]
            ] += reference[
                ..., 0:2
            ]  # cx, cy
            tmp[..., [self.code_index["CX"], self.code_index["CY"]]] = tmp[
                ..., 0:2
            ].sigmoid()

            tmp[..., self.code_index["CZ"]] += reference[..., 2]  # cz
            tmp[..., self.code_index["CZ"]] = tmp[
                ..., self.code_index["CZ"]
            ].sigmoid()

            tmp[..., self.code_index["CX"]] = (
                tmp[..., self.code_index["CX"]]
                * (self.pc_range[3] - self.pc_range[0])
                + self.pc_range[0]
            )  # cx
            tmp[..., self.code_index["CY"]] = (
                tmp[..., self.code_index["CY"]]
                * (self.pc_range[4] - self.pc_range[1])
                + self.pc_range[1]
            )  # cy
            tmp[..., self.code_index["CZ"]] = (
                tmp[..., self.code_index["CZ"]]
                * (self.pc_range[5] - self.pc_range[2])
                + self.pc_range[2]
            )  # cz

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        if self.stage2nd_cfg is not None:

            refined_bbox_preds = []

            for i in range(hs.shape[0]):
                refined_bbox_preds.append(
                    self.stage2nd(
                        cur_bev_feat,
                        outputs_coords[i],
                        self.pc_range,
                        self.bev_h,
                        self.bev_w,
                    )
                )

            refined_bbox_preds = torch.stack(refined_bbox_preds)
        else:
            refined_bbox_preds = None  # type: ignore

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outs = {
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "refined_bbox_preds": refined_bbox_preds,
        }
        return outs
