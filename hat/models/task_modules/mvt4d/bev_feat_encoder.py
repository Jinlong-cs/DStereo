from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.models.base_modules.transformer_attentions import (
    BevDeformableTemporalAttention,
    BevSpatialCrossAtten,
    MSDeformableAttention3D,
)
from hat.models.base_modules.transformer_bricks import TransformerLayerSequence
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
except ImportError:
    Quaternion = None
    quaternion_yaw = None


@OBJECT_REGISTRY.register
class BevFeatEncoder(TransformerLayerSequence):
    """Implements the MVT4D's Bev Feat Encoder.

    Args:
        bev_h: Bev feat height.
        bev_w: Bev feat width.
        bev_z: Bev feat z value.
        bev_num_refs: Number of reference points in bev height dim.
        num_feature_levels: Number of feature maps from FPN:
            Default: 4.
        num_cams: Number of cameras(image views). Default: 6.
        use_cams_embeds: whether using camera embeds.
        positional_encoding: Positional Encoding.
        transformerlayer: Module of transformerlayer
            in TransformerCoder. Default: None.
        num_layers: The number of `TransformerLayer`. Default: 3.
        pc_range: Range of point cloud, same as vcs_range.
    """

    @require_packages(
        "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
    )
    def __init__(
        self,
        bev_h: int = 100,
        bev_w: int = 100,
        bev_z: int = 8,
        bev_num_refs: int = 4,
        num_feature_levels: int = 4,
        num_cams: int = 6,
        use_cams_embeds: bool = True,
        positional_encoding: Optional[nn.Module] = None,
        transformerlayers: Optional[nn.Module] = None,
        num_layers: int = 3,
        pc_range: List[float] = None,
        temporal_layer: Optional[nn.Module] = None,
        max_interval: float = 0.0,
    ):
        super(BevFeatEncoder, self).__init__(
            transformerlayers=transformerlayers, num_layers=num_layers
        )

        self.temporal_layer = temporal_layer

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.num_bev_query = bev_h * bev_w
        self.pc_range = pc_range

        self.bev_query_embedding = nn.Embedding(
            self.num_bev_query, self.embed_dims
        )

        self.positional_encoding = positional_encoding

        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.bev_num_refs = bev_num_refs
        self.use_cams_embeds = use_cams_embeds
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims)
        )
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )

        self.max_interval = max_interval

        self.init_weights()
        self.init_bev_refpoints_buffers()
        self.init_bev_shapes_buffers()

    def init_weights(self) -> None:
        """Initialize the BevFeatEncoder weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if (
                isinstance(m, BevSpatialCrossAtten)
                or isinstance(m, MSDeformableAttention3D)
                or isinstance(m, BevDeformableTemporalAttention)
            ):
                try:
                    m.init_weights()
                except AttributeError:
                    m.init_weight()

        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)

    def init_bev_refpoints_buffers(self) -> None:
        # Z = 4
        H, W, Z, D = (
            self.bev_h,
            self.bev_w,
            self.bev_z,
            self.bev_num_refs,
        )
        zs = (
            torch.linspace(0.5, Z - 0.5, D, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, H, W)
            / Z
        )
        xs = (
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float)
            .view(1, 1, W)
            .expand(D, H, W)
            / W
        )
        ys = (
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float)
            .view(1, H, 1)
            .expand(D, H, W)
            / H
        )

        bev_reference_points_xyz = torch.stack((xs, ys, zs), -1)
        bev_reference_points_xyz = (
            bev_reference_points_xyz.permute(0, 3, 1, 2)
            .flatten(2)
            .permute(2, 0, 1)
            .contiguous()
        )

        self.register_buffer(
            "bev_reference_points_xyz",
            bev_reference_points_xyz,
            persistent=False,
        )

    def init_bev_shapes_buffers(self) -> None:
        bev_query_spatial_shapes = [(self.bev_h, self.bev_w)]
        bev_query_spatial_shapes = torch.as_tensor(
            bev_query_spatial_shapes, dtype=torch.long
        )
        bev_query_level_start_index = torch.cat(
            (
                bev_query_spatial_shapes.new_zeros((1,)),
                bev_query_spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        self.register_buffer(
            "bev_query_spatial_shapes",
            bev_query_spatial_shapes,
            persistent=False,
        )
        self.register_buffer(
            "bev_query_level_start_index",
            bev_query_level_start_index,
            persistent=False,
        )

    def encode_bev_feat_in_spatial(
        self,
        query: torch.Tensor,
        bev_reference_points: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode bev feat by TransformerEncoder format in spatial space.

        Args:
            query: Input query with shape
                `(num_query, bs, embed_dims)`.
            bev_reference_points: The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape (bs, num_query, 2).
                (bs, num_query, Nref, 3)
            pre_bev_feat: bev feat of previous frame
                with shape `(num_query, bs, embed_dims)`,
            pre_img_metas: image metas of previous frame.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims].
        """

        output = query  # torch.Size([40000, 1, 256])

        for _, layer in enumerate(self.layers):
            reference_points_input = (
                bev_reference_points  # torch.Size([1, 40000, 4, 3])
            )
            output = layer(
                output,
                bev_reference_points=reference_points_input,
                **kwargs,
            )

        return output

    def get_pre_ref_points(
        self,
        bev_reference_points,
        img_metas,
        pre_img_metas,
    ):
        cur_T_vcs2global = img_metas["T_vcs2global"]
        cur_T_global2vcs = img_metas["T_global2vcs"]

        start_of_sequence = torch.BoolTensor(
            [
                abs(cur_ts - pre_ts) > self.max_interval
                for cur_ts, pre_ts in zip(
                    img_metas["timestamp"], pre_img_metas["timestamp"]
                )
            ]
        ).to(img_metas["timestamp"].device)

        pre_T_global2vcs = pre_img_metas["T_global2vcs"]
        pre_T_global2vcs[start_of_sequence, ...] = cur_T_global2vcs[
            start_of_sequence, ...
        ]

        T_curvcs2prevcs = pre_T_global2vcs @ cur_T_vcs2global

        lidar_ref_points = bev_reference_points[:, :, 0, :].clone()
        lidar_ref_points[..., 0:1] = (
            lidar_ref_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
            + self.pc_range[0]
        )
        lidar_ref_points[..., 1:2] = (
            lidar_ref_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
            + self.pc_range[1]
        )
        lidar_ref_points[..., 2:3] = (
            lidar_ref_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2])
            + self.pc_range[2]
        )
        bs, num_query, pdim = lidar_ref_points.shape
        lidar_ref_points = lidar_ref_points.view(bs, -1, pdim)

        lidar_ref_points = torch.cat(
            (lidar_ref_points, torch.ones_like(lidar_ref_points[..., :1])), -1
        )
        lidar_ref_points = lidar_ref_points.view(bs, -1, 4).unsqueeze(-1)
        T_curvcs2prevcs = T_curvcs2prevcs.view(bs, 1, 4, 4).repeat(
            1, num_query, 1, 1
        )
        lidar_ref_points = torch.matmul(
            T_curvcs2prevcs, lidar_ref_points
        ).squeeze(-1)[..., :3]

        lidar_ref_points[..., 0:1] = (
            lidar_ref_points[..., 0:1] - self.pc_range[0]
        ) / (self.pc_range[3] - self.pc_range[0])
        lidar_ref_points[..., 1:2] = (
            lidar_ref_points[..., 1:2] - self.pc_range[1]
        ) / (self.pc_range[4] - self.pc_range[1])
        lidar_ref_points[..., 2:3] = (
            lidar_ref_points[..., 2:3] - self.pc_range[2]
        ) / (self.pc_range[5] - self.pc_range[2])

        lidar_ref_points = lidar_ref_points.view(bs, num_query, 1, pdim)

        return lidar_ref_points[..., 0:1, :2], start_of_sequence

    def extract_bev_feat(
        self,
        mlvl_feats: List[torch.Tensor],
        bev_query_embeds: torch.Tensor,
        bev_pos: torch.Tensor,
        img_metas: Dict[str, Any],
        pre_bev_feat: Optional[torch.Tensor] = None,
        pre_img_metas: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Extract bev feat from multi-levels image feats.

        Args:
            mlvl_feats: Multi-levels image feats.
            bev_query_embeds: Bev query embedding.
                with shape `(bev_h*bev_w, embed_dims)`
            bev_pos: Bev positional encoding.
            img_metas: image metas of current frame.
            pre_bev_feat: bev feat of previous frame
                with shape `(num_query, bs, embed_dims)`,
            pre_img_metas: image metas of current frame.
        Returns:
            Tensor: Results with shape (bev_h*bev_w, bs, embed_dims).
        """

        bs = mlvl_feats[0].size(0)
        bev_query = bev_query_embeds.unsqueeze(0).expand(
            bs, -1, -1
        )  # query (bs, 200x200, 256)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        bev_reference_points_xyz_s = self.bev_reference_points_xyz[
            None
        ].repeat(bs, 1, 1, 1)

        feat_flatten = []
        mlvl_feats_spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            B, N, C, feat_h, feat_w = feat.shape
            feat_spatial_shape = (feat_h, feat_w)
            mlvl_feats_spatial_shapes.append(feat_spatial_shape)
            feat = (
                feat.permute(3, 4, 0, 1, 2)
                .contiguous()
                .view(feat_h * feat_w, B * N, C)
            )

            if self.use_cams_embeds:
                feat = feat + self.cams_embeds.repeat(1, B, 1).to(feat.dtype)
            feat = feat + self.level_embeds[None, lvl : lvl + 1, :].to(
                feat.dtype
            )

            feat_flatten.append(feat)

        device = bev_query.device

        mlvl_feats_spatial_shapes = torch.as_tensor(
            mlvl_feats_spatial_shapes, dtype=torch.long, device=device
        )
        mlvl_feats_level_start_index = torch.cat(
            (
                mlvl_feats_spatial_shapes.new_zeros((1,)),
                mlvl_feats_spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        feat_flatten = torch.cat(feat_flatten, 0)

        # encoder
        bev_query = bev_query.permute(1, 0, 2)  # (200x200, bs,  256)

        bev_feat = self.encode_bev_feat_in_spatial(  # type: ignore
            query=bev_query,
            key=feat_flatten,
            value=feat_flatten,
            query_pos=bev_pos,
            mlvl_feats_spatial_shapes=mlvl_feats_spatial_shapes,
            mlvl_feats_level_start_index=mlvl_feats_level_start_index,
            bev_reference_points=bev_reference_points_xyz_s,
            reference_points=bev_reference_points_xyz_s[..., :2],
            spatial_shapes=self.bev_query_spatial_shapes,
            level_start_index=self.bev_query_level_start_index,
            img_metas=img_metas,
        )

        pre_ref_points = None
        start_of_sequence = None
        if pre_bev_feat is not None:
            pre_ref_points, start_of_sequence = self.get_pre_ref_points(
                bev_reference_points_xyz_s,
                img_metas,
                pre_img_metas,
            )

        if self.temporal_layer:
            bev_feat = self.temporal_layer(
                query=bev_feat,
                key=None,
                value=None,
                query_pos=bev_pos,
                bev_reference_points=bev_reference_points_xyz_s,
                reference_points=bev_reference_points_xyz_s[..., :2],
                spatial_shapes=self.bev_query_spatial_shapes,
                level_start_index=self.bev_query_level_start_index,
                pre_bev_feat=pre_bev_feat,
                pre_ref_points=pre_ref_points,
                start_of_sequence=start_of_sequence,
            )

        return bev_feat

    @autocast(enabled=False)
    def forward(
        self,
        mlvl_feats: List[torch.Tensor],
        img_metas: Dict[str, Any],
        pre_bev_feat: Optional[torch.Tensor] = None,
        pre_img_metas: Optional[Dict[str, Any]] = None,
        is_feat_formatted: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            mlvl_feats: Multi-level features, e.g.,
                features produced by FPN.
            img_metas: image metas including calibs.
            pre_bev_feat: previous frame's bev feature.
            pre_img_metas: previous frame's image metas.
            is_feat_formatted: turn to shape (bs, dims, bev_h, bev_w).

        Returns:
            Bev feature with shape (bev_h*bev_w, bs, embed_dims).
        """
        batch_size, _ = mlvl_feats[0].shape[:2]

        bev_mask = mlvl_feats[0].new_zeros(
            (batch_size, self.bev_h, self.bev_w)
        )
        bev_pos = self.positional_encoding(bev_mask)  # type: ignore

        bev_query_embeds = self.bev_query_embedding.weight

        cur_bev_feat = self.extract_bev_feat(
            mlvl_feats,
            bev_query_embeds,
            bev_pos=bev_pos,
            img_metas=img_metas,
            pre_bev_feat=pre_bev_feat,
            pre_img_metas=pre_img_metas,
        )

        if is_feat_formatted:
            cur_bev_feat = cur_bev_feat.permute(1, 2, 0).view(
                -1, self.embed_dims, self.bev_h, self.bev_w
            )

        return cur_bev_feat
