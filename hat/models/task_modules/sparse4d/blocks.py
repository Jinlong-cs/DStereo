# Copyright (c) Horizon Robotics. All rights reserved.
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.core.utils_3d import project_func_pinhole
from hat.models.weight_init import constant_init, xavier_init
from hat.registry import OBJECT_REGISTRY

try:
    from hat.models.task_modules.sparse4d.ops import (
        DeformableAggregationFunction as DAF,
    )
except ImportError:
    DAF = None

__all__ = [
    "LinearFusionModule",
    "DeformableFeatureAggregation",
    "AsymmetricFFN",
    "DenseDepthNet",
]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


@OBJECT_REGISTRY.register
class LinearFusionModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        alpha=0.9,
        beta=10,
        enable_temporal_weight=True,
    ):
        super(LinearFusionModule, self).__init__()
        self.embed_dims = embed_dims
        self.alpha = alpha
        self.beta = beta
        self.enable_temporal_weight = enable_temporal_weight
        self.fusion_layer = nn.Linear(self.embed_dims * 2, self.embed_dims)

    def init_weight(self):
        xavier_init(self.fusion_layer, distribution="uniform", bias=0.0)

    def forward(self, feature_1, feature_2, time_interval=None):
        if self.enable_temporal_weight:
            temp_weight = self.alpha ** torch.abs(time_interval * self.beta)
            feature_2 = torch.transpose(
                feature_2.transpose(0, -1) * temp_weight, 0, -1
            )
        return self.fusion_layer(torch.cat([feature_1, feature_2], dim=-1))


@OBJECT_REGISTRY.register
class DeformableFeatureAggregation(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_groups=8,
        num_levels=4,
        num_cams=6,
        attn_drop=0.0,
        proj_drop=0.0,
        kps_generator=None,
        temporal_fusion_module=None,
        view_pad_mask=False,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
        enable_trans_with_vel=True,
        output_proj_input_dim=None,
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        self.use_deformable_func = use_deformable_func and DAF is not None
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.residual_mode = residual_mode

        self.kps_generator = kps_generator
        self.num_pts = self.kps_generator.num_pts
        self.temp_module = temporal_fusion_module
        out_proj_input_dim = (
            embed_dims
            if output_proj_input_dim is None
            else output_proj_input_dim
        )
        self.output_proj = nn.Linear(out_proj_input_dim, embed_dims)
        self.enable_trans_with_vel = enable_trans_with_vel
        self.view_pad_mask = view_pad_mask

        if use_camera_embed:
            self.camera_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature,
        anchor,
        anchor_embed,
        feature_maps,
        metas: dict,
        feature_queue=None,
        meta_queue=None,
        anchor_encoder=None,
        **kwargs,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        if feature_queue is not None and len(feature_queue) > 0:
            T_cur2temp_list = []
            for meta in meta_queue:
                T_cur2temp_list.append(
                    meta["img_metas"]["T_global_inv"]
                    @ metas["img_metas"]["T_global"]
                )
            key_points, temp_key_points_list = self.kps_generator(
                anchor,
                instance_feature,
                T_cur2temp_list,
                metas["timestamp"],
                [meta["timestamp"] for meta in meta_queue],
            )
            temp_anchors = self.kps_generator.anchor_projection(
                anchor,
                T_cur2temp_list,
                metas["timestamp"],
                [meta["timestamp"] for meta in meta_queue],
                self.enable_trans_with_vel,
            )
            temp_anchor_embeds = [
                anchor_encoder(x)
                if self.use_temporal_anchor_embed
                and anchor_encoder is not None
                else None
                for x in temp_anchors
            ]
            time_intervals = [
                (metas["timestamp"] - x["timestamp"]).to(
                    dtype=instance_feature.dtype
                )
                for x in [metas] + meta_queue
            ]
        else:
            key_points = self.kps_generator(anchor, instance_feature)
            temp_key_points_list = []
            feature_queue = meta_queue = []
            temp_anchor_embeds = temp_anchors = []
            time_intervals = [instance_feature.new_tensor([0])]

        if self.temp_module is not None and len(feature_queue) == 0:
            features = instance_feature.new_zeros(
                [bs, num_anchor, self.num_pts, self.embed_dims]
            )
        else:
            features = None

        if not self.use_temporal_anchor_embed or anchor_encoder is None:
            weights = self._get_weights(instance_feature, anchor_embed, metas)

        for (
            temp_feature_maps,
            temp_metas,
            temp_key_points,
            temp_anchor_embed,
            time_interval,
        ) in zip(
            feature_queue[::-1] + [feature_maps],
            meta_queue[::-1] + [metas],
            temp_key_points_list[::-1] + [key_points],
            temp_anchor_embeds[::-1] + [anchor_embed],
            time_intervals[::-1],
        ):
            if self.use_temporal_anchor_embed and anchor_encoder is not None:
                weights = self._get_weights(
                    instance_feature, temp_anchor_embed, metas
                )  # bs, num_anchor, num_cams, num_scale, num_pt, num_group

            if self.view_pad_mask:
                mask = weights.new_ones(weights.shape[:3]) * (
                    metas["img_metas"]["view_pad_mask"][:, None, :]
                )
                view_pad_mask = torch.where(mask > 0, 0, 1)
                weights = view_pad_mask[..., None, None, None] * weights

            if self.use_deformable_func:
                weights = (
                    weights.permute(0, 1, 4, 2, 3, 5)
                    .contiguous()
                    .reshape(
                        bs,
                        num_anchor * self.num_pts,
                        self.num_cams,
                        self.num_levels,
                        self.num_groups,
                    )
                )
                points_2d = (
                    self.project_points(
                        temp_key_points,
                        temp_metas["projection_mat"],
                        temp_metas.get("image_wh"),
                        temp_metas.get("cam_intrinsic"),
                        temp_metas.get("cam_distcoeffs"),
                    )
                    .permute(0, 2, 3, 1, 4)
                    .reshape(bs, num_anchor * self.num_pts, self.num_cams, 2)
                )
                if isinstance(temp_feature_maps[0], (list, tuple)):
                    start_idx = 0
                    temp_features_next = 0
                    for feat in temp_feature_maps:
                        next_idx = feat[0].shape[1] + start_idx
                        temp_features_next = temp_features_next + DAF.apply(
                            *feat,
                            points_2d[:, :, start_idx:next_idx],
                            weights[:, :, start_idx:next_idx],
                        )
                        start_idx = next_idx
                else:
                    temp_features_next = DAF.apply(
                        *temp_feature_maps, points_2d, weights
                    )
                temp_features_next = temp_features_next.reshape(
                    bs, num_anchor, self.num_pts, -1
                )
            else:
                temp_features_next = self.feature_sampling(
                    temp_feature_maps,
                    temp_key_points,
                    temp_metas["projection_mat"],
                    temp_metas.get("image_wh"),
                    temp_metas.get("cam_intrinsic"),
                    temp_metas.get("cam_distcoeffs"),
                )
                temp_features_next = self.multi_view_level_fusion(
                    temp_features_next, weights
                )

            if features is None:
                features = temp_features_next
            elif self.temp_module is not None:
                features = self.temp_module(
                    features, temp_features_next, time_interval
                )
            else:
                features = features + temp_features_next

        features = features.sum(dim=2)  # fuse multi-point features
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
                if metas.get("cam_intrinsic") is None
                else metas["cam_intrinsic"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]
        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(
        key_points,
        projection_mat,
        image_wh=None,
        cam_intrinsic=None,
        cam_distcoeffs=None,
    ):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        if cam_distcoeffs is None:
            points_2d = points_2d[..., :2] / torch.clamp(
                points_2d[..., 2:3], min=1e-5
            )
        else:
            key_points = points_2d[..., :3]
            # enable proj grad will get worse metrics
            with torch.no_grad():
                points_2d = DeformableFeatureAggregation.project_points_with_distortion(  # noqa E501
                    key_points, cam_intrinsic, cam_distcoeffs
                )
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps,
        key_points,
        projection_mat,
        image_wh=None,
        cam_intrinsic=None,
        cam_distcoeffs=None,
    ):
        if isinstance(feature_maps[0], (list, tuple)):
            start_idx = 0
            output = []
            for feat in feature_maps:
                next_idx = feat[0].shape[1] + start_idx
                output.append(
                    DeformableFeatureAggregation.feature_sampling(
                        feat,
                        key_points,
                        projection_mat[:, start_idx:next_idx],
                        image_wh[:, start_idx:next_idx]
                        if image_wh is not None
                        else None,
                        cam_intrinsic[:, start_idx:next_idx]
                        if cam_intrinsic is not None
                        else None,
                        cam_distcoeffs[:, start_idx:next_idx]
                        if cam_distcoeffs is not None
                        else None,
                    )
                )
                start_idx = next_idx
            output = torch.cat(output, dim=2)
            return output

        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points,
            projection_mat,
            image_wh,
            cam_intrinsic,
            cam_distcoeffs,
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=-4), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims
        return features

    def multi_view_level_fusion(self, features, weights):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, -1)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(bs, num_anchor, self.num_pts, -1)
        return features

    # TODO(jin.yang): support fisheye camera
    @staticmethod
    def project_points_with_distortion(
        key_points, cam_intrinsic, cam_distcoeffs
    ):
        assert cam_intrinsic is not None

        bs, cam_num, num_anchor, num_pts = key_points.shape[:4]
        key_points = key_points.reshape(bs * cam_num, num_anchor * num_pts, 3)
        cam_intrinsic = cam_intrinsic.flatten(0, 1)
        cam_distcoeffs = cam_distcoeffs.flatten(0, 1)
        points_2d = project_func_pinhole(
            key_points, cam_intrinsic, cam_distcoeffs
        )
        points_2d = points_2d.reshape(bs, cam_num, num_anchor, num_pts, 2)
        return points_2d


@OBJECT_REGISTRY.register
class DenseDepthNet(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for _ in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = (depth.T * focal / self.equal_focal).T
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss


@OBJECT_REGISTRY.register
class AsymmetricFFN(nn.Module):
    def __init__(
        self,
        activate,
        in_channels=None,
        pre_norm=False,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__()
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = activate

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm:
            self.pre_norm = nn.LayerNorm(in_channels)
        else:
            self.pre_norm = None

        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if self.in_channels == embed_dims
                else nn.Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + out
