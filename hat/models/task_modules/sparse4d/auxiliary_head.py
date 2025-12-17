# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY

try:
    from hat.models.task_modules.sparse4d.ops import (
        DeformableAggregationFunction as DAF,
    )
except ImportError:
    DAF = None
from .blocks import DeformableFeatureAggregation as DF

__all__ = ["DenseBEV3DAuxHead"]


@OBJECT_REGISTRY.register
class DenseBEV3DAuxHead(nn.Module):
    """Head providing dense 3D auxiliary supervision. \
    img_feature_map -> deform_agg -> feature_encoder -> neck -> head -> loss.

    Args:
        vcs_range: vcs range of the bev feature, (min_x, min_y, max_x, max_y).
        bev_size: size of the bev feature, (num_grid_in_x, num_grid_in_y).
        head: module which input bev feature and output perception result.
        feature_encoder: backbone for bev feature map.
        neck: neck connected to feature_encoder.
        loss: loss module adapted to head.
    """

    def __init__(
        self,
        vcs_range,
        bev_size,
        head=None,
        feature_encoder=None,
        neck=None,
        loss=None,
    ):
        super(DenseBEV3DAuxHead, self).__init__()
        bev_size = np.array(bev_size)
        resolution = bev_size / np.array(
            [vcs_range[2] - vcs_range[0], vcs_range[3] - vcs_range[1]]
        )
        bev_pts_x = np.linspace(
            vcs_range[2] - resolution[0] / 2,
            vcs_range[0] + resolution[0] / 2,
            bev_size[0],
        )
        bev_pts_y = np.linspace(
            vcs_range[3] - resolution[1] / 2,
            vcs_range[1] + resolution[1] / 2,
            bev_size[1],
        )
        bev_pts_x = np.tile(bev_pts_x[:, None], (1, bev_size[1]))
        bev_pts_y = np.tile(bev_pts_y[None], (bev_size[0], 1))
        bev_pts_z = np.zeros_like(bev_pts_x)
        bev_pts = np.stack([bev_pts_x, bev_pts_y, bev_pts_z], axis=-1)
        self.bev_pts = torch.nn.Parameter(
            torch.tensor(bev_pts, dtype=torch.float32).reshape(-1, 3),
            requires_grad=False,
        )
        self.bev_size = bev_size
        self.head = head
        self.feature_encoder = feature_encoder
        self.neck = neck
        self.loss = loss

    def _get_pts2d(self, metas):
        projection_mat = metas["projection_mat"]
        bs, num_cams = projection_mat.shape[:2]
        pts_3d = self.bev_pts[None, None].tile(bs, 1, 1, 1)
        pts_2d = DF.project_points(
            pts_3d,
            projection_mat,
            metas.get("distort"),
            metas.get("cam_intrinsic"),
            metas.get("image_wh"),
        )[:, :, 0]
        return pts_2d

    @autocast(enabled=False)
    def forward(
        self,
        feature_maps,
        metas,
    ):
        pts = self._get_pts2d(metas).permute(0, 2, 1, 3)
        bs, _, num_cams = pts.shape[:3]
        if isinstance(feature_maps[0], (list, tuple)):
            num_levels = feature_maps[0][1].shape[0]
        else:
            num_levels = feature_maps[1].shape[0]
        weights = (
            torch.ones_like(pts[..., :1, None]).tile(1, 1, 1, 4, 1)
            / num_cams
            / num_levels
        )
        if isinstance(feature_maps[0], (list, tuple)):
            start_idx = 0
            bev_feat = 0
            for feat in feature_maps:
                next_idx = feat[0].shape[1] + start_idx
                bev_feat = bev_feat + DAF.apply(
                    *feat,
                    pts[:, :, start_idx:next_idx].contiguous(),
                    weights[:, :, start_idx:next_idx].contiguous(),
                )
                start_idx = next_idx
        else:
            bev_feat = DAF.apply(*feature_maps, pts, weights)
        bev_feat = (
            bev_feat.reshape(bs, *self.bev_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        if self.feature_encoder is not None:
            bev_feat = self.feature_encoder(bev_feat)
            if self.neck is not None:
                bev_feat = self.neck(bev_feat)
        output = self.head(bev_feat)
        return self.loss(output, metas)
