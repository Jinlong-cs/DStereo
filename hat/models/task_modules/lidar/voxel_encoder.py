# Copyright (c) Horizon Robotics. All rights reserved.

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["MeanVFE"]


@OBJECT_REGISTRY.register_module
class MeanVFE(nn.Module):
    """Voxel Feature Extractor.

    Voxel feature extractor ultilized in voxel-based frameworks.
    Related open-sourced repo:
    https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/vfe/mean_vfe.py

    Args:
        num_input_features:  Number of input features.

    """

    def __init__(self, num_input_features=4):
        super(MeanVFE, self).__init__()
        self.num_input_features = num_input_features

    def forward(self, voxel_features, num_voxels):
        """Forward point features within each voxel.

        Args:
            voxel_features: point features within each voxel.
            num_voxels: number of points within each voxel.

        """
        points_mean = voxel_features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        )
        # avoid empty input
        normalizer = torch.clamp_min(num_voxels.view(-1, 1), min=1.0).type_as(
            voxel_features
        )
        points_mean = points_mean / normalizer

        return points_mean.contiguous()
