# Copyright (c) Horizon Robotics. All rights reserved.

# This folder contains some of the encoders module.
# Encoders are typically used for lidar detection network to convert
# lidar point clouds to pseudo-images.

from .anchor_generator import Anchor3DGeneratorStride
from .box_coders import GroundBox3dCoder
from .fusion_module import LidarCameraFusionModule
from .head import AfdetHead
from .lidar_loss import AfdetLidarLoss, AfsegLidarLoss
from .pillar_encoder import PillarFeatureNet, PointPillarScatter
from .postprocess import AfdetPredict, AfsegPredict
from .rad_encoder import (
    RadFeatureConcat,
    RadFeatureExtractor,
    RadGridMaker,
    RadScatter,
)
from .target_assigner import LidarTargetAssigner
from .voxel_encoder import MeanVFE

__all__ = [
    "PillarFeatureNet",
    "PointPillarScatter",
    "RadFeatureExtractor",
    "RadScatter",
    "RadGridMaker",
    "AfdetHead",
    "AfdetPredict",
    "AfsegPredict",
    "AfdetLidarLoss",
    "AfsegLidarLoss",
    "LidarCameraFusionModule",
    "Anchor3DGeneratorStride",
    "LidarTargetAssigner",
    "GroundBox3dCoder",
    "RadFeatureConcat",
    "MeanVFE",
]
