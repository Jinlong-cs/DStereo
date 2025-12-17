# Copyright (c) Horizon Robotics. All rights reserved.

# Lift Splat Shoot encoder
from typing import Dict, List

import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "OccflowHead",
]


@OBJECT_REGISTRY.register
class OccflowHead(nn.Module):
    """Occupancy and flow basic head module.

    This is the basic output head for occupancy prediction and flow
    prediction. For occupancy, it predicts both observed and occluded
    occupancy; for flow, it predicts x and y direction flow vector. The total
    number of future frames to predict is num_waypoints, thus both heads has
    2 * num_waypoints * num_class output channels. Please refer to
    https://horizonrobotics.feishu.cn/file/boxcnvc2zh3rLIaPVMD6ZfIXzdg for
    details.

    Args:
        num_class: number of classes to predict, by default is 1 (only predict
            vehicles).
        in_channels: number of input channels for previous module.
        num_waypoints: number of future frames to predict.
    """

    def __init__(
        self,
        num_class: int,
        in_channels: int,
        num_waypoints: int = 8,
    ):
        super().__init__()
        self.num_class = num_class
        out_channels = 2 * num_waypoints * num_class
        self.occ_head = nn.Sequential(
            ConvModule2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
        )
        self.flow_head = nn.Sequential(
            ConvModule2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
        )
        self.dequant = DeQuantStub()

    def forward(self, x: List) -> Dict:
        """Forward pass for occflow head.

        args:
            agg_encoding: List[x, output_distributions]
            x: List[torch.tensor], multiscale features of previous layers
            output_distributions: Dict, output dict of distribution module.
                Can be empty dict if no distribution module is inserted.
        """
        occ_preds = self.occ_head(x[0])
        flow_preds = self.flow_head(x[0])
        occ_preds = self.dequant(occ_preds)
        flow_preds = self.dequant(flow_preds)
        return {
            "occ_preds": occ_preds,
            "flow_preds": flow_preds,
        }

    def fuse_model(self) -> None:
        oh = getattr(self.occ_head, "0")
        fh = getattr(self.flow_head, "0")
        oh.fuse_model()
        fh.fuse_model()

    def set_qconfig(self) -> None:
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        oh = getattr(self.occ_head, "0")
        fh = getattr(self.flow_head, "0")
        oh.qconfig = qconfig_manager.get_default_qat_out_qconfig()
        fh.qconfig = qconfig_manager.get_default_qat_out_qconfig()
