# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["ANCBevVismaskLoss"]


@OBJECT_REGISTRY.register
class ANCBevVismaskLoss(nn.Module):
    """Calculate bev vismask loss.

    Args:
        vismask_loss_cfg: vismask loss config.
        pred_name: pred name.
        gt_name: vismask gt name.

    """

    def __init__(
        self,
        vismask_loss_cfg: Optional[torch.nn.Module],
        pred_name: str = "pred_bev_vismask_frame0",
        gt_name: str = "gt_bev_elevation_vismask",
    ):
        super(ANCBevVismaskLoss, self).__init__()

        self.vismask_loss = vismask_loss_cfg
        self.pred_name = pred_name
        self.gt_name = gt_name

    def forward(
        self,
        pred_dict: Union[torch.Tensor, Sequence[torch.Tensor]],
        target_dict: torch.Tensor,
    ) -> torch.Tensor:
        result_dict = {}
        target = target_dict[self.gt_name]["vismask"]
        pred = _as_list(pred_dict[self.pred_name])[0]

        vismask_loss = self.vismask_loss(pred, target)
        result_dict["loss_bev_vismask"] = vismask_loss

        return result_dict
