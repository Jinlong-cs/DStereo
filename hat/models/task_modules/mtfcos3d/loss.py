# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, Tuple

import torch

from hat.registry import OBJECT_REGISTRY

__all__ = ["MTFCOS3DLoss"]


@OBJECT_REGISTRY.register
class MTFCOS3DLoss(torch.nn.Module):
    """MTFCOS3D loss wrapper.

    Args:
        cls_loss: class loss.
        reg_loss: reg loss, 2d bbox.
        group_reg_loss: 3d bbox group info.
        centerness_loss: as name defined.
        dir_loss: direction loss
        corner_loss: 3d corners loss.
        mask_l1_loss: L1 loss with mask part of bins.
    Note:
        This class is not universal. Make sure the limit is well known
        before using it.

    """

    def __init__(
        self,
        cls_loss: torch.nn.Module = None,
        reg_loss: torch.nn.Module = None,
        group_reg_loss: torch.nn.Module = None,
        centerness_loss: torch.nn.Module = None,
        centerness3d_loss: torch.nn.Module = None,
        dir_loss: torch.nn.Module = None,
        corner_loss: torch.nn.Module = None,
        mask_l1_loss: torch.nn.Module = None,
    ):
        super().__init__()
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.group_reg_loss = group_reg_loss
        self.centerness_loss = centerness_loss
        self.centerness3d_loss = centerness3d_loss
        self.dir_loss = dir_loss
        self.corner_loss = corner_loss
        self.mask_l1_loss = mask_l1_loss

    def forward(self, pred: Tuple, target: Dict) -> Dict:
        res = {}
        for k, v in target.items():
            if "dir" in k:
                _sub_loss = self.dir_loss(**v)
            elif "cls" in k:
                _sub_loss = self.cls_loss(**v)
            elif "ctrness_3d" in k:
                _sub_loss = self.centerness3d_loss(**v)
            elif "ctrness" in k:
                _sub_loss = self.centerness_loss(**v)
            elif "_3d_group_reg" in k:
                _sub_loss = self.group_reg_loss(**v)
            elif "_2d_reg" in k:
                _sub_loss = self.reg_loss(**v)
            elif "corners" in k:
                _sub_loss = self.corner_loss(**v)
            elif "mask_l1" in k:
                _sub_loss = self.mask_l1_loss(**v)
            else:
                raise NotImplementedError(f"got unexcepted output key: {k}")
            if isinstance(_sub_loss, Dict):
                _sub_loss = list(_sub_loss.values())[-1]
            res.update({"loss_" + k: _sub_loss})
        return res
