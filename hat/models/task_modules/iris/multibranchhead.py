# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Optional

import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from .head import IrisSingleBranchHead

__all__ = ["IrisMutilBranchHead"]


@OBJECT_REGISTRY.register
class IrisMutilBranchHead(nn.Module):
    """Multi-branch regression model for iris status prediction.

    Iris status are divided into 2 groups: left eye iris, right eye iris.
    Each iris status have two score.
    The first one represent visibility score.
    The second one represent invisibility score.

    Args:
        bn_kwargs: Dict for BN layer.
        alpha: Alpha for mobilenetv2.
        bias: Whether to use bias in module.
        use_pool: Whether to use pool in the out layer.
        classfier_num: Num classes of output layer.
    """

    def __init__(
        self,
        bn_kwargs: Optional[dict] = None,
        alpha: float = 1.0,
        bias: bool = True,
        use_pool: bool = True,
        classfier_num: int = 2,
    ):
        super(IrisMutilBranchHead, self).__init__()

        start_stage = 4
        end_stage = 6
        in_chls = [
            None,
            None,
            None,
            None,
            [int(x * alpha) for x in [64, 64, 64]],
            [int(x * alpha) for x in [48, 32]],
        ]
        out_chls = [
            None,
            None,
            None,
            None,
            [int(x * alpha) for x in [64, 64, 48]],
            [int(x * alpha) for x in [32, 32]],
        ]

        self.r_branch = IrisSingleBranchHead(
            bn_kwargs=bn_kwargs,
            alpha=alpha,
            bias=bias,
            classifier_num=classfier_num,
            start_stage=start_stage,
            end_stage=end_stage,
            use_pool=use_pool,
            in_chls=in_chls,
            out_chls=out_chls,
            pre_channels=4,
        )

        self.l_branch = IrisSingleBranchHead(
            bn_kwargs=bn_kwargs,
            alpha=alpha,
            bias=bias,
            classifier_num=classfier_num,
            start_stage=start_stage,
            end_stage=end_stage,
            use_pool=use_pool,
            in_chls=in_chls,
            out_chls=out_chls,
            pre_channels=4,
        )

    def forward(self, x):
        data = x
        l_branch_feat = self.l_branch(data)
        r_branch_feat = self.r_branch(data)
        return [l_branch_feat, r_branch_feat]

    def fuse_model(self):
        for module in [self.l_branch, self.r_branch]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.l_branch, self.r_branch]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
