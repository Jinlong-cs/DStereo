# Copyright (c) Horizon Robotics. All rights reserved.

import torch.nn as nn

from hat.models.task_modules.fcos import FCOSHead
from hat.models.weight_init import bias_init_with_prob, normal_init
from hat.registry import OBJECT_REGISTRY

__all__ = ["AutoAssignHead"]


@OBJECT_REGISTRY.register
class AutoAssignHead(FCOSHead):
    """AutoAssignHead used in AutoAssign. More details can be found \
    in the `paper <https://arxiv.org/abs/2007.03496>`_ ."""

    def _init_weights(self):
        """Initialize weights of the head.

        In particular, we have special initialization for classified conv's and
        regression conv's bias
        """

        super(AutoAssignHead, self)._init_weights()
        if self.share_conv:
            bias_cls = bias_init_with_prob(0.02)
            normal_init(self.conv_cls, std=0.01, bias=bias_cls)
            normal_init(self.conv_reg, std=0.01, bias=4.0)
        else:
            bias_cls = bias_init_with_prob(0.02)
            for m in self.conv_cls.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01, bias=bias_cls)
            for m in self.conv_reg.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01, bias=4.0)
