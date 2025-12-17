import logging
import math

import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["HMR"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class HMR(nn.Module):
    """The structure of HMR.

    "End-to-end Recovery of Human Shape and Pose", CVPR 2017.
    https://arxiv.org/pdf/1712.06584.pdf

    Args:
        backbone: Backbone module.
        head: Head module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None):
        x = self.backbone(x)
        output = self.head(x[-1], init_pose, init_shape, init_cam)
        return output

    def fuse_model(self):
        modules = [
            self.backbone,
            self.head,
        ]

        for module in modules:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        modules = [
            self.backbone,
            self.head,
        ]
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in modules:
            if module is not None and hasattr(module, "set_qconfig"):
                module.set_qconfig()
