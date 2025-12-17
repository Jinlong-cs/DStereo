# Copyright (c) Horizon Robotics. All rights reserved.
import logging

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["GazeModel"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class GazeModel(nn.Module):
    """
    The basic structure of classifier.

    Args:
        backbone (torch.nn.Module): Backbone module.
        head (torch.nn.Module): Head module.
        losses (torch.nn.Module): Losses module.
    """

    def __init__(
        self,
        backbone,
        head,
        losses=None,
        deploy: bool = False,
        compile_eyeldmk: bool = True,
    ):
        super(GazeModel, self).__init__()
        self.backbone = backbone
        self.head = head
        self.losses = losses
        self.deploy = deploy
        self.compile_eyeldmk = compile_eyeldmk

    def forward(self, data):
        image = data["img"]
        target = data.get("gaze_label", None)

        feats = self.backbone(image)

        head_input = {}
        head_input["last_backbone_feat"] = feats[-1]

        preds = self.head(head_input)

        if self.deploy:
            if self.compile_eyeldmk:
                if len(preds["eye_ldmk"]) == 6:  # normal
                    return [
                        preds["gaze"],
                        *[preds["eye_ldmk"][_] for _ in [0, 1, 3, 4]],
                    ]
                elif len(preds["eye_ldmk"]) == 3:  # speed up with groupconv
                    return [preds["gaze"], preds["eye_ldmk"][0]]
            else:
                return preds["gaze"]

        # format 3 vals to 6 vals for loss calculation
        if len(preds["eye_ldmk"]) == 3:
            group, lp, rp = preds["eye_ldmk"]
            ll, li, rl, ri = torch.split(group, 16, 1)
            preds["eye_ldmk"] = [ll, li, lp, rl, ri, rp]

        if target is None:
            return preds

        if not self.training or self.losses is None:
            return preds, target

        loss_dict = self.losses(preds, target)
        preds["loss_dict"] = loss_dict
        total_loss = 0.0
        for _, value in loss_dict.items():
            total_loss += value
        return preds, total_loss

    def fuse_model(self):
        for module in [self.backbone, self.head, self.losses]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
                else:
                    module.qconfig = None
        if self.losses is not None:
            self.losses.qconfig = None

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
        if self.losses is not None:
            self.losses.qconfig = None
