import logging

import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["CycWheelKpsModel"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class CycWheelKpsModel(nn.Module):
    """The basic structure of cyclist wheel landmark model.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        post_process: Postprocess module.
        deploy: true, deploy models.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        postprocess: nn.Module = None,
        deploy: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.postprocess = postprocess
        self.deploy = deploy

    def forward(self, data: dict):
        img = data["img"]
        bone_feats = self.backbone(img)
        neck_feats = self.neck(bone_feats)
        if self.deploy:
            output = self.head(neck_feats)
            res = output["kps_label_pred"], output["kps_pos_offset_pred"]
            if self.postprocess is not None:
                res = self.postprocess(res)
            return res
        else:
            data["features"] = neck_feats
            output = self.head(data)
            if self.postprocess is not None:
                assert not self.training
                keys_output = self.postprocess(output, data)
                output.update(keys_output)

            return output

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.neck, self.head]:
            if module is not None and hasattr(module, "set_qconfig"):
                module.set_qconfig()
