# Copyright (c) Horizon Robotics. All rights reserved.
import logging

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "EyeStatusClassifier",
    "EyeStatusSingleHeadClassifier",
    "EyeStatusMultiTaskClassifier",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class EyeStatusClassifier(nn.Module):
    """
    The basic structure of eye status classifier.

    Args:
        backbone : Backbone module.
        cls_head : Classifier head module.
        ldmk_neck : Ldmk neck module.
        ldmk_head: Ldmk head module.
        heatmap_losses : Heatmap loss module.
        ldmk_losses : Ldmk loss module.
        left_cls_losses : Classifier loss module of left eye status.
        right_cls_losses : Classifier loss module of right eye status.
        num_ldmk : The number of key points in the eyes.
        num_classes : The number of eye state categories.
        deploy: Is deploy model. Defaults to False.
    """

    def __init__(
        self,
        backbone: nn.Module,
        cls_head: nn.Module,
        ldmk_neck: nn.Module = None,
        ldmk_head: nn.Module = None,
        heatmap_losses: nn.Module = None,
        ldmk_losses: nn.Module = None,
        left_cls_losses: nn.Module = None,
        right_cls_losses: nn.Module = None,
        num_ldmk: int = 16,
        num_classes: int = 5,
        deploy: bool = False,
    ):
        super(EyeStatusClassifier, self).__init__()
        self.backbone = backbone
        self.ldmk_neck = ldmk_neck
        self.cls_head = cls_head
        self.ldmk_head = ldmk_head
        self.ldmk_losses = ldmk_losses
        self.heatmap_losses = heatmap_losses
        self.left_cls_losses = left_cls_losses
        self.right_cls_losses = right_cls_losses
        self.num_ldmk = num_ldmk
        self.num_classes = num_classes
        self.deploy = deploy

    def forward(self, data):
        cls_target = data.get("gt_eye_cls_labels", None)
        outputs = self.get_outpus(data)
        if (
            self.deploy
            or cls_target is None
            or self.left_cls_losses is None
            or self.right_cls_losses is None
        ):
            return outputs["l_feat"], outputs["r_feat"]

        losses = self.get_losses(data, outputs)
        return outputs, losses

    def get_feats(self, features):
        l_feat, r_feat = self.cls_head(features[-1])
        return l_feat, r_feat

    def get_outpus(self, data):
        image = data["img"]
        features = self.backbone(image)
        l_feat, r_feat = self.get_feats(features)
        outputs = {
            "l_feat": l_feat,
            "r_feat": r_feat,
        }

        if self.ldmk_head is not None:
            ldmk_preds = self.ldmk_neck(features[1:])
            out_x, out_y, heatmap = self.ldmk_head(ldmk_preds)
            outputs["x_dist"] = out_x
            outputs["y_dist"] = out_y
            outputs["heatmap"] = heatmap
        return outputs

    def get_losses(self, data, outputs):
        losses = self.get_cls_loss(data, outputs)
        if self.ldmk_losses is not None:
            losses.update(self.get_ldmk_loss(data, outputs))
        return losses

    def get_cls_loss(self, data, outputs):
        cls_target = data.get("gt_eye_cls_labels", None)
        with torch.no_grad():
            l_cls_target, r_cls_target = torch.split(
                cls_target, self.num_classes, dim=1
            )
            l_cls_target = l_cls_target.contiguous()
            r_cls_target = r_cls_target.contiguous()

        left_cls_loss = self.left_cls_losses(
            outputs["l_feat"].view(-1, self.num_classes),
            l_cls_target.detach(),
        )

        right_cls_loss = self.right_cls_losses(
            outputs["r_feat"].view(-1, self.num_classes),
            r_cls_target.detach(),
        )

        losses = {
            "left_cls_loss": left_cls_loss,
            "right_cls_loss": right_cls_loss,
        }
        return losses

    def get_ldmk_loss(self, data, outputs):
        losses = {}
        batch_size = data["img"].size()[0]
        ldmk_target_x = data.get("gt_vector_x")
        ldmk_target_y = data.get("gt_vector_y")
        ldmk_target_weight_x = data.get("gt_vector_weight_x")
        ldmk_target_weight_y = data.get("gt_vector_weight_y")

        heatmap_target = data.get("gt_heatmap")
        heatmap_target_weight = data.get("gt_heatmap_weight")
        ldmk_attr = data.get("gt_ldmk_attr")

        out_x = outputs["x_dist"].view(batch_size, self.num_ldmk, -1)
        out_y = outputs["y_dist"].view(batch_size, self.num_ldmk, -1)

        ldmk_loss_x = self.ldmk_losses(
            out_x,
            ldmk_target_x,
            ldmk_target_weight_x * ldmk_attr.view((batch_size, -1, 1)),
        )
        ldmk_loss_y = self.ldmk_losses(
            out_y,
            ldmk_target_y,
            ldmk_target_weight_y * ldmk_attr.view((batch_size, -1, 1)),
        )
        heatmap_loss = self.heatmap_losses(
            outputs["heatmap"],
            heatmap_target,
            heatmap_target_weight * ldmk_attr.view((batch_size, -1, 1, 1)),
        )
        losses["heatmap_loss"] = heatmap_loss * 0.5
        losses["ldmk_loss"] = (ldmk_loss_x * 0.5 + ldmk_loss_y * 0.5) * 0.5
        return losses

    def fuse_model(self):
        for module in [
            self.backbone,
            self.cls_head,
        ]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.cls_head is not None:
            if hasattr(self.cls_head, "set_qconfig"):
                self.cls_head.set_qconfig()

        if self.ldmk_head is not None:
            if hasattr(self.ldmk_head, "set_qconfig"):
                self.ldmk_head.set_qconfig()

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )


@OBJECT_REGISTRY.register
class EyeStatusSingleHeadClassifier(EyeStatusClassifier):
    """
    The structure of eye status classifier in which the \
        cls_head has a single output.

    Args:
        backbone : Backbone module.
        cls_head : Classifier head module.
        ldmk_neck : Ldmk neck module.
        ldmk_head: Ldmk head module(single output).
        heatmap_losses : Heatmap loss module.
        ldmk_losses : Ldmk loss module.
        left_cls_losses : Classifier loss module of left eye status.
        right_cls_losses : Classifier loss module of right eye status.
        num_ldmk : The number of key points in the eyes.
        num_classes : The number of eye state categories.
        deploy: Is deploy model. Defaults to False.
    """

    def __init__(
        self,
        backbone: nn.Module,
        cls_head: nn.Module,
        ldmk_neck: nn.Module = None,
        ldmk_head: nn.Module = None,
        heatmap_losses: nn.Module = None,
        ldmk_losses: nn.Module = None,
        left_cls_losses: nn.Module = None,
        right_cls_losses: nn.Module = None,
        num_ldmk: int = 16,
        num_classes: int = 5,
        deploy: bool = False,
    ):
        super(EyeStatusSingleHeadClassifier, self).__init__(
            backbone,
            cls_head,
            ldmk_neck,
            ldmk_head,
            heatmap_losses,
            ldmk_losses,
            left_cls_losses,
            right_cls_losses,
            num_ldmk,
            num_classes,
            deploy,
        )

    def get_feats(self, features):
        feats = self.cls_head(features[-1])
        batch_size = feats.size()[0]
        l_feat, r_feat = torch.split(feats, batch_size // 2, dim=0)
        return l_feat, r_feat


@OBJECT_REGISTRY.register
class EyeStatusMultiTaskClassifier(EyeStatusClassifier):
    """
    The structure of eye status classifier in which the \
        cls_head has a single output.

    Args:
        backbone : Backbone module.
        cls_head : Classifier head module.
        bin_cls_head : Binary classification head
        ldmk_neck : Ldmk neck module.
        ldmk_head: Ldmk head module(single output).
        heatmap_losses : Heatmap loss module.
        ldmk_losses : Ldmk loss module.
        left_cls_losses : Classifier loss module of left eye status.
        right_cls_losses : Classifier loss module of right eye status.
        num_ldmk : The number of key points in the eyes.
        num_classes : The number of eye state categories.
        deploy: Is deploy model. Defaults to False.
    """

    def __init__(
        self,
        backbone: nn.Module,
        cls_head: nn.Module,
        bin_cls_head: nn.Module,
        ldmk_neck: nn.Module = None,
        ldmk_head: nn.Module = None,
        heatmap_losses: nn.Module = None,
        ldmk_losses: nn.Module = None,
        left_cls_losses: nn.Module = None,
        right_cls_losses: nn.Module = None,
        left_bin_cls_losses: nn.Module = None,
        right_bin_cls_losses: nn.Module = None,
        num_ldmk: int = 16,
        num_classes: int = 5,
        deploy: bool = False,
    ):
        super(EyeStatusMultiTaskClassifier, self).__init__(
            backbone,
            cls_head,
            ldmk_neck,
            ldmk_head,
            heatmap_losses,
            ldmk_losses,
            left_cls_losses,
            right_cls_losses,
            num_ldmk,
            num_classes,
            deploy,
        )
        self.left_bin_cls_losses = left_bin_cls_losses
        self.right_bin_cls_losses = right_bin_cls_losses
        self.bin_cls_head = bin_cls_head

    def get_bin_cls_loss(self, data, outputs):
        losses = {}
        cls_target = data.get("gt_eye_cls_labels", None)
        l_cls_target, r_cls_target = torch.split(
            cls_target, self.num_classes, dim=1
        )
        l_cls_target = l_cls_target.contiguous()
        r_cls_target = r_cls_target.contiguous()
        # get the label for whether to close eyes(first dimension)
        l_bin_cls_target = torch.split(l_cls_target, 1, dim=1)[0]
        r_bin_cls_target = torch.split(r_cls_target, 1, dim=1)[0]

        l_bin_cls_target = l_bin_cls_target.contiguous()
        r_bin_cls_target = r_bin_cls_target.contiguous()

        left_bin_cls_loss = self.left_bin_cls_losses(
            outputs["l_bin_feat"],
            l_bin_cls_target.detach(),
        )

        right_bin_cls_loss = self.right_bin_cls_losses(
            outputs["r_bin_feat"],
            r_bin_cls_target.detach(),
        )
        losses["left_bin_loss"] = left_bin_cls_loss
        losses["right_bin_loss"] = right_bin_cls_loss
        return losses

    def get_losses(self, data, outputs):
        with torch.no_grad():
            losses = self.get_cls_loss(data, outputs)
            if self.ldmk_losses is not None:
                losses.update(self.get_ldmk_loss(data, outputs))
        losses.update(self.get_bin_cls_loss(data, outputs))
        return losses

    def get_outpus(self, data):
        image = data["img"]
        features = self.backbone(image)
        l_feat, r_feat = self.get_feats(features)
        outputs = {
            "l_feat": l_feat,
            "r_feat": r_feat,
        }

        if self.ldmk_head is not None:
            ldmk_preds = self.ldmk_neck(features[1:])
            out_x, out_y, heatmap = self.ldmk_head(ldmk_preds)
            outputs["x_dist"] = out_x
            outputs["y_dist"] = out_y
            outputs["heatmap"] = heatmap

        l_bin_feat, r_bin_feat = self.bin_cls_head(features[-1])
        outputs["l_bin_feat"] = l_bin_feat
        outputs["r_bin_feat"] = r_bin_feat
        return outputs

    def forward(self, data):
        cls_target = data.get("gt_eye_cls_labels", None)
        outputs = self.get_outpus(data)
        if (
            self.deploy
            or cls_target is None
            or self.left_cls_losses is None
            or self.right_cls_losses is None
        ):
            return (
                outputs["l_feat"],
                outputs["r_feat"],
                outputs["l_bin_feat"],
                outputs["r_bin_feat"],
            )

        losses = self.get_losses(data, outputs)
        return outputs, losses

    def fuse_model(self):
        for module in [
            self.backbone,
            self.cls_head,
            self.bin_cls_head,
        ]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.cls_head is not None:
            if hasattr(self.cls_head, "set_qconfig"):
                self.cls_head.set_qconfig()

        if self.bin_cls_head is not None:
            if hasattr(self.bin_cls_head, "set_qconfig"):
                self.bin_cls_head.set_qconfig()

        if self.ldmk_head is not None:
            if hasattr(self.ldmk_head, "set_qconfig"):
                self.ldmk_head.set_qconfig()
