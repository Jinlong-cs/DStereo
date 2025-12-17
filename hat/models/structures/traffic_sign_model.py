from torch import nn

from hat.models.structures.classifier import Classifier
from hat.registry import OBJECT_REGISTRY

__all__ = ["TrafficSignClassifierMultitask"]


@OBJECT_REGISTRY.register
class TrafficSignClassifierMultitask(Classifier):
    """
    The basic structure of classifier for traffic sign.

    Args:
        backbone: Backbone module.
        backbone_extra: Backbone extra moduel.
        prediction_head: Prediction head using features extracted from backbone.
        losses: Losses module.
        desc: Desc module.
    """  # noqa

    def __init__(
        self,
        backbone: nn.Module,
        backbone_extra: nn.Module,
        prediction_head: nn.Module,
        losses: nn.Module,
        desc: nn.Module = None,
    ):
        super(TrafficSignClassifierMultitask, self).__init__(backbone, losses)
        self.backbone_extra = backbone_extra
        self.prediction_head = prediction_head
        self.desc = desc

    def forward(self, data):
        image = data["img"]
        target = data.get("gt_classes", None)
        features = self.backbone(image)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        features = self.backbone_extra(features)
        preds = self.prediction_head(features)
        if self.training and self.desc is not None:
            preds = self.desc(preds)
        if target is None:
            return preds

        if not self.training or self.losses is None:
            return dict(preds=preds, target=target)  # noqa

        losses = self.losses(preds, target)

        return dict(preds=preds, classification_loss=losses)  # noqa

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.backbone is not None:
            if hasattr(self.backbone, "set_qconfig"):
                self.backbone.set_qconfig()
        if self.prediction_head is not None:
            if hasattr(self.prediction_head, "set_qconfig"):
                self.prediction_head.set_qconfig()

        if self.losses is not None:
            self.losses.qconfig = None

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
        if self.losses is not None:
            self.losses.qconfig = None

    def fuse_model(self):
        for module in [self.backbone, self.prediction_head, self.losses]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
