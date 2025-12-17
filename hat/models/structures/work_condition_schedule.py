from torch import nn

from hat.models.structures.classifier import Classifier
from hat.registry import OBJECT_REGISTRY

__all__ = ["WorkConditionClassifier"]


@OBJECT_REGISTRY.register
class WorkConditionClassifier(Classifier):
    """The basic structure of classifier.

    Args:
        backbone (torch.nn.Module): Backbone module.
        backbone_extra: Neck module.
        prediction_head (torch.nn.Module): Prediction head using features extracted from backbone.   # noqa
        losses (torch.nn.Module): Losses module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_extra: nn.Module,
        prediction_head: nn.Module,
        losses: nn.Module,
        desc: nn.Module = None,
    ):
        super(WorkConditionClassifier, self).__init__(backbone, losses)
        self.backbone_extra = backbone_extra
        self.prediction_head = prediction_head
        self.desc = desc

    def forward(self, data):
        image = data["img"]
        target = data.get("labels", None)
        features = self.backbone(image)
        features = self.backbone_extra(features)
        preds = self.prediction_head(features)
        if self.training and self.desc is not None:
            preds = self.desc(preds)
        if target is None:
            return {"pred_cls": preds.flatten(1)}

        if not self.training or self.losses is None:
            return {"preds": preds.flatten(1), "target": target}

        losses = self.losses(preds.flatten(1), target)

        return {"preds": preds.flatten(1), "classification_loss": losses}

    def set_qconfig(self):
        super(WorkConditionClassifier, self).set_qconfig(self)
        if self.prediction_head is not None:
            if hasattr(self.prediction_head, "set_qconfig"):
                self.prediction_head.set_qconfig()

    def fuse_model(self):
        for module in [self.backbone, self.prediction_head, self.losses]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
