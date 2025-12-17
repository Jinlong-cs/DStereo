# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from .metric import EvalMetric

__all__ = ["SmokeClsRecall", "SmokeClsPrecision"]


@OBJECT_REGISTRY.register
class SmokeClsRecall(EvalMetric):
    """Computes smoke cls recall with threshold.

    Args:
        axis: The axis that represents classes
            Defaults to 1.
        cls_id: Target class id
            Defaults to 1.
        thresh: Threshold of the target class.
            Defaults to 0.5.
        name:  Name of this metric instance for display.
            Defaults to "recall".
    """

    def __init__(
        self,
        axis: int = 1,
        cls_id: int = 1,
        thresh: float = 0.5,
        name: str = "smoke_cls_recall",
    ):
        self.axis = axis
        self.cls_id = cls_id
        self.thresh = thresh
        name = f"{name}_cls{cls_id}_thres{float}"
        super().__init__(name)

    def update(self, labels, preds):

        labels = _as_list(labels)
        preds = _as_list(preds)

        for label, pred_label in zip(labels, preds):

            softmax_scores = torch.nn.functional.softmax(pred_label, self.axis)
            preds_score, preds_label = torch.max(softmax_scores, dim=self.axis)
            label = torch.argmax(label, dim=self.axis)

            preds_score = preds_score.squeeze()
            preds_label = preds_label.squeeze()
            label = label.squeeze()
            assert len(label) == len(preds_label)

            self.num_inst += torch.sum(label == self.cls_id)
            self.sum_metric += torch.sum(
                preds_score[label == self.cls_id] > self.thresh
            )


@OBJECT_REGISTRY.register
class SmokeClsPrecision(EvalMetric):
    """Computes smoke cls precision with threshold.

    Args:
        axis: The axis that represents classes
            Defaults to 1.
        cls_id: Target class id
            Defaults to 1.
        thresh: Threshold of the target class.
            Defaults to 0.5.
        name: Name of this metric instance for display.
            Defaults to "precision".
    """

    def __init__(
        self,
        axis: int = 1,
        cls_id: int = 1,
        thresh: float = 0.5,
        name: str = "smoke_cls_precision",
    ):
        self.axis = axis
        self.cls_id = cls_id
        self.thresh = thresh
        name = f"{name}_cls{cls_id}_thres{float}"
        super().__init__(name)

    def update(self, labels, preds):

        labels = _as_list(labels)
        preds = _as_list(preds)

        for _, pred in zip(labels, preds):

            softmax_scores = torch.nn.functional.softmax(pred, self.axis)
            preds_score, preds_label = torch.max(softmax_scores, dim=self.axis)

            self.num_inst += torch.sum(preds_label == self.cls_id)
            self.sum_metric += torch.sum(
                preds_score[preds_label == self.cls_id] > self.thresh
            )
