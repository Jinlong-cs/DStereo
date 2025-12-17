# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, Optional

import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from .metric import EvalMetric

__all__ = [
    "CumulativeAccuracy",
    "SigmoidOrdinalMAE",
    "SigmoidOrdinalAccuracy",
]


@OBJECT_REGISTRY.register
class CumulativeAccuracy(EvalMetric):
    """Computes age accuracy in the condition of acceptable error.

    Args:
        offset: The acceptable age error.
        classes: Num_class of age.
        in_type: Different postprocessors for age prediction.
        name: Name of this metric instance for display. Defaults to None.
    """

    def __init__(
        self,
        offset: int,
        classes: int,
        in_type: Optional[str] = "sigmoid",
        name: Optional[str] = None,
    ):

        if name is None:
            name = "cumulative-acc-{:02d}".format(offset)
        super(CumulativeAccuracy, self).__init__(name)
        self.offset = offset
        self.in_type = in_type
        self.classes = classes

    def update(self, model_outs: Dict):
        labels = _as_list(model_outs["age"])
        preds = _as_list(model_outs["pred_age"])
        for label, pred in zip(labels, preds):
            if self.in_type == "sigmoid":
                pred_label = (pred > 0).sum(axis=1)
            else:
                raise ValueError("unknown type {}...".format(self.in_type))
            label = label.flatten()
            pred_label = pred_label.flatten()
            mask = label != -1
            diff = torch.abs(pred_label - label)

            sum_metric = ((diff <= self.offset) * mask).sum()
            self.sum_metric += sum_metric
            num_inst = mask.sum()
            self.num_inst += num_inst


@OBJECT_REGISTRY.register
class SigmoidOrdinalMAE(EvalMetric):
    """Computes mean absolute error of age prediction.

    Args:
        from_sigmoid: Whether or not the age prediction
            activated by sigmoid. Defaults to False.
        name: Name of this metric instance for display.
            Defaults to 'ord-mae'.
    """

    def __init__(
        self,
        from_sigmoid: Optional[bool] = False,
        name: Optional[str] = "ord-mae",
    ):

        super(SigmoidOrdinalMAE, self).__init__(name)
        self.from_sigmoid = from_sigmoid

    def update(self, model_outs: Dict):
        labels = _as_list(model_outs["age"])
        preds = _as_list(model_outs["pred_age"])
        for pred, label in zip(preds, labels):
            if self.from_sigmoid:
                pred_label = (pred > 0.5).sum(axis=1)
            else:
                pred_label = (pred > 0).sum(axis=1)
            label = label.flatten()
            pred_label = pred_label.flatten()
            mask = label != -1
            diff = torch.abs(pred_label - label) * mask

            sum_metric = diff.sum()
            self.sum_metric += sum_metric
            num_inst = mask.sum()
            self.num_inst += num_inst


@OBJECT_REGISTRY.register
class SigmoidOrdinalAccuracy(EvalMetric):
    """Computes age segmentation accuracy based on the acceptable error.

    Args:
        offset: The acceptable age segmentation error.
        age_bins: Age segmentation.
        from_sigmoid: Whether or not the age prediction
            activated by sigmoid. Defaults to False.
        need_convert: Whether need convert age prediction to
            the segmentation prediction. Defaults to True.
        name: Name of this metric instance for display. Defaults to None.
    """

    def __init__(
        self,
        offset,
        age_bins,
        from_sigmoid=False,
        need_convert=True,
        name: Optional[str] = None,
    ):

        if name is None:
            name = "ord-acc-{}off".format(offset)
        super(SigmoidOrdinalAccuracy, self).__init__(name)
        self.offset = offset
        self.age_bins = age_bins
        self.from_sigmoid = from_sigmoid
        self.need_convert = need_convert

    def _age2class(self, pred_label, label):
        if not isinstance(pred_label, np.ndarray):
            pred_label = pred_label.cpu().numpy()
        if not isinstance(label, np.ndarray):
            label = label.cpu().numpy()
        pred_clss_label = np.digitize(pred_label, self.age_bins, right=True)
        clss_label = np.digitize(label, self.age_bins, right=True)
        clss_label[np.where(label == -1)] = -1

        return pred_clss_label, clss_label

    def update(self, model_outs: Dict):
        labels = _as_list(model_outs["age"])
        preds = _as_list(model_outs["pred_age"])
        for pred, label in zip(preds, labels):
            if self.from_sigmoid:
                pred_label = (pred > 0.5).sum(axis=1)
            else:
                pred_label = (pred > 0).sum(axis=1)
            label = label.flatten()
            pred_label = pred_label.flatten()

            if self.need_convert:
                pred_label, label = self._age2class(pred_label, label)

            mask = label != -1
            diff = abs(pred_label - label) * mask
            sum_metric = (diff <= self.offset).sum()
            self.sum_metric += sum_metric
            num_inst = mask.sum()
            self.num_inst += num_inst
