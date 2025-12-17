# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import List

import torch

from hat.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = [
    "EyeStatusMetrics",
    "BinEyeStatusMetrics",
    "MixEyeStatusMetrics",
]

logger = logging.getLogger(__name__)

EPS = 1e-12

EYE_STATUS_CLASSES_NUM = 5


EYE_STATUS_TARGET_MAP = {
    "close": 0,
    "open": 1,
    "occlude": 2,
    "narrow": 3,
    "lookdown": 4,
}


def compute_precision_recall_by_confuse_mtx(confuse_mtx: torch.Tensor):
    tptn = torch.sum(confuse_mtx, -1).to(torch.float32)
    tpfp = torch.sum(confuse_mtx, 0).to(torch.float32)
    diag = torch.diag(confuse_mtx).to(torch.float32)

    tpfp = tpfp.view(-1)
    tptn = tptn.view(-1)

    pre = diag / (tpfp + EPS)
    recall = diag / (tptn + EPS)
    precision_recall = torch.stack([pre, recall])
    return precision_recall


@OBJECT_REGISTRY.register
class EyeStatusMetrics(EvalMetric):
    """Evaluation eye status classification results.

    Args:
        num_classes: Total class num.
        name: Name of this metric instance for display, also used as
            monitor params for Checkpoint.

    """

    def __init__(
        self,
        num_classes: int,
        name: str = "eye-status",
    ):
        self.num_classes = num_classes

        metric_names = [
            "confuse-mtx",
            "precision-recall",
        ]

        names = []
        for metric_name in metric_names:
            names.append(f"{name}-{metric_name}")

        super(EyeStatusMetrics, self).__init__(names)

    def _check_update_input(self, data):
        assert len(data.size()) > 0
        assert len(data.size()) < 3

        if len(data.size()) == 1:
            data = torch.unsqueeze(data, 0)
        return data

    def _init_states(self):

        self.add_state(
            "confusion_matrix",
            default=torch.zeros(
                (self.num_classes, self.num_classes), dtype=torch.int32
            ),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        label: torch.Tensor,
        preds: torch.Tensor,
    ):
        """
        Update internal buffer with latest predictions.

        Note that the statistics are not available until
        you call self.get() to return the metrics.

        Args:
            preds: model output.
            label: gt.
        """
        label = self._check_update_input(label)
        preds = self._check_update_input(preds)
        assert label.size()[0] == preds.size()[0]

        label_cls = torch.argmax(label, -1).view(-1)
        # filter invalid label [0, 0, 0, 0, 0]
        valid_ind = torch.sum(label, axis=-1) != 0
        valid_num = torch.sum(valid_ind)

        if valid_num != 0:
            pred_cls = torch.argmax(preds, -1)
            pred_cls = pred_cls.view(-1)

            label_cls = label_cls[valid_ind]
            pred_cls = pred_cls[valid_ind]

            indices = self.num_classes * label_cls + pred_cls
            cls_confusion_matrix = torch.bincount(
                indices, minlength=self.num_classes ** 2
            ).reshape(self.num_classes, self.num_classes)

            self.confusion_matrix += cls_confusion_matrix.to(
                self.confusion_matrix
            )

    def compute(self):
        """Get evaluation metrics."""
        precision_recall = compute_precision_recall_by_confuse_mtx(
            self.confusion_matrix
        )
        return self.confusion_matrix, precision_recall


@OBJECT_REGISTRY.register
class BinEyeStatusMetrics(EyeStatusMetrics):
    """Evaluation eye status binary preds results.

    Args:
        bin_inds: List of eval index.
        sigmoid_thresh: Thresh of binary preds.
        verify_bin_classes_only: Only use bin inds samples for eval.
        name: name of this metric instance for display, also used as
            monitor params for Checkpoint.

    """

    def __init__(
        self,
        bin_inds: List[int],
        sigmoid_thresh: float,
        verify_bin_classes_only: bool = True,
        name: str = "bin-eye-status",
    ):
        assert len(bin_inds) == 2
        self.bin_inds = bin_inds
        self.sigmoid_thresh = sigmoid_thresh
        self.verify_bin_classes_only = verify_bin_classes_only

        super(BinEyeStatusMetrics, self).__init__(num_classes=2, name=name)

    def _get_label_cls_and_valid_ind(self, label):
        label_cls = torch.argmax(label, -1).view(-1)
        valid_ind = torch.sum(label, axis=-1) != 0
        bin_label_cls = torch.zeros_like(label_cls, dtype=torch.int32)

        cls1_ind = label_cls == self.bin_inds[0]
        cls2_ind = label_cls == self.bin_inds[1]

        bin_label_cls[cls2_ind] = 1

        if self.verify_bin_classes_only:
            cls_ind = torch.logical_or(cls1_ind, cls2_ind)
            valid_ind = torch.logical_and(valid_ind, cls_ind)

        return bin_label_cls, valid_ind

    def update(
        self,
        label: torch.Tensor,
        preds: torch.Tensor,
    ):
        """
        Update internal buffer with latest predictions.

        Note that the statistics are not available until
        you call self.get() to return the metrics.

        Args:
            preds: model binary output.
            label: gt.

        """
        label = self._check_update_input(label)
        preds = self._check_update_input(preds)
        assert label.size()[0] == preds.size()[0]

        label_cls, valid_ind = self._get_label_cls_and_valid_ind(label)

        valid_num = torch.sum(valid_ind)

        if valid_num != 0:
            preds = torch.sigmoid(preds)
            preds = preds.view(-1)
            pos_ind = preds > self.sigmoid_thresh
            pred_cls = torch.ones_like(preds, dtype=torch.int32)
            pred_cls[pos_ind] = 0
            pred_cls = pred_cls[valid_ind]
            label_cls = label_cls[valid_ind]

            indices = self.num_classes * label_cls + pred_cls
            cls_confusion_matrix = torch.bincount(
                indices, minlength=self.num_classes ** 2
            ).reshape(self.num_classes, self.num_classes)
            self.confusion_matrix += cls_confusion_matrix.to(
                self.confusion_matrix
            )


@OBJECT_REGISTRY.register
class MixEyeStatusMetrics(EyeStatusMetrics):
    """Evaluation mix eye status pred and pred binary results.

    Args:
        num_classes: Total class num.
        name: Name of this metric instance for display, also used as
            monitor params for Checkpoint.
        narrow_thresh: Binary thresh for narrow target.
        close_thresh: Binary thresh for close target.
        open_thresh: Binary thresh for open target.
        close_pred_ind: Index of close target.
        open_pred_ind: Index of open target.
        narrow_pred_ind: Index of narrow target.
    """

    def __init__(
        self,
        num_classes: int,
        name: str = "MixEyeStatusMetrics",
        narrow_thresh: float = 0.5,
        close_thresh: float = 0.5,
        close_thresh_close: float = 0.7,
        open_thresh: float = 0.5,
        close_pred_ind: int = 0,
        open_pred_ind: int = 1,
        narrow_pred_ind: int = 3,
        lookdown_pred_ind: int = 4,
    ):
        self.num_classes = num_classes
        self.narrow_thresh = narrow_thresh
        self.open_thresh = open_thresh
        self.close_thresh = close_thresh
        self.close_thresh_close = close_thresh_close

        # label pos
        self.close_pred_ind = close_pred_ind
        self.open_pred_ind = open_pred_ind
        self.narrow_pred_ind = narrow_pred_ind
        self.lookdown_pred_ind = lookdown_pred_ind

        super(MixEyeStatusMetrics, self).__init__(
            num_classes=num_classes, name=name
        )

    def apply_bin_on_cls(
        self,
        pred_cls: torch.Tensor,
        pred_bin: torch.Tensor,
        valid_pred_idx: int,
        thresh: float,
        greater: bool = True,
    ):

        assert valid_pred_idx < self.num_classes

        valid_batch_ind = pred_cls == valid_pred_idx

        pred_bin = pred_bin.view(-1)
        if greater:
            valid_pred_ind = pred_bin > thresh
        else:
            valid_pred_ind = pred_bin <= thresh

        exchange_ind = torch.logical_and(valid_pred_ind, valid_batch_ind)

        return exchange_ind

    def update(
        self,
        label: torch.Tensor,
        preds: torch.Tensor,
        preds_bin: torch.Tensor,
    ):
        """
        Update internal buffer with latest predictions.

        Note that the statistics are not available until
        you call self.get() to return the metrics.

        Args:
            label: gt.
            preds: model output.
            preds_bin: model binary output.

        """
        label = self._check_update_input(label)
        preds = self._check_update_input(preds)
        preds_bin = self._check_update_input(preds_bin)
        assert label.size()[0] == preds.size()[0]
        assert label.size()[0] == preds_bin.size()[0]

        label_cls = torch.argmax(label, -1).view(-1)
        # filter invalid label [0, 0, 0, 0, 0]
        valid_ind = torch.sum(label, axis=-1) != 0

        valid_num = torch.sum(valid_ind)
        if valid_num != 0:
            preds_bin = torch.sigmoid(preds_bin)

            pred_cls = torch.argmax(preds, -1)
            pred_cls = pred_cls.view(-1)
            preds_bin = preds_bin.view(-1)

            label_cls = label_cls[valid_ind]
            pred_cls = pred_cls[valid_ind]
            preds_bin = preds_bin[valid_ind]

            mix_cls = pred_cls.clone()
            exchange_ind = self.apply_bin_on_cls(
                pred_cls,
                preds_bin,
                self.lookdown_pred_ind,
                self.close_thresh_close,
                greater=True,
            )
            mix_cls[exchange_ind] = 0

            exchange_ind = self.apply_bin_on_cls(
                pred_cls,
                preds_bin,
                self.narrow_pred_ind,
                self.narrow_thresh,
                greater=True,
            )
            mix_cls[exchange_ind] = 0

            exchange_ind = self.apply_bin_on_cls(
                pred_cls,
                preds_bin,
                self.open_pred_ind,
                self.open_thresh,
                greater=True,
            )
            mix_cls[exchange_ind] = 0

            exchange_ind = self.apply_bin_on_cls(
                pred_cls,
                preds_bin,
                self.close_pred_ind,
                self.close_thresh,
                greater=False,
            )
            mix_cls[exchange_ind] = 3

            indices = self.num_classes * label_cls + mix_cls
            cls_confusion_matrix = torch.bincount(
                indices, minlength=self.num_classes ** 2
            ).reshape(self.num_classes, self.num_classes)
            self.confusion_matrix += cls_confusion_matrix.to(
                self.confusion_matrix
            )
