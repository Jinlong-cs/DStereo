# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Optional

import torch

from hat.metrics.landmark import DecodeHeatmap, DecodeVector
from hat.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = [
    "SmokeKpsNME",
    "SoftmaxAccuracy",
    "SoftmaxRecall",
    "SoftmaxPrecision",
]

logger = logging.getLogger(__name__)

SMOKE_KPS_NORM_TYPE = ["norm12", "norm34", "norm1234"]


@OBJECT_REGISTRY.register
class SmokeKpsNME(EvalMetric):
    """A common metric for smoke keypoints detection.

    Noramlized Mean Error is the mean Euclidean distance of gt landmarks and
    predicted ones which is normlized by some kind of distance.

    Args:
        num_ldmk: number of landmarks.
        name: metric name that is shown in log.
        norm_type: normalization type. It must be chosen from
            SMOKE_KPS_NORM_TYPE. Defaults to "norm12".
            In case "norm12", NME is noramlized by inter-ocular distance.
            In case "norm34", NME is normalized by inter-pupil distance.
            In case "norm1234", NME is normalized by sqrt of bbox area.
        mode: parse landmark from "coords" or "heatmap" or "vector".
        decoding_method: only accessible in "heatmap" or "vector" mode.
            "diff", "shift" and "taylor" are supported.
        feat_stride: feature stride of input / output.
    """

    def __init__(
        self,
        num_ldmk: int,
        name: str = "NME",
        norm_type: str = "norm12",
        mode: str = "heatmap",
        is_ignore_idx: bool = True,
        decoding_method: str = "",
        feat_stride: Optional[float] = None,
    ):
        super().__init__(name)
        self.eps = 1e-6
        self.num_ldmk = num_ldmk
        self.norm_type = norm_type.lower()
        assert self.norm_type in SMOKE_KPS_NORM_TYPE
        self.is_ignore_idx = is_ignore_idx
        self.mode = mode.lower()
        self.decode_vector = DecodeVector(decoding_method, feat_stride)
        self.decode_heatmap = DecodeHeatmap(decoding_method, feat_stride)

    def update(self, data):
        gt_ldmk = data["gt_ldmk"]  # N, num_ldmk, 2/3
        ldmk_mask = data["gt_ldmk_attr"].mean(axis=-1)  # N, num_ldmk
        pr_ldmk = self._get_preds(data).reshape(gt_ldmk.shape)

        err = self.compute_error(
            gt_ldmk, pr_ldmk, ldmk_mask, mode=self.norm_type
        )
        self.sum_metric += err.sum()
        self.num_inst += gt_ldmk.shape[0]

    def _get_preds(self, data):
        if self.mode == "vector":
            vector_x = data["pr_vector_x"].squeeze(2)
            vector_y = data["pr_vector_y"].squeeze(3)
            vector = torch.stack([vector_x, vector_y], axis=1)  # N, 2, K, L
            batch_size = vector.shape[0]
            pr_ldmk = torch.zeros(
                (batch_size, self.num_ldmk, 2), device=vector.device
            )
            for i in range(batch_size):
                for j in range(self.num_ldmk):
                    pr_ldmk[i, j] = self.decode_vector(vector[i, :, j])
        elif self.mode == "heatmap":
            heatmap = data["pr_heatmap"]
            batch_size = heatmap.shape[0]
            pr_ldmk = torch.zeros(
                (batch_size, self.num_ldmk, 2), device=heatmap.device
            )
            for i in range(batch_size):
                for j in range(self.num_ldmk):
                    pr_ldmk[i, j] = self.decode_heatmap(heatmap[i, j])
        else:
            raise ValueError(f"Not suppored decoding mode: {self.mode}.")
        return pr_ldmk

    def compute_error(self, label, pred, weight, mode="norm12"):
        """Compute normalized error of landmarks.

        Parameters
        ----------
        label : np.array or mx.nd.NDArray
            GT landmark coordinates with shape (batch_size, num_lmks, dim). e.g. (16, 4, 2) # noqa
        pred : np.array or mx.nd.NDArray
            Predicted landmark coordinates with shape (batch_size, num_lmks, dim). e.g. (16, 4, 2) # noqa
        weight: np.array or mx.nd.NDArray
            Landmark coordinates weight.
        mode: string
            Including
                norm12:error12/smoke-length.
                norm34:error34/smoke-width.
                norm1234: error12/smoke-length + error34/smoke-width.

        Returns
        -------
        nomralized_err: numpy.ndarray
            Normalized error. (batch_size,)
        """
        error = torch.norm(pred - label, dim=2)
        norm_dist = 1
        error34 = 0

        if mode.startswith("norm12"):
            error12 = torch.mean(error[:, :2], dim=1)
            norm_dist = torch.sqrt(
                torch.square(label[:, 0] - label[:, 1]).sum(axis=1)
            )
            normalized_err12 = error12 * weight / (norm_dist + self.eps)
            if mode == "norm12":
                return normalized_err12
        elif mode == "norm34":
            norm_dist = torch.sqrt(
                torch.square(label[:, 2] - label[:, 3]).sum(axis=1)
            )
        else:
            raise ValueError(f"Not suppored decoding mode: {self.mode}.")

        if self.is_ignore_idx:
            error34 = torch.mean(error[:, 2:], dim=1)
        else:
            error34 = torch.mean(error[:, 2:], dim=1)
            _error34 = torch.norm(
                pred[:, 3] - label[:, 2], dim=1
            ) + torch.norm(pred[:, 2] - label[:, 3], dim=1)
            error34 = torch.min([error34, _error34 / 2], axis=0)

        normalized_err34 = error34 * weight / (norm_dist + self.eps)
        if mode == "norm34":
            return normalized_err34

        return (normalized_err12 + normalized_err34) / 2


@OBJECT_REGISTRY.register
class SoftmaxAccuracy(EvalMetric):
    def __init__(
        self,
        name: str = "Acc",
        cls_type: str = "classes",
        thresh: float = 0.5,
    ):
        super().__init__(name)
        self.eps = 1e-6
        self.cls_type = cls_type
        self.thresh = thresh

    def update(self, data):
        if self.cls_type not in ["classes", "visable"]:
            raise ValueError(f"Not suppored cls type: {self.cls_type}.")

        gt_cls = (
            data["gt_classes"]
            if self.cls_type == "classes"
            else data["gt_visable"]
        )
        pr_cls = self._get_preds(data).reshape(gt_cls.shape)
        ldmk_mask = data["gt_ldmk_attr"].mean(axis=-1).reshape(gt_cls.shape)

        self.sum_metric += torch.sum(torch.eq(gt_cls, pr_cls) * ldmk_mask)
        self.num_inst += ldmk_mask.sum()

    def _get_preds(self, data):
        if self.cls_type == "classes":
            pr_cls = data["pr_classes"].argmax(axis=1)
        else:
            pr_cls = torch.ge(data["pr_visable"], self.thresh)
        return pr_cls


@OBJECT_REGISTRY.register
class SoftmaxRecall(EvalMetric):
    def __init__(
        self,
        name: str = "Rec",
        cls_type: str = "classes",
        cls_id: int = 1,
        thresh: float = 0.5,
    ):
        super().__init__(name)
        self.eps = 1e-6
        self.cls_type = cls_type
        self.cls_id = cls_id
        self.thresh = thresh

    def update(self, data):
        if self.cls_type not in ["classes", "visable"]:
            raise ValueError(f"Not suppored cls type: {self.cls_type}.")

        gt_cls = (
            data["gt_classes"]
            if self.cls_type == "classes"
            else data["gt_visable"]
        )
        pr_cls = self._get_preds(data).reshape(gt_cls.shape)
        ldmk_mask = data["gt_ldmk_attr"].mean(axis=-1).reshape(gt_cls.shape)
        gt_cls = gt_cls * ldmk_mask

        self.sum_metric += torch.sum(
            torch.eq(gt_cls, pr_cls) * gt_cls,
            dtype=float,
        )
        self.num_inst += gt_cls.sum()

    def _get_preds(self, data):
        if self.cls_type == "classes":
            pr_cls = torch.ge(data["pr_classes"][:, self.cls_id], self.thresh)
        else:
            pr_cls = torch.ge(data["pr_visable"][:, self.cls_id], self.thresh)
        return pr_cls


@OBJECT_REGISTRY.register
class SoftmaxPrecision(EvalMetric):
    def __init__(
        self,
        name: str = "Pre",
        cls_type: str = "classes",
        cls_id: int = 1,
        thresh: float = 0.5,
    ):
        super().__init__(name)
        self.eps = 1e-6
        self.cls_type = cls_type
        self.cls_id = cls_id
        self.thresh = thresh

    def update(self, data):
        if self.cls_type not in ["classes", "visable"]:
            raise ValueError(f"Not suppored cls type: {self.cls_type}.")

        gt_cls = (
            data["gt_classes"]
            if self.cls_type == "classes"
            else data["gt_visable"]
        )
        pr_cls = self._get_preds(data).reshape(gt_cls.shape)
        ldmk_mask = data["gt_ldmk_attr"].mean(axis=-1).reshape(gt_cls.shape)
        pr_cls = pr_cls * ldmk_mask

        self.sum_metric += torch.sum(
            torch.eq(gt_cls, pr_cls) * pr_cls,
            dtype=float,
        )
        self.num_inst += pr_cls.sum()

    def _get_preds(self, data):
        if self.cls_type == "classes":
            pr_cls = torch.ge(data["pr_classes"][:, self.cls_id], self.thresh)
        else:
            pr_cls = torch.ge(data["pr_visable"][:, self.cls_id], self.thresh)
        return pr_cls
