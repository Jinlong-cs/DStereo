# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY

__all__ = ["PanopticQualityWithAttributes"]

logger = logging.getLogger(__name__)
OFFSET = 1000


@OBJECT_REGISTRY.register
class PanopticQualityWithAttributes(EvalMetric):
    """Evaluate panoptic segmentation results.

    Args:
        name: Name of this metric instance for display, also used as
            monitor params for Checkpoint.
        attributes: The instance attributes to be evaluated.
        ignore_index: The label index that will be ignored in evaluation.
        void_index: The label index corresponding to the background area.
        iou_thr: IoU threshold for matching gt and prediction. Default is 0.5.
        verbose:  Whether to return verbose value for aidi eval.
            Default is False.
    """

    def __init__(
        self,
        name: str = "",
        attributes: Optional[Dict] = None,
        ignore_index: int = 255,
        void_index: int = 0,
        iou_thr: float = 0.5,
        verbose: bool = False,
    ):
        self.name = name
        self.attributes = {} if attributes is None else attributes
        super().__init__(name)
        self.ignore_index = ignore_index
        self.void_index = void_index
        self.iou_thr = iou_thr
        self.verbose = verbose

    def _init_states(self):
        self.add_state(
            "iou",
            default=torch.zeros((1)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "tp",
            default=torch.zeros((1)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fp",
            default=torch.zeros((1)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fn",
            default=torch.zeros((1)),
            dist_reduce_fx="sum",
        )
        for k, v in self.attributes.items():
            self.add_state(
                k,
                default=torch.zeros((len(v), 4)),  # 4 for (tp, fp, tn, iou)
                dist_reduce_fx="sum",
            )

    def update(
        self,
        gt_attributes: Sequence[torch.Tensor],
        gt_id_maps: Sequence[torch.Tensor],
        pred_attributes: Sequence[torch.Tensor],
        pred_id_maps: Sequence[torch.Tensor],
        **kwargs,
    ):
        """
        Update internal buffer with latest predictions.

        Note that the statistics are not available until
        you call self.get() to return the metrics.

        Args:
            gt_attributes: Each element represents the ground truth
                attributeds, has shape (num_gts, attribute_number + 1),
                where each row is [index, attr_1, attr_2, ... , attr_n].
            gt_id_maps: Each element represents the ground truth index map,
                has shape (H, W).
            pred_attributes: Each element represents the predicted
                attributeds, has shape (num_preds, attribute_number + 1),
                where each row is [index, attr_1, attr_2, ... , attr_n].
            pred_id_maps: Each element represents the predicted index map,
                has shape (H, W).

        """
        assert (
            len(gt_attributes)
            == len(gt_id_maps)
            == len(pred_attributes)
            == len(pred_id_maps)
        )
        for gt_attr, gt_seg, pred_attr, pred_seg in zip(
            gt_attributes, gt_id_maps, pred_attributes, pred_id_maps
        ):
            assert (
                len(self.attributes) + 1
                == pred_attr.shape[1]
                == gt_attr.shape[1]
            ), (
                self.attributes,
                pred_attr,
                gt_attr,
            )

            pred_seg = pred_seg.long()
            gt_seg = gt_seg.long()
            assert gt_seg.max() < OFFSET, torch.unique(gt_seg)
            pan_seg = pred_seg * OFFSET + gt_seg
            index, counts = torch.unique(pan_seg, return_counts=True)
            id2count_pan = {int(i): c for i, c in zip(index, counts)}
            index, counts = torch.unique(pred_seg, return_counts=True)
            id2count_pred = {int(i): c for i, c in zip(index, counts)}
            index, counts = torch.unique(gt_seg, return_counts=True)
            id2count_gt = {int(i): c for i, c in zip(index, counts)}
            id2attr_pred = {int(_[0]): _[1:] for _ in pred_attr}
            id2attr_gt = {int(_[0]): _[1:] for _ in gt_attr}

            matched_index_pred = set()
            matched_index_gt = set()
            matched_index_pan = set()
            for index_pan, intersection in id2count_pan.items():
                index_pred, index_gt = index_pan // OFFSET, index_pan % OFFSET
                if (
                    index_pred == self.void_index
                    or index_gt == self.void_index
                ):
                    continue

                if index_gt == self.ignore_index:
                    # predicted segment is false positive if more than
                    # half of the segment correspond to ignore regions
                    if intersection / id2count_pred[index_pred] > 0.5:
                        matched_index_pred.add(index_pred)
                        matched_index_pan.add(index_pan)
                        self.fp += 1
                    continue

                union = (
                    id2count_pred[index_pred]
                    + id2count_gt[index_gt]
                    - intersection
                )
                iou = intersection / union
                if iou > self.iou_thr:
                    matched_index_pred.add(index_pred)
                    matched_index_gt.add(index_gt)
                    matched_index_pan.add(index_pan)
                    self.iou += iou
                    self.tp += 1
                    # calculate tp/fn/fp for each attribute
                    attr_pred = id2attr_pred[index_pred]
                    if index_gt not in id2attr_gt:
                        logging.warning(
                            f"index_gt={index_gt} in gt_id_map, "
                            f"but not in gt_label {id2attr_gt.keys()}"
                        )
                        continue
                    attr_gt = id2attr_gt[index_gt]
                    for i, k in enumerate(self.attributes.keys()):
                        attr = self.__getattribute__(k)
                        attr_pred_i = attr_pred[i]
                        attr_gt_i = attr_gt[i]
                        if attr_pred_i == attr_gt_i:
                            attr[attr_gt_i][0] += 1  # tp
                        else:
                            attr[attr_gt_i][1] += 1  # fn
                            attr[attr_pred_i][2] += 1  # fp
                        attr[attr_gt_i][3] += iou

            for index_pred in id2count_pred.keys():
                if index_pred == self.void_index:
                    continue
                if index_pred not in matched_index_pred:
                    self.fp += 1

            for index_gt in id2count_gt.keys():
                if index_gt in [self.void_index, self.ignore_index]:
                    continue
                if index_gt not in matched_index_gt:
                    self.fn += 1

    def compute(self):
        """Get evaluation metrics."""

        def _tensor_nan_mean(x: torch.Tensor):
            """Compute the mean value, ignoring NaNs."""
            tmp_value = x.cpu().numpy()
            _mean_x = np.nanmean(tmp_value)
            mean_x = x.new_tensor(_mean_x)
            return mean_x

        sq = self.iou / self.tp if self.tp != 0 else self.tp
        rq = self.tp / (self.tp + 0.5 * self.fp + 0.5 * self.fn)
        pq = self.iou / (self.tp + 0.5 * self.fp + 0.5 * self.fn)
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)

        attr_f1 = {}
        attr_prec = {}
        attr_recall = {}
        attr_miou = {}
        for k in self.attributes.keys():
            attr = self.__getattribute__(k)  # [:, (tp, fn, fp)]
            tp = attr[:, 0]
            tp_fn = attr[:, 0] + attr[:, 1]
            tp_fp = attr[:, 0] + attr[:, 2]
            miou = attr[:, 3] / tp_fn
            tp_fp[tp_fp == 0] = 1  # avoid nan
            tp_fp[tp_fn == 0] = 0  # ignore classes without positive samples
            attr_prec[k] = tp / tp_fp
            attr_recall[k] = tp / tp_fn
            num = 2.0 * (attr_prec[k] * attr_recall[k])
            nom = attr_prec[k] + attr_recall[k]
            nom[nom == 0] = 1  # avoid nan
            nom[tp_fn == 0] = 0  # ignore classes without positive samples
            attr_f1[k] = num / nom
            attr_miou[k] = miou

        summary_str = "~~~~ %s Summary metrics ~~~~\n" % (self.name)
        line_format = "{:<18} {:>10} {:>10} {:>10} {:>10} {:>10}\n"

        # logging instance segmentation results
        summary_str += "Instance segmentation:\n"
        summary_str += line_format.format(
            "Scope", "F1-Score", "Precision", "Recall", "mIoU", "PQ"
        )
        pq_str = "{:.2f}".format(pq.cpu().item() * 100)
        sq_str = "{:.2f}".format(sq.cpu().item() * 100)
        rq_str = "{:.2f}".format(rq.cpu().item() * 100)
        prec_str = "{:.2f}".format(precision.cpu().item() * 100)
        recall_str = "{:.2f}".format(recall.cpu().item() * 100)
        summary_str += line_format.format(
            "global", rq_str, prec_str, recall_str, sq_str, pq_str
        )

        if len(self.attributes) == 0:
            logger.info(summary_str)
            if self.verbose:
                return (
                    pq,
                    sq,
                    rq,
                    precision,
                    recall,
                    attr_f1,
                    attr_prec,
                    attr_recall,
                    attr_miou,
                )
            else:
                return pq

        # logging attribute classification results
        summary_str += "-" * 62 + "\n"

        summary_str += "Attribute classification:\n"
        line_format = "{:<18} {:>10} {:>10} {:>10} {:>10}\n"

        per_attr_fpr = []
        for k in self.attributes.keys():
            mf1 = _tensor_nan_mean(attr_f1[k]).cpu().item() * 100
            mprec = _tensor_nan_mean(attr_prec[k]).cpu().item() * 100
            mrecall = _tensor_nan_mean(attr_recall[k]).cpu().item() * 100
            miou = _tensor_nan_mean(attr_miou[k]).cpu().item() * 100
            per_attr_fpr.append([mf1, mprec, mrecall, miou])

        mf1, mprec, mrecall, miou = np.mean(per_attr_fpr, 0)
        f1_str = "{:.2f}".format(mf1)
        prec_str = "{:.2f}".format(mprec)
        recall_str = "{:.2f}".format(mrecall)
        miou_str = "{:.2f}".format(miou)
        summary_str += line_format.format(
            "global", f1_str, prec_str, recall_str, miou_str
        )

        summary_str += "Per Class Results:\n"
        for i, k in enumerate(self.attributes.keys()):
            mf1, mprec, mrecall, miou = per_attr_fpr[i]
            prec_str = "{:.2f}".format(mprec)
            recall_str = "{:.2f}".format(mrecall)
            f1_str = "{:.2f}".format(mf1)
            miou_str = "{:.2f}".format(miou)
            summary_str += line_format.format(
                k, f1_str, prec_str, recall_str, miou_str
            )

        summary_str += "-" * 62 + "\n"
        summary_str += "Detailed Results:\n"
        for k in self.attributes.keys():
            attr_f1_k = attr_f1[k] * 100
            attr_prec_k = attr_prec[k] * 100
            attr_recall_k = attr_recall[k] * 100
            attr_miou_k = attr_miou[k] * 100

            summary_str += f"attr: {k}\n"
            for (
                attr_name,
                attr_f1_k_i,
                attr_prec_k_i,
                attr_recall_k_i,
                attr_miou_k_i,
            ) in zip(
                self.attributes[k],
                attr_f1_k,
                attr_prec_k,
                attr_recall_k,
                attr_miou_k,
            ):
                prec_str = "{:.2f}".format(attr_prec_k_i.cpu().item())
                recall_str = "{:.2f}".format(attr_recall_k_i.cpu().item())
                f1_str = "{:.2f}".format(attr_f1_k_i.cpu().item())
                miou_str = "{:.2f}".format(attr_miou_k_i.cpu().item())
                summary_str += line_format.format(
                    attr_name, f1_str, prec_str, recall_str, miou_str
                )

        logger.info(summary_str)

        if self.verbose:
            return (
                pq,
                sq,
                rq,
                precision,
                recall,
                attr_f1,
                attr_prec,
                attr_recall,
                attr_miou,
            )
        else:
            return pq
