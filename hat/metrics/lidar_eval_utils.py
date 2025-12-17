import bisect
import math
import os
import os.path as osp
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Tuple
from xml.etree import ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable

from hat.core.box_torch_ops import boxes_iou3d_gpu_horizon
from hat.registry import OBJECT_REGISTRY

__all__ = ["IoUMatching", "DetEvaluation", "VisFPNCurve", "VisFPNDist"]


def group_by_key(detections: Dict, key: str, group_idx: bool = False) -> Dict:
    groups = defaultdict(list)
    for idx, detection in enumerate(detections):
        if group_idx:
            groups[detection[key]].append([idx, detection])
        else:
            groups[detection[key]].append(detection)
    return groups


def in_area(x: float, y: float, area: List[float]) -> bool:
    return area[0] < x < area[1] and area[2] < y < area[3]


def filter_by_area(
    gt: List[Dict], predictions: List[Dict], area: List[float]
) -> Tuple[List, List]:
    """Filter examples by area size.

    Args:
        gt: list of dictionaries in the format described below.
        predictions: list of dictionaries in the format described below.
        ranges: list of range in the format described below
        gt = [{
            'token': 'pandar128_2021_2_22_2000/pcl_with_r/1612608682800.bin',
            'box': [4.81605492e+01, -4.02222855e-01, -1.25083796e+00,
                    2.07142004e+00,  4.33924437e+00,  1.73003797e+00,
                    -1.57852292e+00],
            'name': 'Car',
        }]
        predictions = [{
            'token': 'pandar128_2021_2_22_2000/pcl_with_r/1612608682800.bin',
            'box': [4.81605492e+01, -4.02222855e-01, -1.25083796e+00,
                    2.07142004e+00,  4.33924437e+00,  1.73003797e+00,
                    -1.57852292e+00],
            'name': 'Car',
            'score': 0.3077029437237213
        }]
        area = [x1, x2, y1, y2]
    """
    assert len(area) == 4
    assert area[0] < area[1] and area[2] < area[3]

    new_gt = []
    new_predictions = []
    for x in gt:
        if in_area(x["box"][0], x["box"][1], area):
            new_gt.append(x)
    for x in predictions:
        if in_area(x["box"][0], x["box"][1], area):
            new_predictions.append(x)
    return new_gt, new_predictions


def filter_by_area_seg(
    points: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    area: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter examples by area segmentation.

    Args:
        points: Shape nx3.
        labels: Shape n.
        preds: Shape n.
        area = [x1, x2, y1, y2]
    Returns:
        labels: Filtered labels with shape m.
        preds: Filtered preds with shape m.
    """
    assert points.shape[0] == labels.shape[0] == preds.shape[0]
    assert area[0] < area[2] and area[1] < area[3]
    mask = np.logical_and(
        np.logical_and(area[0] < points[:, 0], points[:, 0] < area[2]),
        np.logical_and(area[1] < points[:, 1], points[:, 1] < area[3]),
    )
    labels = labels[mask]
    preds = preds[mask]

    return labels, preds


def check_dist(
    box: np.ndarray,
    range_limit: List[float] = None,
    all_range_limit: List[float] = None,
) -> bool:
    """Check whether a box is within the range_limit and/or all_range_limit.

    Either could be None and thus can ignored.
    """
    if range_limit is None:
        range_limit = [0, 1000]
    if all_range_limit is None:
        all_range_limit = [1000]
    x = box["box"][0]
    y = box["box"][1]
    r1, r2 = range_limit
    if len(all_range_limit) == 4:
        # all_range_limit is a rectangle area [x1, y1, x2, y2]
        x1, y1, x2, y2 = all_range_limit
        range_limit_flag = r1 < math.sqrt(x ** 2 + y ** 2) < r2
        all_range_limit_flag = (x1 < x < x2) and (y1 < y < y2)
        return range_limit_flag and all_range_limit_flag
    else:
        # all_range_limit is a circle area with radius r
        r = all_range_limit[0]
        r1 = max(0, r1)
        r2 = min(r, r2)
        return r1 < math.sqrt(x ** 2 + y ** 2) < r2


def filter_by_range(
    gt: List[Dict],
    predictions: List[Dict],
    range_limit: List[float] = None,
    all_range_limit: List[float] = None,
) -> Tuple[List, List]:
    """Filter examples by range.

    Args:
        gt: list of dictionaries in the format described below.
        predictions: list of dictionaries in the format described below.
        range_limit: list of range in the format described below
        gt = [{
            'token': 'pandar128_2021_2_22_2000/pcl_with_r/1612608682800.bin',
            'box': [4.81605492e+01, -4.02222855e-01, -1.25083796e+00,
                    2.07142004e+00,  4.33924437e+00,  1.73003797e+00,
                    -1.57852292e+00],
            'name': 'Car',
        }]
        predictions = [{
            'token': 'pandar128_2021_2_22_2000/pcl_with_r/1612608682800.bin',
            'box': [4.81605492e+01, -4.02222855e-01, -1.25083796e+00,
                    2.07142004e+00,  4.33924437e+00,  1.73003797e+00,
                    -1.57852292e+00],
            'name': 'Car',
            'score': 0.3077029437237213
        }]
        range_limit: [r1, r2] like [0, 20] or [20, 50]..., predefined distance
            intervals, for detailed range specific evaluation. Can be none for
            initial filtering.
        all_range_limit: An overall range limit that is superimposed over
            range_limit, could either be a rectangle limit area
            [x1, y1, x2, y2] or a circle limit area with radius r.
    """
    assert range_limit is None or len(range_limit) == 2
    assert (
        all_range_limit is None
        or len(all_range_limit) == 4
        or len(all_range_limit) == 1
    )
    if range_limit:
        assert range_limit[0] < range_limit[1]
        assert range_limit[0] >= 0

    new_gt = []
    new_predictions = []
    for x in gt:
        if check_dist(x, range_limit, all_range_limit):
            new_gt.append(x)
    for x in predictions:
        if check_dist(x, range_limit, all_range_limit):
            new_predictions.append(x)
    return new_gt, new_predictions


def get_envelope(precisions: np.ndarray) -> np.ndarray:
    """Compute the precision envelope."""
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    return precisions


def get_ap(recalls: np.ndarray, precisions: np.ndarray) -> np.ndarray:
    """Calculate average precision.

    Args:
        recalls: this is not a 41 point or 11 point recall, this is an
                element-wise recall
        precisions: element-wise precision
    Returns:
        average precision.
    """

    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    precisions = get_envelope(precisions)

    # to calculate area under PR curve, look for points where X axis (recall)
    # changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


def get_ap_waymo(recalls: List[float], precisions: List[float]) -> float:
    """Calculate average precision with waymo method.

    Args:
        recalls: This is not a 41 point or 11 point recall, this is an
                element-wise recall.
        precisions: Element-wise precision.
    Returns:
        average precision.
    """

    recall_precision = OrderedDict()
    recall_precision[0.0] = 1.0
    recall_precision[1.0] = 0.0
    for i, precision in enumerate(precisions):
        if recalls[i] in recall_precision:
            recall_precision[recalls[i]] = max(
                recall_precision[recalls[i]], precision
            )
        else:
            recall_precision[recalls[i]] = precision

    kError = 1e-6
    recall_delta = 0.05
    last_recall = 1.0
    max_precision = 0.0
    precision_recall = []
    ap = 0
    # Iterate from high recall to low recall.
    for it in reversed(recall_precision.items()):
        while last_recall - it[0] > recall_delta + kError:
            last_recall -= recall_delta
            precision_recall.append([max_precision, last_recall])
        # Update precision
        max_precision = max(it[1], max_precision)
        precision_recall.append([max_precision, it[0]])
        last_recall = it[0]
    precision_recall[-1][0] = precision_recall[-2][0]
    for i in range(1, len(precision_recall)):
        ap += (
            0.5
            * (precision_recall[i - 1][1] - precision_recall[i][1])
            * (precision_recall[i - 1][0] + precision_recall[i][0])
        )

    return ap


def recall_precision(
    gts_all: Dict,
    preds: Dict,
    class_name: str = "",
    dump_dir: str = None,
    bev_metric: bool = False,
    ignore_abnormal_recall: bool = False,
) -> Tuple[Any]:
    """Return recalls, precisions, ap, aph of current class.

    Args:
        gts_all: dictionaries of list of dictionaries of current class in the
            format described below.
        preds: list of dictionaries of current class in the format
                     described below.
        class_name: current class name.
        dump_dir: if dump_dir is not none, then fp ,tp and iou will be dumped
        bev_metric: use bev metric if True
        ignore_abnormal_recall: if True, recalls larger than 1 will be ignored

    Returns:
        recalls: This is not a 41 point or 11 point recall, this is an
                 element-wise recall.
        precisions: Element-wise precision.
        ap: average precision of current class
        aph: average precision with heading of current class
        Format for gts_all and preds:
        gts_all = {
            'gts': [{'token', 'box', 'name'}]
            'fn_gts': [{'token', 'box', 'name'}]
            'fn_bev_gts': [{'token', 'box', 'name'}]
        }
        preds = [{
            'token', 'box', 'name', 'score',
            'match_result': [max_overlap, tp, tp_h, fp],
            'match_result_bev': [max_overlap_bev, tp_bev, tp_h_bev, fp_bev]
        }]
    """
    gts = gts_all["gts"]
    num_gts = len(gts)

    # go down dets and mark TPs and FPs
    # obj-wise
    fp_list = []
    # function will return list with value if dump_dir is not None, else
    # return empty list
    iou_list = []

    if bev_metric:
        max_overlap = np.array([p["match_result_bev"][0] for p in preds])
        tp = np.array([p["match_result_bev"][1] for p in preds])
        tp_h = np.array([p["match_result_bev"][2] for p in preds])
        fp = np.array([p["match_result_bev"][3] for p in preds])
    else:
        max_overlap = np.array([p["match_result"][0] for p in preds])
        tp = np.array([p["match_result"][1] for p in preds])
        tp_h = np.array([p["match_result"][2] for p in preds])
        fp = np.array([p["match_result"][3] for p in preds])

    if dump_dir:
        for prediction_index, prediction_match in enumerate(preds):
            # Record IOU
            if max_overlap[prediction_index] > 0:
                iou_list.append(max_overlap[prediction_index])
            # Record FP
            if dump_dir and fp[prediction_index] == 1.0:
                if prediction_match["score"] > 0.4:
                    fp_list.append(
                        {
                            "token": prediction_match["token"],
                            "box": prediction_match["box"],
                            "name": prediction_match["name"],
                            "score": prediction_match["score"],
                        }
                    )
        # Record FN
        fn_dict = gts_all["fn_bev_gts"] if bev_metric else gts_all["fn_gts"]
        fn_dict = group_by_key(fn_dict, "token")

        with open(
            osp.join(dump_dir, "fp_list_{}.pkl".format(class_name)), "wb"
        ) as f:
            pickle.dump(fp_list, f)
        with open(
            osp.join(dump_dir, "fn_dict_{}.pkl".format(class_name)), "wb"
        ) as f:
            pickle.dump(fn_dict, f)
        with open(
            osp.join(dump_dir, "iou_list_{}.pkl".format(class_name)), "wb"
        ) as f:
            pickle.dump(iou_list, f)

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp_h = np.cumsum(tp_h, axis=0)
    tp = np.cumsum(tp, axis=0)
    if num_gts == 0:
        recalls = np.zeros(tp.shape)
    else:
        recalls = tp / float(num_gts)
    if ignore_abnormal_recall:
        recalls[recalls > 1] = 1
    assert np.all(0 <= recalls) & np.all(recalls <= 1)

    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    precisions_h = tp_h / np.maximum(tp + fp, np.finfo(np.float64).eps)
    assert np.all(0 <= precisions) & np.all(precisions <= 1)

    ap = get_ap_waymo(recalls, precisions)
    aph = get_ap_waymo(recalls, precisions_h)

    return recalls, precisions, ap, aph, fp, tp


def angle_diff(x: float, y: float, period: float = 2 * np.pi) -> float:
    """Get the min angle difference between 2 angles: the angle from y to x.

    Args:
        x: To angle.
        y: From angle.
        period: Periodicity in radians for assessing angle difference.

    Returns: <float>. Signed smallest between-angle difference in range
        (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


def calculate_fp_metrics(
    tp: List[float],
    fp: List[float],
    recalls: List[float],
    recall_thresholds: List[float],
) -> Tuple[List, List]:
    fp_rates = []
    fp_nums = []
    for i in range(len(recall_thresholds)):
        idx = bisect.bisect_left(recalls, recall_thresholds[i])
        if idx >= len(fp):
            idx = min(idx, len(fp) - 1)
            fp_rates.append(np.nan)
            fp_nums.append(np.nan)
            continue
        fp_rates.append(fp[idx] / (tp[idx] + fp[idx]))
        fp_nums.append(fp[idx])
    return fp_rates, fp_nums


def get_average_precisions(
    gts: List,
    predictions: List,
    range_limit: List,
    total_frame_num: int,
    recall_thresholds: Tuple[float] = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85),
    pr_curve_name: str = None,
    bev_metric: bool = False,
    dump_dir: str = None,
    ignore_abnormal_recall: bool = False,
) -> Tuple[np.array]:
    """Return an array with an average precision per class.

    Args:
        gts: dictionaries of list of dictionaries in the format described
            below.
        predictions: list of dictionaries in the format described below.
        iou_thresholds: IOU threshold used to calculate TP / FN
        pr_curve_name: if pr_curve_name is None, then no pr curve will be drawn
        dump_dir: if dump_dir is not none, then fp ,tp and iou will be dumped

    Returns an array with an average precision per class.
    Format for gts and predictions:
    gt = {
        'token': 'pandar128_2021_2_22_2000/pcl_with_r/1612608682800.bin',
        'box': [4.81605492e+01, -4.02222855e-01, -1.25083796e+00,
                2.07142004e+00,  4.33924437e+00,  1.73003797e+00,
                -1.57852292e+00],
        'name': 'Car',
    }
    gts = {
        'class1': {'gts': [gt], 'fn_gts': [fn_gt], 'fn_bev_gts': [fn_bev_gt]}
        'class2': {'gts': [gt], 'fn_gts': [fn_gt], 'fn_bev_gts': [fn_bev_gt]}
        ...
    }
    prediction = {
        'token', 'box', 'name', 'score',
        'match_result': [max_overlap, tp, tp_h, fp],
        'match_result_bev': [max_overlap_bev, tp_bev, tp_h_bev, fp_bev]
    }
    predictions = {
        'class1': [prediction_match]
        'class2': [prediction_match]
        ...
    }
    """
    class_names = list(gts.keys())
    average_precisions = np.zeros(len(class_names))
    average_precision_headings = np.zeros(len(class_names))
    fp_metrics = {
        "recall": np.zeros(len(class_names) + 1),
    }
    for i in range(len(recall_thresholds)):
        fp_metrics.update(
            {
                f"fp_rate@R{recall_thresholds[i]}": np.zeros(
                    len(class_names) + 1
                ),
                f"avg_fp_num@R{recall_thresholds[i]}": np.zeros(
                    len(class_names) + 1
                ),
            }
        )

    if pr_curve_name:
        # setup plot details
        colors = [
            "navy",
            "turquoise",
            "darkorange",
            "cornflowerblue",
            "teal",
            "gold",
            "g",
            "r",
        ]
        plt.figure(figsize=(7, 8))
        lines = []
        labels = []

    for class_id, class_name in enumerate(predictions):
        preds_class = predictions[class_name]
        gts_class = gts[class_name]
        (
            recalls,
            precisions,
            average_precision,
            average_precision_heading,
            fp,
            tp,
        ) = recall_precision(
            gts_class,
            preds_class,
            class_name=class_name,
            dump_dir=dump_dir,
            bev_metric=bev_metric,
            ignore_abnormal_recall=ignore_abnormal_recall,
        )

        if recalls.size == 0:
            fp_metrics["recall"][class_id] = 0
        else:
            fp_metrics["recall"][class_id] = recalls[-1]
        fp_rates, fp_nums = calculate_fp_metrics(
            tp, fp, recalls, recall_thresholds=recall_thresholds
        )
        for i in range(len(recall_thresholds)):
            fp_metrics[f"fp_rate@R{recall_thresholds[i]}"][
                class_id
            ] = fp_rates[i]
            fp_metrics[f"avg_fp_num@R{recall_thresholds[i]}"][class_id] = (
                fp_nums[i] / total_frame_num
            )
        average_precisions[class_id] = average_precision
        average_precision_headings[class_id] = average_precision_heading
        if pr_curve_name:
            (l,) = plt.plot(recalls, precisions, color=colors[class_id], lw=2)
            lines.append(l)
            labels.append(class_name)
    fp_metrics["recall"][-1] = np.mean(fp_metrics["recall"][:-1])
    for i in range(len(recall_thresholds)):
        nan_mask = np.isnan(
            fp_metrics[f"fp_rate@R{recall_thresholds[i]}"][:-1]
        )
        if np.sum(~nan_mask) > 0:
            fp_metrics[f"fp_rate@R{recall_thresholds[i]}"][-1] = np.mean(
                fp_metrics[f"fp_rate@R{recall_thresholds[i]}"][:-1][~nan_mask]
            )
            fp_metrics[f"avg_fp_num@R{recall_thresholds[i]}"][-1] = np.sum(
                fp_metrics[f"avg_fp_num@R{recall_thresholds[i]}"][:-1][
                    ~nan_mask
                ]
            )
        else:
            fp_metrics[f"fp_rate@R{recall_thresholds[i]}"][-1] = np.nan
            fp_metrics[f"avg_fp_num@R{recall_thresholds[i]}"][-1] = np.nan
    if pr_curve_name:
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.15)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            f"Precision-Recall curve for range "
            f"{range_limit[0]}~{range_limit[1]}m"
        )
        plt.legend(lines, labels, loc="best", prop={"size": 8})
        plt.savefig(pr_curve_name)

    return average_precisions, average_precision_headings, fp_metrics


def draw_block(
    im: np.ndarray,
    center: List[float],
    radius: float,
    k: float = 1,
    color: Tuple[int] = (0, 0, 255),
) -> np.ndarray:
    """Draw block on white image.

    Args:
        im: input image, must have 3 channels
        center: Draw center coords
        radius: Block radius
    """
    x, y = int(center[0]), int(center[1])

    left, right = radius, radius + 1
    top, bottom = radius, radius + 1

    masked_im = im[y - top : y + bottom, x - left : x + right, :]
    masked_block = np.ones_like(masked_im) * k
    color_substract = np.array([1 - c / 255 for c in color])
    masked_block = (masked_block * color_substract).astype(np.uint8)
    if min(masked_block.shape) > 0 and min(masked_im.shape) > 0:
        masked_im -= masked_block
    return im


class ConfusionMatrix:
    """Class for confusion matrix with various convenient methods."""

    def __init__(self, num_classes: int, ignore_idx: int = None):
        """Initialize a ConfusionMatrix object.

        :param num_classes: Number of classes in the confusion matrix.
        :param ignore_idx: Index of the class to be ignored in the confusion
            matrix.
        """
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx

        self.global_cm = None

    def update(self, gt_array: np.ndarray, pred_array: np.ndarray) -> None:
        """Update the global confusion matrix.

        :param gt_array: An array containing the ground truth labels.
        :param pred_array: An array containing the predicted labels.
        """
        cm = self._get_confusion_matrix(gt_array, pred_array)

        if self.global_cm is None:
            self.global_cm = cm
        else:
            self.global_cm += cm

    def _get_confusion_matrix(
        self, gt_array: np.ndarray, pred_array: np.ndarray
    ) -> np.ndarray:
        """
        Obtain the confusion matrix for the segmentation of a single pc.

        :param gt_array: An array containing the ground truth labels.
        :param pred_array: An array containing the predicted labels.
        :return: N x N array where N is the number of classes.
        """
        assert all(
            (gt_array >= 0) & (gt_array < self.num_classes)
        ), f"Error: Array for ground truth must be between 0 and "  # noqa [F541]
        f"{self.num_classes-1} (inclusive)."

        assert all(
            (pred_array >= 0) & (pred_array < self.num_classes)
        ), f"Error: Array for predictions must be between 0 and "  # noqa [F541]
        f"{self.num_classes-1} (inclusive)."

        label = self.num_classes * gt_array.astype("int") + pred_array
        count = np.bincount(label, minlength=self.num_classes ** 2)

        # Make confusion matrix (rows = gt, cols = preds).
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)

        # For the class to be ignored, set both the row and column to 0
        # (adapted from
        # https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py).
        if self.ignore_idx is not None:
            confusion_matrix[self.ignore_idx, :] = 0
            confusion_matrix[:, self.ignore_idx] = 0

        return confusion_matrix

    def get_per_class_iou(self) -> List[float]:
        """
        Get the IOU of each class in a confusion matrix.

        :return: An array in which the IOU of a particular class sits at the
            array index corresponding to the class index.
        """
        conf = self.global_cm.copy()

        # Get the intersection for each class.
        intersection = np.diagonal(conf)

        # Get the union for each class.
        ground_truth_set = conf.sum(axis=1)
        predicted_set = conf.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        # Get the IOU for each class.
        # In case we get a division by 0, ignore / hide the error(adapted from
        # https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py).
        with np.errstate(divide="ignore", invalid="ignore"):
            iou_per_class = intersection / (union.astype(np.float32))

        return iou_per_class

    def get_mean_iou(self) -> float:
        """
        Get the mean IOU (mIOU) over the classes.

        :return: mIOU over the classes.
        """
        iou_per_class = self.get_per_class_iou()
        miou = float(np.nanmean(iou_per_class))
        return miou

    def get_freqweighted_iou(self) -> float:
        """
        Get the frequency-weighted IOU over the classes.

        :return: Frequency-weighted IOU over the classes.
        """
        conf = self.global_cm.copy()

        # Get the number of points per class (based on ground truth).
        num_points_per_class = conf.sum(axis=1)

        # Get the total number of points in the eval set.
        num_points_total = conf.sum()

        # Get the IOU per class.
        iou_per_class = self.get_per_class_iou()

        # Weight the IOU by frequency and sum across the classes.
        freqweighted_iou = float(
            np.nansum(num_points_per_class * iou_per_class) / num_points_total
        )

        return freqweighted_iou


def dump_log_file(log_file: str, result: Dict):
    maphs = np.zeros(8)
    with open(log_file, "w") as f:
        line = "Python pipeline result:"
        print(line)
        f.write("{}\n".format(line))
        for i, key in enumerate(result):
            if key.startswith("All"):
                mean_8 = result[key][1]
                continue
            if key in ["fp_metrics"]:
                continue
            maphs[i - 1] = result[key][1]
            cat = key.split(" ")[0]
            line = "{} APH: {}".format(cat, round(result[key][1] * 100, 2))
            print(line)
            f.write("{}\n".format(line))
        maphs = np.delete(maphs, -1)
        mean_7 = maphs.mean()
        maphs = np.delete(maphs, 4)
        mean_6 = maphs.mean()
        line = "All 8 class APH: {}".format(round(mean_8 * 100, 2))
        print(line)
        f.write("{}\n".format(line))
        line = "7 class(no blur) APH: {}".format(round(mean_7 * 100, 2))
        print(line)
        f.write("{}\n".format(line))
        line = "6 class(no cons/blur) APH: {}".format(round(mean_6 * 100, 2))
        print(line)
        f.write("{}\n".format(line))
        print(" ")
        f.write("\n")


class HTMLGenerator:
    """Generate HTML file using ElementalTree."""

    def __init__(self, output_file: str):
        self.html = ET.Element("html")
        self.head = ET.Element("head")
        self.html.append(self.head)
        self.body = ET.Element("body")
        self.html.append(self.body)
        self.output_file = output_file

    def add_style(self, style: str):
        """
        Add a CSS style to the HTML.

        Style is string like '.bigtab{font-size:160%}'.
        """
        self.style = ET.Element("style")
        self.style.text = style
        self.head.append(self.style)

    def add_heading(self, text: str, size: int = 1):
        """
        Add a heading to the HTML.

        if size == 0, add text (<p>);
        if 1<=size<=6, add heading (<h>).
        """
        if size == 0:
            title = ET.Element("p")
        else:
            assert 1 <= size <= 6
            title = ET.Element(f"h{size}")
        title.text = text
        self.body.append(title)

    def add_imgs(self, img_list: List, width: int = 500):
        """
        Add image/images in a row to the HTML.

        img_list could be single image path or list of image paths.
        """
        div = ET.Element("div", attrib={"class": "images"})
        if not isinstance(img_list, list):
            img_list = [img_list]
        for img_path in img_list:
            img_full_path = osp.join(osp.dirname(self.output_file), img_path)
            if not osp.isfile(img_full_path):
                print(f"Image {img_path} does not exist. Skipped.")
                continue
            img = ET.Element(
                "img", attrib={"src": f"{img_path}", "width": f"{str(width)}"}
            )
            div.append(img)
        self.body.append(div)

    def add_table(
        self,
        table: str,
        style: str = None,
        title: str = None,
        caption: str = None,
    ):
        """Add table to the HTML.

        table is the variable name of the table in code
        and must be a Pandas.DataFrame, style is the name of style applied to
        the table, title and caption could be added above and below the table.
        """
        assert isinstance(
            table, pd.DataFrame
        ), "Table should be a pandas.DataFrame object"
        if title:
            self.add_heading(title, 0)
        tab = ET.fromstring(table.to_html(classes=style))
        self.body.append(tab)
        if caption:
            self.add_heading(caption, 0)

    def write(self):
        ET.ElementTree(self.html).write(
            self.output_file, encoding="unicode", method="html"
        )


class BaseMatchingModule(ABC):
    """Matching function template."""

    def __init__(self, class_names: list):
        self.class_names = class_names

    @abstractmethod
    def matching_function(
        self,
        preds: list,
        gts_by_token: dict,
        gts_matched: dict,
        gts_matched_bev: dict,
        class_id: list,
    ) -> list:
        return None

    def prediction_matching(
        self, gts_all: list, preds_all: list
    ) -> Tuple[Dict, Dict]:
        """Match predictions with gts.

        Get TP/FP/FN information for each class.
        Outputs will have matching information and grouped by classes.

        Args:
        gts_all = [{'token', 'box', 'name'}]
        preds_all = [{'token', 'box', 'name', 'score'}]

        Returns:
        gts_checked_all = {
            'class1': {'gts': [gt], 'fn_gts': [fn_gt],
                    'fn_bev_gts': [fn_bev_gt]}
            'class2': {'gts': [gt], 'fn_gts': [fn_gt],
                    'fn_bev_gts': [fn_bev_gt]}
            ...
        }
        preds_match_all = {
            'class1': [preds_match]
            'class2': [preds_match]
            ...
        }
        For details about content in each class,
        please refer to function prediction_matching_in_class.
        """
        gts_by_class_name = group_by_key(gts_all, "name")
        preds_by_class_name = group_by_key(preds_all, "name")
        preds_match_all = {}
        gts_checked_all = {}
        for class_id, class_name in enumerate(self.class_names):
            if class_name in preds_by_class_name:
                gts_in_class = gts_by_class_name[class_name]
                preds_in_class = preds_by_class_name[class_name]

                print("Matching class {}...".format(class_name))
                gts_checked, preds_match = self.prediction_matching_in_class(
                    gts_in_class, preds_in_class, class_id
                )

                gts_checked_all[class_name] = gts_checked
                preds_match_all[class_name] = preds_match

        return gts_checked_all, preds_match_all

    def prediction_matching_in_class(
        self, gts: list, preds: list, class_id: int
    ):
        """Do matching between gts and preds.

        Match within a class for both 3D IOU and bev IOU.
        For gts, add FN gt list to gts
        For predictions, record max_IOU, TP, TP_H, FP for each pred.

        Args:
            gts = [{'token', 'box', 'name'}]
            preds = [{'token', 'box', 'name', 'score'}]
        Returns:
            gts_checked = {
                'gts': [{'token', 'box', 'name'}]
                'fn_gts': [{'token', 'box', 'name'}]
                'fn_bev_gts': [{'token', 'box', 'name'}]
            }
            preds_match = [{
                'token', 'box', 'name', 'score',
                'match_result': [max_overlap, tp, tp_h, fp],
                'match_result_bev': [max_overlap_bev, tp_bev, tp_h_bev, fp_bev]
            }]
        """
        preds_match = []

        gts_by_token = group_by_key(gts, "token")
        gts_matched = {
            token: np.zeros(len(boxes))
            for token, boxes in gts_by_token.items()
        }
        gts_matched_bev = {
            token: np.zeros(len(boxes))
            for token, boxes in gts_by_token.items()
        }
        preds = sorted(preds, key=lambda x: x["score"], reverse=True)

        preds_match = self.matching_function(
            preds, gts_by_token, gts_matched, gts_matched_bev, class_id
        )
        # record FN
        fn_dict = []
        fn_bev_dict = []
        for token in gts_matched:
            cur_frame = gts_by_token[token]
            cur_frame_matched = gts_matched[token]
            cur_frame_matched_bev = gts_matched_bev[token]
            for i in range(len(cur_frame_matched)):
                if not cur_frame_matched[i]:
                    fn_dict.append(cur_frame[i])
                if not cur_frame_matched_bev[i]:
                    fn_bev_dict.append(cur_frame[i])

        gts_checked = {
            "gts": gts,
            "fn_gts": fn_dict,
            "fn_bev_gts": fn_bev_dict,
        }
        return gts_checked, preds_match


@OBJECT_REGISTRY.register_module
class IoUMatching(BaseMatchingModule):
    def __init__(self, class_names: list, iou_thresholds: list):
        """Iou Matching function.

        Args:
            class_names: object class names.
            iou_thresholds: iou_threshold for every class.
        """
        super(IoUMatching, self).__init__(class_names)
        self.iou_thresholds = iou_thresholds

    @staticmethod
    def normalize_angle(theta: float):
        return theta - 2 * np.pi * np.floor((theta + np.pi) / (2 * np.pi))

    def compute_heading_accuracy(self, pd_heading, gt_heading):
        pd_heading = self.normalize_angle(pd_heading)
        gt_heading = self.normalize_angle(gt_heading)
        diff_heading = abs(pd_heading - gt_heading)
        # Normalize heading error to [0, PI] (+PI and -PI are the same).
        if diff_heading > np.pi:
            diff_heading = 2.0 * np.pi - diff_heading
        # Clamp the range to avoid numerical errors.
        return min(1.0, max(0.0, 1.0 - diff_heading / np.pi))

    def matching_function(
        self,
        preds: list,
        gts_by_token: dict,
        gts_matched: dict,
        gts_matched_bev: dict,
        class_id: list,
    ) -> List:
        preds_match = []
        iou_threshold = self.iou_thresholds[class_id]
        for _, pred in enumerate(preds):
            pred_match = deepcopy(pred)
            pred_match.update(
                {
                    "match_result": [0, 0, 0, 0],  # max_overlap, TP, TPH, FP
                    "match_result_bev": [0, 0, 0, 0],
                }
            )

            token = pred["token"]

            max_overlap = -np.inf
            jmax = -1

            try:
                gts_in_token = gts_by_token[token]  # gt_boxes per sample
                # gt flags per sample
                gts_matched_in_token = gts_matched[token]
                # gt flags per sample
                gts_matched_bev_in_token = gts_matched_bev[token]
            except KeyError:
                # Key error happens when there are no gt boxes in current frame
                gts_in_token = []
                gts_matched_in_token = None
                gts_matched_bev_in_token = None

            if len(gts_in_token) > 0:
                overlaps, bev_overlaps = boxes_iou3d_gpu_horizon(
                    torch.tensor([gt["box"] for gt in gts_in_token])
                    .float()
                    .cuda(),
                    torch.tensor([pred["box"]]).float().cuda(),
                )
                overlaps, bev_overlaps = overlaps.numpy(), bev_overlaps.numpy()
                # 3D
                max_overlap = np.max(overlaps)
                jmax = np.argmax(overlaps)
                if (
                    max_overlap > iou_threshold
                    and gts_matched_in_token[jmax] == 0
                ):
                    gts_matched_in_token[jmax] = 1
                    heading_accuracy = self.compute_heading_accuracy(
                        pred["box"][-1], gts_in_token[jmax]["box"][-1]
                    )
                    pred_match["match_result"] = [
                        max_overlap,
                        1,
                        heading_accuracy,
                        0,
                    ]
                else:
                    pred_match["match_result"] = [max_overlap, 0, 0, 1]
                # bev
                if bev_overlaps is not None:
                    max_bev_overlap = np.max(bev_overlaps)
                    bev_jmax = np.argmax(bev_overlaps)
                    if (
                        max_bev_overlap > iou_threshold
                        and gts_matched_bev_in_token[bev_jmax] == 0
                    ):
                        gts_matched_bev_in_token[bev_jmax] = 1
                        heading_accuracy = self.compute_heading_accuracy(
                            pred["box"][-1], gts_in_token[bev_jmax]["box"][-1]
                        )
                        pred_match["match_result_bev"] = [
                            max_bev_overlap,
                            1,
                            heading_accuracy,
                            0,
                        ]
                    else:
                        pred_match["match_result_bev"] = [
                            max_bev_overlap,
                            0,
                            0,
                            1,
                        ]
            else:
                # max_overlap, TP, TPH, FP
                pred_match["match_result"] = [0, 0, 0, 1]
                pred_match["match_result_bev"] = [
                    0,
                    0,
                    0,
                    1,
                ]
            preds_match.append(pred_match)
        return preds_match


class BaseEvalModule(ABC):
    """Workflow template."""

    def __init__(
        self,
        range_limit: list,
        af_work_dir: str,
        result_name: str = None,
        filter_by_x: bool = False,
        filter_by_y: bool = False,
    ):
        if not isinstance(range_limit[0], list):
            self._range = [range_limit]
        else:
            self._range = range_limit
        assert not (
            filter_by_x and filter_by_y
        ), "filter_by_x and filter_by_y could not be True together"
        self._filter_by_x = filter_by_x
        self._filter_by_y = filter_by_y
        self.af_work_dir = af_work_dir
        self.result_name = result_name

    def _filter_boxes(self, boxes: list, range_limit: list):
        """Filter list of gts or preds based on range.

        boxes: list of boxes from either gts or preds
        """
        r1, r2 = range_limit
        if self._filter_by_x:
            range_filter = filter(lambda p: r1 <= p["box"][0] < r2, boxes)
        elif self._filter_by_y:
            range_filter = filter(lambda p: r1 <= p["box"][1] < r2, boxes)
        else:
            range_filter = filter(
                lambda p: r1
                <= math.sqrt(p["box"][0] ** 2 + p["box"][1] ** 2)
                < r2,
                boxes,
            )
        return list(range_filter)

    def filter_by_range_gts(self, gts: list, range_limit: list):
        gts_new = {}
        for class_name in gts:
            gts_new[class_name] = {}
            gts_new[class_name]["gts"] = self._filter_boxes(
                gts[class_name]["gts"], range_limit
            )
            gts_new[class_name]["fn_gts"] = self._filter_boxes(
                gts[class_name]["fn_gts"], range_limit
            )
            gts_new[class_name]["fn_bev_gts"] = self._filter_boxes(
                gts[class_name]["fn_bev_gts"], range_limit
            )
        return gts_new

    def filter_by_range_predictions(
        self, predictions: list, range_limit: list
    ):
        predictions_new = {}
        for class_name in predictions:
            predictions_new[class_name] = self._filter_boxes(
                predictions[class_name], range_limit
            )
        return predictions_new

    def filter_by_range(self, gts, predictions: list, range_limit: list):
        self.filter_by_range_gts(gts, range_limit)
        self.filter_by_range_predictions(predictions, range_limit)
        return self.filter_by_range_gts(
            gts, range_limit
        ), self.filter_by_range_predictions(predictions, range_limit)

    def filter_by_range_generator_gts(self, gts: list, ranges: list):
        for range_limit in ranges:
            yield self.filter_by_range_gts(gts, range_limit)

    def filter_by_range_generator_predictions(
        self, predictions: list, ranges: list
    ):
        for range_limit in ranges:
            yield self.filter_by_range_predictions(predictions, range_limit)

    def run_all_ranges(self, gts: Dict, predictions: Dict, **kwargs):
        """Start method for each workflow.

        If range_limit is a single range, filter gts and preds and run method;
        If range_limit is multiple range intervals, run method concurrently on
            all range intervals.
        """
        total_frame_num = 0
        self.class_names = list(gts.keys())
        for class_name in gts:
            total_frame_num += len(gts[class_name]["gts"])
        self.total_frame_num = total_frame_num

        if len(self._range) == 1:
            gts_filtered, predictions_filtered = self.filter_by_range(
                gts, predictions, self._range[0]
            )
            return self.run_within_range(
                gts_filtered, predictions_filtered, self._range[0], **kwargs
            )
        else:
            with ProcessPoolExecutor(max_workers=4) as executor:
                evaluation = partial(self.run_within_range, **kwargs)
                results = executor.map(
                    evaluation,
                    self.filter_by_range_generator_gts(gts, self._range),
                    self.filter_by_range_generator_predictions(
                        predictions, self._range
                    ),
                    self._range,
                )

            return results

    def run(self, gts: Dict, predictions: Dict, **kwargs):
        results = self.run_all_ranges(gts, predictions, **kwargs)
        return self.summarize(results, **kwargs)

    def summarize(self, results: Dict, **kwargs):
        return results

    @abstractmethod
    def run_within_range(
        self, gts: Dict, predictions: Dict, range_limit: List, **kwargs
    ):
        pass


@OBJECT_REGISTRY.register_module
class DetEvaluation(BaseEvalModule):
    """Do basic AP/APH evaluation across different range.

    intervals defined by 'range_interval'.
    Please note that the first interval must be the full range.
    """

    def __init__(
        self,
        range_limit,
        af_work_dir,
        filter_by_x=False,
        filter_by_y=False,
        bev_metric=False,
        vis_pr_curve=False,
        failure_case=False,
        result_name="aph_result",
        ignore_abnormal=False,
    ):
        """Initialize the workflow.

        Args:
            range_limit: Range limit for evaluation.
            af_work_dir: Working directory for evaluation.
            filter_by_x: Filter by x if True.
            filter_by_y: Filter by y if True.
            bev_metric: Use BEV metric if True.
            vis_pr_curve: Draw PR curves in each range interval if True.
            failure_case: Dump failure cases in the full range if True.
            ignore_abnormal: Ignore abnormal recall if True.
        """
        super(DetEvaluation, self).__init__(
            range_limit, af_work_dir, result_name, filter_by_x, filter_by_y
        )
        self.full_range = self._range[0]
        self.bev_metric = bev_metric
        self.vis_pr_curve = vis_pr_curve
        self.dump_failure_case = failure_case
        self.result_name = result_name
        self.class_names = []
        if self.vis_pr_curve:
            self.pr_curve_dirname = osp.join(self.af_work_dir, "pr_curves")
            os.makedirs(self.pr_curve_dirname, exist_ok=True)
        if self.dump_failure_case:
            self.dump_dir = osp.join(self.af_work_dir, "failure_case")
            os.makedirs(self.dump_dir, exist_ok=True)
        self.ignore_abnormal = ignore_abnormal

    def result_to_dict(self, aps, aphs, fp_metrics, limit_range):
        key = "{}m~{}m".format(int(limit_range[0]), int(limit_range[1]))
        result_dict = {}
        html_output = {}
        result_dict["All AP/APH"] = [np.mean(aps), np.mean(aphs)]
        html_output["All AP/APH"] = "{}/{}".format(
            round(np.mean(aps) * 100, 2), round(np.mean(aphs) * 100, 2)
        )
        for class_id, class_name in enumerate(self.class_names):
            class_key = f"{class_name} AP/APH"
            result_dict[class_key] = [aps[class_id], aphs[class_id]]
            html_output[class_key] = "{}/{}".format(
                round(aps[class_id] * 100, 2), round(aphs[class_id] * 100, 2)
            )
        pt = PrettyTable()
        title = ["metric"] + self.class_names + ["ALL"]
        pt.field_names = title
        for k in title:
            pt.float_format[k] = ".2"
        for k in sorted(fp_metrics.keys()):
            pt.add_row(
                [k]
                + [
                    fp_metrics[k][class_id]
                    for class_id in range(len(self.class_names) + 1)
                ]
            )
        result_dict["fp_metrics"] = pt.get_string()
        return key, result_dict, html_output

    def run_within_range(self, gts, predictions, range_limit):
        """Do evaluation within a certain range.

        range_limit: [r1, r2] like [0, 20] or [20, 50]..., predefined distance
            intervals, for detailed range specific evaluation.
        """
        if self.vis_pr_curve:
            pr_curve_name = osp.join(
                self.pr_curve_dirname,
                "pr_curve_{}_{}.jpg".format(
                    int(range_limit[0]), int(range_limit[1])
                ),
            )
        else:
            pr_curve_name = None
        # only dump failure cases in full range
        if self.dump_failure_case and range_limit == self.full_range:
            dump_dir = self.dump_dir
        else:
            dump_dir = None
        aps, aphs, fp_metrics = get_average_precisions(
            gts,
            predictions,
            range_limit,
            total_frame_num=self.total_frame_num,
            pr_curve_name=pr_curve_name,
            dump_dir=dump_dir,
            bev_metric=self.bev_metric,
            ignore_abnormal_recall=self.ignore_abnormal,
        )
        key, result_dict, html_output = self.result_to_dict(
            aps, aphs, fp_metrics, range_limit
        )
        return key, result_dict, html_output

    def summarize(self, results):
        """Override base method.

        Returns:
        aph_results: AP/APH table for HTML.
        result_dicts: regular AP/APH result with fp_metrics.
        """
        html_outputs = {}
        result_dicts = {}
        for key, result_dict, aph_result in results:
            result_dicts[key] = result_dict
            html_outputs[key] = aph_result
        html_outputs = pd.DataFrame(html_outputs)
        html_outputs = html_outputs.rename(columns={"0m~1000m": "All range"})
        return html_outputs, result_dicts


@OBJECT_REGISTRY.register_module
class VisFPNCurve(BaseEvalModule):
    """Draw FPR and FNR curve within the given range on each merged class.

    For this example, all 7 classes will be merged into 3 classes which are
    'Vehicle', 'Cyclist' and 'Pedestrian', thus 3 graphs will be generated.
    """

    def __init__(
        self,
        range_limit,
        af_work_dir,
        class_map=None,
        score_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ):
        """Intialize the workflow.

        class_map: Dict that map original class to new class,
            used for class merging.
        score_thresholds: List of score threholds values used to draw FPR and
            FNR curves.
        """
        super(VisFPNCurve, self).__init__(range_limit, af_work_dir)
        self.class_map = class_map
        self.score_thresholds = score_thresholds
        self.fpn_curve_dirname = osp.join(self.af_work_dir, "fpn_curves")
        os.makedirs(self.fpn_curve_dirname, exist_ok=True)
        self.new_class_names = list(set(class_map.values()))
        self.fpn_curve_names = [
            osp.join(
                self.fpn_curve_dirname,
                "fpn_curve_{}_{}_{}.jpg".format(
                    int(self._range[0][0]), int(self._range[0][1]), class_name
                ),
            )
            for class_name in self.new_class_names
        ]

    def draw_roc_curve(
        self, fpr_by_scores, fnr_by_scores, roc_curve_name, title
    ):
        plt.figure(figsize=(7, 8))
        lines = []
        labels = []
        (l1,) = plt.plot(
            self.score_thresholds[: len(fpr_by_scores)],
            fpr_by_scores,
            "navy",
            lw=2,
        )
        (l2,) = plt.plot(
            self.score_thresholds[: len(fnr_by_scores)],
            fnr_by_scores,
            "turquoise",
            lw=2,
        )
        lines = [l1, l2]
        labels = ["FPR", "FNR"]

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.15)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Score Thresholds")
        plt.ylabel("Rate")
        plt.title(title)
        plt.legend(lines, labels, loc="best", prop={"size": 8})
        plt.savefig(roc_curve_name)

    def run_within_range(self, gts, predictions, range_limit):
        # remap classes
        gts_new = {}
        predictions_new = {}
        for class_name in self.new_class_names:
            gts_new[class_name] = {"gts": [], "fn_gts": [], "fn_bev_gts": []}
            predictions_new[class_name] = []
        for class_name in predictions:
            gts_new[self.class_map[class_name]]["gts"].extend(
                gts[class_name]["gts"]
            )
            gts_new[self.class_map[class_name]]["fn_gts"].extend(
                gts[class_name]["fn_gts"]
            )
            gts_new[self.class_map[class_name]]["fn_bev_gts"].extend(
                gts[class_name]["fn_bev_gts"]
            )
            predictions_new[self.class_map[class_name]].extend(
                predictions[class_name]
            )
        gts = gts_new
        predictions = predictions_new

        for class_id, class_name in enumerate(predictions):
            preds_class = predictions[class_name]
            gts_class = gts[class_name]

            fpr_by_scores = []
            fnr_by_scores = []
            for score_thres in self.score_thresholds:
                preds_class = [
                    p for p in preds_class if p["score"] > score_thres
                ]
                if not preds_class:
                    continue

                tp = np.array([p["match_result"][1] for p in preds_class])
                fp = np.array([p["match_result"][3] for p in preds_class])
                fp = np.cumsum(fp, axis=0)
                tp = np.cumsum(tp, axis=0)

                fp_rate = fp[-1] / len(preds_class)
                tp_rate = tp[-1] / len(gts_class["gts"])
                fn_rate = 1.0 - tp_rate
                fpr_by_scores.append(fp_rate)
                fnr_by_scores.append(fn_rate)
            near = range_limit[0]
            far = range_limit[1]
            title = f"FP/FN curve for {near}~{far}m {class_name}"
            self.draw_roc_curve(
                fpr_by_scores,
                fnr_by_scores,
                self.fpn_curve_names[class_id],
                title=title,
            )


@OBJECT_REGISTRY.register_module
class VisFPNDist(BaseEvalModule):
    """Get FP/FN numbers and percentage for each given range intervals.

    It will also draw FP/FN distribution map on all frames.
    """

    def __init__(
        self,
        range_limit,
        af_work_dir,
        pc_range,
        voxel_size=(0.15, 0.15),
        fp_threshold=0.4,
        normalize_denominator=None,
        draw_fp_dist=True,
        draw_fn_dist=True,
        result_name="fpn_result",
    ):
        """Initialize workflow.

        pc_range: distribution map visualization range
        voxel_size: distribution map voxel size
        fp_threshold: distribution map FP threshold
        normalize_denominator: fix normalize factor for comparision
        draw_fp_dist: draw FP distribution map if True
        draw_fn_dist: draw FN distribution map if True
        """
        super(VisFPNDist, self).__init__(
            range_limit,
            af_work_dir,
            result_name,
        )
        self.pc_range = np.array(pc_range)
        self.voxel_size = np.array(voxel_size)
        self.fp_threshold = fp_threshold
        w = (pc_range[2] - pc_range[0]) / voxel_size[0]
        h = (pc_range[3] - pc_range[1]) / voxel_size[1]
        self.bev_shape = np.array([h, w, 3], dtype=(np.int32))
        fpn_dist_dirname = osp.join(self.af_work_dir, "fpn_dist")
        os.makedirs(fpn_dist_dirname, exist_ok=True)
        self.draw_fp_dist = draw_fp_dist
        self.draw_fn_dist = draw_fn_dist
        self.normalize_denominator = normalize_denominator
        if self.draw_fp_dist:
            self.fp_dist_name = osp.join(fpn_dist_dirname, "fp_dist.jpg")
        if self.draw_fn_dist:
            self.fn_dist_name = osp.join(fpn_dist_dirname, "fn_dist.jpg")

    def _point2pixel(self, point):
        """Convert coords in bev into coords on image."""
        x1, y1, x2, y2 = self.pc_range
        h, w, _ = self.bev_shape
        x, y = point
        x = int(w * (x - x1) / (x2 - x1))
        y = int(h * (y - y1) / (y2 - y1))
        return (x, y)

    def _length2pixel(self, length):
        """Convert length in bev into length on image."""
        x1, _, x2, _ = self.pc_range
        _, w, _ = self.bev_shape
        pix = int(w * length / (x2 - x1))
        return pix

    def _draw_axis(self, bev_map):
        x1, y1, x2, y2 = self.pc_range
        cv2.line(
            bev_map,
            self._point2pixel((0, y1)),
            self._point2pixel((0, y2)),
            (0, 0, 0),
        )
        cv2.line(
            bev_map,
            self._point2pixel((x1, 0)),
            self._point2pixel((x2, 0)),
            (0, 0, 0),
        )
        return bev_map

    def _draw_dist_circle(self, bev_map):
        for range_limit in self._range:
            r = range_limit[1]
            cv2.circle(
                bev_map,
                self._point2pixel((0, 0)),
                self._length2pixel(r),
                (0, 200, 0),
            )
            cv2.putText(
                bev_map,
                f"{r}m",
                self._point2pixel((r + 1, -1)),
                0,
                0.7,
                color=(0, 200, 0),
            )
        return bev_map

    def _draw_percentage_result(self, bev_map, percentage_result):
        for range_interval, result in zip(self._range, percentage_result):
            # generate percentage text coords for current range interval if the
            # coords could lay inside the map
            r1, r2 = range_interval
            r2 = min(r2, self.pc_range[2])
            if r1 + 10 <= r2:
                pixel_coord = self._point2pixel(((r1 + r2) / 2, 2))
                pixel_coord = (
                    pixel_coord[1],
                    self.bev_shape[1] - pixel_coord[0],
                )
                result = round(result * 100, 1)
                cv2.putText(
                    bev_map,
                    f"{result}%",
                    pixel_coord,
                    0,
                    0.9,
                    color=(0, 0, 0),
                )
        return bev_map

    def draw_bev_dist(self, points, path, percentage_result):
        """Compose distribution map."""
        dist_count = np.zeros(shape=self.bev_shape[:2], dtype=np.int32)
        for point in points:
            x, y = self._point2pixel(point)
            dist_count[y, x] += 1
        # normalize
        if self.normalize_denominator is None:
            dist_count = dist_count / dist_count.max() * 255
        else:
            dist_count = dist_count / self.normalize_denominator * 255

        # initialize bev map
        bev_map = np.ones(shape=self.bev_shape, dtype=np.int16) * 255
        # draw dist heatmap
        for y in range(self.bev_shape[0]):
            for x in range(self.bev_shape[1]):
                if dist_count[y, x] > 0:
                    bev_map = draw_block(
                        bev_map,
                        (x, y),
                        2,
                        k=dist_count[y, x],
                    )
        bev_map = bev_map.clip(0, 255).astype(np.uint8)
        # draw x, y axis
        bev_map = self._draw_axis(bev_map)
        # draw distance circle
        bev_map = self._draw_dist_circle(bev_map)
        # rotate bev map 90 degree
        bev_map = cv2.transpose(bev_map)
        bev_map = cv2.flip(bev_map, 0)
        # add percentage result
        bev_map = self._draw_percentage_result(bev_map, percentage_result)
        # write image
        cv2.imwrite(path, bev_map)

    def run_within_range(self, gts, predictions, range_limit):
        # gather fp and fn
        fp_num = 0
        fn_num = 0
        fp_points = []
        fn_points = []
        for class_name in gts:
            preds = predictions[class_name]
            fp_filter = filter(
                lambda p: p["match_result"][-1] == 1
                and p["score"] > self.fp_threshold,
                preds,
            )
            fps = list(fp_filter)
            fp_num += len(fps)
            if self.draw_fp_dist:
                fp_points.extend([(p["box"][0], p["box"][1]) for p in fps])
            fns = gts[class_name]["fn_gts"]
            fn_num += len(fns)
            if self.draw_fn_dist:
                fn_points.extend([(g["box"][0], g["box"][1]) for g in fns])
        result = {"fp_num": fp_num, "fn_num": fn_num}
        key = f"{range_limit[0]}m~{range_limit[1]}m"
        return key, result, fp_points, fn_points

    def summarize(self, results):
        """Generate FP/FN tables and draw distribution maps.

        Returns:
        num_result: FP/FN numbers table for HTML
        percenrage_result: FP/FN percentage table for HTML
        """
        result = {}
        fp_points_all = []
        fn_points_all = []
        for key, result_dict, fp_points, fn_points in results:
            result[key] = result_dict
            fp_points_all.extend(fp_points)
            fn_points_all.extend(fn_points)
        # convert pandas
        num_result = pd.DataFrame(result)
        num_result["all_range"] = num_result.sum(axis=1)
        percentage_result = num_result.div(num_result["all_range"], axis=0)
        percentage_result = percentage_result.rename(
            index={"fp_num": "fp%", "fn_num": "fn%"}
        )
        # draw fp/fn dist
        if self.draw_fp_dist:
            self.draw_bev_dist(
                fp_points_all, self.fp_dist_name, percentage_result.loc["fp%"]
            )
        if self.draw_fn_dist:
            self.draw_bev_dist(
                fn_points_all, self.fn_dist_name, percentage_result.loc["fn%"]
            )
        return num_result, percentage_result
