# Copyright (c) Horizon Robotics. All rights reserved.

import glob
import json
import logging
import math
import os
import pickle
from collections import defaultdict
from typing import List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

try:
    from aidisdk.experiment import Image, Line, Table
except ImportError:
    Image = None
    Line = None
    Table = None

from matplotlib import pyplot as plt

from hat.metrics.detection2d.utils import calap
from hat.metrics.metric import EvalMetric
from hat.metrics.metric_3dv_utils import (
    ct_rot_matching,
    rotate_iou,
    rotate_iou_matching,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.aidi import EvalResult
from hat.utils.distributed import get_dist_info

__all__ = ["ANCBEVDiscreteObjectEval"]

logger = logging.getLogger(__name__)


def ct_matching(
    det_locs: np.ndarray,
    gt_locs: np.ndarray,
    det_scores: np.ndarray,
    ct_match_mode: Optional[str] = "percentage",
    gt_ignore_mask: Optional[Sequence[int]] = None,
    thresholds: Optional[Union[dict, float]] = None,
) -> Mapping:
    """Match GT and pred boxes by center location on vcs.

    The hyper-parameters refer to https://jira.hobot.cc:8443/browse/MSD-6559.
    Args:
        det_locs: the predict location, shape:[N, 2].
        gt_locs: the gt location, shape: [K,3].
        det_scores: the pred score, shape: [N,].
        ct_match_mode: match mode, default "percentage", or
            "distance": according to center location
        gt_ignore_mask: the gt igonre list.
        thresholds: the mapping of gt distance(meter) and matching
            threshold(percent of gt distance).
    Returns:
        Mapping: results based on location center matching.
    """
    det_ct, gt_ct = det_locs, gt_locs
    overlaps = np.zeros(shape=gt_ct.shape[0])
    det_assign = np.zeros(shape=gt_ct.shape[0], dtype=np.int) - 1
    matched_det = np.zeros(shape=det_ct.shape[0], dtype=np.int)

    valid_gt = np.ones(gt_ct.shape[0], dtype=bool)
    det_sorted_idx = np.argsort(det_scores)[::-1]
    det_ignored_mask = np.zeros(shape=det_ct.shape[0], dtype=bool)

    if ct_match_mode != "distance":
        assert isinstance(thresholds, dict)
        thresholds = sorted(thresholds.items(), key=lambda x: x[0])

    for dt_idx in det_sorted_idx:
        if not np.any(valid_gt):
            # all the gt instances have been matched.
            break
        dxy = (
            np.sum((det_ct[dt_idx, [0, 1]] - gt_ct[:, [0, 1]]) ** 2, axis=1)
            ** 0.5
        )
        gt_dist = np.sum(gt_ct[:, [0, 1]] ** 2, axis=1) ** 0.5

        dxy[valid_gt == 0] = np.inf
        min_ind = np.argmin(dxy)
        min_dxy = dxy[min_ind]
        if ct_match_mode == "percentage":
            for critical_dist, thresh_percent in thresholds:
                if gt_dist[min_ind] < critical_dist:
                    dxy_thresh = gt_dist[min_ind] * thresh_percent
                    break
                dxy_thresh = gt_dist[min_ind] * thresh_percent
        elif ct_match_mode == "distance":
            dxy_thresh = thresholds
        else:
            raise NotImplementedError(
                f"please confirm {ct_match_mode} is percentage or distance"
            )

        if min_dxy < dxy_thresh:
            if gt_ignore_mask and gt_ignore_mask[min_ind]:
                det_ignored_mask[dt_idx] = True
                matched_det[dt_idx] = -1
            else:
                det_assign[min_ind] = dt_idx
                matched_det[dt_idx] = 1
                overlaps[min_ind] = min_dxy
                valid_gt[min_ind] = 0
    redundant = np.where(matched_det == 0)[0]
    return (
        {"overlaps": overlaps, "det_assign": det_assign},
        redundant,
        det_ignored_mask,
    )


def voc_ap(res, score_threshold=0, save_pr=False, save_dir=None):
    if len(res["tp"]) == 0:
        return 0
    scores_matched = np.array(res["tp"])[:, 0]
    redundant = np.array(res["fp"])[:, 0]

    redundant = sorted(redundant, reverse=True)
    scores_matched = sorted(scores_matched, reverse=True)
    gts_total = len(res["fn"]) + len(scores_matched)

    recalls = np.array([])
    precisions = np.array([])
    tp, fp = 0, 0
    Num_matched = len(scores_matched)
    Num_redundant = len(redundant)
    for score in np.linspace(1, score_threshold, 100):
        while tp < Num_matched:
            if scores_matched[tp] >= score:
                tp += 1
            else:
                break
        while fp < Num_redundant:
            if redundant[fp] >= score:
                fp += 1
            else:
                break
        precision = tp / (tp + fp + 1e-6)
        recalls = np.append(recalls, tp)
        precisions = np.append(precisions, precision)
    recalls /= gts_total

    if save_dir and save_pr:
        plt.plot(recalls, precisions)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(save_dir, "all_pr.png"))
        np.save(os.path.join(save_dir, "precision.npy"), precisions)
        np.save(os.path.join(save_dir, "recall.npy"), recalls)
        np.save(os.path.join(save_dir, "scores_matched.npy"), scores_matched)
        np.save(os.path.join(save_dir, "redundant.npy"), redundant)
        np.save(os.path.join(save_dir, "gt_miss.npy"), res["fn"])

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def summarize_coco_ap(res: dict, frames_num: int):
    """Summarize results with coco method.

    Args:
        res: Metric info.
        example:
                dict(
                    'tp': np.array(shape=(tp_ins_num, 9)),
                    'fp': np.array(shape=(fp_ins_num, 3)),
                    'fn': np.array(shape=(fn_ins_num, 2)),
                ).
            The dim 1 elements of the tp value includes of
                (det_score, dx, dy, dxy, dw, dh, drot, gt_x, gt_y).
            The dim 1 elements of the fp value includes of
                (det_score, det_x, det_y).
            The dim 1 elements of the fn value includes of
                (gt_x, gt_y).
        frames_num: Length of val dataset.
    """
    res_all = defaultdict(list)
    gt_count = 0
    gt_count += len(res["fn"])
    for i in range(len(res["fp"])):
        res_all["tp"].append(0)
        res_all["fp"].append(1)
        res_all["conf"].append(res["fp"][i][0])
    for i in range(len(res["tp"])):
        res_all["tp"].append(1)
        res_all["fp"].append(0)
        res_all["conf"].append(res["tp"][i][0])
        gt_count += 1

    tp = np.array(res_all["tp"])
    fp = np.array(res_all["fp"])
    conf = np.array(res_all["conf"])

    argsort = np.argsort(-conf)
    conf = conf[argsort]
    tp = tp[argsort]
    fp = fp[argsort]

    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)

    if gt_count == 0:
        recalls = np.zeros(tp.shape)
    else:
        recalls = tp / (gt_count + 1e-6)
    assert np.all(0 <= recalls) & np.all(recalls <= 1)

    precisions = tp / (tp + fp + 1e-6)
    assert np.all(0 <= precisions) & np.all(precisions <= 1)
    ap, recall, precision = calap(recalls, precisions)
    recall = np.array(recall)
    precision = np.array(precision)

    fppi = fp / float(frames_num)
    fppi += 1e-6
    detection_rate = (tp - fp) / (gt_count + 1e-6)

    results = {}
    results["coco_ap"] = ap
    results["recalls"] = np.around(recall.astype(np.float64), 3).tolist()
    results["precisions"] = np.around(precision.astype(np.float64), 3).tolist()
    results["fppi"] = np.around(fppi.astype(np.float64), 3).tolist()
    results["conf"] = np.around(conf.astype(np.float64), 3).tolist()
    results["detection_rate"] = np.around(
        detection_rate.astype(np.float64), 3
    ).tolist()
    results["fp"] = np.around(fp.astype(np.float64), 3).tolist()
    results["tp"] = np.around(tp.astype(np.float64), 3).tolist()

    results["target_recalls"] = []
    target_recalls = np.linspace(0.5, 0.95, 9)
    for target_recall in target_recalls:
        idx = max(min((recall < target_recall).sum(), len(recall) - 1), 0)
        results["target_recalls"].append(
            {
                "threshold": results["conf"][idx] if len(recall) > 0 else 0,
                "recall": results["recalls"][idx] if len(recall) > 0 else 0,
                "precision": results["precisions"][idx]
                if len(recall) > 0
                else 0,
                "num_tp": results["tp"][idx] if len(recall) > 0 else 0,
                "num_fp": results["fp"][idx] if len(recall) > 0 else 0,
                "num_gt": float(gt_count) if len(recall) > 0 else 0,
            }
        )

    results["target_precisions"] = []
    target_precisions = np.linspace(0.5, 0.95, 9)
    for target_precision in target_precisions:
        idx = max((precision >= target_precision).sum() - 1, 0)
        results["target_precisions"].append(
            {
                "threshold": results["conf"][idx] if len(precision) > 0 else 0,
                "precision": results["precisions"][idx]
                if len(precision) > 0
                else 0,
                "recall": results["recalls"][idx] if len(precision) > 0 else 0,
                "num_tp": results["tp"][idx] if len(precision) > 0 else 0,
                "num_fp": results["fp"][idx] if len(precision) > 0 else 0,
                "num_gt": float(gt_count) if len(precision) > 0 else 0,
            }
        )
    return results


def is_in_range(loc, eval_vcs_range=None):
    if eval_vcs_range is None:
        return 1
    x, y = loc
    if (
        x < eval_vcs_range[0]
        or x > eval_vcs_range[2]
        or y < eval_vcs_range[1]
        or y > eval_vcs_range[3]
    ):
        return 0
    return 1


def eval_metric(
    det_res: Mapping,
    annotation: Mapping,
    iou_threshold: float,
    gt_max_depth: float,
    yaw_amplitude: int,
    match_mode: str = None,
    ct_match_mode: Optional[str] = "percentage",
    eval_vcs_range: Optional[Sequence[float]] = None,
    ct_matching_thresholds: Optional[dict] = None,
    compute_foreground_prec: bool = False,
) -> Mapping:
    """Eval the metric between GT and pred boxes.

    Based on bev3d_bbox_eval(). For more details please refer to
    (hat/metrics/metric_3dv_utils.py)

    Args:
        det_res: The predict discrete obj boxes info.
        annotation: The ground truth discrete obj boxes info.
        iou_threshold: Threshold for IoU.
        gt_max_depth: Max depth for gts.
        yaw_amplitude: The amplitude of yaw, the
            yaw range of (stopline,crosswalk,...) is (-90,90), so the amplitude
            of yaw is 180, arrow is 360.
        match_mode: match mode, default is iou, or
            "ct_rot": acrroding to center location and rotation angle.
        ct_match_mode: when match mode is "ct", ct_match_mode have two:
            "percentage": according percent of gt distance
            "distance": according to center location
        eval_vcs_range: The vcs range of you care.
        ct_matching_thresholds: the mapping of gt distance(meter)
            and matching threshold(percent of gt distance).
            Only use when match_mode == "ct".
        compute_foreground_prec: Whether to compute classification
            precision, if True, fp predictions and all gts will be returned.

    Returns:
        Dict contains the results.
    """

    all_dets = defaultdict(list)
    all_gts = defaultdict(list)
    for det in det_res:
        if not is_in_range(det["pred_loc"], eval_vcs_range):
            continue
        all_dets[det["timestamp"]].append(det)

    for gt in annotation["annotations"]:
        gt_depth = abs(gt["vcs_discobj_loc"][0])  # vcs: abs(x) = depth
        if gt_depth > gt_max_depth:
            continue
        if not is_in_range(gt["vcs_discobj_loc"], eval_vcs_range):
            gt["vcs_discobj_ignore"] = 1
        all_gts[gt["timestamp"]].append(gt)  # 需过滤eval_vcs_range

    gt_missed = []
    det_redundant = []
    matched = []
    frame_results = defaultdict(lambda: defaultdict())

    if compute_foreground_prec:
        all_valid_gt = {}
        miss_det_fp = {}

    for timestamp in annotation["timestamps"]:
        single_frame_gt_missed = []
        single_frame_det_redundant = []
        single_frame_matched = []
        gts = all_gts[timestamp]
        dets = all_dets[timestamp]
        det_bbox3d, det_scores, det_locs = [], [], []
        gt_bbox3d, gt_locs, gt_ignores = [], [], []

        if compute_foreground_prec:
            all_valid_gt[timestamp] = []
            miss_det_fp[timestamp] = []

        if len(gts) == 0:
            for det in dets:
                single_frame_det_redundant.append(
                    [det["pred_score"].flatten()[0]] + det["pred_loc"].tolist()
                )
                if compute_foreground_prec:
                    dim = det["pred_wh"]
                    yaw = det["pred_yaw"]
                    loc = det["pred_loc"]
                    score = det["pred_score"]
                    # the first value is a flag which defined 0
                    # for pred for differentiable with gt
                    miss_det_fp[timestamp].extend(
                        [[0, score, loc[0], loc[1], dim[0], dim[1], -yaw]]
                    )
            # update output results
            frame_results[timestamp] = {
                "det_redundant": single_frame_det_redundant,
                "gt_missed": single_frame_gt_missed,
                "single_frame_matched": single_frame_matched,
            }
            det_redundant.extend(single_frame_det_redundant)
            gt_missed.extend(single_frame_gt_missed)
            matched.extend(single_frame_matched)
            continue

        for gt in gts:
            dim = gt["vcs_discobj_wh"]
            yaw = gt["vcs_discobj_yaw"]
            loc = gt["vcs_discobj_loc"]
            # [x, y, w, h, -yaw], -yaw means change the yaw from \
            # counterclockwise -> clockwise
            bbox3d = [loc[0], loc[1], dim[0], dim[1], -yaw]
            gt_bbox3d.append(bbox3d)
            gt_locs.append(loc)
            gt_ignores.append(int(gt["vcs_discobj_ignore"]))

            if compute_foreground_prec and int(gt["vcs_discobj_ignore"]) == 0:
                # the first value is a flag which defined 1
                # for gt for differentiable with pred
                all_valid_gt[timestamp].extend([[1, 1.0] + bbox3d])

        if len(dets) == 0:
            for gt, gt_ignore in zip(gts, gt_ignores):
                if gt_ignore:
                    continue
                loc = gt["vcs_discobj_loc"]
                single_frame_gt_missed.append(loc.tolist())
            # update output results
            frame_results[timestamp] = {
                "det_redundant": single_frame_det_redundant,
                "gt_missed": single_frame_gt_missed,
                "single_frame_matched": single_frame_matched,
            }
            det_redundant.extend(single_frame_det_redundant)
            gt_missed.extend(single_frame_gt_missed)
            matched.extend(single_frame_matched)
            continue

        for det in dets:
            dim = det["pred_wh"]
            yaw = det["pred_yaw"]
            loc = det["pred_loc"]
            bbox3d = [loc[0], loc[1], dim[0], dim[1], -yaw]
            det_bbox3d.append(bbox3d)
            det_scores.append(det["pred_score"])
            det_locs.append(det["pred_loc"])

        det_bbox3d = np.array(det_bbox3d)
        det_scores = np.array(det_scores)
        gt_bbox3d = np.array(gt_bbox3d)
        det_locs = np.array(det_locs)
        gt_locs = np.array(gt_locs)

        assert det_bbox3d.shape[0] == det_scores.shape[0]

        if match_mode == "ct_rot":
            (matched_dict, redundant, _) = ct_rot_matching(
                det_bbox3d,
                det_locs,
                gt_bbox3d,
                gt_locs,
                det_scores,
                gt_ignores,
            )
        elif match_mode == "ct":
            (matched_dict, redundant, _) = ct_matching(
                det_locs,
                gt_locs,
                det_scores,
                ct_match_mode,
                gt_ignores,
                ct_matching_thresholds,
            )
        else:
            (matched_dict, redundant, _) = rotate_iou_matching(
                det_bbox3d,
                det_locs,
                gt_bbox3d,
                gt_locs,
                det_scores,
                iou_threshold,
                gt_ignores,
            )

        single_frame_det_redundant.extend(
            [[det_scores[i]] + det_locs[i].tolist() for i in redundant]
        )

        if compute_foreground_prec:
            for i in redundant:
                det = dets[i]
                dim = det["pred_wh"]
                yaw = det["pred_yaw"]
                loc = det["pred_loc"]
                score = det["pred_score"]
                miss_det_fp[timestamp].extend(
                    [[0, score, loc[0], loc[1], dim[0], dim[1], -yaw]]
                )

        det_assigns = matched_dict["det_assign"]
        inds = np.array(list(range(len(det_assigns))))
        gt_miss_idx = det_assigns == -1
        gt_ignores = np.array(gt_ignores)
        gt_miss_idx[gt_ignores == 1] = False
        single_frame_gt_missed.extend(
            [gt_locs[i].tolist() for i in range(len(inds)) if gt_miss_idx[i]]
        )

        mask = det_assigns != -1
        det_assigns, inds = det_assigns[mask], inds[mask]

        if len(det_assigns) == 0:
            # update output results
            frame_results[timestamp] = {
                "det_redundant": single_frame_det_redundant,
                "gt_missed": single_frame_gt_missed,
                "single_frame_matched": single_frame_matched,
            }
            det_redundant.extend(single_frame_det_redundant)
            gt_missed.extend(single_frame_gt_missed)
            matched.extend(single_frame_matched)

            continue

        pred_score = np.array([dets[i]["pred_score"] for i in det_assigns])
        pred_dim = np.array([dets[i]["pred_wh"] for i in det_assigns])
        pred_loc = np.array([dets[i]["pred_loc"] for i in det_assigns])
        pred_yaw_rad = np.array([dets[i]["pred_yaw"] for i in det_assigns])
        pred_yaw = np.rad2deg(pred_yaw_rad)

        gt_dim = np.array([gts[i]["vcs_discobj_wh"] for i in inds])
        gt_loc = np.array([gts[i]["vcs_discobj_loc"] for i in inds])
        gt_yaw_rad = np.array([gts[i]["vcs_discobj_yaw"] for i in inds])
        gt_yaw = np.rad2deg(gt_yaw_rad)

        dx = np.abs(pred_loc[:, 0] - gt_loc[:, 0])
        dy = np.abs(pred_loc[:, 1] - gt_loc[:, 1])
        dxy = (dx ** 2 + dy ** 2) ** 0.5
        dw = np.abs(pred_dim[:, 0] - gt_dim[:, 0])
        dh = np.abs(pred_dim[:, 1] - gt_dim[:, 1])
        abs_rot = np.abs(gt_yaw - pred_yaw) % yaw_amplitude
        drot = np.minimum(abs_rot, yaw_amplitude - abs_rot)

        single_frame_matched.extend(
            [
                [
                    pred_score[i],
                    dx[i],
                    dy[i],
                    dxy[i],
                    dw[i],
                    dh[i],
                    drot[i],
                ]
                + gt_loc[i].tolist()
                for i in range(len(dx))
            ]
        )
        # update output results
        frame_results[timestamp] = {
            "det_redundant": single_frame_det_redundant,
            "gt_missed": single_frame_gt_missed,
            "single_frame_matched": single_frame_matched,
        }
        det_redundant.extend(single_frame_det_redundant)
        gt_missed.extend(single_frame_gt_missed)
        matched.extend(single_frame_matched)

    Num_all = len(gt_missed) + len(det_redundant) + len(matched)
    result = np.zeros((Num_all, 9))
    i_num = 0
    for gt in gt_missed:
        result[i_num, :2] = gt
        i_num += 1
    for det in det_redundant:
        result[i_num, :3] = det
        i_num += 1
    for mat in matched:
        result[i_num] = mat
        i_num += 1

    if compute_foreground_prec:
        miss_result = {}
        for timestamp in miss_det_fp.keys():
            _miss_det_fp = np.array(miss_det_fp[timestamp])
            _all_valid_gt = np.array(all_valid_gt[timestamp])
            if len(_all_valid_gt) and len(_miss_det_fp):
                miss_result[timestamp] = np.concatenate(
                    [_miss_det_fp, _all_valid_gt], axis=0
                )
            elif len(_all_valid_gt):
                miss_result[timestamp] = _all_valid_gt
            else:
                miss_result[timestamp] = _miss_det_fp
    else:
        miss_result = None
    return result, miss_result, frame_results


def collect_result_by_dep(results, dep_thresh):
    results_dep = [{} for _ in range(len(dep_thresh) + 1)]
    for name, res in results.items():
        res = np.array(res)
        res_np = res[:, -2] if res.size > 1 else res[:, 0]
        if dep_thresh[0] < 0:
            dep_inds = np.sum(res_np[:, np.newaxis] > dep_thresh, axis=-1)
        else:
            dep_inds = np.sum(abs(res_np[:, np.newaxis]) > dep_thresh, axis=-1)
        for i in range(len(dep_thresh) + 1):
            results_dep[i][name] = res[dep_inds == i]

    return results_dep


def collect_result_by_dist(results: dict, dist_thresh: list) -> list:
    """Split results by distance.

    Args:
        results: metric results.
            example:
                dict(
                    'tp': np.array(shape=(tp_ins_num, 9)),
                    'fp': np.array(shape=(fp_ins_num, 3)),
                    'fn': np.array(shape=(fn_ins_num, 2)),
                ).
            The dim 1 elements of the tp value includes of
                (det_score, dx, dy, dxy, dw, dh, drot, gt_x, gt_y).
            The dim 1 elements of the fp value includes of
                (det_score, det_x, det_y).
            The dim 1 elements of the fn value includes of
                (gt_x, gt_y).
        dist_thresh: distance range to validation.
    Returns:
        squence of results(dict) in each distance range.

    """
    results_dist = [{} for _ in range(len(dist_thresh) + 1)]
    for name, res in results.items():
        res = np.array(res)
        # res[:, -2:] indicate vcs location of each instance.
        res_np = (
            np.linalg.norm(res[:, -2:], axis=1) if len(res.shape) > 1 else res
        )
        dist_inds = np.sum(abs(res_np[:, np.newaxis]) > dist_thresh, axis=-1)
        for i in range(len(dist_thresh) + 1):
            results_dist[i][name] = res[dist_inds == i]

    return results_dist


@OBJECT_REGISTRY.register
class ANCBEVDiscreteObjectEval(EvalMetric):
    """The BEV discrete object detection eval metrics.

    The BEV discrete object detection metric calculation is based on real3d
    eval metrics. For more details please refer to Real3DEval
    (hat/metrics/real3d.py).

    Args:
        name: Name of this metric instance for display
        eval_category_ids: The categories to be evaluation.
        score_threshold: Threshold for score.
        iou_threshold: Threshold for IoU.
        gt_max_depth: Max depth for gts.
        metrics: Tuple of eval metrics, using
            ("dx", "dy", "dxy", "dw", "dh", "drot") if not special.
        depth_intervals: Depth range to validation, using (20, 50, 70)
            if not special.
        save_dir: Path to save the predictions.
        save_eval_result_path: Path to save eval result.
        vis_image_dir: Save dir of visual images.
        upload_image2aidi: Whether upload visunal images to aidi experiment
            manager or not.
        match_mode: match mode, default is iou. stopline and speedbump
            use "ct_rot": acrroding to center location and rotation angle.
        ct_match_mode: when match mode is "ct", ct_match_mode have two:
            "percentage": according percent of gt distance
            "distance": according to center location
        yaw_amplitude: The amplitude of yaw, the yaw range of
            (stopline,crosswalk,...) is (-90,90), so the amplitude
            of yaw is 180, arrow is 360.
        eval_vcs_range: The vcs range of you care, default is Dict().
            Order is [-x, -y, x, y].
        ap_score_threshold: Threshold for score while calculate AP.
        dist_intervals: distance range to validation.
        ct_matching_thresholds: the mapping of gt distance(meter)
            and matching threshold(percent of gt distance). Only use
            when match_mode == "ct".
        compute_foreground_prec: Whether to get
            classification precision.
        result_prefix: Prefix of aidi eval result.
    """

    def __init__(
        self,
        id2label: dict,
        eval_category_ids: Optional[Sequence[int]],
        depth_intervals: dict,
        eval_vcs_range: dict,
        score_threshold: float = 0.1,
        iou_threshold: float = 0.2,
        gt_max_depth: int = 100,
        metrics: Optional[Sequence[str]] = (
            "dx",
            "dy",
            "dxy",
            "dw",
            "dh",
            "drot",
        ),
        save_dir: Optional[str] = None,
        save_eval_result_path: Optional[str] = None,
        vis_image_dir: Optional[str] = None,
        upload_image_2_aidi: bool = False,
        match_mode: Optional[dict] = None,
        ct_match_mode: Optional[str] = "percentage",
        yaw_amplitude: int = 180,
        ap_score_threshold: float = 0.0,
        dist_intervals: Optional[Sequence[float]] = None,
        ct_matching_thresholds: Optional[Sequence[dict]] = None,
        compute_foreground_prec: bool = False,
        name: str = "BEV_disc_obj",
        annos_key: str = "annos_bev_discrete_obj",
        result_prefix: str = "",
    ):
        self.name = name
        self.task_name = name.split("bev_")[-1]
        self.eval_category_ids = eval_category_ids
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.gt_max_depth = gt_max_depth
        self.metrics = metrics
        self.match_mode = match_mode
        self.ct_match_mode = ct_match_mode
        self.yaw_amplitude = yaw_amplitude
        self.id2label = id2label
        self.ap_score_threshold = ap_score_threshold
        self.ct_matching_thresholds = ct_matching_thresholds
        self.compute_foreground_prec = compute_foreground_prec
        self.annos_key = annos_key

        assert (
            id2label[0] in depth_intervals
        ), f"{id2label[0]} must in depth_intervals"
        assert (
            id2label[0] in eval_vcs_range
        ), f"{id2label[0]} must in eval_vcs_range"
        assert (
            self.ap_score_threshold <= self.score_threshold
        ), f"ap_score_threshold must be less than \
            or equal to score_threshold, but get \
            {self.ap_score_threshold} > {self.score_threshold}"

        self.eval_vcs_range = eval_vcs_range
        self.depth_intervals = depth_intervals
        self.dist_intervals = dist_intervals
        self.result_prefix = result_prefix
        self.eps = 1e-9
        self.metric_index = 0
        self.save_dir = save_dir
        self.vis_image_dir = vis_image_dir
        self.upload_image_2_aidi = upload_image_2_aidi
        self.save_eval_result_path = save_eval_result_path
        self.collect_result = ["fn", "tp", "fp"]
        if self.compute_foreground_prec:
            self.collect_result.append("fp_miscls")
        super(ANCBEVDiscreteObjectEval, self).__init__(name)

    def _init_states(self):
        for name in self.get_names():
            self.add_state(
                name,
                default=[],
                dist_reduce_fx="cat",
            )
        self.add_state(
            "frames_num", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        self.add_state("frame_results", default=[], dist_reduce_fx="cat")

    def save_result(self, batch, output):
        with open(self.det_save_path, "ab") as f:
            pickle.dump([batch, output], f)

    def reset(self):
        super().reset()
        if self.save_dir:
            rank, _ = get_dist_info()
            self.det_save_path = os.path.join(
                self.save_dir, str(self.metric_index), f"dets/{rank}.pkl"
            )
            pkl_save_root = os.path.dirname(self.det_save_path)
            if not os.path.exists(pkl_save_root):
                os.makedirs(pkl_save_root, exist_ok=True)
        if self.upload_image_2_aidi:
            rank, _ = get_dist_info()
            self.images_save_path = os.path.join(
                os.path.dirname(self.save_eval_result_path),
                f"images/{rank}.pkl",
            )
            images_save_root = os.path.dirname(self.images_save_path)
            if not os.path.exists(images_save_root):
                os.makedirs(images_save_root, exist_ok=True)

    def get(self):
        """Get evaluation metrics."""
        values = self.compute()
        return self.name, values

    def get_misclassified_pred(self, misclassified_data, timestamps):
        """Get all-category misclassified prediction."""
        all_miss_det_clsid = {}
        all_miss_gt_clsid = {}
        all_miss_det_box = {}
        all_miss_gt_box = {}
        # get all fps and all gts
        for timestamp in timestamps:
            all_miss_det_clsid[timestamp] = []
            all_miss_gt_clsid[timestamp] = []
            all_miss_det_box[timestamp] = np.zeros((0, 7), dtype=np.float32)
            all_miss_gt_box[timestamp] = np.zeros((0, 7), dtype=np.float32)
            for cid in self.eval_category_ids:
                val = misclassified_data[cid][timestamp]
                if len(val) == 0:
                    continue
                miss_det_idx = val[:, 0] == 0
                miss_gt_idx = val[:, 0] == 1
                # only preserve high score prediction.
                vaild_det_idx = val[miss_det_idx][:, 1] >= self.score_threshold
                all_miss_det_box[timestamp] = np.append(
                    all_miss_det_box[timestamp],
                    val[miss_det_idx][vaild_det_idx],
                    axis=0,
                )
                all_miss_det_clsid[timestamp].extend(
                    [cid] * vaild_det_idx.sum()
                )
                # preserve all gts.
                all_miss_gt_box[timestamp] = np.append(
                    all_miss_gt_box[timestamp], val[miss_gt_idx], axis=0
                )
                all_miss_gt_clsid[timestamp].extend([cid] * miss_gt_idx.sum())
        # dict for misclassification per category, score and depth are in it.
        miscls_fp = {}
        for cid in self.eval_category_ids:
            miscls_fp[cid] = np.zeros((0, 3), dtype=np.float32)
        # compute iou between fps and gts,
        # if iou>0 between different class pair<fp, gt>, then this fp can be
        # treated as misclassified pred.
        for timestamp in timestamps:
            miss_det_box = all_miss_det_box[timestamp]
            miss_gt_box = all_miss_gt_box[timestamp]
            miss_det_clsid = all_miss_det_clsid[timestamp]
            miss_gt_clsid = all_miss_gt_clsid[timestamp]
            if len(miss_det_box) and len(miss_gt_box):
                valid_gt = np.ones(miss_gt_box.shape[0], dtype=bool)
                pairwise_iou = rotate_iou(
                    miss_det_box[:, 2:], miss_gt_box[:, 2:]
                )
                for dt_idx in range(miss_det_box.shape[0]):
                    det_iou = pairwise_iou[dt_idx] * valid_gt.astype(float)
                    max_ind = np.argmax(det_iou)
                    max_iou = det_iou[max_ind]
                    dt_cls = miss_det_clsid[dt_idx]
                    gt_cls = miss_gt_clsid[max_ind]
                    # if iou=0 or fp and gt are the same class
                    # this fp does not belong to misclassification.
                    if max_iou == 0 or dt_cls == gt_cls:
                        continue
                    valid_gt[max_ind] = 0
                    miscls_fp[dt_cls] = np.append(
                        miscls_fp[dt_cls],
                        miss_det_box[dt_idx][1:4][None, :],
                        axis=0,
                    )

        for cid in self.eval_category_ids:
            val = getattr(self, f"{cid}_misfg_prec_data") + [
                torch.Tensor(miscls_fp[cid]).cuda()
            ]
            setattr(self, f"{cid}_misfg_prec_data", val)

    def update(self, input, pred):
        """Update.

        Args:
            timestamp (torch.Tensor): Input image timestamp, (batch_size,).
            annos_bev_discobj (dict): A dict contains GT bbox info.

                .. code-block:: none

                    "vcs_discobj_loc": GT bbox center coord,
                        (batch_size, max_num,2).

                    "vcs_discobj_wh": GT bbox width and height,
                        (batch_size, max_num,2).

                    "vcs_discobj_valid": Valid GT bbox, (batch_size, max_num).

                    "vcs_discobj_cls": GT bbox category, (batch_size, max_num).

                    "vcs_discobj_yaw": GT bbox yaw, (batch_size, max_num).

            pred_ct (torch.Tensor): Predict bbox center, (batch_size, topk, 2).
            pred_wh (torch.Tensor): Predict bbox width and height,
                (batch_size, topk, 2).
            pred_score (torch.Tensor): Predict bbox score, (batch_size, topk).
            pred_yaw (torch.Tensor): Predict bbox yaw, (batch_size, topk).
            pred_bev_discobj_cls_id (torch.Tensor): Predict bbox category,
                (batch_size, topk).
        """
        batch = {
            "timestamp": input["timestamp"],
            "annos_bev_discrete_obj": input[self.annos_key],
        }
        output = {
            "pred_loc": pred["pred_ct"],
            "pred_wh": pred["pred_wh"],
            "pred_score": pred["pred_score"],
            "pred_yaw": pred["pred_yaw"],
            "pred_bev_discobj_cls_id": pred["pred_bev_discobj_cls_id"],
        }

        if self.save_dir:
            self.save_result(batch, output)

        batch_timestamps = batch["timestamp"].cpu().numpy()
        self.frames_num[0] += len(batch_timestamps)
        batch_timestamps = [str(int(_ts * 1000)) for _ts in batch_timestamps]
        eval_timestamps = batch_timestamps
        det_objs_by_cid = {cid: [] for cid in self.eval_category_ids}
        gt_objs_by_cid = {cid: [] for cid in self.eval_category_ids}

        # process model's output
        assert "pred_wh" in output.keys()
        batch_size, num_objs = output["pred_wh"].shape[:2]
        for bs in range(batch_size):
            for obj_idx in range(num_objs):
                pred_items = {
                    key: val[bs][obj_idx].cpu().numpy()
                    for key, val in output.items()
                }
                pred_items["timestamp"] = batch_timestamps[bs]
                cid = pred_items["pred_bev_discobj_cls_id"].tolist()
                if cid in det_objs_by_cid:
                    det_objs_by_cid[cid] += [pred_items]

        # process ground truth
        assert (
            "annos_bev_discrete_obj" in batch
        ), "Please confirm the annos in batch, \
                check BevDiscreteObjectTargetGenerator in auto3dv"
        annotations = batch["annos_bev_discrete_obj"]  # in vcs

        for cid in self.eval_category_ids:
            for bs in range(batch_size):
                # cid==-2 means the instance is labeled ignore.
                cur_cid_idxs = (
                    (
                        (annotations["vcs_discobj_cls"][bs] == cid)
                        | (annotations["vcs_discobj_cls"][bs] == -2)
                    )
                    .nonzero()
                    .squeeze(-1)
                )
                gt_cur_cid = {
                    key: val[bs][cur_cid_idxs]
                    for key, val in annotations.items()
                }
                num_objs_gt = gt_cur_cid[list(gt_cur_cid.keys())[0]].shape[0]
                for gt_obj_idx in range(num_objs_gt):
                    gt_items = {
                        key: val[gt_obj_idx].cpu().numpy()
                        for key, val in gt_cur_cid.items()
                    }
                    gt_items["timestamp"] = batch_timestamps[bs]
                    gt_objs_by_cid[cid] += [gt_items]

        if self.compute_foreground_prec:
            # if need to compute classification precision,
            # we need to get all-category fps
            misclassified_data = {}

        frame_results = defaultdict(lambda: defaultdict(list))
        for cid in self.eval_category_ids:
            gt = {
                "timestamps": eval_timestamps,
                "annotations": gt_objs_by_cid[cid],
            }

            det = []
            for item in det_objs_by_cid[cid]:
                if item["pred_score"] >= self.ap_score_threshold:
                    det.append(item)

            res, miss_res, cid_frame_results = eval_metric(
                det,
                gt,
                self.iou_threshold,
                self.gt_max_depth,
                self.yaw_amplitude,
                self.match_mode.get(self.id2label[cid], "iou")
                if isinstance(self.match_mode, dict)
                else self.match_mode,
                self.ct_match_mode,
                self.eval_vcs_range.get(self.id2label[cid], None),
                ct_matching_thresholds=self.ct_matching_thresholds,
                compute_foreground_prec=self.compute_foreground_prec,
            )

            val = getattr(self, f"{cid}") + [torch.Tensor(res).cuda()]
            setattr(self, f"{cid}", val)
            if self.compute_foreground_prec:
                misclassified_data[cid] = miss_res
            for ts in cid_frame_results.keys():
                for state in cid_frame_results[ts].keys():
                    frame_results[ts][state].extend(
                        cid_frame_results[ts][state]
                    )
        if self.upload_image_2_aidi:
            self.generate_images(frame_results)
        if self.compute_foreground_prec:
            self.get_misclassified_pred(misclassified_data, eval_timestamps)

    def get_names(self):
        names = []
        for cid in self.eval_category_ids:
            names.append(f"{cid}")
        if self.compute_foreground_prec:
            for cid in self.eval_category_ids:
                names.append(f"{cid}_misfg_prec_data")
        return names

    def print_log(self, format_metrics):
        names, values = [], []
        json_dict = {}
        log_line = "\n %s Metircs:\n" % (self.name)

        counts_show = ["tp", "fn", "fp", "Recall", "Precision", "AP", "fppi"]
        if self.compute_foreground_prec:
            counts_show.append("Foreground_Precision")

        summary = {}
        tables = []
        plots = []
        images = []

        for cid, cate_metrics in format_metrics.items():
            if (
                len(cate_metrics.get("tp", []))
                + len(cate_metrics.get("fn", []))
                == 0
            ):
                continue
            Num_tp, Num_fn, Num_fp, Num_fp_miscls = 0, 0, 0, 0

            if cid == "all":
                category = cid
                eval_vcs_range = self.eval_vcs_range.get(
                    self.id2label[0], None
                )
                depth_intervals = []
                for cls_id in self.id2label.keys():
                    cls_depth = self.depth_intervals.get(
                        self.id2label[cls_id], None
                    )
                    if cls_depth is None:
                        continue
                    depth_intervals = list(
                        set(depth_intervals).union(cls_depth)
                    )
                depth_intervals.sort()
            else:
                category = self.id2label[cid]
                eval_vcs_range = self.eval_vcs_range.get(
                    self.id2label[cid], None
                )
                depth_intervals = self.depth_intervals.get(
                    self.id2label[cid], None
                )
            log_line += f"category: {category}\n"
            log_line += f"eval_vcs_range:{eval_vcs_range}\n"
            json_dict.setdefault(category, {})
            json_dict[category]["eval_vcs_range"] = str(eval_vcs_range)
            names.append("category")
            values.append(category)

            columns = ["dep_interval"]
            table_data = []
            log_line += "dep_interval".ljust(15)
            for metric_name in self.metrics:
                log_line += metric_name.ljust(10)
                columns.append(metric_name)
            for num_name in counts_show:
                log_line += num_name.ljust(10)
                columns.append(num_name)

            log_line += "\n"

            if eval_vcs_range:
                depth_intervals = (
                    [eval_vcs_range[0]]
                    + list(depth_intervals)
                    + [eval_vcs_range[2]]
                )
            else:
                depth_intervals = [0] + [20, 50, 70] + [self.gt_max_depth]
            full_depth_intervals = [
                f"({start},{end})"
                for start, end in zip(
                    depth_intervals[:-1], depth_intervals[1:]
                )
            ]
            cate_metrics_dep = collect_result_by_dep(
                cate_metrics, depth_intervals
            )

            for i_dep, dep in enumerate(full_depth_intervals):
                log_line += str(dep).ljust(15)
                dep_data = {"dep_interval": str(dep)}
                if len(cate_metrics_dep[i_dep + 1]["tp"]) == 0:
                    for metric_name in self.metrics:
                        log_line += str(0).ljust(10)
                        dep_data.update({metric_name: 0})
                        names.append(f"{dep}_{metric_name}")
                        values.append(0)
                        json_dict[category].setdefault(f"dep_{dep}", {})
                        json_dict[category][f"dep_{dep}"][metric_name] = 0
                    Num_tp_ = 0
                    Num_fp_miscls_ = 0
                    Num_fn_ = len(cate_metrics_dep[i_dep + 1]["fn"])
                    scores_fp = cate_metrics_dep[i_dep + 1]["fp"][:, 0]
                    Num_fp_ = np.sum(scores_fp >= self.score_threshold)
                    show_nums_ = [
                        0,
                        Num_fn_,
                        Num_fp_,
                        0,
                        0,
                        0,
                        np.round(Num_fp_ / int(self.frames_num.item()), 3),
                    ]
                    if self.compute_foreground_prec:
                        show_nums_.append(0)
                    for counts_show_num, num in zip(counts_show, show_nums_):
                        log_line += str(num).ljust(10)
                        if isinstance(num, np.floating):
                            num = float(num)
                        elif isinstance(num, np.integer):
                            num = int(num)
                        dep_data.update({counts_show_num: num})
                        names.append(f"{dep}_{counts_show_num}")
                        values.append(num)
                        json_dict[category].setdefault(f"dep_{dep}", {})
                        json_dict[category][f"dep_{dep}"][
                            counts_show_num
                        ] = np.float(num)
                    log_line += "\n"
                else:
                    ap = np.round(
                        summarize_coco_ap(
                            cate_metrics_dep[i_dep + 1],
                            int(self.frames_num.item()),
                        )["coco_ap"],
                        3,
                    )
                    scores_tp = cate_metrics_dep[i_dep + 1]["tp"][:, 0]
                    scores_fp = cate_metrics_dep[i_dep + 1]["fp"][:, 0]
                    fn = cate_metrics_dep[i_dep + 1]["fn"]
                    Num_tp_ = np.sum(scores_tp >= self.score_threshold)
                    Num_fp_ = np.sum(scores_fp >= self.score_threshold)
                    gt_totals = len(scores_tp) + len(fn)
                    Num_fn_ = gt_totals - Num_tp_
                    indices_matched = scores_tp >= self.score_threshold
                    gt_matched = cate_metrics_dep[i_dep + 1]["tp"][
                        indices_matched
                    ]

                    for i_metric, metric_name in enumerate(self.metrics):
                        val = round(
                            np.sum(gt_matched[:, i_metric + 1])
                            / (Num_tp_ + self.eps),
                            3,
                        )
                        log_line += str(val).ljust(10)
                        dep_data.update({metric_name: val})
                        names.append(f"{dep}_{metric_name}")
                        values.append(val)
                        json_dict[category].setdefault(f"dep_{dep}", {})
                        json_dict[category][f"dep_{dep}"][
                            metric_name
                        ] = np.float(val)

                    recall_ = round(Num_tp_ / (gt_totals + self.eps), 3)
                    precision_ = round(
                        Num_tp_ / (Num_tp_ + Num_fp_ + self.eps), 3
                    )
                    show_nums_ = [
                        Num_tp_,
                        Num_fn_,
                        Num_fp_,
                        recall_,
                        precision_,
                        ap,
                        np.round(Num_fp_ / int(self.frames_num.item()), 3),
                    ]

                    if self.compute_foreground_prec:
                        Num_fp_miscls_ = len(
                            cate_metrics_dep[i_dep + 1]["fp_miscls"]
                        )
                        fg_prec_ = round(
                            Num_tp_ / (Num_tp_ + Num_fp_miscls_ + self.eps), 3
                        )
                        show_nums_.append(fg_prec_)

                    for counts_show_num, num in zip(counts_show, show_nums_):
                        log_line += str(num).ljust(10)
                        if isinstance(num, np.floating):
                            num = float(num)
                        elif isinstance(num, np.integer):
                            num = int(num)
                        dep_data.update({counts_show_num: num})
                        names.append(f"{dep}_{counts_show_num}")
                        values.append(num)
                        json_dict[category].setdefault(f"dep_{dep}", {})
                        json_dict[category][f"dep_{dep}"][
                            counts_show_num
                        ] = np.float(num)
                    log_line += "\n"
                Num_tp += Num_tp_
                Num_fp += Num_fp_
                Num_fn += Num_fn_
                if self.compute_foreground_prec:
                    Num_fp_miscls += Num_fp_miscls_
                table_data.append(dep_data)
            if self.dist_intervals is not None and eval_vcs_range:
                (
                    dist_log_line,
                    dist_names,
                    dist_values,
                ) = self.get_dist_interval_result(
                    category,
                    cate_metrics,
                    eval_vcs_range,
                    json_dict,
                    counts_show,
                )
                log_line += dist_log_line
                names += dist_names
                values += dist_values
                reformat_dist_log_line = dist_log_line.split("\n")
                dist_columns = reformat_dist_log_line[0].strip().split()
                dist_table_data = []
                for line in reformat_dist_log_line[1:]:
                    _data = {}
                    reformat_line = line.strip().split()
                    for k, v in zip(dist_columns, reformat_line):
                        if k != "dist_interval":
                            v = float(v)
                        _data.update({k: v})
                    dist_table_data.append(_data)

                tables.append(
                    Table(
                        name=f"{self.task_name} distance interval, category: {category}, eval_vcs_range: {eval_vcs_range}",  # noqa
                        columns=dist_columns,
                        data=dist_table_data,
                    )
                )

            for num in counts_show:
                log_line += num.ljust(10)
            log_line += "\n"

            recall = round(Num_tp / (Num_tp + Num_fn + self.eps), 3)
            precision = round(Num_tp / (Num_tp + Num_fp + self.eps), 3)

            summarize_results = summarize_coco_ap(
                cate_metrics, int(self.frames_num.item())
            )
            if category == "all":
                # update info with max detection rate to summary.
                det_rate = summarize_results["detection_rate"]

                if len(det_rate) == 0:
                    max_rate_score = max_rate_pre = max_rate_rec = 0.0
                else:
                    max_det_rate = max(det_rate)
                    max_det_rate_index = det_rate.index(max_det_rate)
                    max_rate_score = summarize_results["conf"][
                        max_det_rate_index
                    ]
                    max_rate_pre = summarize_results["precisions"][
                        max_det_rate_index
                    ]
                    max_rate_rec = summarize_results["recalls"][
                        max_det_rate_index
                    ]
                summary.update(
                    {
                        f"{self.task_name}: {category} Threshod@MaxDetRate": max_rate_score,  # noqa
                        f"{self.task_name}: {category} Recall@MaxDetRate": max_rate_rec,  # noqa
                        f"{self.task_name}: {category} Precision@MaxDetRate": max_rate_pre,  # noqa
                    }
                )
                # update results with target recall/precison to tabels.
                for key in ["target_recalls", "target_precisions"]:
                    tables.append(
                        Table(
                            name=f"{self.task_name} {key}: category: {category}, vcs_range:{eval_vcs_range}",  # noqa
                            columns=list(summarize_results[key][0].keys()),
                            data=summarize_results[key],
                        )
                    )

            ap = round(
                summarize_results["coco_ap"],
                3,
            )

            for num in counts_show:
                names.append(num)
            show_nums = [
                Num_tp,
                Num_fn,
                Num_fp,
                recall,
                precision,
                ap,
                np.round(Num_fp / int(self.frames_num.item()), 3),
            ]

            if self.compute_foreground_prec:
                fg_prec = round(
                    Num_tp / (Num_tp + Num_fp_miscls + self.eps), 3
                )
                show_nums.append(fg_prec)

            for counts_show_num, num in zip(counts_show, show_nums):
                if category == "all" and counts_show_num in [
                    "Precision",
                    "Recall",
                    "AP",
                ]:
                    summary.update(
                        {
                            f"{self.task_name}: {category} {counts_show_num}": float(  # noqa
                                num
                            )
                        }
                    )
                values.append(num)
                names.append(counts_show_num)
                log_line += str(num).ljust(10)
                json_dict[category][counts_show_num] = np.float(num)
            log_line += "\n"
            tables.append(
                Table(
                    name=f"{self.task_name} depth interval, category: {category}, eval_vcs_range: {eval_vcs_range}",  # noqa
                    columns=columns,
                    data=table_data,
                )
            )
            # update plot info
            plot_info = {
                "category": category,
                "result_prefix": self.result_prefix,
                "results": summarize_results,
            }
            for name, num in zip(counts_show, show_nums):
                plot_info[name] = num
            plots.append(plot_info)

        logger.info(log_line)
        if self.save_eval_result_path:
            eval_path_dir = os.path.dirname(self.save_eval_result_path)
            if not os.path.exists(eval_path_dir):
                os.makedirs(eval_path_dir, exist_ok=True)
            with open(self.save_eval_result_path, "w") as fw:
                fw.write(json.dumps(json_dict))
        plots = self.generate_plots(plots)
        images = None
        if self.upload_image_2_aidi:
            images = self.gather_images()
        return EvalResult(
            summary=summary,
            tables=tables,
            plots=plots,
            images=images,
        )

    def compute(self):
        self.metric_index += 1
        format_metrics = {"all": {}}
        for cid in self.eval_category_ids:
            format_metrics[cid] = {}
            val = getattr(self, f"{cid}")
            if len(val) == 0:
                continue

            if isinstance(val, List):
                val_npy = val[0].cpu().numpy()
                if len(val) > 1:
                    for v in val[1:]:
                        val_npy = np.append(val_npy, v.cpu().numpy(), axis=0)
            if isinstance(val, torch.Tensor):
                val_npy = val.cpu().numpy()

            fn_index = np.sum(np.abs(val_npy[:, 2:]), axis=1) < 1e-6
            fn = val_npy[fn_index, :2]
            val_npy = val_npy[fn_index == 0]
            fp_index = np.sum(np.abs(val_npy[:, 3:]), axis=1) < 1e-6
            fp = val_npy[fp_index, :3]
            tp = val_npy[fp_index == 0]

            tmp_res = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

            if self.compute_foreground_prec:
                val = getattr(self, f"{cid}_misfg_prec_data")
                if len(val) == 0:
                    val_npy = np.zeros((0, 3), dtype=np.float32)
                else:
                    if isinstance(val, List):
                        val_npy = val[0].cpu().numpy()
                        if len(val) > 1:
                            for v in val[1:]:
                                val_npy = np.append(
                                    val_npy, v.cpu().numpy(), axis=0
                                )
                    if isinstance(val, torch.Tensor):
                        val_npy = val.cpu().numpy()
                tmp_res["fp_miscls"] = val_npy

            for name in self.collect_result:
                if name not in format_metrics[cid]:
                    format_metrics[cid][name] = tmp_res[name]
                else:
                    format_metrics[cid][name] = np.append(
                        format_metrics[cid][name], tmp_res[name], axis=0
                    )

                if name not in format_metrics["all"]:
                    format_metrics["all"][name] = tmp_res[name]
                else:
                    format_metrics["all"][name] = np.append(
                        format_metrics["all"][name], tmp_res[name], axis=0
                    )

        format_metrics_show = self.print_log(format_metrics)
        return format_metrics_show

    def get_dist_interval_result(
        self, category, cate_metrics, eval_vcs_range, json_dict, counts_show
    ):
        """Compute results within each distance intervals.

        Args:
            category(str): the label name.
            cate_metrics(dict): the metrics results of current category.
            eval_vcs_range(list): the eval vcs range of current category.
                the sequence (bottom, right, top, left).
            json_dict(dict): the dict to be update here which will
                save as json finally.
            counts_show(list): a list of metric names.
        """
        names, values = [], []
        log_line = "dist_interval".ljust(15)

        for metric_name in self.metrics:
            log_line += metric_name.ljust(10)
        for num_name in counts_show:
            log_line += num_name.ljust(10)
        log_line += "\n"

        dist_intervals = self.dist_intervals
        dist_bottom = round(
            math.sqrt(
                eval_vcs_range[0] ** 2
                + max(abs(eval_vcs_range[1]), abs(eval_vcs_range[3])) ** 2
            ),
            1,
        )
        dist_top = round(
            math.sqrt(
                eval_vcs_range[2] ** 2
                + max(abs(eval_vcs_range[1]), abs(eval_vcs_range[3])) ** 2
            ),
            1,
        )
        dist_max = max(dist_bottom, dist_top)
        dist_intervals = [0] + list(dist_intervals) + [dist_max]
        full_dist_intervals = [
            f"({start},{end})"
            for start, end in zip(dist_intervals[:-1], dist_intervals[1:])
        ]
        cate_metrics_dist = collect_result_by_dist(
            cate_metrics, dist_intervals
        )
        for i_dist, dist in enumerate(full_dist_intervals):
            log_line += str(dist).ljust(15)

            if len(cate_metrics_dist[i_dist + 1]["tp"]) == 0:
                for metric_name in self.metrics:
                    log_line += str(0).ljust(10)
                    json_dict[category].setdefault(f"dist_{dist}", {})
                    json_dict[category][f"dist_{dist}"][metric_name] = 0
                Num_tp_ = 0
                Num_fn_ = len(cate_metrics_dist[i_dist + 1]["fn"])
                scores_fp = cate_metrics_dist[i_dist + 1]["fp"][:, 0]
                Num_fp_ = np.sum(scores_fp >= self.score_threshold)
                for counts_show_num, num in zip(
                    counts_show,
                    [
                        0,
                        Num_fn_,
                        Num_fp_,
                        0,
                        0,
                        0,
                        np.round(Num_fp_ / int(self.frames_num.item()), 3),
                    ],
                ):
                    log_line += str(num).ljust(10)
                    names.append(f"{dist}_{counts_show_num}")
                    values.append(num)
                    json_dict[category].setdefault(f"dist_{dist}", {})
                    json_dict[category][f"dist_{dist}"][
                        counts_show_num
                    ] = np.float(num)
                log_line += "\n"
            else:
                ap = np.round(
                    summarize_coco_ap(
                        cate_metrics_dist[i_dist + 1],
                        int(self.frames_num.item()),
                    )["coco_ap"],
                    3,
                )
                scores_tp = cate_metrics_dist[i_dist + 1]["tp"][:, 0]
                scores_fp = cate_metrics_dist[i_dist + 1]["fp"][:, 0]
                fn = cate_metrics_dist[i_dist + 1]["fn"]
                Num_tp_ = np.sum(scores_tp >= self.score_threshold)
                Num_fp_ = np.sum(scores_fp >= self.score_threshold)
                gt_totals = len(scores_tp) + len(fn)
                Num_fn_ = gt_totals - Num_tp_
                indices_matched = scores_tp >= self.score_threshold
                gt_matched = cate_metrics_dist[i_dist + 1]["tp"][
                    indices_matched
                ]

                for i_metric, metric_name in enumerate(self.metrics):
                    val = round(
                        np.sum(gt_matched[:, i_metric + 1])
                        / (Num_tp_ + self.eps),
                        3,
                    )
                    log_line += str(val).ljust(10)
                    names.append(f"{dist}_{metric_name}")
                    values.append(val)
                    json_dict[category].setdefault(f"dist_{dist}", {})
                    json_dict[category][f"dist_{dist}"][
                        metric_name
                    ] = np.float(val)

                recall_ = round(Num_tp_ / (gt_totals + self.eps), 3)
                precision_ = round(Num_tp_ / (Num_tp_ + Num_fp_ + self.eps), 3)

                for counts_show_num, num in zip(
                    counts_show,
                    [
                        Num_tp_,
                        Num_fn_,
                        Num_fp_,
                        recall_,
                        precision_,
                        ap,
                        np.round(Num_fp_ / int(self.frames_num.item()), 3),
                    ],
                ):
                    log_line += str(num).ljust(10)
                    names.append(f"{dist}_{counts_show_num}")
                    values.append(num)
                    json_dict[category].setdefault(f"dist_{dist}", {})
                    json_dict[category][f"dist_{dist}"][
                        counts_show_num
                    ] = np.float(num)
                log_line += "\n"
        return log_line, names, values

    def generate_plots(self, plots_info):
        """Generate plots.

        Args:
            plots_info(List[dict]): Plot info for each plot.
        """

        plots = []
        for info in plots_info:
            category = info["category"]
            # result_prefix = info["result_prefix"]
            results = info["results"]
            recalls = results["recalls"]
            precisions = results["precisions"]
            fppi = results["fppi"]
            conf = results["conf"]
            detection_rate = results["detection_rate"]

            tab1_data = []
            x1_recall, idx1 = np.unique(np.array(recalls), return_index=True)
            x1_recall = x1_recall.tolist()
            y1_precision = np.array(precisions)[idx1].tolist()
            for idx in range(len(x1_recall)):
                data_dict = {
                    "recall": x1_recall[idx],
                    "precision": y1_precision[idx],
                }
                tab1_data.append(data_dict)
            tab2_data = []
            x2_fppi, idx2 = np.unique(np.array(fppi), return_index=True)
            y2_recall = np.array(recalls)[idx2]
            valid_index = y2_recall < 1.0
            x2_fppi = x2_fppi[valid_index].tolist()
            y2_recall = y2_recall[valid_index].tolist()
            for idx in range(len(x2_fppi)):
                data_dict = {"fppi": x2_fppi[idx], "recall": y2_recall[idx]}
                tab2_data.append(data_dict)
            tab3_data = []
            x3_conf, idx3 = np.unique(np.array(conf), return_index=True)
            x3_conf = x3_conf.tolist()
            y3_recall = np.array(recalls)[idx3].tolist()
            y3_precision = np.array(precisions)[idx3].tolist()
            y3_detection_rate = np.array(detection_rate)[idx3].tolist()
            for idx in range(len(x3_conf)):
                data_dict = {
                    "threshold": x3_conf[idx],
                    "recall": y3_recall[idx],
                    "precision": y3_precision[idx],
                    "detection_rate": y3_detection_rate[idx],
                }
                tab3_data.append(data_dict)

            table1 = Table(
                name=f"{self.task_name}: {category}: recall vs precision",  # noqa
                columns=["recall", "precision"],
                data=tab1_data,
            )
            table2 = Table(
                name=f"{self.task_name}: {category}: fppi vs recall",  # noqa
                columns=["fppi", "recall"],
                data=tab2_data,
            )
            table3 = Table(
                name=f"{self.task_name}: {category}: thr vs recall & precision & detection_rate",  # noqa
                columns=[
                    "threshold",
                    "recall",
                    "precision",
                    "detection_rate",
                ],
                data=tab3_data,
            )
            plot_1 = {
                "Table": table1,
                "Line": Line(
                    x="recall",
                    y="precision",
                    stroke=f"recall-precision",  # noqa
                ),
            }
            plot_2 = {
                "Table": table2,
                "Line": Line(
                    x="fppi",
                    y="recall",
                    stroke=f"fppi-recall",  # noqa
                ),
            }
            plot_3 = {
                "Table": table3,
                "Line": Line(
                    x="threshold",
                    y="recall",
                    stroke=f"threshold-recall",  # noqa
                ),
            }
            plot_3["Line"].add(
                y="precision",
                stroke=f"threshold-precision",  # noqa
            )
            plot_3["Line"].add(
                y="detection_rate",
                stroke=f"threshold-detection_rate",  # noqa
            )
            plots += [plot_1, plot_2, plot_3]
        return plots

    def generate_images(self, frame_results):
        """Calculate various extreme scores within one frame.

        Args:
            frame_results(dict): Metric results.

                .. code-block: json

                    {
                        timestamp:{
                            "det_redundant": single_frame_det_redundant,
                            "gt_missed": single_frame_gt_missed,
                            "single_frame_matched": single_frame_matched,

                        }
                    }

        """
        images = []
        for ts, value in frame_results.items():
            image_path = glob.glob(
                os.path.join(self.vis_image_dir, f"*{int(ts)}.jpg")
            )
            if len(image_path) == 0:
                continue
            image_path = image_path[0]
            if not os.path.exists(image_path):
                continue
            tp_score = max([det[0] for det in value["matched"]] + [-1])
            tp_rot = max([det[6] for det in value["matched"]] + [-1])
            tp_dxy = max([det[3] for det in value["matched"]] + [-1])
            fp_score = max([det[0] for det in value["det_redundant"]] + [-1])
            fn_dist = max(
                [-np.linalg.norm(gt) for gt in value["gt_missed"] + [-100]]
            )

            image = Image(
                name="/".join(image_path.split("/")[-2:]),
                attrs={
                    "tp_score": float(tp_score),
                    "fn_dist": float(fn_dist),
                    "fp_score": float(fp_score),
                    "tp_dxy": float(tp_dxy),
                    "tp_rot": float(tp_rot),
                },
            )
            image.add_slice(data_or_path=image_path)
            images.append(image)
        with open(self.images_save_path, "ab") as fw:
            pickle.dump(images, fw)

    def gather_images(self):
        images = []
        logger.info("Begin gather images......")
        image_pkl_files = glob.glob(
            os.path.join(
                os.path.dirname(self.save_eval_result_path), "images/*.pkl"
            )
        )
        for pkl_file in image_pkl_files:
            logger.info(f"Begin load {pkl_file}")
            with open(pkl_file, "rb") as fr:
                while True:
                    try:
                        item = pickle.load(fr)
                        images += item
                    except EOFError:
                        break
        logger.info(
            f"All images have been gathered, total num is {len(images)}"
        )

        return images
