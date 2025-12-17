#!/usr/bin/env python
import argparse
import json
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.interpolate.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from six import iteritems
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from hat.metrics.bev.online_mapping_utils import (
    chamfer_distance,
    get_target_categorys,
)

try:
    from horizon_plugin_pytorch import om_chamfer_distance
except ImportError:
    om_chamfer_distance = None

import warnings

warnings.filterwarnings("ignore")

om_chamfer_distance_valid = True
select_chamfer_distance = om_chamfer_distance
if om_chamfer_distance is None:
    om_chamfer_distance_valid = False
    select_chamfer_distance = chamfer_distance
    warnings.warn(
        "Fail to find om_chamfer_distance, "
        "please update horizon_plugin_pytorch>=0.16.5. "
    )


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def local_chamfer_distance(x, y, dist_thr=10):
    r"""Chamfer distance between two point clouds.

    Args:
        x: numpy array [n_points_x, n_dims]
            first point cloud
        y: numpy array [n_points_y, n_dims]
            second point cloud
        dist_thr: default dist value.
    Returns:
        chamfer_dist: float
            computed bidirectional Chamfer distance:
                sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} +
                sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """  # noqa

    if len(x) == 0 or len(y) == 0:
        return np.full((len(x)), dist_thr), np.full((len(y)), dist_thr)

    x_nn = NearestNeighbors(
        n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric="l2"
    ).fit(x)
    min_y2x = x_nn.kneighbors(y)[0][:, 0]
    y_nn = NearestNeighbors(
        n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric="l2"
    ).fit(y)
    min_x2y = y_nn.kneighbors(x)[0][:, 0]
    return min_x2y, min_y2x


def get_lane_pts_per_metre(gt_lane, pred_lane):
    """Get matched points from gt and pred lanes, get a point every 1 meter.

    Args:
        gt_lane: ndarray, gt lanes' points
        pred_lane: ndarray, pred lanes' points

    Returns:
        list: points per 1 meter from gt and pred matched lanes

    """
    start_point = max(min(gt_lane[:, 0]), min(pred_lane[:, 0]))
    end_point = min(max(gt_lane[:, 0]), max(pred_lane[:, 0]))
    lane_gt_x = np.round(gt_lane[:, 0], 6)
    lane_gt_y = np.round(gt_lane[:, 1], 6)
    lane_pred_x = np.round(pred_lane[:, 0], 6)
    lane_pred_y = np.round(pred_lane[:, 1], 6)
    tp_gt_pts = []
    tp_pred_pts = []
    for i in range(int(start_point), int(end_point) + 1, 1):
        sub_gt_lane_x = lane_gt_x[
            np.intersect1d(
                np.where(lane_gt_x >= i - 2),
                np.where(lane_gt_x <= i + 2),
            )
        ]
        sub_pred_lane_x = lane_pred_x[
            np.intersect1d(
                np.where(lane_pred_x >= i - 2),
                np.where(lane_pred_x <= i + 2),
            )
        ]
        if len(sub_gt_lane_x) < 2 or len(sub_pred_lane_x) < 2:
            continue
        sub_gt_lane_y = []
        sub_pred_lane_y = []
        for j in sub_gt_lane_x:
            sub_gt_lane_y.append(lane_gt_y[np.where(lane_gt_x == j)][0])
        for j in sub_pred_lane_x:
            sub_pred_lane_y.append(lane_pred_y[np.where(lane_pred_x == j)][0])

        f_gt = interp1d(
            sub_gt_lane_x,
            np.array(sub_gt_lane_y),
            bounds_error=False,
        )
        f_pred = interp1d(
            sub_pred_lane_x,
            np.array(sub_pred_lane_y),
            bounds_error=False,
        )
        gt_pt = np.round(f_gt(i), 4)
        pred_pt = np.round(f_pred(i), 4)
        if np.isnan(gt_pt) or np.isnan(pred_pt):
            continue
        tp_gt_pts.append(gt_pt)
        tp_pred_pts.append(pred_pt)
    return tp_gt_pts, tp_pred_pts


def bench(
    pred_lanes,
    gt_lanes,
    head_groups,
    cd_list,
    cd_threshold=0.5,
):
    """Get gt and pred lane statistic.

    Args:
        gt_lane: list, gt lanes' points.
        pred_lane: list, pred lanes' points.
        head_groups: om heads config.
        category: om output category.
        cd_list: cd_threshold list, to calculate
            different lane statistic from different cd_threshold.
        cd_threshold: classification threshold.

    Returns:
        get lane statistic dict

    """
    while [] in gt_lanes:
        gt_lanes.remove([])
    while [] in pred_lanes:
        pred_lanes.remove([])
    num_gt_lane = len(gt_lanes)
    num_pred_lane = len(pred_lanes)
    gt_all = (
        np.vstack(gt_lanes)[:, :2] if num_gt_lane > 0 else np.empty((0, 2))
    )
    pred_all = (
        np.vstack(pred_lanes)[:, :2] if num_pred_lane > 0 else np.empty((0, 2))
    )
    cd_g2p_list, cd_p2g_list = local_chamfer_distance(gt_all, pred_all)
    num_gt_pt = gt_all.shape[0]
    num_pred_pt = pred_all.shape[0]
    if om_chamfer_distance_valid:
        gt_all = torch.tensor(gt_all, dtype=torch.float32, device="cpu")
        pred_all = torch.tensor(pred_all, dtype=torch.float32, device="cpu")

    cd_g2p = (
        select_chamfer_distance(gt_all, pred_all, direction="x_to_y")
        if num_gt_lane > 0
        else 0
    )
    cd_p2g = (
        select_chamfer_distance(gt_all, pred_all, direction="y_to_x")
        if num_pred_lane > 0
        else 0
    )

    cost_mat = np.zeros((num_gt_lane, num_pred_lane))
    for i in range(num_gt_lane):
        gt_lane = gt_lanes[i][:, :2]
        if om_chamfer_distance_valid:
            gt_lane = torch.tensor(gt_lane, dtype=torch.float32, device="cpu")
        for j in range(num_pred_lane):
            pred_lane = pred_lanes[j][:, :2]
            if om_chamfer_distance_valid:
                pred_lane = torch.tensor(
                    pred_lane, dtype=torch.float32, device="cpu"
                )
            cost_mat[i, j] = (
                select_chamfer_distance(gt_lane, pred_lane, direction="bi")
                / 2.0
            )
    gt_inds, pred_inds = linear_sum_assignment(cost_mat)
    tp_gt_ids = []
    tp_pred_ids = []
    tp_gt_pts = []
    tp_pred_pts = []
    stats_with_cdlist_dict = {}

    # get sub heads
    # lane_pts_attr = {
    #     "vcs": 2,
    #     "vcs_origin": 2,
    #     "cls": 1,
    #     "prob": 1,
    #     "sin": 1,
    #     "cos": 1,
    #     "embedding": self.embedding_dim=4,
    #     "bev": 2,
    #     "offset_r": 1,
    # }
    sub_heads = []
    sub_heads_attr_index = {}
    start_index = 15
    for head, head_infos in head_groups.items():
        if not head_infos.get("multi", False):
            sub_heads.append(head)
            sub_heads_attr_index[head] = start_index
            start_index += 2

    for cd_score in cd_list:
        num_tp_lane = 0
        fn_gt_ids = list(range(num_gt_lane))
        fp_pred_ids = list(range(num_pred_lane))
        num_gt_pt_inst_list = []
        num_pred_pt_inst_list = []
        cd_g2p_inst_list = []
        cd_p2g_inst_list = []
        sub_task_pts_list = {}
        for head in sub_heads:
            sub_task_pts_list[f"gt_{head}_cls_inst"] = []
            sub_task_pts_list[f"pred_{head}_cls_inst"] = []
            sub_task_pts_list[f"gt_{head}_cls_pts"] = []
            sub_task_pts_list[f"pred_{head}_cls_pts"] = []
        # process sub task
        for gt_ind in range(num_gt_lane):
            for head in sub_heads:
                attr_index = sub_heads_attr_index[head]
                sub_task_pts_list[f"gt_{head}_cls_pts"] += gt_lanes[gt_ind][
                    :, attr_index
                ].tolist()
                sub_task_pts_list[f"pred_{head}_cls_pts"] += gt_lanes[gt_ind][
                    :, attr_index + 1
                ].tolist()
                gt_subtask_category = np.argmax(
                    np.bincount(
                        gt_lanes[gt_ind][:, attr_index].astype(np.uint8)
                    )
                )
                pred_subtask_category = np.argmax(
                    np.bincount(
                        gt_lanes[gt_ind][:, attr_index + 1].astype(np.uint8)
                    )
                )
                sub_task_pts_list[f"gt_{head}_cls_inst"].append(
                    gt_subtask_category
                )
                sub_task_pts_list[f"pred_{head}_cls_inst"].append(
                    pred_subtask_category
                )

        for gt_ind, pred_ind in zip(gt_inds, pred_inds):
            # process main task
            if cost_mat[gt_ind, pred_ind] < cd_score:
                num_tp_lane += 1
                gt_lane = gt_lanes[gt_ind][:, :2]
                pred_lane = pred_lanes[pred_ind][:, :2]
                num_gt_pt_inst = gt_lane.shape[0]
                num_pred_pt_inst = pred_lane.shape[0]
                if om_chamfer_distance_valid:
                    gt_lane = torch.tensor(
                        gt_lane,
                        dtype=torch.float32,
                        device="cpu",
                    )
                    pred_lane = torch.tensor(
                        pred_lane,
                        dtype=torch.float32,
                        device="cpu",
                    )
                cd_g2p_inst = select_chamfer_distance(
                    gt_lane, pred_lane, direction="x_to_y"
                )
                cd_p2g_inst = select_chamfer_distance(
                    gt_lane, pred_lane, direction="y_to_x"
                )
                num_gt_pt_inst_list.append(num_gt_pt_inst)
                num_pred_pt_inst_list.append(num_pred_pt_inst)
                cd_g2p_inst_list.append(cd_g2p_inst)
                cd_p2g_inst_list.append(cd_p2g_inst)

                if cd_score == cd_threshold:
                    tp_gt_ids.append(gt_ind)
                    tp_pred_ids.append(pred_ind)
                    fn_gt_ids.remove(gt_ind)
                    fp_pred_ids.remove(pred_ind)
                    tp_gt_pts, tp_pred_pts = get_lane_pts_per_metre(
                        gt_lane, pred_lane
                    )

        stats_with_cdlist_dict[str(cd_score)] = {
            "cd_g2p_inst_list": cd_g2p_inst_list,
            "cd_p2g_inst_list": cd_p2g_inst_list,
            "num_gt_pt_inst_list": num_gt_pt_inst_list,
            "num_pred_pt_inst_list": num_pred_pt_inst_list,
            "num_gt_lane": num_gt_lane,
            "num_pred_lane": num_pred_lane,
            "num_tp_lane": num_tp_lane,
        }
        if cd_score == cd_threshold:
            stats_with_cdlist_dict[str(cd_score)].update(
                {"tp_gt_pts": tp_gt_pts, "tp_pred_pts": tp_pred_pts}
            )
            for key in sub_task_pts_list:
                sub_task_pts_list[key] = np.array(sub_task_pts_list[key])
            stats_with_cdlist_dict[str(cd_score)].update(sub_task_pts_list)

    result_dict = {}
    result_dict["num_gt_pt"] = num_gt_pt
    result_dict["num_pred_pt"] = num_pred_pt
    result_dict["cd_g2p"] = cd_g2p
    result_dict["cd_p2g"] = cd_p2g
    result_dict["cd_g2p_list"] = cd_g2p_list
    result_dict["cd_p2g_list"] = cd_p2g_list
    result_dict["stats_with_cdlist_dict"] = stats_with_cdlist_dict

    return result_dict


def eval_sub(
    sub_head,
    head_groups,
    target_categorys,
    laneline_stats_dict,
    cd_threshold,
    output_stats,
):
    # get main categories of sub_head
    category_list = []
    group = head_groups[sub_head]["group"]
    for head, head_infos in head_groups.items():
        multi_head = head_infos.get("multi", False)
        if multi_head and head_infos["group"] == group:
            if "cls_list" in head_infos:
                category_list = head_infos["cls_list"]
            else:
                category_list = list(head_infos["cls_remap"].keys())
            category_list = [head + "_" + k for k in category_list]

    gt_cls_inst_list = []
    pred_cls_inst_list = []
    gt_cls_pts_list = []
    pred_cls_pts_list = []
    for category in target_categorys:
        if category not in category_list:
            continue
        for lane in laneline_stats_dict[category]:
            gt_cls_inst_list += (
                lane["stats_with_cdlist_dict"][str(cd_threshold)][
                    f"gt_{sub_head}_cls_inst"
                ]
                .astype(np.int)
                .tolist()
            )
            pred_cls_inst_list += (
                lane["stats_with_cdlist_dict"][str(cd_threshold)][
                    f"pred_{sub_head}_cls_inst"
                ]
                .astype(np.int)
                .tolist()
            )
            gt_cls_pts_list += (
                lane["stats_with_cdlist_dict"][str(cd_threshold)][
                    f"gt_{sub_head}_cls_pts"
                ]
                .astype(np.int)
                .tolist()
            )
            pred_cls_pts_list += (
                lane["stats_with_cdlist_dict"][str(cd_threshold)][
                    f"pred_{sub_head}_cls_pts"
                ]
                .astype(np.int)
                .tolist()
            )

    # get cls dims
    head_infos = head_groups[sub_head]
    num_class = max([v for _, v in head_infos["cls_remap"].items()])
    # point level
    if gt_cls_pts_list and max(gt_cls_pts_list) >= 0:
        head_conf = confusion_matrix(
            gt_cls_pts_list,
            pred_cls_pts_list,
            labels=range(0, num_class + 2),
        )
        head_conf = head_conf[1:, 1:]
    else:
        head_conf = np.zeros((num_class + 1, num_class + 1))
    num_gt_pts_per_category = head_conf.sum(axis=1)
    num_pred_pts_per_category = head_conf.sum(axis=0)
    pts_recall = np.diag(head_conf) / (
        num_gt_pts_per_category.astype(np.float64) + 1e-6
    )
    pts_presion = np.diag(head_conf) / (
        num_pred_pts_per_category.astype(np.float64) + 1e-6
    )
    pts_f1_scores = (
        2 * pts_recall * pts_presion / (pts_recall + pts_presion + 1e-6)
    )
    # compute all class pts metric
    pts_precision_all = np.sum(np.diag(head_conf)) / (
        np.sum(num_pred_pts_per_category.astype(np.float64)) + 1e-6
    )
    # instance level
    if gt_cls_inst_list and max(gt_cls_inst_list) >= 0:
        head_conf = confusion_matrix(
            gt_cls_inst_list,
            pred_cls_inst_list,
            labels=range(0, num_class + 2),
        )
        head_conf = head_conf[1:, 1:]
    else:
        head_conf = np.zeros((num_class + 1, num_class + 1))
    num_gt_inst_per_category = head_conf.sum(axis=1)
    num_pred_inst_per_category = head_conf.sum(axis=0)
    recall = np.diag(head_conf) / (
        num_gt_inst_per_category.astype(np.float64) + 1e-6
    )
    presion = np.diag(head_conf) / (
        num_pred_inst_per_category.astype(np.float64) + 1e-6
    )
    f1_scores = 2 * recall * presion / (recall + presion + 1e-6)
    # compute all cls instance metric
    inst_precision_all = np.sum(np.diag(head_conf)) / (
        np.sum(num_pred_inst_per_category.astype(np.float64)) + 1e-6
    )
    inst_num_all = np.sum(num_gt_inst_per_category)

    # get head categories
    head_categorys = []
    if "cls_list" in head_groups[sub_head]:
        head_categorys = head_groups[sub_head]["cls_list"]
    else:
        head_categorys = list(head_groups[sub_head]["cls_remap"].keys())

    for i in range(len(head_categorys)):
        category = sub_head + "_" + head_categorys[i]
        output_stats.update({category: {}})

        output_stats[category]["Precision_pts"] = round(pts_presion[i], 4)
        output_stats[category]["Recall_pts"] = round(pts_recall[i], 4)
        output_stats[category]["F-score_pts"] = round(pts_f1_scores[i], 4)
        output_stats[category]["GT_pts_num"] = round(
            num_gt_pts_per_category[i], 4
        )
        output_stats[category]["Pred_pts_num"] = round(
            num_pred_pts_per_category[i], 4
        )
        output_stats[category]["Precision_inst"] = round(presion[i], 4)
        output_stats[category]["Recall_inst"] = round(recall[i], 4)
        output_stats[category]["F-score_inst"] = round(f1_scores[i], 4)
        output_stats[category]["GT_inst_num"] = round(
            num_gt_inst_per_category[i], 4
        )
        output_stats[category]["Pred_inst_num"] = round(
            num_pred_inst_per_category[i], 4
        )
    # update all cls metric
    output_stats[sub_head] = {}
    output_stats[sub_head]["Precision_pts"] = round(pts_precision_all, 4)
    output_stats[sub_head]["Precision_inst"] = round(inst_precision_all, 4)
    output_stats[sub_head]["Inst_num"] = int(inst_num_all)
    return output_stats


def om_eval(
    head_groups,
    target_categorys,
    laneline_stats_dict,
    cd_threshold,
    eval_result_file=None,
):
    """Om metric with configs.

    Args:
        head_groups: om heads config.
        target_categorys: om output category.
        laneline_stats_dict: om lane statistics dict.
        cd_threshold: classification threshold.
        eval_result_file: result output path.

    """
    output_stats = {}
    fuse_region_metric = {}
    summary_str1 = "~~~~ OnlineMapping Summary Metrics ~~~~\n"
    metric_title1 = (
        "{:>20} {:>9} {:>9} {:>9} {:>9} {:>9} "
        + "{:>9} {:>9} {:>9} {:>9} {:>9} {:>9}\n"
    )
    summary_str1 += metric_title1.format(
        " ",
        "CD",
        "CD_g2p",
        "CD_p2g",
        "CD_inst",
        "CD_g2p_inst",
        "CD_p2g_inst",
        "Precision",
        "Recall",
        "F-score",
        "num_gt",
        "num_pred",
    )

    summary_str2 = "~~~~ OnlineMapping Summary Errors ~~~~\n"
    metric_title2 = "{:>20} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}\n"  # noqa
    summary_str2 += metric_title2.format(
        " ",
        "avg_error",
        "96_error",
        "9676_error",
        "sample_pts",
        "evs_avg",
        "evs_96",
        "evs_9976",
        "evs_R",
        "evs_P",
        "num_gt_pt",
        "num_pred_pt",
    )

    for category in target_categorys:
        output_stats.update({category: {}})
        if not laneline_stats_dict[category]:
            continue
        num_gt_pt = np.array(
            [i["num_gt_pt"] for i in laneline_stats_dict[category]]
        )
        num_pred_pt = np.array(
            [i["num_pred_pt"] for i in laneline_stats_dict[category]]
        )
        cd_g2p = np.array([i["cd_g2p"] for i in laneline_stats_dict[category]])
        cd_p2g = np.array([i["cd_p2g"] for i in laneline_stats_dict[category]])
        # compute matched error
        num_gt_lane = np.array(
            [
                i["stats_with_cdlist_dict"][str(cd_threshold)]["num_gt_lane"]
                for i in laneline_stats_dict[category]
            ]
        )
        num_pred_lane = np.array(
            [
                i["stats_with_cdlist_dict"][str(cd_threshold)]["num_pred_lane"]
                for i in laneline_stats_dict[category]
            ]
        )
        num_tp_lane = np.array(
            [
                i["stats_with_cdlist_dict"][str(cd_threshold)]["num_tp_lane"]
                for i in laneline_stats_dict[category]
            ]
        )
        num_gt_pt_inst_list = np.array(
            [
                i["stats_with_cdlist_dict"][str(cd_threshold)][
                    "num_gt_pt_inst_list"
                ]
                for i in laneline_stats_dict[category]
            ]
        )
        num_pred_pt_inst_list = np.array(
            [
                i["stats_with_cdlist_dict"][str(cd_threshold)][
                    "num_pred_pt_inst_list"
                ]
                for i in laneline_stats_dict[category]
            ]
        )
        cd_g2p_inst_list = np.array(
            [
                i["stats_with_cdlist_dict"][str(cd_threshold)][
                    "cd_g2p_inst_list"
                ]
                for i in laneline_stats_dict[category]
            ]
        )
        cd_p2g_inst_list = np.array(
            [
                i["stats_with_cdlist_dict"][str(cd_threshold)][
                    "cd_p2g_inst_list"
                ]
                for i in laneline_stats_dict[category]
            ]
        )
        tp_gt_pts = np.array(
            [
                i["stats_with_cdlist_dict"][str(cd_threshold)]["tp_gt_pts"]
                for i in laneline_stats_dict[category]
            ]
        )
        tp_pred_pts = np.array(
            [
                i["stats_with_cdlist_dict"][str(cd_threshold)]["tp_pred_pts"]
                for i in laneline_stats_dict[category]
            ]
        )

        if (np.sum(num_gt_pt) + np.sum(num_pred_pt)) == 0:
            continue
        cd = (np.sum(cd_g2p * num_gt_pt) + np.sum(cd_p2g * num_pred_pt)) / (
            np.sum(num_gt_pt) + np.sum(num_pred_pt) + 1e-6
        )
        cd_g2p_v = np.sum(cd_g2p * num_gt_pt) / (np.sum(num_gt_pt) + 1e-6)
        cd_p2g_v = np.sum(cd_p2g * num_pred_pt) / (np.sum(num_pred_pt) + 1e-6)
        num_gt_lane = np.sum(num_gt_lane)
        num_pred_lane = np.sum(num_pred_lane)
        recall = np.sum(num_tp_lane) / (num_gt_lane + 1e-6)
        precision = np.sum(num_tp_lane) / (num_pred_lane + 1e-6)

        f_score = 2 * recall * precision / (recall + precision + 1e-6)

        num_gt_pt_inst_list = np.hstack(num_gt_pt_inst_list)
        num_pred_pt_inst_list = np.hstack(num_pred_pt_inst_list)
        num_gt_pt = np.sum(num_gt_pt)
        num_pred_pt = np.sum(num_pred_pt)
        cd_g2p_inst_list = np.hstack(cd_g2p_inst_list)
        cd_p2g_inst_list = np.hstack(cd_p2g_inst_list)
        cd_inst = (
            np.sum(cd_g2p_inst_list * num_gt_pt_inst_list)
            + np.sum(num_pred_pt_inst_list * cd_p2g_inst_list)
        ) / (
            np.sum(num_gt_pt_inst_list) + np.sum(num_pred_pt_inst_list) + 1e-6
        )
        cd_g2p_inst = np.sum(cd_g2p_inst_list * num_gt_pt_inst_list) / (
            np.sum(num_gt_pt_inst_list) + 1e-6
        )
        cd_p2g_inst = np.sum(cd_p2g_inst_list * num_pred_pt_inst_list) / (
            np.sum(num_pred_pt_inst_list) + 1e-6
        )
        tp_gt_pts = np.hstack(tp_gt_pts)
        tp_pred_pts = np.hstack(tp_pred_pts)
        if tp_gt_pts.size <= 0 or tp_pred_pts.size <= 0:
            average_error = 0
            percentile_96_error = 0
            percentile_9976_error = 0
            sample_points = 0
        else:
            average_error = np.nanmean(abs(tp_gt_pts - tp_pred_pts))
            percentile_96_error = np.nanpercentile(
                abs(tp_gt_pts - tp_pred_pts), 96
            )
            percentile_9976_error = np.nanpercentile(
                abs(tp_gt_pts - tp_pred_pts), 99.76
            )
            sample_points = len(np.hstack(tp_gt_pts))
        # compute evs aligned error
        cd_g2p_list = np.hstack(
            [i["cd_g2p_list"] for i in laneline_stats_dict[category]]
        )
        cd_p2g_list = np.hstack(
            [i["cd_p2g_list"] for i in laneline_stats_dict[category]]
        )
        cd_p2g_mask = cd_p2g_list < 0.5
        valid_cd_p2g = cd_p2g_list[cd_p2g_mask]
        if len(valid_cd_p2g) > 0:
            evs_avg_error = np.nanmean(valid_cd_p2g)
            evs_96_error = np.nanpercentile(valid_cd_p2g, 96)
            evs_9976_error = np.nanpercentile(valid_cd_p2g, 99.76)
            evs_precision = np.sum(cd_p2g_mask) / (len(cd_p2g_list) + 1e-6)
        else:
            evs_avg_error = 0
            evs_96_error = 0
            evs_9976_error = 0
            evs_precision = 0
        cd_g2p_mask = cd_g2p_list < 0.5
        valid_cd_g2p = cd_g2p_list[cd_g2p_mask]
        if len(valid_cd_g2p) > 0:
            evs_recall = np.sum(cd_g2p_mask) / (len(cd_g2p_list) + 1e-6)
        else:
            evs_recall = 0

        summary_str1 += metric_title1.format(
            category,
            "{:.3f}".format(cd),
            "{:.3f}".format(cd_g2p_v),
            "{:.3f}".format(cd_p2g_v),
            "{:.3f}".format(cd_inst),
            "{:.3f}".format(cd_g2p_inst),
            "{:.3f}".format(cd_p2g_inst),
            "{:.3f}".format(precision),
            "{:.3f}".format(recall),
            "{:.3f}".format(f_score),
            "{:}".format(int(num_gt_lane)),
            "{:}".format(int(num_pred_lane)),
        )

        summary_str2 += metric_title2.format(
            category,
            "{:.3f}".format(average_error),
            "{:.3f}".format(percentile_96_error),
            "{:.3f}".format(percentile_9976_error),
            "{:}".format(int(sample_points)),
            "{:.3f}".format(evs_avg_error),
            "{:.3f}".format(evs_96_error),
            "{:.3f}".format(evs_9976_error),
            "{:.3f}".format(evs_recall),
            "{:.3f}".format(evs_precision),
            "{:}".format(int(num_gt_pt)),
            "{:}".format(int(num_pred_pt)),
        )
        fuse_region_metric[category + "_cd_inst"] = cd_inst
        fuse_region_metric[category + "_P"] = precision
        fuse_region_metric[category + "_R"] = recall
        # fuse_region_metric[category + "_avg_E"] = average_error
        # fuse_region_metric[category + "_96_E"] = percentile_96_error
        # fuse_region_metric[category + "_pt_num"] = int(sample_points)
        fuse_region_metric[category + "_gt_pt_num"] = int(num_gt_pt)
        fuse_region_metric[category + "_pred_pt_num"] = int(num_pred_pt)
        fuse_region_metric[category + "_evs_avg"] = evs_avg_error
        fuse_region_metric[category + "_evs_96"] = evs_96_error
        # fuse_region_metric[category + "_evs_9976"] = evs_9976_error
        fuse_region_metric[category + "_evs_P"] = evs_precision
        fuse_region_metric[category + "_evs_R"] = evs_recall

        output_stats[category]["CD"] = round(cd, 4)
        output_stats[category]["CD_g2p"] = round(cd_g2p_v, 4)
        output_stats[category]["CD_p2g"] = round(cd_p2g_v, 4)
        output_stats[category]["CD_inst"] = round(cd_inst, 4)
        output_stats[category]["CD_g2p_inst"] = round(cd_g2p_inst, 4)
        output_stats[category]["CD_p2g_inst"] = round(cd_p2g_inst, 4)
        output_stats[category]["Precision"] = round(precision, 4)
        output_stats[category]["Recall"] = round(recall, 4)
        output_stats[category]["F-score"] = round(f_score, 4)
        output_stats[category]["num_gt_lane"] = int(num_gt_lane)
        output_stats[category]["num_pred_lane"] = int(num_pred_lane)
        output_stats[category]["average_error(m)"] = round(average_error, 4)
        output_stats[category]["96_percentile_error(m)"] = round(
            percentile_96_error, 4
        )
        output_stats[category]["9976_percentile_error(m)"] = round(
            percentile_9976_error, 4
        )
        output_stats[category]["sample_points"] = int(sample_points)
        output_stats[category]["evs_avg(m)"] = round(evs_avg_error, 4)
        output_stats[category]["evs_96(m)"] = round(evs_96_error, 4)
        output_stats[category]["evs_9976(m)"] = round(evs_9976_error, 4)
        output_stats[category]["evs_P"] = round(evs_precision, 4)
        output_stats[category]["evs_R"] = round(evs_recall, 4)

    # update sub task
    summary_str3 = "~~~~ OnlineMapping Summary Subtask Metric ~~~~\n"
    metric_title3 = "{:>20} {:>15} {:>15} {:>15}\n"
    summary_str3 += metric_title3.format(
        " ",
        "Precision_pts",
        "Precision_inst",
        "Inst_num",
    )
    sub_heads = []
    for head, head_infos in head_groups.items():
        if not head_infos.get("multi", False):
            sub_heads.append(head)
    for sub_head in sub_heads:
        output_stats = eval_sub(
            sub_head,
            head_groups,
            target_categorys,
            laneline_stats_dict,
            cd_threshold,
            output_stats,
        )
        summary_str3 += metric_title3.format(
            sub_head,
            "{:.3f}".format(output_stats[sub_head]["Precision_pts"]),
            "{:.3f}".format(output_stats[sub_head]["Precision_inst"]),
            "{}".format(int(output_stats[sub_head]["Inst_num"])),
        )
        fuse_region_metric[sub_head + "_pts_P"] = output_stats[sub_head][
            "Precision_pts"
        ]
        fuse_region_metric[sub_head + "_inst_P"] = output_stats[sub_head][
            "Precision_inst"
        ]
        fuse_region_metric[sub_head + "_inst_num"] = output_stats[sub_head][
            "Inst_num"
        ]

    if eval_result_file:
        os.makedirs(os.path.dirname(eval_result_file), exist_ok=True)
        with open(eval_result_file, "w") as f:
            json.dump(output_stats, f, cls=NpEncoder)

    metric_str = summary_str1 + "\n" + summary_str3 + "\n" + summary_str2
    return metric_str, fuse_region_metric


def find_nearst_lane(lanes, x_min, x_max, y_min, y_max):
    """Get nearst lane according to coordinate range."""
    lane_y_list = []
    for i in range(len(lanes)):
        idx = np.intersect1d(
            np.where(lanes[i][:, 0] > x_min), np.where(lanes[i][:, 0] < x_max)
        )
        lane_y_list.append(lanes[i][:, 1][idx].mean())
    lane_y_list = np.array(lane_y_list, dtype=np.float64)
    lane_idx = np.intersect1d(
        np.where(lane_y_list > y_min), np.where(lane_y_list < y_max)
    )
    return lanes[lane_idx]


def get_roi_lane(lanes, rois):
    """Get roi lane according to coordinate range."""
    lanes_roi = [[] for i in range(len(lanes))]
    for i in range(len(lanes)):
        if len(lanes[i]) == 0:
            lanes_roi[i] = []
        else:
            filter_all = 0
            for roi in rois:
                y_min, y_max, x_min, x_max = roi
                bound_x = np.logical_and(
                    lanes[i][:, 0] > x_min, lanes[i][:, 0] < x_max
                )
                bound_y = np.logical_and(
                    lanes[i][:, 1] > y_min, lanes[i][:, 1] < y_max
                )
                filter = np.logical_and(bound_x, bound_y)
                filter_all = filter.astype(np.int) + filter_all
            if np.sum(filter_all) == 0:
                lanes_roi[i] = []
            else:
                lanes_roi[i] = lanes[i][filter_all > 0]
    return lanes_roi


def get_high_resolution_lane(lanes, interval=0.2):
    """Get high resolution lane according to interval."""
    high_resolution_lanes = [[] for i in range(len(lanes))]
    for n in range(len(lanes)):
        if lanes[n].shape[0] == 1:
            continue
        pts = lanes[n]
        for i in range(len(pts) - 1):
            start_pt = pts[i]
            end_pt = pts[i + 1]
            high_resolution_lanes[n].append(start_pt.tolist())
            if (max(pts[:, 0]) - min(pts[:, 0])) > (
                max(pts[:, 1]) - min(pts[:, 1])
            ):
                f = interp1d(
                    [start_pt[0], end_pt[0]],
                    [start_pt[1], end_pt[1]],
                    bounds_error=False,
                )
                num_sample = int(abs(end_pt[0] - start_pt[0]) / 0.2)
                for j in range(1, num_sample):
                    x = start_pt[0] + j * interval
                    y = float(f(x))
                    new_pt = np.copy(start_pt)
                    new_pt[0] = x
                    new_pt[1] = y
                    high_resolution_lanes[n].append(new_pt.tolist())
            else:
                f = interp1d(
                    [start_pt[1], end_pt[1]],
                    [start_pt[0], end_pt[0]],
                    bounds_error=False,
                )
                num_sample = int(abs(end_pt[1] - start_pt[1]) / 0.2)
                for j in range(1, num_sample):
                    y = start_pt[1] + j * interval
                    x = float(f(y))
                    new_pt = np.copy(start_pt)
                    new_pt[0] = x
                    new_pt[1] = y
                    high_resolution_lanes[n].append(new_pt.tolist())
        high_resolution_lanes[n] = np.array(high_resolution_lanes[n])
    return high_resolution_lanes


def get_pr_curve(lane_stats_dict, category, save_path):
    """Get precision and recall curve.

    According to cd_threshold from lane_stats_dict.
    """
    cd_threshold_list = []
    cd_inst_list = []
    precision_list = []
    recall_list = []
    for k, lane_dict in iteritems(lane_stats_dict):
        cd_g2p_inst_list = np.hstack(lane_dict["cd_g2p_inst_list"])
        cd_p2g_inst_list = np.hstack(lane_dict["cd_p2g_inst_list"])
        num_gt_pt_inst_list = np.hstack(lane_dict["num_gt_pt_inst_list"])
        num_pred_pt_inst_list = np.hstack(lane_dict["num_pred_pt_inst_list"])
        cd_inst = (
            np.sum(cd_g2p_inst_list * num_gt_pt_inst_list)
            + np.sum(num_pred_pt_inst_list * cd_p2g_inst_list)
        ) / (np.sum(num_gt_pt_inst_list) + np.sum(num_pred_pt_inst_list))
        num_gt_lane = np.sum(lane_dict["num_gt_lane"])
        num_pred_lane = np.sum(lane_dict["num_pred_lane"])
        num_tp_lane = np.sum(lane_dict["num_tp_lane"])
        recall = np.sum(num_tp_lane) / (num_gt_lane + 1e-6)
        precision = np.sum(num_tp_lane) / (num_pred_lane + 1e-6)
        cd_threshold_list.append(float(k))
        cd_inst_list.append(round(cd_inst, 3))
        precision_list.append(round(precision, 3))
        recall_list.append(round(recall, 3))
    plt.plot(cd_threshold_list, cd_inst_list, label="cd_inst", color="r")
    plt.plot(cd_threshold_list, precision_list, label="precision", color="g")
    plt.plot(cd_threshold_list, recall_list, label="recall", color="b")
    for x, y in zip(cd_threshold_list, cd_inst_list):
        plt.text(x, y, str(y), fontsize=10)
    for x, y in zip(cd_threshold_list, precision_list):
        plt.text(x, y, str(y), fontsize=10)
    for x, y in zip(cd_threshold_list, recall_list):
        plt.text(x, y, str(y), fontsize=10)
    plt.xlabel("cd threshold")
    plt.title(category)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="path of the configure file")
    parser.add_argument("--infer_path", help="path of the input files")
    parser.add_argument("--eval_path", help="path of the output files")
    parser.add_argument("--head_groups", default="", help="om_head_groups")
    parser.add_argument(
        "--metric_save_path", default="", help="om_head_groups"
    )
    parser.add_argument("--metric_cfg", default="", help="metric configs")
    args = parser.parse_args()
    return args


def eval_thread(
    profile_dict,
    rois,
    datas,
    target_categorys,
    head_groups,
    thread_id,
    results_list,
):
    cd_threshold = profile_dict["cd_threshold"]
    cd_list = profile_dict["cd_list"]
    upsample_value = profile_dict["upsample_value"]
    # init output result dict
    result_dict = {}
    for category in target_categorys:
        result_dict[category] = []

    for sub_data in datas:
        infer_path, _, image_name = sub_data
        gt_file = os.path.join(infer_path, image_name + "_gt.json")
        pred_file = os.path.join(infer_path, image_name + ".json")

        gt = json.load(open(gt_file, "r"))
        pred = json.load(open(pred_file, "r"))
        for category in target_categorys:
            gt_lanes = [np.array(i) for i in gt.get(category, [])]
            pred_lanes = [np.array(i) for i in pred.get(category, [])]

            if upsample_value > 0:
                gt_lanes = get_high_resolution_lane(
                    gt_lanes, interval=upsample_value
                )
                pred_lanes = get_high_resolution_lane(
                    pred_lanes, interval=upsample_value
                )
            if rois is not None:
                gt_lanes = get_roi_lane(np.array(gt_lanes), rois)
                pred_lanes = get_roi_lane(np.array(pred_lanes), rois)
            laneline_stats = bench(
                pred_lanes,
                gt_lanes,
                head_groups,
                cd_list,
                cd_threshold=cd_threshold,
            )
            if laneline_stats is not None:
                result_dict[category].append(laneline_stats)
    results_list[thread_id] = result_dict


def cal_fine_metric(
    profile_path=None,
    infer_root_src=None,
    eval_result_output=None,
    om_head_groups=None,
    metric_cfg=None,
    metric_save_path=None,
):
    profile_dict = yaml.safe_load(open(profile_path, "r"))
    # update with input config
    if metric_cfg is not None:
        metric_cfg = eval(metric_cfg)
        for k, v in metric_cfg.items():
            profile_dict[k] = v
    cd_threshold = profile_dict["cd_threshold"]
    if infer_root_src in [None, "None"]:
        infer_root_src = profile_dict["infer_root"]
    cd_list = profile_dict["cd_list"]
    if eval_result_output in [None, "None"]:
        eval_result_output = profile_dict["eval_result_output"]
    os.makedirs(
        os.path.join(eval_result_output),
        exist_ok=True,
    )

    # get head groups info
    if om_head_groups:
        head_groups = eval(om_head_groups)
    else:
        head_groups = profile_dict["om_head_groups"]
        head_blacklist = profile_dict["head_blacklist"]
        # filter head config
        head_groups = {
            k: v for k, v in head_groups.items() if k not in head_blacklist
        }
    # parse main category infos from head_groups
    target_categorys = get_target_categorys(head_groups)
    for key in profile_dict.get("black_list", []):
        if key in target_categorys:
            target_categorys.pop(key)
    # get data list
    data_list = []
    for pack_name in tqdm(os.listdir(infer_root_src)):
        infer_path = os.path.join(infer_root_src, pack_name)
        sub_data_list = [
            os.path.splitext(i)[0]
            for i in os.listdir(infer_path)
            if "_gt" not in i
        ]
        for image_name in sub_data_list:
            data_list.append([infer_path, pack_name, image_name])

    # split into different threads
    thread_num = profile_dict.get("thread_num", 1)
    thread_datas = [[] for _ in range(thread_num)]
    for i in range(len(data_list)):
        thread_index = i % thread_num
        thread_datas[thread_index].append(data_list[i])

    # define fuse str title
    sum_metric_str = ""
    region_dict = profile_dict["region"]
    fuse_results = {}
    for region_name, region_roi in region_dict.items():
        if region_roi is not None:
            if not isinstance(region_roi[0], (list, tuple)):
                region_roi = [region_roi]
        laneline_stats_dict = {}
        for category in target_categorys:
            laneline_stats_dict.update({category: []})
        sum_metric_str += f"\n\n---------- {region_name} -----------\n"
        thread_list = []
        manager = mp.Manager()
        return_dict = manager.dict()
        for i in range(thread_num):
            thread = mp.Process(
                target=eval_thread,
                args=(
                    profile_dict,
                    region_roi,
                    thread_datas[i],
                    target_categorys,
                    head_groups,
                    i,
                    return_dict,
                ),
            )
            thread_list.append(thread)
            thread.start()
        for i in range(len(thread_datas)):
            thread_list[i].join()
        for thread_results in return_dict.values():
            for category, stats_list in thread_results.items():
                laneline_stats_dict[category] += stats_list
        for category in target_categorys:
            lane_stats_list = [
                i["stats_with_cdlist_dict"]
                for i in laneline_stats_dict[category]
            ]
            lane_stats_dict_with_cd = {}
            for cd in cd_list:
                lane_stats_dict_with_cd[str(cd)] = {}
                for key in lane_stats_list[0][str(cd_threshold)]:
                    lane_stats_dict_with_cd[str(cd)].update({key: []})
            for lane_stats in lane_stats_list:
                for k, thedict in iteritems(lane_stats):
                    for sub_k, sub_v in iteritems(thedict):
                        lane_stats_dict_with_cd[k][sub_k].append(sub_v)

            curve_name = category + "_" + region_name
            get_pr_curve(
                lane_stats_dict_with_cd,
                curve_name,
                save_path=os.path.join(
                    eval_result_output, curve_name + ".png"
                ),
            )
        sub_region_metric_str, fuse_region_metric = om_eval(
            head_groups,
            target_categorys,
            laneline_stats_dict,
            cd_threshold,
            os.path.join(eval_result_output, region_name + ".json"),
        )
        sum_metric_str += sub_region_metric_str
        fuse_results[region_name] = fuse_region_metric
    # show config
    sum_metric_str += "\nmetric_cfg:\n{}\n".format(metric_cfg)
    # update fuse region metric str
    sum_metric_str += "\n\n!!!!!!!!!!!!!!!!!! summary !!!!!!!!!!!!!!!!!!!!!\n"
    sum_metric_str += "{:>18}".format(" ")
    for region_name in region_dict.keys():
        region_name = region_name.replace("host_next", "hn")
        sum_metric_str += " {:>10}".format(region_name)
    sum_metric_str += "\n"
    base_region_name = list(region_dict.keys())[0]
    remap_keys_dict = profile_dict.get("remap_print_keys", {})
    for key in fuse_results[base_region_name].keys():
        print_key = key
        for k, v in remap_keys_dict.items():
            if k in print_key:
                print_key = print_key.replace(k, v)
                break
        sum_metric_str += "{:>25}".format(print_key)
        for region_name in region_dict.keys():
            if key in fuse_results[region_name]:
                value = fuse_results[region_name][key]
                if value < 1e-6:
                    sum_metric_str += "{:>10}".format("-")
                    continue
                if "num" in key:
                    sum_metric_str += "{:>10}".format("{}".format(int(value)))
                else:
                    sum_metric_str += "{:>10}".format("{:.3f}".format(value))
            else:
                sum_metric_str += "{:>10}".format("-")
        sum_metric_str += "\n"
    # save fuse results
    save_fuse_results = {
        "results": fuse_results,
        "remap_keys": remap_keys_dict,
    }
    if metric_save_path:
        metric_save_dir, _ = os.path.split(metric_save_path)
        os.makedirs(metric_save_dir, exist_ok=True)
        with open(metric_save_path, "w") as f:
            json.dump(save_fuse_results, f, cls=NpEncoder)

    return sum_metric_str


if __name__ == "__main__":
    args = parse_arguments()
    metric_str = cal_fine_metric(
        profile_path=args.profile,
        infer_root_src=args.infer_path,
        eval_result_output=args.eval_path,
        om_head_groups=args.head_groups,
        metric_save_path=args.metric_save_path,
        metric_cfg=args.metric_cfg,
    )
    print(metric_str)
