# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import torch

from hat.utils.apply_func import convert_numpy

__all__ = [
    "predict_preprocess",
    "gt_preprocess",
    "cal_vcs_pts",
    "nms_pts",
]


def predict_preprocess(batch_id, group, pred):
    """Perform activation func and convert numpy."""
    pred_stats = {}
    pred_logits = pred[f"pred_crosspoint_cls_{group}_frame0"][0][
        batch_id : batch_id + 1
    ]
    pred_logits = pred_logits.sigmoid()
    pred_stats["prob"], pred_stats["cls"] = torch.max(
        pred_logits, dim=1, keepdim=True
    )
    pred_stats["x"] = pred[f"pred_crosspoint_x_{group}_frame0"][0][
        batch_id : batch_id + 1
    ].sigmoid()
    pred_stats["y"] = pred[f"pred_crosspoint_y_{group}_frame0"][0][
        batch_id : batch_id + 1
    ].sigmoid()
    pred_stats = {k: convert_numpy(v[0]) for k, v in pred_stats.items()}
    return pred_stats


def gt_preprocess(batch_id, group, label):
    """Get single instance and convert numpy."""
    label_stats = {}
    label_stats["cls"] = label[group]["cls"][batch_id]
    label_stats["prob"], label_stats["cls"] = torch.max(
        label_stats["cls"], dim=0, keepdim=True
    )
    label_stats["x"] = label[group]["x"][batch_id]
    label_stats["y"] = label[group]["y"][batch_id]
    label_stats = {k: convert_numpy(v) for k, v in label_stats.items()}
    return label_stats


def cal_vcs_pts(
    stats,
    ap_score_thresh,
    vcs_range,
    meter_per_out_pixel_h,
    meter_per_out_pixel_w,
    ignore_index,
    is_gt,
):
    """Convert model output to crosspoints.

    Args:
        stats: Prediction or gt stats, include cls, pro, x, y.
        ap_score_thresh: Score threshold when calculate AP.
        vcs_range: Visbile range of bev, (bottom, right, top, left)
            in order.
        meter_per_out_pixel_h: Vertical vcs meters per pixel representation.
        meter_per_out_pixel_w: Horizontal vcs meters per pixel representation.
        ignore_index: Ignored cls_id in train and val stage.
        is_gt: If calculate groundtruth points or not.

    Returns:
        List: Crosspts, each as [x, y, cls_id, cls_score].
    """
    if stats is None:
        return None
    pts = []
    positive_mask = (
        (stats["prob"] == 1) if is_gt else (stats["prob"] >= ap_score_thresh)
    )
    non_ignore_mask = stats["cls"] != ignore_index
    mask = positive_mask * non_ignore_mask
    _, ids_x, ids_y = np.where(mask > 0)

    prob = stats["prob"][mask]
    x = stats["x"][mask]
    y = stats["y"][mask]
    cls = stats["cls"][mask]

    _, _, top_offset, left_offset = vcs_range
    for ins, (id_x, id_y) in enumerate(zip(ids_x, ids_y)):
        cls_id = cls[ins]
        score = prob[ins]
        coord_x = top_offset - (x[ins] + id_x) * meter_per_out_pixel_h
        coord_y = left_offset - (y[ins] + id_y) * meter_per_out_pixel_w
        pts.append([coord_x, coord_y, cls_id, score])
    return pts


def nms_pts(pred_pts, nms_thresh):
    """Crosspt nms.

    Args:
        pred_pts: Predict crosspt, each as [x, y, cls_id, cls_score].
        nms_thresh: Nms threshold, (x_thresh, y_thresh), unit: m.
    """
    pred_pts = sorted(pred_pts, key=lambda x: x[3], reverse=True)
    for i in range(len(pred_pts)):
        if pred_pts[i][2] == -1:
            continue
        for j in range(i + 1, len(pred_pts)):
            if pred_pts[j][2] == -1:
                continue

            x_dis = abs(pred_pts[i][0] - pred_pts[j][0])
            y_dis = abs(pred_pts[i][1] - pred_pts[j][1])
            if x_dis < nms_thresh[0] and y_dis < nms_thresh[1]:
                pred_pts[j][2] = -1
    nms_pred_pts = [pts for pts in pred_pts if pts[2] > -1]
    return nms_pred_pts
