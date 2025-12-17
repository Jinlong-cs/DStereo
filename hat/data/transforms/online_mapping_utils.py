# Copyright (c) Horizon Robotics. All rights reserved.

import math
import os
import time
import warnings

import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.neighbors import NearestNeighbors

from hat.core.affine import get_vcs2bev_img_mat
from hat.core.bev_elevation_utils import decimal_div
from hat.data.transforms.real3d import draw_heatmap
from hat.utils.package_helper import check_packages_available

try:
    from horizon_plugin_pytorch import om_ogc
except ImportError:
    om_ogc = None

import copy

from hat.visualize.online_mapping import (
    arrange_imgs,
    draw_clip_raw_gt,
    draw_dilate_mask,
    draw_ipm_img,
    draw_multi_head,
    draw_raw_gt,
    draw_raw_img,
    draw_single_head,
    get_bev_pts,
    get_cls_color_table,
)

try:
    from horizon_plugin_pytorch import om_pt2lineseg
except ImportError:
    om_pt2lineseg = None

__all__ = [
    "get_gt_online_mapping",
    "get_navnet",
    "chamfer_distance",
    "convert_gt",
    "convert_pred",
    "get_offset",
    "get_smoothing",
    "get_vcs_lane",
]


def ogc_diversity(
    pixel1: np.array,
    pixel2: np.array,
    pose_weight: float = 0.1,
) -> float:
    """
    Compute the diversity between two om pixels.

    Args:
        pixel1: [prob, row, col, channel, r, sin, cos, embedding]
        pixel2: [prob, row, col, channel, r, sin, cos, embedding]
        pose_weight: the weight of pose diversity

    Returns:
        diversity value
    """

    # embedding feature distance
    fea_dist = np.linalg.norm(pixel1[7:] - pixel2[7:])
    # direction and pose distance
    # notice: real sin = -sin, real cos = -cos, front x, left y
    x1 = -pixel1[1] - pixel1[4] * pixel1[6]
    y1 = -pixel1[2] - pixel1[4] * pixel1[5]
    x2 = -pixel2[1] - pixel2[4] * pixel2[6]
    y2 = -pixel2[2] - pixel2[4] * pixel2[5]
    dx = x2 - x1
    dy = y2 - y1
    dist_2_1 = np.fabs(pixel1[6] * dx + pixel1[5] * dy)
    dist_1_2 = np.fabs(pixel2[6] * dx + pixel2[5] * dy)
    pose_dist = np.fmax(dist_1_2, dist_2_1)
    return fea_dist + pose_weight * pose_dist


def ogc_cluster(
    pred_cls: np.array,
    pred_prob: np.array,
    pred_r: np.array,
    pred_sin: np.array,
    pred_cos: np.array,
    pred_embedding: np.array,
    cls_thr: float = 0.5,
    radius_l: int = 9,
    radius_t: int = 2,
    min_num: int = 1,
    pose_weights: float = None,
    cluster_thr: float = 0.9,
):
    """
    OGC (Offset Growth Cluster) for Online Mapping Post Process.

    Notice: Generally, the channel C means set num, the default value is 2

    Args:
        pred_cls: [C, H, W], class label
        pred_prob: [C, H, W], class confidence
        pred_r: [C, H, W], offset radius
        pred_sin: [C, H, W], offset sin value
        pred_cos:  [C, H, W], offset cos value
        pred_embedding: [C, H, W, D], embedding features for instance
        cls_thr: used to select valid pixels from pred_probs
        radius_l: the radius of longitudinal searching
        radius_t: the radius of transverse searching
        min_num: the minimum number of clustering points
        pose_weight: the weight of computing pose diversity
        cluster_thr: the threshold of point similarly

    Returns:
        cluster_result: [H, W], index of different lanes on each pixel
    """

    # init
    C, H, W = pred_cls.shape
    cluster_result = np.zeros((C, H, W), dtype=np.int32)
    cluster_id = 0
    bins = 18
    # embedding features dims
    dims = pred_embedding.shape[-1]

    # cluster for each class
    pos = pred_prob > cls_thr
    cls_labels = np.unique(pred_cls).tolist()
    cls_labels.sort()
    for cls_label in cls_labels:
        # generate valid pixels mask
        mask = np.logical_and(pos, pred_cls == cls_label)
        valid_num = mask.astype(np.int32).sum()
        if valid_num < min_num:
            continue
        # generate current pose weight
        pose_weight = 0.1
        if pose_weights is not None:
            if isinstance(pose_weights, float):
                pose_weight = pose_weights
            elif isinstance(pose_weights, (list, tuple)):
                pose_weight = pose_weights[int(cls_label) - 1]
            elif isinstance(pose_weights, dict):
                pose_weight = list(pose_weights.values())[int(cls_label) - 1]
            else:
                raise AssertionError(
                    "Invalid ogc pose_weights:{}".format(pose_weights)
                )
        # creat empty map used to store valid key points information
        count_map = np.zeros((H, W), dtype=np.uint8)
        index_map = np.zeros((H, W, C), dtype=np.int32)
        # valid pixels: [prob, row, col, channel, r, sin, cos, embedding]
        pixels = np.zeros((valid_num, 7 + dims), dtype=np.float32)
        # init flags as -1: UNCLASSIFIED; -2: NOISE, >=0: CLUSTER_ID
        flags = [-1] * valid_num
        # extract all valid pixels
        index = 0
        for channel in range(C):
            for row in range(H):
                for col in range(W):
                    if not mask[channel, row, col]:
                        continue
                    index_map[row, col, count_map[row, col]] = index + 1
                    count_map[row, col] += 1
                    pixels[index, 0] = pred_prob[channel, row, col]
                    pixels[index, 1] = float(row)
                    pixels[index, 2] = float(col)
                    pixels[index, 3] = float(channel)
                    pixels[index, 4] = pred_r[channel, row, col]
                    pixels[index, 5] = pred_sin[channel, row, col]
                    pixels[index, 6] = pred_cos[channel, row, col]
                    pixels[index, 7:] = pred_embedding[channel, row, col, :]
                    index += 1
        # core search loop
        cluster_centers = []
        for i in range(valid_num):
            if flags[i] != -1:
                continue
            search_list = []
            search_list.append(i)
            cluster_center = pixels[i, 7:].copy()
            for search_index in search_list:
                flags[search_index] = cluster_id
                anchor_pixel = pixels[search_index, :]
                anchor_row = int(anchor_pixel[1] + 0.5)
                anchor_col = int(anchor_pixel[2] + 0.5)
                # compute the direction on img coordinate system
                direction = np.arctan2(anchor_pixel[5], anchor_pixel[6])
                direction = direction + np.pi * 0.5
                if direction < 0:
                    direction += np.pi
                else:
                    direction = math.fmod(direction, np.pi)
                # compute the radial search space base on the direction
                radial_bin = int(direction / np.pi * bins)
                radial_bin = max(min(radial_bin, bins - 1), 0)
                alpha = np.pi / bins * (radial_bin + 0.5)
                a = np.sin(alpha)
                b = -np.cos(alpha)
                radius_space = []
                for r in range(-radius_l, radius_l + 1):
                    for c in range(-radius_l, radius_l + 1):
                        dist = np.fabs(a * r + b * c)
                        if dist < radius_t + 0.5 and (r != 0 or c != 0):
                            radius_space.append([r, c])
                # search radius_space
                for d_r, d_c in radius_space:
                    row, col = anchor_row + d_r, anchor_col + d_c
                    # check in range
                    if row < 0 or col < 0 or row >= H or col >= W:
                        continue
                    # check has valid pixels
                    if count_map[row, col] <= 0:
                        continue
                    # check if belong to the same cluster
                    for channel in range(C):
                        pixel_index = index_map[row, col, channel] - 1
                        if pixel_index < 0 or flags[pixel_index] != -1:
                            continue
                        pixel = pixels[pixel_index, :]
                        d = ogc_diversity(anchor_pixel, pixel, pose_weight)
                        if d < cluster_thr:
                            flags[pixel_index] = cluster_id
                            count_map[row, col] -= 1
                            search_list.append(pixel_index)
                            cluster_center += pixel[7:]
            # process noise and assign cluster id to cluster_result
            if len(search_list) < min_num:
                for search_index in search_list:
                    flags[search_index] = -2
            else:
                cluster_id += 1
                for search_index in search_list:
                    pixel = pixels[search_index, :]
                    row = int(pixel[1] + 0.5)
                    col = int(pixel[2] + 0.5)
                    channel = int(pixel[3] + 0.5)
                    cluster_result[channel, row, col] = cluster_id
                cluster_center /= float(len(search_list))
                cluster_centers.append(cluster_center)
        # merge clusters
        channel_cluster_num = len(cluster_centers)
        if channel_cluster_num > 1:
            visited = [False] * channel_cluster_num
            merge_id_map = [-1] * channel_cluster_num
            new_cluster_num = 0
            for i in range(channel_cluster_num):
                if visited[i]:
                    continue
                visited[i] = True
                merge_id_map[i] = new_cluster_num
                for j in range(i + 1, channel_cluster_num):
                    if visited[j]:
                        continue
                    dist = np.linalg.norm(
                        cluster_centers[i] - cluster_centers[j]
                    )
                    if dist < cluster_thr:
                        visited[j] = True
                        merge_id_map[j] = new_cluster_num
                new_cluster_num += 1
            if new_cluster_num < channel_cluster_num:
                id_offset = cluster_id - channel_cluster_num
                for i in range(valid_num):
                    if flags[i] < 0:
                        continue
                    pixel = pixels[i, :]
                    row = int(pixel[1] + 0.5)
                    col = int(pixel[2] + 0.5)
                    channel = int(pixel[3] + 0.5)
                    new_id = merge_id_map[flags[i] - id_offset] + 1
                    cluster_result[channel, row, col] = new_id + id_offset
                cluster_id -= channel_cluster_num - new_cluster_num

    return cluster_result


def chamfer_distance(
    x: np.array,
    y: np.array,
    metric: str = "l2",
    direction: str = "bi",
    dist_thresh: int = 10,
) -> float:
    r"""Chamfer distance between two point clouds.

    Args:
        x: [n_points_x, n_dims]
            first point cloud
        y: [n_points_y, n_dims]
            second point cloud
        metric:  default ‘l2’
            metric to use for distance computation.
            Any metric from scikit-learn or scipy.spatial.distance can be used.
        direction: direction of Chamfer distance.
            "y_to_x": computes average minimal distance
                        from every point in y to x
            "x_to_y": computes average minimal distance
                        from every point in x to y
            "bi": compute both
    Returns:
        chamfer_dist: computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} +
            sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """  # noqa

    if direction == "y_to_x":
        if len(x) == 0:
            chamfer_dist = dist_thresh
        else:
            x_nn = NearestNeighbors(
                n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
            ).fit(x)
            min_y_to_x = x_nn.kneighbors(y)[0]
            chamfer_dist = np.mean(min_y_to_x)
    elif direction == "x_to_y":
        if len(y) == 0:
            chamfer_dist = dist_thresh
        else:
            y_nn = NearestNeighbors(
                n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
            ).fit(y)
            min_x_to_y = y_nn.kneighbors(x)[0]
            chamfer_dist = np.mean(min_x_to_y)
    elif direction == "bi":
        if len(x) == 0:
            chamfer_dist_y_to_x = dist_thresh
        else:
            x_nn = NearestNeighbors(
                n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
            ).fit(x)
            min_y_to_x = x_nn.kneighbors(y)[0]
            chamfer_dist_y_to_x = np.mean(min_y_to_x)
        if len(y) == 0:
            chamfer_dist_x_to_y = dist_thresh
        else:
            y_nn = NearestNeighbors(
                n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
            ).fit(y)
            min_x_to_y = y_nn.kneighbors(x)[0]
            chamfer_dist_x_to_y = np.mean(min_x_to_y)
        chamfer_dist = chamfer_dist_x_to_y + chamfer_dist_y_to_x
    else:
        raise ValueError(
            'Invalid direction type. Supported types: "y_x", "x_y", "bi"'
        )

    return chamfer_dist


def adjust_roi_weight(gt_stats, roi_weight_cfg, out_h, out_w, vcs_range):
    """Adjust multi type roi weight.

    More details refer to:
    https://horizonrobotics.feishu.cn/wiki/LaTZwVAysiFtvNkZeCDc00pFnHc.

    Args:
        gt_stats: all gt data.
        roi_weight_cfg: all configurations related to ROI weights.Example:
            "roi_name_1":{
                "region": roi range of bev, (bottom, right, top, left)
                    in order.
                "weight": loss weight.
                "save_roi_mask": build specific type roi mask in gt_stats.
            }
            "roi_name_2":{
                ...
            }
        out_h: the height of output head.
        out_w: the width of output head.
        vcs_range: visbile range of bev, (bottom, right, top, left)
            in order.
    """

    for group in gt_stats:
        weight = gt_stats[group]["weight"]
        for roi_name, region_cfg in roi_weight_cfg.items():
            positive_flag = 1
            save_roi_mask = region_cfg.get("save_roi_mask", False)
            for region in region_cfg["region"]:
                top, left, bottom, right = get_roi_vcs_range_box(
                    (out_h, out_w), vcs_range, region
                )
                weight[:, top:bottom, left:right] *= region_cfg["weight"]

                if save_roi_mask:
                    type_mask = gt_stats[group][roi_name]
                    type_mask[:, top:bottom, left:right] = positive_flag
    return gt_stats


def height_distance(
    x, y, x_height, y_height, metric="l2", direction="y_to_x", dist_thresh=10
):
    r"""Height distance between two point clouds."""  # noqa

    if direction == "y_to_x":
        if len(x) == 0:
            height_err = np.array([0, 0, 0, dist_thresh]).reshape(-1, 4)
        else:
            x_nn = NearestNeighbors(
                n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
            ).fit(x)
            min_y_to_x_id = x_nn.kneighbors(y)[1]
            height_err = np.concatenate(
                [y, y_height, y_height - x_height[min_y_to_x_id.reshape(-1)]],
                -1,
            )
    elif direction == "x_to_y":
        if len(y) == 0:
            height_err = np.array([0, 0, 0, dist_thresh]).reshape(-1, 4)
        else:
            y_nn = NearestNeighbors(
                n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
            ).fit(y)
            min_x_to_y_id = y_nn.kneighbors(x)[1]
            height_err = np.concatenate(
                [x, x_height, x_height - y_height[min_x_to_y_id.reshape(-1)]],
                -1,
            )
    else:
        raise ValueError(
            'Invalid direction type. Supported types: "y_x", "x_y", "bi"'
        )

    return height_err


def embedding_post_process(
    embedding: np.array,
    bin_seg: np.array,
    cluster_alg: str = "dbscan",
    band_width: float = 1.5,
    max_num_lane: int = None,
) -> np.array:
    """
    First use mean shift to find dense cluster center.

    Args:
        embedding: [H, W, embed_dim]
        bin_seg: [H, W], each pixel is 0 or 1, 0 for background pixel

    Returns:
        cluster_result: [H, W], index of different lanes on each pixel
    """
    cluster_result = np.zeros(bin_seg.shape, dtype=np.int32)
    cluster_list = embedding[bin_seg > 0]  # 64*64*2, 4
    if len(cluster_list) == 0:
        return cluster_result
    if cluster_alg == "mean_shift":
        alg = MeanShift(bandwidth=band_width, bin_seeding=True, n_jobs=-1)
    elif cluster_alg == "dbscan":
        alg = DBSCAN(eps=band_width, min_samples=1)
    else:
        raise NotImplementedError
    alg.fit(cluster_list)
    labels = alg.labels_
    cluster_result[bin_seg > 0] = labels + 1
    return cluster_result


def direct_decoder(cls, r, sin, cos, direction, max_bg_depth=3):
    x = -r * cos
    y = -r * sin
    offset = np.concatenate((x[..., np.newaxis], y[..., np.newaxis]), axis=-1)
    p, h, w, feat_n = direction.shape

    id_map = np.arange(p * h * w).reshape(p, h, w).astype(np.int32)

    def _get_inv_id(id):
        p0 = id // (h * w)
        res = id % (h * w)
        h0 = res // w
        w0 = res % w
        return (p0, h0, w0)

    w_coord, h_coord = np.meshgrid(
        np.linspace(0.5, w - 0.5, w), np.linspace(0.5, h - 0.5, h)
    )
    neighbor_num = feat_n // 2

    pred_neighbors = np.zeros((neighbor_num, p, h, w)).astype(np.int32)
    valid_ids = cls > 0
    for i in range(neighbor_num):
        direct = direction[:, :, :, i * 2 : i * 2 + 2]
        h_pred = np.expand_dims(h_coord, 0) + direct[:, :, :, 0]
        h_pred[h_pred < 0] = 0
        h_pred[h_pred > h - 1e-3] = h - 1e-3
        w_pred = np.expand_dims(w_coord, 0) + direct[:, :, :, 1]
        w_pred[w_pred < 0] = 0
        w_pred[w_pred > w - 1e-3] = w - 1e-3
        pred_ = np.stack([h_pred, w_pred], -1).reshape(-1, 2)
        h_id = (h_pred).astype(np.int32).reshape(-1, 1)
        w_id = (w_pred).astype(np.int32).reshape(-1, 1)
        pred_id = np.concatenate([h_id, w_id], -1)
        tmp_idx = np.stack([h_id, w_id])
        offset_hw = offset[:, h_id.squeeze(), w_id.squeeze(), :]
        dist = offset_hw + pred_id + 0.5 - pred_
        mask = np.logical_and(
            h_id
            == np.stack([h_coord, h_coord]).astype(np.int32).reshape(-1, 1),
            w_id
            == np.stack([w_coord, w_coord]).astype(np.int32).reshape(-1, 1),
        ).reshape(p, h, w)
        dist = np.sum(dist ** 2, axis=-1)

        # shape phw
        patch_id = np.argmin(
            np.where(
                valid_ids[:, h_id.reshape(-1), w_id.reshape(-1)], dist, 1000
            ),
            axis=0,
        )

        pred_neighbors[i] = id_map[
            patch_id, h_id.reshape(-1), w_id.reshape(-1)
        ].reshape(p, h, w)

        pred_neighbors[i, mask] = -1
    positive_ids = id_map[cls > 0]
    valid_cls = cls[cls > 0]
    positive_id_instance = np.zeros_like(positive_ids)
    is_visited = np.zeros_like(positive_ids)
    positive_id_mapping = {pid: _get_inv_id(pid) for pid in positive_ids}
    positive_id_index_mapping = {
        pid: pid_i for pid_i, pid in enumerate(positive_ids)
    }

    def visit(id, l_id, bg_depth=0):
        if id not in positive_ids:
            if bg_depth > max_bg_depth:
                return None
            bg_depth_current = bg_depth + 1
            tmp_id = None
            tmp_idx = _get_inv_id(id)
        else:
            bg_depth_current = 0
            tmp_id = positive_id_index_mapping[id]
            tmp_idx = positive_id_mapping[id]
        if tmp_id is not None:
            if is_visited[tmp_id]:
                positive_id_instance[
                    positive_id_instance == positive_id_instance[tmp_id]
                ] = l_id
            else:
                is_visited[tmp_id] = 1
                positive_id_instance[tmp_id] = l_id

                if pred_neighbors[0, tmp_idx[0], tmp_idx[1], tmp_idx[2]] > -1:
                    visit(
                        pred_neighbors[0, tmp_idx[0], tmp_idx[1], tmp_idx[2]],
                        l_id,
                        bg_depth_current,
                    )
                if pred_neighbors[1, tmp_idx[0], tmp_idx[1], tmp_idx[2]] > -1:
                    visit(
                        pred_neighbors[1, tmp_idx[0], tmp_idx[1], tmp_idx[2]],
                        l_id,
                        bg_depth_current,
                    )
        else:

            if pred_neighbors[0, tmp_idx[0], tmp_idx[1], tmp_idx[2]] > -1:
                visit(
                    pred_neighbors[0, tmp_idx[0], tmp_idx[1], tmp_idx[2]],
                    l_id,
                    bg_depth_current,
                )
            if pred_neighbors[1, tmp_idx[0], tmp_idx[1], tmp_idx[2]] > -1:
                visit(
                    pred_neighbors[1, tmp_idx[0], tmp_idx[1], tmp_idx[2]],
                    l_id,
                    bg_depth_current,
                )
        return None

    instance_id = 1
    for pid_i in range(len(positive_ids)):
        if is_visited[pid_i] > 0:
            continue
        visit(positive_ids[pid_i], instance_id, 0)
        instance_id += 1

    pred_instance = np.zeros((p, h, w)).astype(np.int32)

    label_ids = np.unique(positive_id_instance)
    import copy

    new_label = copy.deepcopy(positive_id_instance)
    label_idx = 1
    for _, label_i in enumerate(label_ids):
        class_ids = valid_cls[positive_id_instance == label_i]
        for cls_i in np.unique(class_ids):
            ind = np.logical_and(
                positive_id_instance == label_i, valid_cls == cls_i
            )
            new_label[ind] = label_idx
            label_idx = label_idx + 1
    for pid_i, pid in enumerate(positive_ids):
        tmp_idx = positive_id_mapping[pid]
        pred_instance[tmp_idx[0], tmp_idx[1], tmp_idx[2]] = new_label[pid_i]
    return pred_instance


def get_offset(
    pred_r: np.array, pred_sin: np.array, pred_cos: np.array, eps=1e-5
):
    pred_x = -pred_r * pred_cos
    pred_y = -pred_r * pred_sin
    pred_offset = np.concatenate(
        (pred_x[..., np.newaxis], pred_y[..., np.newaxis]), axis=-1
    )
    pred_direction = np.arctan(pred_sin / (pred_cos + eps)) / np.pi + 0.5
    return pred_offset, pred_direction


def convert_pred(
    pred_cls: torch.tensor,
    pred_instance: torch.tensor,
    pred_r: torch.tensor,
    pred_sin: torch.tensor,
    pred_cos: torch.tensor,
    cls_threshold: np.array,
    cls_loss_type: str,
    cluster_alg: str,
    cluster_bw: float,
    ogc_cluster_pose_weights: dict,
    cls_num: int,
    cluster_all: bool,
):
    if cls_loss_type == "ce_loss":
        pred_score = torch.softmax(pred_cls, dim=1)[:, 1:, ...]
        cls_num -= 1
    elif cls_loss_type == "set_loss":
        pred_a, pred_b = torch.chunk(pred_cls, 2, dim=1)
        pred_score = torch.cat(
            (torch.softmax(pred_a, dim=1), torch.softmax(pred_b, dim=1))
        )[:, 1:, ...]
        slice_a, slice_b = torch.chunk(pred_instance, 2, dim=1)
        pred_instance = torch.cat((slice_a, slice_b))  # 1,8,64,64 -> 2,4,64,64
        cls_num -= 1
    elif cls_loss_type == "focal_loss":
        pred_score = torch.sigmoid(pred_cls)
    else:
        raise NotImplementedError

    pred_prob, pred_cls = torch.max(pred_score, dim=1)
    pred_cls = pred_cls.to(torch.int32) + 1
    if cluster_alg == "ogc":
        pred_prob = pred_prob.cpu()
        pred_cls = pred_cls.cpu()
        pred_r = pred_r[0].cpu()
        pred_sin = pred_sin[0].cpu()
        pred_cos = pred_cos[0].cpu()
        pred_instance = pred_instance.permute((0, 2, 3, 1)).cpu()
    else:
        pred_prob = pred_prob.cpu().numpy()
        pred_cls = pred_cls.cpu().numpy()
        pred_r = pred_r[0].cpu().numpy()
        pred_sin = pred_sin[0].cpu().numpy()
        pred_cos = pred_cos[0].cpu().numpy()
        pred_instance = pred_instance.permute((0, 2, 3, 1)).cpu().numpy()

    if cluster_alg == "mean_shift":
        bin_seg = (pred_prob > cls_threshold).astype(np.int64)
        pred_cluster = embedding_post_process(
            pred_instance, bin_seg, cluster_alg="mean_shift"
        )
    elif cluster_alg == "dbscan":
        pos = pred_prob >= cls_threshold
        cls_labels = np.unique(pred_cls).tolist()
        cls_labels.sort()
        global_instance_id = 0
        pred_cluster = np.zeros_like(pred_cls)
        for cls_label in cls_labels:
            bin_seg = np.logical_and(pos, pred_cls == cls_label).astype(
                np.int64
            )
            sub_cluster = embedding_post_process(
                pred_instance, bin_seg, band_width=cluster_bw
            )
            for sub_cluster_id in np.unique(sub_cluster):
                ins = np.logical_and(bin_seg, sub_cluster == sub_cluster_id)
                if ins.sum() == 0:
                    continue
                global_instance_id += 1
                pred_cluster[ins] = global_instance_id
    elif cluster_alg == "ogc":
        if om_ogc is None:
            check_packages_available("horizon_plugin_pytorch>=0.16.3")

        # notice: cls_num means foreground class num
        pred_cluster = om_ogc(
            pred_cls,
            pred_prob,
            pred_r,
            pred_sin,
            pred_cos,
            pred_instance,
            ogc_cluster_pose_weights,
            cls_num,
            cls_thr=cls_threshold,
            radius_l=9,
            radius_t=2,
            min_num=1,
            cluster_thr=cluster_bw,
            merge=True,
        )
        pred_cls = pred_cls.numpy()
        pred_prob = pred_prob.numpy()
        pred_r = pred_r.numpy()
        pred_sin = pred_sin.numpy()
        pred_cos = pred_cos.numpy()
        pred_instance = pred_instance.numpy()
        pred_cluster = pred_cluster.numpy()

    pred_cluster_all = None
    if cluster_all:
        pred_cluster_all = embedding_post_process(
            pred_instance, np.ones_like(pred_prob), cluster_alg="mean_shift"
        )

    return (
        pred_cls,
        pred_prob,
        pred_cluster,
        pred_r,
        pred_sin,
        pred_cos,
        pred_cluster_all,
        pred_instance,
    )


def convert_gt(
    label_cls: torch.tensor,
    label_instance: torch.tensor,
    label_r: torch.tensor,
    label_sin: torch.tensor,
    label_cos: torch.tensor,
):
    label_cls = label_cls.cpu().numpy()
    label_prob = (label_cls > 0).astype(np.int64)
    label_instance = label_instance.cpu().numpy()
    label_r = label_r.cpu().numpy()
    label_sin = label_sin.cpu().numpy()
    label_cos = label_cos.cpu().numpy()
    return label_cls, label_prob, label_instance, label_r, label_sin, label_cos


def get_online_mappting_dict(lanes, cls_id_2_category, fill_crosswalk):
    # return x_vcs, y_vcs, cls, prob, sin, cos
    online_mapping_dict = {}
    for lane in lanes:
        if len(lane) <= 2:
            continue
        lane = np.array(lane)
        counts = np.bincount(lane[:, 4].astype(np.int64))
        cls_id = np.argmax(counts)
        category = cls_id_2_category[cls_id]
        if fill_crosswalk and category == "crosswalks":
            lane = np.hstack([lane[:, 2:4], lane[:, 4:]])
        else:
            lane = np.hstack([lane[:, :2], lane[:, 4:]])
        if category not in online_mapping_dict:
            online_mapping_dict.update({category: []})
        online_mapping_dict[category].append(lane)
    return online_mapping_dict


def get_seq_lane(lane, meter_per_out_pixel):
    """Get sequence lane."""
    step = meter_per_out_pixel
    near_threshold = step * 0.8
    min_dist_threshold = step
    mask = np.zeros(len(lane))
    seq_ids = []
    last_direction = None
    current_pt_id = 0
    mask[current_pt_id] = 1
    seq_ids.append(current_pt_id)
    current_pt = lane[current_pt_id, :2]
    valid_idx = np.where(mask == 0)[0]
    dist = np.linalg.norm(lane[:, :2] - current_pt, axis=1)
    mask[valid_idx[dist[valid_idx] < near_threshold]] = 1
    while np.sum(mask) < len(lane):
        pt = lane[current_pt_id, :]
        target_pt_plus = (pt[0] - step * pt[6], pt[1] + step * pt[7])
        target_pt_minus = (pt[0] + step * pt[6], pt[1] - step * pt[7])
        valid_idx = np.where(mask == 0)[0]
        dist_plus = np.linalg.norm(lane[:, :2] - target_pt_plus, axis=1)
        min_dist_plus = dist_plus[valid_idx].min()
        dist_minus = np.linalg.norm(lane[:, :2] - target_pt_minus, axis=1)
        min_dist_minus = dist_minus[valid_idx].min()
        if min_dist_plus <= min_dist_minus:
            dist = dist_plus
            min_dist = min_dist_plus
            direction = 1
        else:
            dist = dist_minus
            min_dist = min_dist_minus
            direction = -1
        if min_dist <= min_dist_threshold or last_direction is None:
            current_pt_id = valid_idx[dist[valid_idx].argmin()]
            last_direction = lane[current_pt_id, 6:8] * direction
        else:
            target_pt = (
                pt[0] - step * last_direction[0],
                pt[1] + step * last_direction[1],
            )
            dist = np.linalg.norm(lane[:, :2] - target_pt, axis=1)
            min_dist = dist[valid_idx].min()
            current_pt_id = valid_idx[dist[valid_idx].argmin()]
            last_direction = None
        mask[current_pt_id] = 1
        seq_ids.append(current_pt_id)
        current_pt = lane[current_pt_id, :2]
        valid_idx = np.where(mask == 0)[0]
        dist = np.linalg.norm(lane[:, :2] - current_pt, axis=1)
        mask[valid_idx[dist[valid_idx] < near_threshold]] = 1
    return lane[seq_ids, :]


def get_vcs_lane(
    cls: np.array,
    prob: np.array,
    instance: np.array,
    offset: np.array,
    sin: np.array,
    cos: np.array,
    embedding: np.array,
    out_h: int,
    out_w: int,
    top: float,
    left: float,
    meter_per_out_pixel: float,
    target_categorys: dict,
    fill_crosswalk: bool = False,
    cls_threshold: float = 0.5,
    nms: bool = False,
    add_height: bool = False,
    height: np.array = None,
):
    instance_id_dict = {
        instance_id: i for i, instance_id in enumerate(np.unique(instance))
    }
    num_pred_lane = len(instance_id_dict)
    lanes = [[] for _ in range(num_pred_lane)]
    for h in np.arange(0, out_h):
        for w in np.arange(0, out_w):
            for ch in np.arange(0, cls.shape[0]):
                if not (cls[ch, h, w] > 0 and prob[ch, h, w] >= cls_threshold):
                    continue
                instance_id = instance_id_dict[instance[ch, h, w]]
                x_vcs_origin = (
                    top - h * meter_per_out_pixel - meter_per_out_pixel / 2
                )
                y_vcs_origin = (
                    left - w * meter_per_out_pixel - meter_per_out_pixel / 2
                )
                x_vcs = (
                    x_vcs_origin + offset[ch, h, w, 0] * meter_per_out_pixel
                )
                y_vcs = (
                    y_vcs_origin + offset[ch, h, w, 1] * meter_per_out_pixel
                )
                pt = [
                    x_vcs,
                    y_vcs,
                    x_vcs_origin,
                    y_vcs_origin,
                    cls[ch, h, w],
                    prob[ch, h, w],
                    sin[ch, h, w],
                    cos[ch, h, w],
                ]
                if embedding is not None:
                    pt = pt + embedding[ch, h, w, :].tolist()
                pt = pt + [h, w, np.linalg.norm(offset[ch, h, w, :])]
                if add_height:
                    pt = pt + [height[ch, h, w]]
                lanes[instance_id].append(pt)

    if nms:
        assert not add_height, "current nms not support height"
        prob_with_r = False
        nms_lanes = []
        for lane in lanes:
            if len(lane) == 0:
                continue
            nms_lane = []
            bev = np.zeros((out_h, out_w))
            for pt in lane:
                bev[pt[-3], pt[-2]] = (
                    pt[5] * (1 - pt[-1]) if prob_with_r else pt[5]
                )
            bev_tensor = torch.tensor(bev).unsqueeze(0)
            mp_vertical = (
                torch.nn.functional.max_pool2d(
                    bev_tensor,
                    kernel_size=(1, 3),
                    stride=(1, 1),
                    padding=(0, 1),
                )
                .numpy()
                .squeeze(0)
            )
            mp_horizon = (
                torch.nn.functional.max_pool2d(
                    bev_tensor,
                    kernel_size=(3, 1),
                    stride=(1, 1),
                    padding=(1, 0),
                )
                .numpy()
                .squeeze(0)
            )
            ap_vertical = (
                torch.nn.functional.avg_pool2d(
                    bev_tensor,
                    kernel_size=(3, 5),
                    stride=(1, 1),
                    padding=(1, 2),
                )
                .numpy()
                .squeeze(0)
            )
            ap_horizon = (
                torch.nn.functional.avg_pool2d(
                    bev_tensor,
                    kernel_size=(5, 3),
                    stride=(1, 1),
                    padding=(2, 1),
                )
                .numpy()
                .squeeze(0)
            )
            for pt in lane:
                w = pt[-2]
                h = pt[-3]
                if ap_horizon[h, w] > ap_vertical[h, w]:
                    if mp_vertical[h, w] == (
                        pt[5] * (1 - pt[-1]) if prob_with_r else pt[5]
                    ):
                        nms_lane.append(pt)
                else:
                    if mp_horizon[h, w] == (
                        pt[5] * (1 - pt[-1]) if prob_with_r else pt[5]
                    ):
                        nms_lane.append(pt)
            nms_lanes.append(nms_lane)
        lanes = nms_lanes

    # sort lanes
    sorted_lanes = []
    for lane in lanes:
        if len(lane) == 0:
            continue
        lane = np.array(lane)
        lane_x_vcs_list = lane[:, 0]
        lane_y_vcs_list = lane[:, 1]
        lane_x_vcs_min = np.min(lane_x_vcs_list)
        lane_x_vcs_max = np.max(lane_x_vcs_list)
        lane_y_vcs_min = np.min(lane_y_vcs_list)
        lane_y_vcs_max = np.max(lane_y_vcs_list)
        lane_y_vcs_range = lane_y_vcs_max - lane_y_vcs_min
        lane_x_vcs_range = lane_x_vcs_max - lane_x_vcs_min
        if lane_x_vcs_range >= lane_y_vcs_range:
            sorted_lane = lane[lane[:, 0].argsort()]
        else:
            sorted_lane = lane[lane[:, 1].argsort()]
        sorted_lanes.append(sorted_lane)

    # sequence lanes
    seq_lanes = []
    for lane in sorted_lanes:
        seq_lane = get_seq_lane(lane, meter_per_out_pixel)
        seq_lanes.append(seq_lane)

    cls_id_2_category = {v: k for k, v in target_categorys.items()}
    online_mapping_dict = get_online_mappting_dict(
        sorted_lanes, cls_id_2_category, fill_crosswalk
    )
    online_mapping_dict_seq = get_online_mappting_dict(
        seq_lanes, cls_id_2_category, fill_crosswalk
    )

    return online_mapping_dict, online_mapping_dict_seq


def get_smoothing(online_mapping: dict, deg: int = 3):
    result_dict = {}
    for (category, edges) in online_mapping.items():
        result_dict.update({category: []})
        lanes = [np.array(lane) for lane in edges]
        for lane in lanes:
            if len(lane) < deg:
                continue
            xs_vcs = lane[:, 0]
            ys_vcs = lane[:, 1]
            lane_x_vcs_min = np.min(xs_vcs)
            lane_x_vcs_max = np.max(xs_vcs)
            lane_y_vcs_min = np.min(ys_vcs)
            lane_y_vcs_max = np.max(ys_vcs)
            x_vcs_diff = lane_x_vcs_max - lane_x_vcs_min
            y_vcs_diff = lane_y_vcs_max - lane_y_vcs_min
            if x_vcs_diff >= y_vcs_diff:
                start_idx = np.where(xs_vcs == lane_x_vcs_min)[0][0]
            else:
                start_idx = np.where(ys_vcs == lane_y_vcs_min)[0][0]

            start_vcs_pt = lane[start_idx]
            distance = np.linalg.norm(lane - start_vcs_pt, axis=1)
            lane_sort = lane[np.argsort(distance)]
            px = np.polyfit(distance, lane_sort[:, 0], deg=deg)
            xs_px = np.polyval(px, distance)
            py = np.polyfit(distance, lane_sort[:, 1], deg=deg)
            ys_py = np.polyval(py, distance)
            lane_smoothing = np.vstack([xs_px, ys_py]).transpose()
            result_dict[category].append(lane_smoothing)
    return result_dict


def in_region(pt_vcs, region=(72.4, -30, 51.2, -51.2)):
    top, bottom, left, right = region
    x_vcs = pt_vcs[0]
    y_vcs = pt_vcs[1]
    if x_vcs > top or x_vcs < bottom or y_vcs > left or y_vcs < right:
        return False
    return True


def get_poly_lanes(online_mapping_gt_origin, keep_height=False):
    online_mapping_gt = online_mapping_gt_origin.copy()
    height_offset = 1 if keep_height else 0
    for category, lanes in online_mapping_gt_origin.items():
        lanes_poly = [
            np.concatenate(
                [
                    lane[:, : 2 + height_offset],
                    lane[-1:, 3 : 5 + height_offset],
                ],
                axis=0,
            )
            for lane in lanes
        ]
        online_mapping_gt[category] = lanes_poly  # 3类 ，每类几条线，一条线几个点
    return online_mapping_gt


def get_navnet(
    gt_online_mapping_dir: str,
    target_categorys: dict,
    region: tuple = None,
    poly: bool = False,
):
    online_mapping = {}
    if not os.path.exists(gt_online_mapping_dir):
        return online_mapping
    for category in [
        i.split(".")[0]
        for i in os.listdir(gt_online_mapping_dir)
        if i.endswith("txt")
    ]:
        if target_categorys and category not in target_categorys:
            continue
        navnetmap_file = os.path.join(
            gt_online_mapping_dir, "{}.txt".format(category)
        )
        assert os.path.exists(navnetmap_file)
        if category == "crosswalks":
            navnetmap = load_navnetmap(navnetmap_file, region=None, poly=poly)
            navnetmap = [
                np.vstack([lane, np.hstack([lane[-1, -3:], lane[0, :3]])])
                for lane in navnetmap
            ]
        else:
            navnetmap = load_navnetmap(
                navnetmap_file, region=region, poly=poly
            )
        if navnetmap:
            online_mapping[category] = navnetmap
    return online_mapping


def get_label(
    lane,
    cls_label,
    instance_label,
    out_h,
    out_w,
    top,
    left,
    meter_per_out_pixel,
    meter_step,
    n_min_points,
    mat_vcs2out,
    with_offset=True,
):
    gt_online_mapping_cls = np.zeros((out_h, out_w), dtype=np.uint8)
    gt_online_mapping_instance = np.zeros((out_h, out_w), dtype=np.uint8)
    gt_online_mapping_r = np.zeros((out_h, out_w))
    gt_online_mapping_sin = np.zeros((out_h, out_w))
    gt_online_mapping_cos = np.zeros((out_h, out_w))
    if with_offset:
        lane_x_vcs_min = np.min(lane[:, 0])
        lane_x_vcs_max = np.max(lane[:, 0])
        lane_y_vcs_min = np.min(lane[:, 1])
        lane_y_vcs_max = np.max(lane[:, 1])
        f_linear_x = interp1d(lane[:, 0], lane[:, 1], bounds_error=False)
        f_linear_y = interp1d(lane[:, 1], lane[:, 0], bounds_error=False)
        for h_origin in np.arange(0, out_h):
            for w_origin in np.arange(0, out_w):
                x_vcs_max = top - h_origin * meter_per_out_pixel
                x_vcs_min = x_vcs_max - meter_per_out_pixel
                y_vcs_max = left - w_origin * meter_per_out_pixel
                y_vcs_min = y_vcs_max - meter_per_out_pixel
                x_vcs_origin = x_vcs_max - meter_per_out_pixel / 2
                y_vcs_origin = y_vcs_max - meter_per_out_pixel / 2

                if (
                    x_vcs_max < lane_x_vcs_min
                    or x_vcs_min > lane_x_vcs_max
                    or y_vcs_max < lane_y_vcs_min
                    or y_vcs_min > lane_y_vcs_max
                ):
                    continue

                xs = np.arange(x_vcs_min, x_vcs_max, meter_step)
                ys = f_linear_x(xs)
                tmp = np.vstack([xs, ys]).transpose()
                tmp = tmp[~np.isnan(tmp[:, 1]), :]
                f_linear_x_pt_vcs_list = [
                    pt for pt in tmp if y_vcs_min <= pt[1] <= y_vcs_max
                ]

                ys = np.arange(y_vcs_min, y_vcs_max, meter_step)
                xs = f_linear_y(ys)
                tmp = np.vstack([xs, ys]).transpose()
                tmp = tmp[~np.isnan(tmp[:, 0]), :]
                f_linear_y_pt_vcs_list = [
                    pt for pt in tmp if x_vcs_min <= pt[0] <= x_vcs_max
                ]

                if (
                    max(
                        len(f_linear_x_pt_vcs_list),
                        len(f_linear_y_pt_vcs_list),
                    )
                    >= n_min_points
                ):
                    if len(f_linear_x_pt_vcs_list) >= len(
                        f_linear_y_pt_vcs_list
                    ):
                        x_fit = [
                            pt[0] - x_vcs_origin
                            for pt in f_linear_x_pt_vcs_list
                        ]
                        y_fit = [
                            pt[1] - y_vcs_origin
                            for pt in f_linear_x_pt_vcs_list
                        ]
                        p1 = np.polyfit(x_fit, y_fit, 1)
                        a, b, c = p1[0], -1, p1[1]
                    else:
                        x_fit = [
                            pt[0] - x_vcs_origin
                            for pt in f_linear_y_pt_vcs_list
                        ]
                        y_fit = [
                            pt[1] - y_vcs_origin
                            for pt in f_linear_y_pt_vcs_list
                        ]
                        p1 = np.polyfit(y_fit, x_fit, 1)
                        a, b, c = -1, p1[0], p1[1]
                    distance = abs(c) / (math.sqrt(a * a + b * b))
                    sin = b / (math.sqrt(a * a + b * b)) * math.copysign(1, c)
                    cos = a / (math.sqrt(a * a + b * b)) * math.copysign(1, c)
                    r = distance / meter_per_out_pixel
                    gt_online_mapping_cls[h_origin, w_origin] = cls_label
                    gt_online_mapping_instance[
                        h_origin, w_origin
                    ] = instance_label
                    gt_online_mapping_r[h_origin, w_origin] = r
                    gt_online_mapping_sin[h_origin, w_origin] = sin
                    gt_online_mapping_cos[h_origin, w_origin] = cos
    else:
        bev_pts = get_bev_pts(lane, mat_vcs2out)
        gt_online_mapping_cls = cv2.polylines(
            gt_online_mapping_cls,
            np.int32([bev_pts]),
            isClosed=False,
            color=cls_label,
            thickness=1,
        )
        gt_online_mapping_instance = cv2.polylines(
            gt_online_mapping_instance,
            np.int32([bev_pts]),
            isClosed=False,
            color=instance_label,
            thickness=1,
        )

    return (
        gt_online_mapping_cls,
        gt_online_mapping_instance,
        gt_online_mapping_r,
        gt_online_mapping_sin,
        gt_online_mapping_cos,
    )


def get_dir_vector(curve_line):
    start, end = curve_line[0], curve_line[-1]
    return end - start


def curve_angle(curve_line1, curve_line2):
    dir_vec1, dir_vec2 = get_dir_vector(curve_line1), get_dir_vector(
        curve_line2
    )
    costheta = np.dot(dir_vec1, dir_vec2) / (
        np.linalg.norm(dir_vec1) * np.linalg.norm(dir_vec2)
    )
    return costheta


def is_intersected(line1, line2):
    """
    Judge two lines are intersected.

    Args:
        line1: list of points.
        line2: list of points.
    """
    # a, b, c, d: point, [x, y]
    a, b = line1[0], line1[-1]
    c, d = line2[0], line2[-1]
    ac, ad = c - a, d - a
    bc, bd = c - b, d - b
    return (
        (a[0] in line2 and a[1] in line2)
        or (b[0] in line2 and b[1] in line2)
        or (c[0] in line1 and c[1] in line1)
        or (d[0] in line1 and d[1] in line1)
        or np.linalg.norm(bc, ord=2) < 1.6
        or np.linalg.norm(ad, ord=2) < 1.6
        or np.linalg.norm(ac, ord=2) < 1.6
        or np.linalg.norm(bd, ord=2) < 1.6
    )


def distance_point_to_curve_ends(pt, curve):
    return min(np.linalg.norm(curve[0] - pt), np.linalg.norm(curve[-1] - pt))


def merge_lines(online_mapping_gt):
    merged_mapping_gt = {}
    for component_type, curve_lines in online_mapping_gt.items():
        if component_type == "crosswalks":
            merged_mapping_gt[component_type] = curve_lines
            continue

        curve_lines = sorted(curve_lines, key=lambda r: -len(r))
        adj_mat = np.zeros((len(curve_lines), len(curve_lines)))
        for i, line1 in enumerate(curve_lines):
            for j, line2 in enumerate(curve_lines[i + 1 :]):
                adj_mat[i, i + 1 + j] = is_intersected(line1, line2)

        head_idxes = np.where(adj_mat.sum(axis=1) > 0)[0]
        tail_idxes = []
        tail_to_head = {}
        for head in head_idxes[::-1]:
            tails = np.where(adj_mat[head] == 1)[0]
            for tail in tails[::-1]:
                if tail_to_head.get(tail, None) in tails:
                    continue
                elif tail in tail_idxes:
                    tail = tail_to_head[tail]
                    if tail in tail_to_head.keys():
                        # TODO.yy. check whether a line merge with 3 lines.
                        tail = tail_to_head[tail]

                if len(curve_lines[tail]) > len(curve_lines[head]):
                    head, tail = tail, head
                dis_head = distance_point_to_curve_ends(
                    curve_lines[head][0], curve_lines[tail]
                )
                dis_tail = distance_point_to_curve_ends(
                    curve_lines[head][-1], curve_lines[tail]
                )
                if dis_head > dis_tail:
                    merged = np.vstack((curve_lines[head], curve_lines[tail]))
                else:
                    merged = np.vstack((curve_lines[tail], curve_lines[head]))

                _, indexes = np.unique(merged, axis=0, return_index=True)
                curve_lines[head] = np.array(
                    [merged[index] for index in sorted(indexes)]
                )

                tail_idxes.append(tail)
                tail_to_head[tail] = head

        # pop useless(tail) curves.
        for tail in sorted(set(tail_idxes))[::-1]:
            curve_lines.pop(tail)
        merged_mapping_gt[component_type] = sorted(
            curve_lines, key=lambda r: len(r)
        )
    return merged_mapping_gt


def resample_laneline_in_x(input_lane, x_steps, out_vis=False):
    # at least two points are included
    assert input_lane.shape[0] >= 2

    x_min = np.min(input_lane[:, 0]) - 5
    x_max = np.max(input_lane[:, 0]) + 5

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate(
            [input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)],
            axis=1,
        )

    f_y = interp1d(
        input_lane[:, 0], input_lane[:, 1], fill_value="extrapolate"
    )
    f_z = interp1d(
        input_lane[:, 0], input_lane[:, 2], fill_value="extrapolate"
    )

    y_values = f_y(x_steps)
    z_values = f_z(x_steps)

    if out_vis:
        output_visibility = np.logical_and(x_steps >= x_min, x_steps <= x_max)
        return y_values, z_values, output_visibility.astype(np.float32) + 1e-9
    return y_values, z_values


def get_delete_nearby_lines_(
    online_mapping_gt, x_min, y_min, x_max, y_max, num_samples_x, x_samples
):
    if not (
        "solid_lanes" in online_mapping_gt and "roadedges" in online_mapping_gt
    ):
        return online_mapping_gt

    dist_th = 1.5
    ratio_th = 0.25
    mean_dist_th = 1.5

    for (category, gt_lanes) in online_mapping_gt.items():
        # sort by x_vcs
        gt_lanes = [lane[lane[:, 0].argsort()] for lane in gt_lanes]
        cnt_gt = len(gt_lanes)
        gt_visibility_mat = np.zeros((cnt_gt, num_samples_x))
        for i in range(cnt_gt):
            x_min = np.min(np.array(gt_lanes[i])[:, 0])
            x_max = np.max(np.array(gt_lanes[i])[:, 0])
            y_values, z_values, visibility_vec = resample_laneline_in_x(
                np.array(gt_lanes[i]), x_samples, out_vis=True
            )
            gt_lanes[i] = np.vstack([y_values, z_values]).T
            gt_visibility_mat[i, :] = np.logical_and(
                y_values >= y_min,
                np.logical_and(
                    y_values <= y_max,
                    np.logical_and(x_samples >= x_min, x_samples <= x_max),
                ),
            )
            gt_visibility_mat[i, :] = np.logical_and(
                gt_visibility_mat[i, :], visibility_vec
            )
        if category == "solid_lanes":
            solid_lanes = gt_lanes
            solid_lanes_visibility_mat = gt_visibility_mat
        if category == "roadedges":
            roadedges = gt_lanes
            roadedges_visibility_mat = gt_visibility_mat

    mask_solid_lanes = np.ones(len(solid_lanes))
    for i in range(len(solid_lanes)):
        for j in range(len(roadedges)):
            y_dist = np.abs(solid_lanes[i][:, 0] - roadedges[j][:, 0])
            euclidean_dist = np.sqrt(y_dist ** 2)
            euclidean_dist[
                np.logical_or(
                    solid_lanes_visibility_mat[i, :] < 0.5,
                    roadedges_visibility_mat[j, :] < 0.5,
                )
            ] = dist_th
            num_match = np.sum(euclidean_dist < dist_th)
            ratio = num_match / np.sum(solid_lanes_visibility_mat[i, :])
            mean_dist = np.mean(
                euclidean_dist[solid_lanes_visibility_mat[i, :].astype(bool)]
            )
            if ratio >= ratio_th and mean_dist <= mean_dist_th:
                mask_solid_lanes[i] = 0
                break
    online_mapping_gt["solid_lanes"] = [
        solid_lane
        for (solid_lane, mask_solid_lane) in zip(
            online_mapping_gt["solid_lanes"], mask_solid_lanes
        )
        if mask_solid_lane
    ]
    return online_mapping_gt


def get_delete_nearby_lines(online_mapping_gt, x_min, y_min, x_max, y_max):
    num_samples_x = int(x_max - x_min)
    x_samples = np.linspace(
        int(x_min), int(x_max), num=num_samples_x, endpoint=False
    )
    num_samples_y = int(y_max - y_min)
    y_samples = np.linspace(
        int(y_min), int(y_max), num=num_samples_y, endpoint=False
    )

    online_mapping_gt = get_delete_nearby_lines_(
        online_mapping_gt, x_min, y_min, x_max, y_max, num_samples_x, x_samples
    )

    for (category, gt_lanes) in online_mapping_gt.items():
        online_mapping_gt[category] = [lane[:, ::-1] for lane in gt_lanes]
    online_mapping_gt = get_delete_nearby_lines_(
        online_mapping_gt, y_min, x_min, y_max, x_max, num_samples_y, y_samples
    )
    for (category, gt_lanes) in online_mapping_gt.items():
        online_mapping_gt[category] = [lane[:, ::-1] for lane in gt_lanes]
    return online_mapping_gt


def get_direction_from_id(
    gt_online_mapping_instance,
    gt_online_mapping_offset,
    gt_online_mapping_directions,
    neighborhood=8,
    num_directions=2,
):
    assert neighborhood in [4, 8]
    if neighborhood == 4:
        h_offsets = [-1, 0, 0, 1]
        w_offsets = [0, -1, 1, 0]
    elif neighborhood == 8:
        h_offsets = [-1, -1, -1, 0, 0, 1, 1, 1]
        w_offsets = [-1, 0, 1, -1, 1, -1, 0, 1]
    else:
        raise NotImplementedError

    patches, h, w = gt_online_mapping_instance.shape
    neighbor_ids = np.zeros((patches, num_directions, h, w))
    lane_ids = np.unique(gt_online_mapping_instance)
    lane_ids = lane_ids[lane_ids > 0]
    helper_occ = np.zeros((h + 2, w + 2))
    helper_offset = np.zeros((h + 2, w + 2, 2))
    for id in lane_ids:
        lane_indicator = gt_online_mapping_instance == id
        occupancy = np.sum(lane_indicator, axis=0)
        patch_id = np.argmax(lane_indicator, axis=0)
        helper_occ[1:-1, 1:-1] = occupancy
        tmp_offset = np.zeros((h, w, 2))
        for patch_i in range(patches):
            tmp_offset[patch_id == patch_i] = gt_online_mapping_offset[
                patch_i
            ][patch_id == patch_i]
        helper_offset[1:-1, 1:-1] = tmp_offset
        for n_i in range(neighborhood):
            is_neighbored = np.logical_and(
                occupancy
                == helper_occ[
                    1 + h_offsets[n_i] : 1 + h_offsets[n_i] + h,
                    1 + w_offsets[n_i] : 1 + w_offsets[n_i] + w,
                ],
                occupancy,
            )
            if np.sum(is_neighbored) < 0.5:
                continue
            # offsets: arr(p,t) + arr(cur_p,p) = arr(cur_p, t)
            offsets = (
                helper_offset[
                    1 + h_offsets[n_i] : 1 + h_offsets[n_i] + h,
                    1 + w_offsets[n_i] : 1 + w_offsets[n_i] + w,
                ]
                + np.array([h_offsets[n_i], w_offsets[n_i]])
            )
            for d_i in range(num_directions):
                for p_i in range(patches):
                    valid = np.logical_and(
                        neighbor_ids[p_i, d_i] == 0,
                        np.logical_and(patch_id == p_i, is_neighbored),
                    )
                    neighbor_ids[p_i, d_i, valid] = n_i + 1
                    gt_online_mapping_directions[
                        p_i, valid, d_i * 2 : d_i * 2 + 2
                    ] = offsets[valid]
                    is_neighbored[valid] = 0
    return gt_online_mapping_directions


def smooth_direction(
    directions,
    gt_online_mapping_instance,
    gt_online_mapping_offset,
    neighbor_range=1,
):
    # directions: n_patch, h, w, 4; arr(p, n_t)
    occupancy = gt_online_mapping_instance > 0
    neighbor_offsets = range(-neighbor_range, neighbor_range + 1)
    p, h, w, c = directions.shape
    new_directions = np.zeros_like(directions)
    ww, hh = np.meshgrid(neighbor_offsets, neighbor_offsets)
    grids = np.stack([hh, ww])
    weight_map = np.zeros((p, h, w))

    lane_ids = np.unique(gt_online_mapping_instance)
    for lane_id in lane_ids:
        line_weight_map = np.zeros((p, h, w))
        tmp_direction = np.zeros((p, h, w, c))
        if lane_id == 0:
            continue
        occupancy = gt_online_mapping_instance == lane_id
        for p_i, h_i, w_i in zip(*np.where(occupancy)):
            current_offset = (
                gt_online_mapping_offset[p_i, h_i, w_i].reshape(-1, 1, 1)
                - grids
            )
            loc_direction = directions[p_i, h_i, w_i]

            cur_direction = np.concatenate(
                [
                    loc_direction[i * 2 : i * 2 + 2].reshape(-1, 1, 1) - grids
                    for i in range(c // 2)
                ],
                0,
            ).transpose(1, 2, 0)
            weight = 1.0 / (np.sum(current_offset ** 2, axis=0) + 1e-3)
            draw_heatmap(
                line_weight_map[p_i],
                weight,
                [w_i, h_i],
                [tmp_direction[p_i]],
                [cur_direction],
            )

        valid_mask = line_weight_map > weight_map
        tmp = tmp_direction[valid_mask]
        tmp_direction[valid_mask] = new_directions[valid_mask]
        new_directions[valid_mask] = tmp
        tmp = line_weight_map[valid_mask]
        line_weight_map[valid_mask] = weight_map[valid_mask]
        weight_map[valid_mask] = tmp
        # for cur_p in range(p):
        #     valid_mask = line_weight_map > weight_map[cur_p]
        #     tmp = tmp_direction[valid_mask]
        #     tmp_direction[valid_mask] = new_directions[cur_p, valid_mask]
        #     new_directions[cur_p, valid_mask] = tmp
        #     tmp = line_weight_map[valid_mask]
        #     line_weight_map[valid_mask] = weight_map[cur_p, valid_mask]
        #     weight_map[cur_p, valid_mask] = tmp

    return new_directions


def out_region(start, end, region):
    """Check if a line segment is out of the region.

    Args:
        start: the start point of the line segment.
        end: the end point of the line segment.
        region: target region [top, bottom, left, right].
    """
    top, bottom, left, right = region
    x_vcs_start, x_vcs_end = start[0], end[0]
    y_vcs_start, y_vcs_end = start[1], end[1]
    if (
        (x_vcs_start > top and x_vcs_end > top)
        or (x_vcs_start < bottom and x_vcs_end < bottom)
        or (y_vcs_start > left and y_vcs_end > left)
        or (y_vcs_start < right and y_vcs_end < right)
    ):
        return True
    return False


def load_navnetmap(navnetmap_file, region=None, poly=True):
    """Load navnet map txt."""
    with open(navnetmap_file, "r") as fin:
        lanes = []
        new_line = True
        start_vertices = []
        end_vertices = []
        for line in fin.readlines():
            if "p1:" in line:
                new_line = False
                start_vertices.append(
                    [float(x.strip()) for x in line.split(" ")[1:]]
                )
            elif "p2:" in line:
                new_line = False
                end_vertices.append(
                    [float(x.strip()) for x in line.split(" ")[1:]]
                )
            elif ":" in line and not new_line:
                if region is None:
                    if len(start_vertices) >= 1:
                        lanes.append(np.hstack((start_vertices, end_vertices)))
                else:
                    start_vertices_ = []
                    end_vertices_ = []
                    for start_vertice, end_vertice in zip(
                        start_vertices, end_vertices
                    ):
                        if out_region(start_vertice, end_vertice, region):
                            continue
                        start_vertices_.append(start_vertice)
                        end_vertices_.append(end_vertice)
                    if len(start_vertices_) >= 1:
                        lanes.append(
                            np.hstack((start_vertices_, end_vertices_))
                        )
                start_vertices = []
                end_vertices = []
                new_line = True
        if not new_line:
            if region is None:
                if len(start_vertices) >= 1:
                    lanes.append(np.hstack((start_vertices, end_vertices)))
            else:
                start_vertices_ = []
                end_vertices_ = []
                for start_vertice, end_vertice in zip(
                    start_vertices, end_vertices
                ):
                    if out_region(start_vertice, end_vertice, region):
                        continue
                    start_vertices_.append(start_vertice)
                    end_vertices_.append(end_vertice)
                if len(start_vertices_) >= 1:
                    lanes.append(np.hstack((start_vertices_, end_vertices_)))

        if poly:
            lanes = [
                np.concatenate([lane[:, :2], lane[-1:, 3:5]], axis=0)
                for lane in lanes
            ]
    return lanes


def get_dist_pt2lineseg(
    p: np.array, start_point: np.array, end_point: np.array
):
    """Compute the distance between a point and multi line segments.

    Args:
        p: with shape[1, 2], the candinate point.
        start_point: with shape [N, 2], the start point of line segments.
        end_point: with shape [N, 2], the end point of line segments.
    """

    ab = end_point - start_point
    pa = start_point - p
    pb = end_point - p
    bp = p - end_point
    # compute segments length
    d_ab = (np.hypot(ab[..., 0], ab[..., 1])[..., None]) + 1e-8
    d_pa = (np.hypot(pa[..., 0], pa[..., 1])[..., None]) + 1e-8
    d_pb = (np.hypot(pb[..., 0], pb[..., 1])[..., None]) + 1e-8
    # normalize segments vector to unit vector
    direction_ab = np.divide(ab, d_ab)
    direction_pa = np.divide(pa, d_pa)
    direction_pb = np.divide(pb, d_pb)
    # calculate the vertical position from the point to the line segment
    s = np.multiply(pa, direction_ab).sum(axis=-1)
    t = np.multiply(bp, direction_ab).sum(axis=-1)
    h = np.maximum.reduce([s, t, np.zeros(s.shape)])
    # calculate the distance from the point to the line
    c = pa[..., 0] * direction_ab[..., 1] - pa[..., 1] * direction_ab[..., 0]
    if len(p) == 1:
        direction_pc = (
            np.array([[0, 1], [-1, 0]]).dot(direction_ab.transpose())
            * np.sign(c)
        ).transpose()
    else:
        unit_vector = np.tile(np.array([[0, 1], [-1, 0]]), (len(p), 1, 1))
        direction_pc = (
            unit_vector.dot(direction_ab.squeeze().transpose()).transpose(
                0, 2, 1
            )
            * np.sign(c)[..., None]
        )
    r = np.hypot(h, c)
    direction = direction_pc
    direction[s > 0] = direction_pa[s > 0]
    direction[t > 0] = direction_pb[t > 0]
    return r, direction


def rect_contain(rect, pt):
    """Check if a point is in a rectangle.

    Args:
        rect: [x_min, y_min, x_max, y_max]
        pt: [x, y]
    """
    return (rect[0] < pt[0] < rect[2]) and (rect[1] < pt[1] < rect[3])


def line_line_intersect(p1, p2, p3, p4):
    """Compute the intersection of two line segments.

    Args:
        p1, p2: one line segment
        p3, p4: another line segment
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:  # parallel
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:  # out of range
        return None
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:  # out of range
        return None
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return [x, y]


def rect_line_intersect(rect, start_pt, end_pt):
    """Compute the intersection between a line segment and a rectangle.

    Args:
        rect: [x_min, y_min, x_max, y_max]
        start_pt: [x, y]
        end_pt: [x, y]
    """
    x_min, y_min, x_max, y_max = rect
    rect_line_segments = [
        [[x_min, y_min], [x_max, y_min]],
        [[x_max, y_min], [x_max, y_max]],
        [[x_max, y_max], [x_min, y_max]],
        [[x_min, y_max], [x_min, y_min]],
    ]
    points = []
    for segment in rect_line_segments:
        pt = line_line_intersect(start_pt, end_pt, segment[0], segment[1])
        if pt is not None:
            points.append(pt)
    return points


def get_init_gt(
    head_groups, out_h, out_w, global_ignore_index, roi_weight_cfg
):
    """Generate init gt maps.

    Args:
        head_groups: output head infos of each group.
        out_h: the height of output head.
        out_w: the width of output head.
        global_ignore_index: global ignore class index.
        roi_weight_cfg: all configurations related to ROI weights.
    """
    gt_stats = {}
    for head, head_infos in head_groups.items():
        group = head_infos["group"]
        group_head = head_groups[group]
        max_patch_segment = group_head.get("max_patch_segment", 1)
        head_shape = (max_patch_segment, out_h, out_w)
        multi_head = head_infos.get("multi", False)
        group = head_infos["group"]
        if group not in gt_stats:
            gt_stats[group] = {}
        if multi_head:
            gt_stats[group]["cls"] = np.zeros(head_shape, dtype=np.int)
            gt_stats[group]["instance"] = np.zeros(head_shape, dtype=np.int)
            gt_stats[group]["r"] = np.full(head_shape, 100, dtype=np.float)
            gt_stats[group]["sin"] = np.zeros(head_shape, dtype=np.float)
            gt_stats[group]["cos"] = np.zeros(head_shape, dtype=np.float)
            gt_stats[group]["dilate"] = np.zeros(head_shape, dtype=np.int)
            gt_stats[group]["weight"] = np.zeros(head_shape, dtype=np.float)
            if roi_weight_cfg is not None:
                for roi_name, region_cfg in roi_weight_cfg.items():
                    save_roi_mask = region_cfg.get("save_roi_mask", False)
                    if save_roi_mask:
                        gt_stats[group][roi_name] = np.zeros(
                            head_shape, dtype=np.float
                        )
        else:
            # if has background cls
            gt_stats[group][head] = np.full(
                head_shape, global_ignore_index, dtype=np.int
            )
    return gt_stats


def remap_origin_data_laebls(ori_key, ori_instance):
    """Remap the class label of origin data.

    Args:
        ori_key: category.
        ori_instance: instance.
    """
    if ori_key == "solid_lanes":
        if ori_instance.get("lane_flag", "unknown") in ["double", "triple"]:
            ori_instance["lane_direction"] = "bidirectional"
    return ori_instance


def parse_total_element_origin_data(online_mapping_gt_origin, head_groups):
    """Parse origin om data according head groups info.

    Args:
        online_mapping_gt_origin: origin om gt data dict.
        head_groups: output head infos of each group.
    """

    # get target categories of origin om data
    target_categories = []
    for _, head_infos in head_groups.items():
        head_keys = head_infos["key"].split("#")
        main_key = head_keys[0]
        if main_key not in target_categories:
            target_categories.append(main_key)

    # define roadedge closed status map
    curb_status_map = {"open": 0, "closed": 1}
    # extract target categories instance and corresponding labels
    online_mapping_gt = {}
    for ori_key in online_mapping_gt_origin:
        if ori_key not in target_categories:
            continue
        if ori_key not in online_mapping_gt:
            online_mapping_gt[ori_key] = []
        for ori_instance in online_mapping_gt_origin[ori_key]:
            # remap raw labels
            ori_instance = remap_origin_data_laebls(ori_key, ori_instance)
            # get a valid instance
            pts = ori_instance["pts"]
            pts = np.concatenate([pts[:, :2], pts[-1:, 3:5]], axis=0)
            if len(pts) < 2:
                continue
            target_gt = {"pts": pts, "heads": {}}
            # get roadedge closed status
            curb_status = -1
            if "curb_status" in ori_instance:
                curb_status_name = ori_instance["curb_status"]
                if curb_status_name in curb_status_map:
                    curb_status = curb_status_map[curb_status_name]
            target_gt["curb_status"] = curb_status
            for head in head_groups:
                head_keys = head_groups[head]["key"].split("#")
                main_key = head_keys[0]
                if ori_key != main_key:
                    continue
                # assign target group info
                target_gt["group"] = head_groups[head]["group"]
                # get cls label from origin data according sub keys list
                sub_keys = head_keys[1:]
                sub_value = ori_instance
                for sub_key in sub_keys:
                    if sub_key not in sub_value:
                        sub_value = None
                        break
                    sub_value = sub_value[sub_key]
                # remap origin cls label to gt cls label
                label_remap_dict = head_groups[head]["cls_remap"]
                ignore_index = head_groups[head].get("ignore_index", -1)
                label = ignore_index
                if sub_value in label_remap_dict:
                    label = label_remap_dict[sub_value]
                target_gt["heads"][head] = label
            online_mapping_gt[ori_key].append(target_gt)

    online_mapping_gt["ignores"] = (
        [i["pts"] for i in online_mapping_gt_origin["ignores"]]
        if "ignores" in online_mapping_gt_origin.keys()
        else []
    )

    return online_mapping_gt


def split_single_line(rect, start_pt, end_pt):
    """Split a line segment with rect.

    Args:
        rect: [x_min, y_min, x_max, y_max]
        start_pt: [x, y]
        end_pt: [x, y]
    """
    split_pts = []
    split_status = []
    start_in = rect_contain(rect, start_pt)
    end_in = rect_contain(rect, end_pt)
    if start_in and end_in:
        # the whole line segments in region
        split_pts = [start_pt]
        split_status = [1]
    else:
        # find intersections
        pts = rect_line_intersect(rect, start_pt, end_pt)
        num = len(pts)
        if num == 1:
            if start_in:
                split_pts = [start_pt, pts[0]]
                split_status = [1, 0]
            elif end_in:
                split_pts = [start_pt, pts[0]]
                split_status = [0, 1]
        elif num == 2:
            x, y = start_pt
            dist1 = (x - pts[0][0]) ** 2 + (y - pts[0][1]) ** 2
            dist2 = (x - pts[1][0]) ** 2 + (y - pts[1][1]) ** 2
            if dist1 < dist2:
                split_pts = [start_pt, pts[0], pts[1]]
                split_status = [0, 1, 0]
            else:
                split_pts = [start_pt, pts[1], pts[0]]
                split_status = [0, 1, 0]
        else:
            split_pts = [start_pt]
            split_status = [0]
    return split_pts, split_status


def split_by_region(online_mapping_gt, vcs_region):
    """Split line segments by vcs region box.

    Args:
        online_mapping_gt: raw online mapping gt
        vcs_region: vcs region rectangle, [min_x, min_y, max_x, max_y]
    """
    for category, instances in online_mapping_gt.items():
        if category == "ignores":
            continue
        for instance in instances:
            pts = instance["pts"]
            pts_num = pts.shape[0]
            split_pts = []
            split_status = []
            for i in range(pts_num - 1):
                segments, segments_status = split_single_line(
                    vcs_region, pts[i], pts[i + 1]
                )
                split_pts += segments
                split_status += segments_status
            # add end point
            if np.linalg.norm(pts[-1] - split_pts[-1]) > 0.1:
                split_pts.append(pts[-1])
                split_status.append(0)
            # check the last pt is invalid
            split_status[-1] = 0
            # merge
            new_pts = np.zeros((len(split_status), 3))
            new_pts[:, :2] = np.array(split_pts)
            new_pts[:, -1] = np.array(split_status)
            instance["pts"] = new_pts
    return online_mapping_gt


def process_horizontal_line(online_mapping_gt, om_horizontal_cfg):
    """Process horizontal instances.

    More details refer to:
    https://horizonrobotics.feishu.cn/docx/Boqzd96wLohFwsxMiNQcluQnnKg.

    Args:
        online_mapping_gt: Raw online mapping gt.
        om_horizontal_cfg: Horizontal instance process cfg.
            "horizontal_ignore_roi": Horizontal instances in this region
                will be regard as negative.
            "horizontal_degree_thresh": Instances that form this angle
                the Y-axis are identified as horizontal.
            "ratio_thresh": Instances that are within the horizontal
                region by ratio_thresh will be removed.
    """

    for category, instances in online_mapping_gt.items():
        if category == "ignores":
            continue
        new_instances = []
        for instance in instances:
            in_horizontal_region = False
            # get in-vcs-region part of instance
            valid_pt_id = np.where(instance["pts"][:-1][:, -1] >= 1)[
                0
            ].tolist()
            if not valid_pt_id:
                continue
            valid_pt_id.append(max(valid_pt_id) + 1)
            ins_valid = instance["pts"][valid_pt_id]

            # judge horizontal or vertical
            (
                line_x_vcs_min,
                line_x_vcs_max,
                line_y_vcs_min,
                line_y_vcs_max,
            ) = cal_xy_minmax(ins_valid)
            line_x_range = line_x_vcs_max - line_x_vcs_min
            line_y_range = line_y_vcs_max - line_y_vcs_min
            if line_x_range / np.maximum(line_y_range, 1e-5) > np.tan(
                np.deg2rad(om_horizontal_cfg["horizontal_degree_thresh"])
            ):
                instance["horizontal_flag"] = False
                new_instances.append(instance)
                continue
            else:
                instance["horizontal_flag"] = True

            # judge if in horizontal region
            for roi in om_horizontal_cfg["horizontal_ignore_roi"]:
                bottom, right, top, left = roi
                overlap_x = np.minimum(line_x_vcs_max, top) - np.maximum(
                    line_x_vcs_min, bottom
                )
                overlap_y = np.minimum(line_y_vcs_max, left) - np.maximum(
                    line_y_vcs_min, right
                )
                in_region_ratio_x = overlap_x / np.maximum(line_x_range, 1e-5)
                in_region_ratio_y = overlap_y / np.maximum(line_y_range, 1e-5)
                if (
                    in_region_ratio_x > om_horizontal_cfg["ratio_thresh"]
                    and in_region_ratio_y > om_horizontal_cfg["ratio_thresh"]
                ):
                    in_horizontal_region = True
                    break

            if not in_horizontal_region:
                new_instances.append(instance)

        online_mapping_gt[category] = new_instances

    return online_mapping_gt


def cal_xy_minmax(line_pts):
    """Cal x and y axis coord min/max value of line."""
    line_x_vcs_list = line_pts[:, 0]
    line_y_vcs_list = line_pts[:, 1]
    line_x_vcs_min = np.min(line_x_vcs_list)
    line_x_vcs_max = np.max(line_x_vcs_list)
    line_y_vcs_min = np.min(line_y_vcs_list)
    line_y_vcs_max = np.max(line_y_vcs_list)
    return line_x_vcs_min, line_x_vcs_max, line_y_vcs_min, line_y_vcs_max


def cal_xy_range(line_pts, output_minmax=False):
    """Cal x and y axis coord range of line."""
    (
        line_x_vcs_min,
        line_x_vcs_max,
        line_y_vcs_min,
        line_y_vcs_max,
    ) = cal_xy_minmax(line_pts)
    line_y_vcs_range = line_y_vcs_max - line_y_vcs_min
    line_x_vcs_range = line_x_vcs_max - line_x_vcs_min

    if output_minmax:
        return (
            line_x_vcs_min,
            line_x_vcs_max,
            line_y_vcs_min,
            line_y_vcs_max,
            line_x_vcs_range,
            line_y_vcs_range,
        )
    else:
        return line_x_vcs_range, line_y_vcs_range


def sample_segment(
    start_pt,
    end_pt,
    resolution=0.05,
):
    """Sample pts from a line segment.

    Args:
        start_pt: the start point of the line segment.
        end_pt: the end point of the line segment.
        resolution: the resolution of dense sample.
    """
    segment = end_pt - start_pt
    length = np.linalg.norm(segment)
    sample_num = max(int(length / resolution + 1), 2)
    sample_t = np.linspace(0, 1, num=sample_num, endpoint=True)
    sample_t = sample_t.reshape(-1, 1)
    sample_pts = start_pt + sample_t * segment
    return sample_pts


def multi_line_segment_intersection(segments, start_pt, end_pt):
    """Check if has inter.

    Args:
        segments: segments list. N * 4
        start_pt: the start point of another line segment.
        end_pt: the end point of another line segment
    """

    x1, y1 = start_pt
    x2, y2 = end_pt
    x3, y3 = segments[:, 0], segments[:, 1]
    x4, y4 = segments[:, 2], segments[:, 3]

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    ua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
    ua = np.divide(ua, denom, where=(denom != 0))
    ub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
    ub = np.divide(ub, denom, where=(denom != 0))
    result = (denom == 0) | (ua <= 0) | (ua >= 1) | (ub <= 0) | (ub >= 1)
    result = ~result
    return result


def disable_behind(occ_segments, pts, occ_flag, origin_pts, sample_res):
    """Disable instance behind closed curbs.

    Args:
        occ_segments: segments list of curbs.
        pts: [x, y, flag] the polyline segments of a instance.
        occ_flag: if this pts segment from occ_segments.
        origin_pts: used to build rays when calculating occlusion.
        sample_res: the resolution of sampling points from line segment.
    """
    origin_pts = origin_pts.reshape(1, -1, 2)
    origin_num = origin_pts.shape[1]
    pts_num = pts.shape[0]
    new_pts = []
    for i in range(pts_num - 1):
        # skip out-region line segments
        if pts[i][-1] < 1:
            new_pts.append(pts[i])
            continue
        start_pt = pts[i, :2]
        end_pt = pts[i + 1, :2]
        # find self segment indexes
        self_mask = []
        if occ_flag:
            segment = np.array(
                [start_pt[0], start_pt[1], end_pt[0], end_pt[1]]
            )
            dist = np.linalg.norm(segment - occ_segments, axis=1)
            self_mask = dist < 0.05
        sample_pts = sample_segment(start_pt, end_pt, resolution=sample_res)
        sample_pts = sample_pts.reshape(-1, 2)
        sample_num = sample_pts.shape[0]
        # extend sampled pts with different origin_pts
        sample_pts_extend = np.copy(sample_pts).reshape(-1, 1, 2)
        sample_dx = sample_pts_extend[:, :, 0] - origin_pts[:, :, 0]
        sample_dy = sample_pts_extend[:, :, 1] - origin_pts[:, :, 1]
        sample_angle = np.arctan2(sample_dy, sample_dx)
        sample_sin = np.sin(sample_angle)
        sample_cos = np.cos(sample_angle)
        sample_pts_extend = np.repeat(sample_pts_extend, origin_num, axis=1)
        sample_pts_extend[:, :, 0] -= sample_cos * 0.05
        sample_pts_extend[:, :, 1] -= sample_sin * 0.05
        last_occ_status = 0
        for j in range(sample_num - 1):
            # init as occluded
            occ_status = 1
            for k in range(origin_num):
                origin_pt = origin_pts[0, k, :]
                sample_pt = sample_pts_extend[j, k, :]
                intersections = multi_line_segment_intersection(
                    occ_segments, origin_pt, sample_pt
                )
                intersections[self_mask] = False
                if not np.any(intersections):
                    occ_status = -1
                    break
            # correct the status of start point
            if j == 1 and last_occ_status == -1 and occ_status == 1:
                new_pts[-1][-1] = -1
                last_occ_status = 1
            if occ_status != last_occ_status:
                new_pt = [sample_pts[j][0], sample_pts[j][1], -occ_status]
                new_pts.append(new_pt)
                last_occ_status = occ_status

    # add end point
    if np.linalg.norm(pts[-1][:2] - new_pts[-1][:2]) > 0.01:
        pts[-1, -1] = new_pts[-1][-1]
        new_pts.append(pts[-1])
    return np.array(new_pts)


def refine_occ_anchor_points(closed_segments, anchor_points):
    """Refine anchor points within the range of closed segments.

    Args:
        closed_segments: segments list of curbs.
        anchor_points: predetermined anchor points.

    Returns:
        origin_pts: refined anchor points.
    """
    # split the anchor points x into negative and positive
    pts_x_neg = []
    pts_x_pos = []
    for pt_x in anchor_points:
        if pt_x < 0:
            pts_x_neg.append(pt_x)
        else:
            pts_x_pos.append(pt_x)
    pts_x_neg = sorted(pts_x_neg, reverse=True)
    pts_x_pos = sorted(pts_x_pos)

    # find the border
    buffer = 0.1
    border_x_neg = -1e6
    border_x_pos = 1e6
    for segment in closed_segments:
        x1, y1, x2, y2 = segment
        if min(y1, y2) > 0 or max(y1, y2) < 0 or abs(y1 - y2) < 1e-6:
            continue
        border_x = x1 - (x2 - x1) * y1 / (y2 - y1)
        if border_x >= 0:
            border_x_pos = min(border_x_pos, border_x)
        else:
            border_x_neg = max(border_x_neg, border_x)

    refined_pts = []
    border_x_neg += buffer
    border_x_pos -= buffer
    for pts_x in pts_x_neg:
        if pts_x < border_x_neg:
            refined_pts.append(border_x_neg)
            break
        else:
            refined_pts.append(pts_x)
    for pts_x in pts_x_pos:
        if pts_x > border_x_pos:
            refined_pts.append(border_x_pos)
            break
        else:
            refined_pts.append(pts_x)

    origin_pts = []
    for pt_x in refined_pts:
        origin_pts.append([pt_x, 0])
    origin_pts = np.array(origin_pts)
    return origin_pts


def process_roadedges_occ(online_mapping_gt, roadedge_occ_cfg):
    """Porcess om gt with curbs occlusion.

    Args:
        online_mapping_gt: raw online mapping gt.
        roadedge_occ_cfg: config of filting gt with roadedge.
    """
    if roadedge_occ_cfg is None:
        return online_mapping_gt, np.array([])

    # gather all roadedge segments
    check_close = roadedge_occ_cfg["check_close"]
    occ_segments = []
    for category, instances in online_mapping_gt.items():
        if category != "roadedges":
            continue
        for instance in instances:
            if check_close and instance["curb_status"] != 1:
                continue
            instance["occ_flag"] = True
            pts = instance["pts"]
            pts_num = pts.shape[0]
            for i in range(pts_num - 1):
                # skip out-region line segments
                if pts[i][-1] < 1:
                    continue
                # store line segment
                start_pt = pts[i]
                end_pt = pts[i + 1]
                occ_segments.append(
                    [start_pt[0], start_pt[1], end_pt[0], end_pt[1]]
                )
    if not occ_segments:
        return online_mapping_gt, np.array(occ_segments)
    occ_segments = np.array(occ_segments)
    occ_segments = occ_segments.reshape(-1, 4)

    # refine anchor_points to make sure pts within roadedges ranges
    anchor_points = roadedge_occ_cfg["anchor_points"]
    origin_pts = refine_occ_anchor_points(occ_segments, anchor_points)
    if len(origin_pts) == 0:
        return online_mapping_gt, occ_segments

    # remove instance segments behind curb
    black_list = roadedge_occ_cfg.get("black_list", [])
    sample_res = roadedge_occ_cfg.get("sample_res", 0.5)
    keep_horizon = roadedge_occ_cfg.get("keep_horizon", False)
    horizon_thresh = roadedge_occ_cfg.get("horizon_thresh", 28)
    keep_horizon_ratio = roadedge_occ_cfg.get("keep_horizon_ratio", 0.5)
    for category, instances in online_mapping_gt.items():
        if category == "ignores" or category in black_list:
            continue
        for instance in instances:
            if len(instance["pts"]) < 2:
                continue
            # if keep horizon or not
            if keep_horizon:
                pts = instance["pts"]
                pts_num = pts.shape[0]
                if "horizon_flag" in instance:
                    horizon_flag = instance["horizon_flag"]
                else:
                    segments = pts[1:, :2] - pts[0:-1, :2]
                    segments_len = np.linalg.norm(segments, axis=1)
                    horizon_flag = np.zeros((pts_num - 1,))
                    for i in range(pts_num - 1):
                        x1, y1 = pts[i][:2]
                        x2, y2 = pts[i + 1][:2]
                        angle = np.abs(np.arctan((x2 - x1) / (y2 - y1)))
                        if (angle < np.deg2rad(horizon_thresh)) or (
                            angle > np.pi - np.deg2rad(horizon_thresh)
                        ):
                            horizon_flag[i] = 1
                        else:
                            horizon_flag[i] = 0
                    horizon_flag = np.logical_and(
                        horizon_flag, pts[:-1, -1]
                    ).astype(bool)
                    valid_flag = pts[:-1, -1].astype(bool)

                    instance_horizon_ratio = np.sum(
                        segments_len[horizon_flag]
                    ) / (np.sum(segments_len[valid_flag]) + 1e-5)
                    horizon_flag = instance_horizon_ratio > keep_horizon_ratio
                if horizon_flag:
                    continue

            # if this insatnce can apply occlusion to others
            occ_flag = instance.get("occ_flag", False)
            instance["pts"] = disable_behind(
                occ_segments, instance["pts"], occ_flag, origin_pts, sample_res
            )

    return online_mapping_gt, occ_segments


def filter_short_instance(online_mapping_gt, om_short_filter_cfg=None):
    """Filter short instance in om gt.

    More details refer to:
    https://horizonrobotics.feishu.cn/wiki/RTASwJIeximljIkqWn2cC0j5nYt

    Args:
        online_mapping_gt: raw online mapping gt.
        om_short_filter_cfg: filter short instance config. Contains:
        "short_filter_thresh": instance will be filtered if the length
            below short_filter_thresh.
        "short_keep_ratio": if within short_filter_thresh, the ratio of
            effective length to total length in one segment is higher
            than short_keep_ratio, the instance will be retained.
    """
    if om_short_filter_cfg is None:
        return online_mapping_gt
    filter_threshold = om_short_filter_cfg["short_filter_thresh"]
    keep_ratio = om_short_filter_cfg["short_keep_ratio"]
    for category, instances in online_mapping_gt.items():
        if category == "ignores":
            continue
        for instance in instances:
            if len(instance["pts"]) < 2:
                continue
            pts = instance["pts"]
            segments = pts[1:, :2] - pts[0:-1, :2]
            segments_len = np.linalg.norm(segments, axis=1)
            origin_len = np.sum(segments_len)
            valid_list = []
            for i in range(pts.shape[0] - 1):
                if pts[i][-1] > 0:
                    valid_list.append(segments_len[i])
            valid_len = np.sum(valid_list)
            valid_ratio = valid_len / (origin_len + 1e-6)
            if valid_len < filter_threshold:
                # filter valid length shorter than 1m or origin short instance
                if (valid_len < 1) or (valid_ratio < keep_ratio):
                    pts[:, -1] = -1
    return online_mapping_gt


def split_instance_with_space(online_mapping_gt, space_thr=5):
    """Porcess om gt with closed curbs.

    Args:
        online_mapping_gt: raw online mapping gt.
        space_thr: the space thr of merge or split.
    """
    for category, instances in online_mapping_gt.items():
        if category == "ignores":
            continue
        instance_num = len(instances)
        for i in range(instance_num):
            instance = instances[i]
            # TODO: fan.zhao, process instance with loop
            pts = instance["pts"]
            pts_num = len(pts)
            if np.linalg.norm(pts[0, :2] - pts[-1, :2]) < 0.1:
                continue
            # compute accumulated length
            segments = pts[1:, :2] - pts[0:-1, :2]
            segments_length = np.linalg.norm(segments, axis=1)
            segments_length = np.cumsum(segments_length)
            segments_length = np.insert(segments_length, 0, 0)
            # find valid segments
            valid_segments = []
            last_valid_index = -1
            for j in range(pts_num):
                if pts[j, -1] > 0 and last_valid_index < 0:
                    last_valid_index = j
                elif pts[j, -1] < 1 and last_valid_index >= 0:
                    valid_segments.append([last_valid_index, j])
                    last_valid_index = -1
                elif j == pts_num - 1 and last_valid_index >= 0:
                    valid_segments.append([last_valid_index, j])
            # extend valid segments and merge small space
            valid_num = len(valid_segments)
            for j in range(valid_num):
                start_index, end_index = valid_segments[j]
                # foraward
                if j > 0:
                    last_end_index = valid_segments[j - 1][1]
                    length = (
                        segments_length[start_index]
                        - segments_length[last_end_index]
                    )
                    if length < space_thr:
                        pts[last_end_index:start_index, -1] = 2
                # backward
                if j < valid_num - 1:
                    next_start_index = valid_segments[j + 1][0]
                    length = (
                        segments_length[next_start_index]
                        - segments_length[end_index]
                    )
                    if length < space_thr:
                        pts[end_index:next_start_index, -1] = 2
            # split segments with large space into new instance
            split_indexes = [0]
            for j in range(pts_num - 1):
                if pts[j, -1] > 0 and pts[j + 1, -1] < 1:
                    split_indexes.append(j + 1)
            split_indexes.append(pts_num - 1)
            split_instance_num = len(split_indexes) - 1
            for j in range(1, split_instance_num):
                new_insatnce = copy.deepcopy(instance)
                left = split_indexes[j]
                right = split_indexes[j + 1] + 1
                if right - left < 1:
                    continue
                new_insatnce["pts"] = new_insatnce["pts"][left:right, :]
                online_mapping_gt[category].append(new_insatnce)
            left = 0
            right = split_indexes[1] + 1
            instance["pts"] = instance["pts"][left:right, :]

    return online_mapping_gt


def get_valid_mask(mat_vcs2out, online_mapping_gt, bev_height, bev_width):
    """Generate valid mask.

    Args:
        mat_vcs2out: transform matrix from vcs to model output.
        online_mapping_gt: parsed om gt data.
        bev_height: the height of output.
        bev_width: the width of output.
    """

    mask = np.zeros((bev_height, bev_width), dtype="uint8")

    # draw mask img
    for category, instances in online_mapping_gt.items():
        if category == "ignores":
            continue
        for instance in instances:
            pts_num = instance["pts"].shape[0]
            for i in range(pts_num):
                # skip out-region line segments
                if instance["pts"][i][-1] < 1:
                    continue
                x1, y1 = instance["pts"][i][:2]
                pt1 = np.squeeze(
                    mat_vcs2out
                    @ np.array([x1, y1, 1.0], dtype=np.float).reshape((3, 1))
                )
                pt1 = (int(pt1[0] + 0.5), int(pt1[1] + 0.5))
                cv2.circle(mask, pt1, 3, (1), (-1))
                if i < pts_num - 1:
                    x2, y2 = instance["pts"][i + 1][:2]
                    pt2 = np.squeeze(
                        mat_vcs2out
                        @ np.array([x2, y2, 1.0], dtype=np.float).reshape(
                            (3, 1)
                        )
                    )
                    pt2 = (int(pt2[0] + 0.5), int(pt2[1] + 0.5))
                    mask = cv2.line(mask, pt1, pt2, (1), 3)
    return mask


def azimuthAngle(x1, y1, x2, y2):
    """Compute line angle.

    Args:
        x1, y1: the start point
        x2, y2: the end point
    """
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return angle * 180 / math.pi


def get_direction_label(pt1, pt2, raw_direction_label):
    """Compute line direction label.

    Args:
        pt1: the start point if line segment.
        pt2: the end point if line segment.
        raw_direction_label: raw direction label, 4 means bidirectional.
    """

    # bidirectional
    if raw_direction_label != 0:
        return raw_direction_label
    x1, y1 = pt1[:2]
    x2, y2 = pt2[:2]
    lane_angle = azimuthAngle(x1, -y1, x2, -y2)
    lane_direction = -1
    if lane_angle >= 45 and lane_angle < 135:
        lane_direction = 0
    elif lane_angle >= 135 and lane_angle < 225:
        lane_direction = 2
    elif lane_angle >= 225 and lane_angle < 315:
        lane_direction = 1
    else:
        lane_direction = 3

    return lane_direction


def sort_by_dist2ego(online_mapping_gt):
    """Sort om gt by distance from ego car.

    Args:
        online_mapping_gt: parsed om gt data.
    """
    for category, instances in online_mapping_gt.items():
        if category == "ignores":
            continue
        instances_sort = sorted(
            instances, key=lambda x: abs(x["pts"][:, 1].mean()), reverse=True
        )
        online_mapping_gt[category] = instances_sort
    return online_mapping_gt


def get_line_segment(online_mapping_gt):
    """Convert om gt dict to stacked arrays.

    Args:
        online_mapping_gt: parsed om gt data.
    """
    # non-merged instance IDs are placed after merged instance IDs
    # 10 as buffer
    max_merge_ins_num = 10
    for k, v in online_mapping_gt.items():
        if k == "ignores":
            continue
        max_merge_ins_num += len(v)
    line_id = max_merge_ins_num
    segment_id = 0
    line_segments = []
    line_segments_infos = []
    for category, instances in online_mapping_gt.items():
        if category == "ignores":
            continue
        for instance in instances:
            line_info = copy.deepcopy(instance)
            line_info.pop("pts")
            pts = instance["pts"]
            pts_num = len(pts)
            with_lane_direction = "lane_direction" in instance["heads"]
            for idx in range(pts_num - 1):
                # skip out-region line segments
                if pts[idx][-1] < 1:
                    continue
                pt1 = pts[idx][:2]
                pt2 = pts[idx + 1][:2]
                # check distance
                length = np.linalg.norm(pt1 - pt2)
                if length < 1e-4 or length > 1e4:
                    continue
                # update lane direction label
                if with_lane_direction:
                    raw_label = instance["heads"]["lane_direction"]
                    lane_direction = get_direction_label(pt1, pt2, raw_label)
                    line_info["heads"]["lane_direction"] = lane_direction
                line_segments.append(
                    [
                        pt1[0],
                        pt1[1],
                        pt2[0],
                        pt2[1],
                        segment_id,
                        instance["merged_ins_id"]
                        if category == "solid_lanes"
                        and "merged_ins_id" in instance
                        else line_id,
                    ]
                )
                # update line group infos
                segment_id += 1
                line_segments_infos.append(copy.deepcopy(line_info))
            # update line index
            line_id += 1

    line_segments = np.array(line_segments)
    return line_segments, line_segments_infos


def del_nearby_lines(line_segments, delete_thr):
    """Delete nearby lines.

    Args:
        line_segments: all line segments array.
        delete_thr: the threshold of delete near lines.
    """

    start = line_segments[:, :2]
    end = line_segments[:, 2:4]
    num_segments = line_segments.shape[0]
    cost_mat = np.zeros((num_segments, num_segments))
    for i, segment in enumerate(line_segments):
        dist_start, _ = get_dist_pt2lineseg(
            np.array([segment[0:2]]), start, end
        )
        dist_end, _ = get_dist_pt2lineseg(np.array([segment[2:4]]), start, end)
        min_dist = np.min(np.vstack([dist_start, dist_end]), axis=0)
        cost_mat[i] = min_dist
        cost_mat[i, i] = 10000

    all_idx = np.linspace(0, num_segments - 1, num_segments, dtype=np.int)
    instance = line_segments[:, -1]
    unique_instance = sorted(set(instance))
    keep_idx = np.empty((0,), dtype=np.int)
    mask_inst = np.ones(num_segments, dtype=np.bool)
    for inst in unique_instance:
        mask_inst[instance == inst] = False
        mask_inst[keep_idx] = True

        if not np.any(mask_inst):
            continue

        tmp = np.min(cost_mat[instance == inst][:, mask_inst], axis=1)
        keep_idx = np.hstack(
            [
                keep_idx,
                all_idx[instance == inst][tmp > delete_thr],
            ]
        )
    line_segments = line_segments[keep_idx]
    return line_segments


def get_grids(region, res_h, res_w, out_h, out_w):
    """Generate grids center position values.

    Args:
        region: vcs region rectangle, [top, bottom, left, right].
        res_h: the resolution of height grid.
        res_w: the resolution of width grid.
        out_w: the width of final output feature map.
        out_h: the height of final output feature map.
    """

    top, bottom, left, right = region
    coord_ys, coord_xs = np.meshgrid(
        np.linspace(
            left - res_w / 2,
            right + res_w / 2,
            out_w,
        ),
        np.linspace(
            top - res_h / 2,
            bottom + res_h / 2,
            out_h,
        ),
    )
    grids = np.concatenate(
        [coord_xs[:, :, np.newaxis], coord_ys[:, :, np.newaxis]],
        axis=2,
    )
    return grids


def remove_outlier_instance(online_mapping_gt):
    """Remove instances with all points outside the perception range.

    Args:
        online_mapping_gt: Raw online mapping gt.
    """

    for category, instances in online_mapping_gt.items():
        if category == "ignores":
            continue
        new_instances = []
        for instance in instances:
            if np.max(instance["pts"][:, -1]) <= 0:
                continue
            new_instances.append(instance)
        online_mapping_gt[category] = new_instances
    return online_mapping_gt


def split_roadedge(online_mapping_gt):
    """Split U-shape or close roadedge.

    More details refer to:
    https://horizonrobotics.feishu.cn/docx/IsQ8d9fXlopWdPxhEUrcD8bVnme.

    Args:
        online_mapping_gt: Raw online mapping gt.
    """

    # close roadedge start and end point distance threshold
    close_edge_thresh = 0.35

    # U-shape roadedge start and end point distance
    # smaller than ushape_ratio * segment total length
    ushape_ratio = 0.15

    # only split roadedge longer than length_thresh
    length_thresh = 20

    # segment length within curve part usually smaller than it
    curve_segment_thresh = 3

    roadedge_pair_id = 1
    for category, instances in online_mapping_gt.items():
        if category != "roadedges":
            continue
        for instance in instances:
            pts = instance["pts"]
            pts_num = len(pts)
            start_end_dist = np.linalg.norm(pts[0, :2] - pts[-1, :2])

            # compute accumulated length
            segments = pts[1:, :2] - pts[:-1, :2]
            segments_length = np.linalg.norm(segments, axis=1)
            total_length = np.sum(segments_length)

            # only split close or U-shape roadedge
            if (
                start_end_dist >= close_edge_thresh
                and start_end_dist >= ushape_ratio * total_length
            ):
                continue
            if total_length < length_thresh:
                continue

            # reorder pts to make sure start point
            # not whin curve part of roadage
            if start_end_dist < close_edge_thresh:
                new_start_id = None
                for i in range(pts_num - 1):
                    if segments_length[i] > curve_segment_thresh:
                        new_start_id = i
                if new_start_id is None:
                    continue
                pts_new = pts.copy()
                pts_new = np.vstack(
                    [pts_new[new_start_id:], pts_new[:new_start_id]]
                )
                segments = pts_new[1:, :2] - pts_new[:-1, :2]
                segments_length = np.linalg.norm(segments, axis=1)
                pts = pts_new

            # find start and end pt of curve part
            start_pts_id = -1
            change_pts_id = []
            mid_pts_id = []
            for i in range(pts_num):
                if (
                    start_pts_id < 0
                    and i < pts_num - 2
                    and segments_length[i] + segments_length[i + 1]
                    < curve_segment_thresh
                ):
                    start_pts_id = i
                elif (
                    start_pts_id >= 0
                    and i < pts_num - 1
                    and segments_length[i] > curve_segment_thresh
                ) or (i == pts_num - 1 and start_pts_id >= 0):
                    end_pts_id = i
                    change_pts_id.append([start_pts_id, end_pts_id])
                    start_pts_id = -1

            # find middle point of curve part
            for start_id, end_id in change_pts_id:
                arc_length = np.sum(segments_length[start_id:end_id])
                half_arc_length = arc_length / 2
                cum_arc_length = 0
                for j in range(start_id, end_id - 1):
                    cum_arc_length_next = cum_arc_length + segments_length[j]
                    if (
                        cum_arc_length < half_arc_length
                        and cum_arc_length_next >= half_arc_length
                    ):
                        if abs(half_arc_length - cum_arc_length) < abs(
                            half_arc_length - cum_arc_length_next
                        ):
                            mid_pts_id.append(j)
                        else:
                            mid_pts_id.append(j + 1)
                        break
                    cum_arc_length = cum_arc_length_next

            # split roadedge into two instance
            if len(mid_pts_id) == 2:
                pts_1 = pts[mid_pts_id[0] : mid_pts_id[1] + 1]
                pts_2 = np.vstack(
                    [pts[mid_pts_id[1] :], pts[1 : mid_pts_id[0] + 1]]
                )
            elif len(mid_pts_id) == 1:
                pts_1 = pts[: mid_pts_id[0] + 1]
                pts_2 = pts[mid_pts_id[0] :]
            else:
                continue
            instance["pts"] = pts_1
            instance["roadedge_pair_id"] = roadedge_pair_id
            new_instance = instance.copy()
            new_instance["pts"] = pts_2
            new_instance["roadedge_pair_id"] = roadedge_pair_id
            instances.append(new_instance)
            roadedge_pair_id += 1

    return online_mapping_gt


def near_instance_assign(
    gt_stats,
    head_groups,
    head_multi,
    group_patches,
    assign_near_instance_cfg,
    line_segments,
    line_segments_infos,
    line_assign_info,
):
    """Assign near instances to diffrent channel.

    More details refer to:
    https://horizonrobotics.feishu.cn/docx/GmHjdTxmSoXc8BxiSSMcTOfSndb.

    Args:
        gt_stats: All gt data.
        head_groups: Output head infos of each group.
        head_multi: Whether each head is the main head.
        group_patches: Max patch segment of each group.
        assign_near_instance_cfg: Near instance gt assign config.
        line_segments: all line segments array.
        line_segments_infos: group infos of each line segment.

    """
    for group in gt_stats:
        if group_patches[group] != 2:
            continue
        # mean filer method to find near instances
        neighbor_ins_pair = []
        group_gt_stats = gt_stats[group]
        cls_map = group_gt_stats["cls"][0]
        instance_map = group_gt_stats["instance"][0]
        cls_map_binary = cls_map.copy()
        cls_map_binary[cls_map_binary > 0] = 1
        cls_map_blur = cv2.blur(
            np.array(cls_map_binary, np.float),
            assign_near_instance_cfg["assign_blur_window"],
        )
        multi_ins_mask = np.where(
            cls_map_blur >= assign_near_instance_cfg["assign_blur_thresh"]
        )

        stats_dict = {}
        for h, w in zip(multi_ins_mask[0], multi_ins_mask[1]):
            ins_id = instance_map[h, w]
            if ins_id == 0:
                continue
            if ins_id not in stats_dict:
                stats_dict[ins_id] = {
                    "count": 0,
                    "x": 0,
                    "y": 0,
                    "match_ins": {},
                    "is_matched": False,
                }
            stats_dict[ins_id]["count"] += 1
            stats_dict[ins_id]["x"] += h
            stats_dict[ins_id]["y"] += w
            match_ins_id_list = []
            if h - 1 >= 0:
                match_ins_id_list.append(instance_map[h - 1, w])
            if h + 1 < instance_map.shape[0]:
                match_ins_id_list.append(instance_map[h + 1, w])
            if w - 1 >= 0:
                match_ins_id_list.append(instance_map[h, w - 1])
            if w + 1 < instance_map.shape[1]:
                match_ins_id_list.append(instance_map[h, w + 1])
            for match_ins_id in match_ins_id_list:
                if match_ins_id == 0 or match_ins_id == ins_id:
                    continue
                if match_ins_id not in stats_dict[ins_id]["match_ins"]:
                    stats_dict[ins_id]["match_ins"][match_ins_id] = 0
                stats_dict[ins_id]["match_ins"][match_ins_id] += 1

        for key in stats_dict:
            stats_dict[key]["x"] /= stats_dict[key]["count"]
            stats_dict[key]["y"] /= stats_dict[key]["count"]

        key_order = sorted(
            stats_dict, key=lambda i: stats_dict[i]["count"], reverse=True
        )
        reorder_stats_dict = {i: stats_dict[i] for i in key_order}

        for ins_id in reorder_stats_dict:
            max_match_id = 0
            max_match_cnt = 0
            for match_id, cnt in reorder_stats_dict[ins_id][
                "match_ins"
            ].items():
                if (
                    cnt > max_match_cnt
                    and match_id in reorder_stats_dict
                    and not reorder_stats_dict[match_id]["is_matched"]
                ):
                    max_match_id = match_id
                    max_match_cnt = cnt

            if max_match_id > 0:
                ins_xy = (
                    reorder_stats_dict[ins_id]["x"]
                    + reorder_stats_dict[ins_id]["y"]
                )
                match_ins_xy = (
                    reorder_stats_dict[max_match_id]["x"]
                    + reorder_stats_dict[max_match_id]["y"]
                )
                if ins_xy <= match_ins_xy:
                    neighbor_ins_pair.append([ins_id, max_match_id])
                else:
                    neighbor_ins_pair.append([max_match_id, ins_id])
                reorder_stats_dict[ins_id]["is_matched"] = True
                reorder_stats_dict[max_match_id]["is_matched"] = True

        # combine instance list from above two methods
        split_instance_list = [pair[1] for pair in neighbor_ins_pair]
        split_instance_list = list(set(split_instance_list))

        # assign nearly U-shape or close roadedge to different channel
        if group == "roadedge":
            ushape_roadge_pair = {}
            ushape_split_list = []
            for line_segment, line_info in zip(
                line_segments, line_segments_infos
            ):
                if "roadedge_pair_id" in line_info:
                    roadedge_pair_id = line_info["roadedge_pair_id"]
                    if roadedge_pair_id not in ushape_roadge_pair:
                        ushape_roadge_pair[roadedge_pair_id] = []
                    ushape_roadge_pair[roadedge_pair_id].append(
                        line_segment[-1]
                    )
            for pair_instances in ushape_roadge_pair.values():
                pair_instances = list(set(pair_instances))
                if len(pair_instances) < 2:
                    continue
                first_id = pair_instances[0] + 1
                second_id = pair_instances[1] + 1
                pair_first_xy = np.hstack(
                    [
                        np.where(group_gt_stats["instance"][0] == first_id),
                        np.where(group_gt_stats["instance"][1] == first_id),
                    ]
                )
                pair_first_avg_xy = (
                    np.sum(pair_first_xy) / pair_first_xy.shape[1]
                )
                pair_second_xy = np.hstack(
                    [
                        np.where(group_gt_stats["instance"][0] == second_id),
                        np.where(group_gt_stats["instance"][1] == second_id),
                    ]
                )
                pair_second_avg_xy = (
                    np.sum(pair_second_xy) / pair_second_xy.shape[1]
                )
                split_id = (
                    second_id
                    if pair_first_avg_xy <= pair_second_avg_xy
                    else first_id
                )
                if (
                    first_id not in split_instance_list
                    and second_id not in split_instance_list
                ):
                    ushape_split_list.append(split_id)

            split_instance_list.extend(ushape_split_list)

        # move the GT states of specific instances to ch-2
        # dilate is also taking into account
        for ins_id in split_instance_list:
            mask = instance_map == ins_id
            # upadte line assign info
            move_num = mask.sum()
            line_assign_info[group][ins_id][0] -= move_num
            line_assign_info[group][ins_id][1] += move_num
            # compute dilate mask
            dilate_mask = np.zeros_like(mask, np.uint8)
            dilate_mask[mask] = 1
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilate_mask = cv2.morphologyEx(
                dilate_mask, cv2.MORPH_DILATE, morph_kernel, iterations=1
            )
            dilate_mask[mask] = 0
            move_dilate_mask = np.logical_and(
                dilate_mask == 1, group_gt_stats["dilate"][0] == 1
            )
            for head in head_groups:
                if not head.startswith(group):
                    continue
                if head_multi[head]:
                    group_gt_stats["cls"][1, mask] = group_gt_stats["cls"][
                        0, mask
                    ]
                    group_gt_stats["instance"][1, mask] = group_gt_stats[
                        "instance"
                    ][0, mask]
                    group_gt_stats["r"][1, mask] = group_gt_stats["r"][0, mask]
                    group_gt_stats["sin"][1, mask] = group_gt_stats["sin"][
                        0, mask
                    ]
                    group_gt_stats["cos"][1, mask] = group_gt_stats["cos"][
                        0, mask
                    ]
                    group_gt_stats["r"][1, move_dilate_mask] = group_gt_stats[
                        "r"
                    ][0, move_dilate_mask]
                    group_gt_stats["sin"][
                        1, move_dilate_mask
                    ] = group_gt_stats["sin"][0, move_dilate_mask]
                    group_gt_stats["cos"][
                        1, move_dilate_mask
                    ] = group_gt_stats["cos"][0, move_dilate_mask]

                    group_gt_stats["cls"][0, mask] = 0
                    group_gt_stats["instance"][0, mask] = 0
                    group_gt_stats["r"][0, mask] = 0
                    group_gt_stats["sin"][0, mask] = 0
                    group_gt_stats["cos"][0, mask] = 0
                    group_gt_stats["r"][0, move_dilate_mask] = 0
                    group_gt_stats["sin"][0, move_dilate_mask] = 0
                    group_gt_stats["cos"][0, move_dilate_mask] = 0
                else:
                    group_gt_stats[head][1, mask] = group_gt_stats[head][
                        0, mask
                    ]
                    group_gt_stats[head][0, mask] = 0

            group_gt_stats["dilate"][1, mask] = group_gt_stats["dilate"][
                0, mask
            ]
            group_gt_stats["dilate"][1, move_dilate_mask] = group_gt_stats[
                "dilate"
            ][0, move_dilate_mask]
            group_gt_stats["dilate"][0, mask] = 0
            group_gt_stats["dilate"][0, move_dilate_mask] = 0

        # refine instance existed in different channels
        max_ins_id = len(line_assign_info[group])
        for ins_id in range(max_ins_id):
            ins_assign_info = line_assign_info[group][ins_id]
            ins_assign_pixel_num = sum(ins_assign_info)
            if ins_assign_pixel_num <= 0:
                continue
            max_pixel_num = max(ins_assign_info)
            if max_pixel_num >= ins_assign_pixel_num:
                continue
            for channel in range(len(ins_assign_info)):
                # clear
                if ins_assign_info[channel] < max_pixel_num:
                    mask = group_gt_stats["instance"][channel] == ins_id
                    group_gt_stats["instance"][channel][mask] = 0
                    group_gt_stats["cls"][channel][mask] = 0
                    group_gt_stats["dilate"][channel][mask] = 0
    return gt_stats


def update_gt_stats(
    gt_stats,
    head_groups,
    grids,
    valid_mask,
    line_segments,
    line_segments_infos,
    out_h,
    out_w,
    res_h,
    res_w,
    map_dilate,
    dilate_weight,
    vcs_range,
    roi_weight_cfg=None,
    global_ignore_index=-1,
    assign_near_instance_cfg=None,
    om_horizontal_cfg=None,
):
    """Update gt_stats map.

    Args:
        gt_stats: inited gt map data
        head_groups: output head infos of each group.
        grids: grids center position values of output map
        valid_mask: valid region mask of output map.
        line_segments: all line segments array.
        line_segments_infos: group infos of each line segment.
        out_h: the height of final output feature map.
        out_w: the width of final output feature map.
        res_h: the resolution of height grid.
        res_w: the resolution of width grid.
        map_dilate: the dialte radius of generating om gt.
        dilate_weight: weight list of different dilate region.
        vcs_range: visbile range of bev, (bottom, right, top, left)
            in order.
        roi_weight_cfg: all configurations related to ROI weights.
        global_ignore_index: global ignore class index.
        assign_near_instance_cfg: near instance gt assign config.
        om_horizontal_cfg: horizontal instance process cfg.
    """
    # get group patch info
    head_multi = {}
    group_patches = {}
    for head, head_infos in head_groups.items():
        group = head_infos["group"]
        group_head = head_groups[group]
        max_patch_segment = group_head.get("max_patch_segment", 1)
        multi_head = head_infos.get("multi", False)
        head_multi[head] = multi_head
        group_patches[group] = max_patch_segment

    # used to record instance assign info
    max_line_instance = int(max(line_segments[:, 5])) + 1
    line_assign_info = {}
    for group, patch in group_patches.items():
        line_assign_info[group] = []
        for _ in range(max_line_instance + 1):
            line_assign_info[group].append([0] * patch)

    # update valid mask and distance threshold
    inner_h_thr = res_h * 0.5
    inner_w_thr = res_w * 0.5
    unit_res = max(res_h, res_w)
    inner_dist_thr = max(inner_h_thr, inner_w_thr)
    outer_h_thr = inner_h_thr
    outer_w_thr = inner_w_thr
    outer_dist_thr = inner_dist_thr
    if map_dilate > 0:
        outer_h_thr = outer_h_thr + res_h * map_dilate
        outer_w_thr = outer_w_thr + res_w * map_dilate
        outer_dist_thr = max(outer_h_thr, outer_w_thr)
        kernel_size = map_dilate * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        valid_mask = cv2.dilate(valid_mask, kernel)
    # compute distance between grids and line segments with cuda
    if om_pt2lineseg is None:
        start = line_segments[:, :2]
        end = line_segments[:, 2:4]
        warnings.warn(
            "Fail to find om_pt2lineseg in horizon_plugin_pytorch, "
            "please update horizon_plugin_pytorch>=0.16.5. "
        )
    else:
        grids_tensor = torch.tensor(grids, dtype=torch.float32, device="cpu")
        lines_tensor = torch.tensor(
            line_segments[:, :4], dtype=torch.float32, device="cpu"
        )
        mask_tensor = torch.tensor(valid_mask, dtype=torch.uint8, device="cpu")
        dists, directions, selects = om_pt2lineseg(
            grids_tensor, lines_tensor, mask_tensor, outer_dist_thr
        )
        dists = dists.numpy()
        directions = directions.numpy()
        selects = selects.numpy()

    for h in np.arange(0, out_h):
        for w in np.arange(0, out_w):
            if valid_mask[h, w] == 0:
                continue
            if om_pt2lineseg is None:
                coord_x, coord_y = grids[h, w, :]
                dists, directions = get_dist_pt2lineseg(
                    np.array([[coord_x, coord_y]]), start, end
                )
                if not dists.size:
                    continue
                keep_flag = dists < outer_dist_thr
                if not np.any(keep_flag):
                    continue
                line_segments_keep = line_segments[keep_flag]
                dist_keep = dists[keep_flag]
                direction_keep = directions[keep_flag]
            else:
                select_num = selects[h, w, 0]
                if select_num <= 0:
                    continue
                select_indexes = selects[h, w, 1 : select_num + 1]
                line_segments_keep = line_segments[select_indexes]
                dist_keep = dists[h, w, select_indexes]
                direction_keep = directions[h, w, select_indexes]

            phi_keep = np.arctan2(direction_keep[:, 1], direction_keep[:, 0])
            delta_x_keep = dist_keep * np.cos(phi_keep)
            delta_y_keep = dist_keep * np.sin(phi_keep)
            x_plus_y_keep = delta_x_keep + delta_y_keep
            compose_keep = list(
                zip(line_segments_keep, dist_keep, phi_keep, x_plus_y_keep)
            )
            compose_keep = sorted(
                compose_keep, key=lambda x: x[-1], reverse=True
            )
            channel_assigned = {
                group: [False] * patches
                for group, patches in group_patches.items()
            }
            for line_segment, min_dist, phi, _ in compose_keep:
                segment_id = int(line_segment[4])
                line_id = int(line_segment[5])
                if min_dist > np.min(
                    dist_keep[line_segments_keep[:, 5] == line_segment[5]]
                ):
                    continue
                line_group_info = line_segments_infos[segment_id]
                group = line_group_info["group"]

                if False in channel_assigned[group]:
                    if "assign_channel" in line_group_info:
                        ch = line_group_info["assign_channel"]
                    else:
                        ch = channel_assigned[group].index(False)
                else:
                    break

                # check occupancy
                grid_r = gt_stats[group]["r"][ch, h, w] * unit_res
                if min_dist > grid_r:
                    continue
                # check if the foot of a perpendicular is inside the ellipse
                # if in ellipse, x^2/(a^2) + y^2/(b^2) < 1
                dist_h = (min_dist * np.cos(phi)) ** 2
                dist_w = (min_dist * np.sin(phi)) ** 2
                inner_line = (
                    dist_h / (inner_h_thr ** 2) + dist_w / (inner_w_thr ** 2)
                ) < 1
                dilate_line = (
                    dist_h / (outer_h_thr ** 2) + dist_w / (outer_w_thr ** 2)
                ) < 1
                if inner_line:
                    gt_stats[group]["dilate"][ch, h, w] = 2
                elif dilate_line:
                    gt_stats[group]["dilate"][ch, h, w] = 1
                else:
                    continue
                for head, head_label in line_group_info["heads"].items():
                    if head.startswith("_"):
                        continue
                    if head_multi[head]:
                        ins_id = line_id + 1
                        if inner_line:
                            gt_stats[group]["cls"][ch, h, w] = head_label
                            gt_stats[group]["instance"][ch, h, w] = ins_id
                            line_assign_info[group][ins_id][ch] += 1
                        gt_stats[group]["r"][ch, h, w] = min_dist / unit_res
                        gt_stats[group]["sin"][ch, h, w] = -np.sin(phi)
                        gt_stats[group]["cos"][ch, h, w] = -np.cos(phi)
                    else:
                        if inner_line:
                            gt_stats[group][head][ch, h, w] = head_label
                if inner_line:
                    channel_assigned[group][ch] = True

    if assign_near_instance_cfg is not None:
        gt_stats = near_instance_assign(
            gt_stats,
            head_groups,
            head_multi,
            group_patches,
            assign_near_instance_cfg,
            line_segments,
            line_segments_infos,
            line_assign_info,
        )

    for group in gt_stats:
        mask = gt_stats[group]["dilate"]
        gt_stats[group]["r"][mask == 0] = 0
        if map_dilate > 0:
            group_dw = dilate_weight[group]
            gt_stats[group]["weight"][mask == 0] = group_dw["background"]
            gt_stats[group]["weight"][mask == 1] = group_dw["dilated"]
            gt_stats[group]["weight"][mask == 2] = group_dw["foreground"]
            # assign_near_instance only work on lane and
            # a grid must support at most two lanes
            if (
                assign_near_instance_cfg is not None
                and group == "lane"
                and group_patches[group] == 2
            ):
                gt_stats[group]["weight"][1][
                    mask[1] == 0
                ] = assign_near_instance_cfg["lane_ch2_weight"]["background"]
                gt_stats[group]["weight"][1][
                    mask[1] == 1
                ] = assign_near_instance_cfg["lane_ch2_weight"]["dilated"]
                gt_stats[group]["weight"][1][
                    mask[1] == 2
                ] = assign_near_instance_cfg["lane_ch2_weight"]["foreground"]
        if (
            assign_near_instance_cfg is not None
            and group == "lane"
            and group_patches[group] == 2
        ):
            near_ins_mask = gt_stats[group]["cls"][1] > 0
            gt_stats[group]["weight"][
                :, near_ins_mask
            ] *= assign_near_instance_cfg["near_instance_weight"]

    if roi_weight_cfg is not None:
        gt_stats = adjust_roi_weight(
            gt_stats, roi_weight_cfg, out_h, out_w, vcs_range
        )

        if (
            om_horizontal_cfg is not None
            and "horizontal_enchance_roi" in om_horizontal_cfg
        ):
            for enhance_roi in om_horizontal_cfg["horizontal_enchance_roi"]:
                top, left, bottom, right = get_roi_vcs_range_box(
                    (out_h, out_w), vcs_range, enhance_roi
                )
                weight = gt_stats[group]["weight"]
                weight[:, top:bottom, left:right] *= om_horizontal_cfg[
                    "enchance_weight"
                ]

    return gt_stats


def parse_gt_pts(gt_stats, top, left, res_h, res_w):
    """Generate gt lane pts from gt stats map.

    Args:
        gt_stats: all gt data.
        top: the top border of vcs range.
        left: the left border of vcs range.
        res_h: the resolution of height grid.
        res_w: the resolution of width grid.
    """
    unit_res = max(res_h, res_w)
    group_lanes = {}
    for group in gt_stats:
        instance_data = gt_stats[group]["instance"]
        unique_id = np.unique(instance_data).tolist()
        instance_id_dict = {
            instance_id: i
            for i, instance_id in enumerate(np.unique(unique_id))
        }
        num_pred_lane = len(instance_id_dict)
        lanes = [[] for _ in range(num_pred_lane)]
        cls_data = gt_stats[group]["cls"]
        r_data = gt_stats[group]["r"]
        sin_data = gt_stats[group]["sin"]
        cos_data = gt_stats[group]["cos"]
        out_c, out_h, out_w = cls_data.shape
        for h in np.arange(0, out_h):
            for w in np.arange(0, out_w):
                for ch in np.arange(0, out_c):
                    label = cls_data[ch, h, w]
                    if label == 0:
                        continue
                    instance_id = instance_id_dict[instance_data[ch, h, w]]
                    x_vcs_origin = top - h * res_h - res_h / 2
                    y_vcs_origin = left - w * res_w - res_w / 2
                    r = r_data[ch, h, w]
                    sin = sin_data[ch, h, w]
                    cos = cos_data[ch, h, w]
                    vcs_x = x_vcs_origin - r * cos * unit_res
                    vcs_y = y_vcs_origin - r * sin * unit_res
                    pt = [vcs_x, vcs_y, h, w, ch, label]
                    lanes[instance_id].append(pt)
        # convert to np array
        lanes_array = []
        for lane in lanes:
            if lane:
                lanes_array.append(np.array(lane))
        group_lanes[group] = lanes_array

    return group_lanes


def process_four_line(
    online_mapping_gt, head_groups, assign_near_instance_cfg
):
    """Process four line case.

    More details refer to:
    https://horizonrobotics.feishu.cn/docx/OTbEdxJ2MoISTcxopDXcbwn3nmc.

    Args:
        online_mapping_gt: Raw online mapping gt.
        head_groups: Output head infos of each group.
        assign_near_instance_cfg: Near instance gt assign config.
    """

    bidirection_idx = head_groups["lane_direction"]["cls_list"].index(
        "bidirection"
    )
    double_idx = head_groups["lane_double"]["cls_list"].index("double")
    for category, instances in online_mapping_gt.items():
        if category != "solid_lanes":
            continue
        new_instances = []
        for i, instance_i in enumerate(instances):
            if instance_i["heads"]["lane_double"] != double_idx:
                continue
            pts_num_minus = instance_i["pts"].shape[0] - 1
            middle_segment = instance_i["pts"][
                pts_num_minus // 2 : pts_num_minus // 2 + 2
            ]
            middle_pt = np.mean(middle_segment, axis=0)[:2]

            for j, instance_j in enumerate(instances):
                if i == j or instance_j["heads"]["lane_double"] == double_idx:
                    continue
                pts = instance_j["pts"]
                for seg_idx in range(len(pts) - 1):
                    segment = pts[seg_idx : seg_idx + 2][:, :2]
                    status, distance, _ = point_to_line_distance(
                        middle_pt[:2], segment[0], segment[1]
                    )
                    if (
                        status == 1
                        and distance
                        < assign_near_instance_cfg["four_lane_dist_thresh"]
                    ):
                        instance_j["heads"]["lane_direction"] = bidirection_idx
                        instance_j["heads"]["lane_double"] = double_idx
                        instance_i["four_lane_flag"] = True
                        break

        for ins in instances:
            if "four_lane_flag" not in ins:
                new_instances.append(ins)
        online_mapping_gt[category] = new_instances

    return online_mapping_gt


def process_double_line(
    online_mapping_gt, head_groups, assign_near_instance_cfg
):
    """Double line near assign.

    More details refer to:
    https://horizonrobotics.feishu.cn/docx/OTbEdxJ2MoISTcxopDXcbwn3nmc.

    Args:
        online_mapping_gt: Raw online mapping gt.
        head_groups: Output head infos of each group.
        assign_near_instance_cfg: Near instance gt assign config.
    """

    bidirection_idx = head_groups["lane_direction"]["cls_list"].index(
        "bidirection"
    )
    double_idx = head_groups["lane_double"]["cls_list"].index("double")
    for category, instances in online_mapping_gt.items():
        if category != "solid_lanes":
            continue
        for i, instance_i in enumerate(instances):
            if (
                instance_i["heads"]["lane_direction"] != bidirection_idx
                and instance_i["heads"]["lane_double"] != double_idx
            ):
                continue
            pts_num_minus = instance_i["pts"].shape[0] - 1
            middle_segment = instance_i["pts"][
                pts_num_minus // 2 : pts_num_minus // 2 + 2
            ]
            middle_pt = np.mean(middle_segment, axis=0)[:2]

            for j, instance_j in enumerate(instances):
                if i == j or (
                    instance_j["heads"]["lane_direction"] != bidirection_idx
                    and instance_j["heads"]["lane_double"] != double_idx
                ):
                    continue
                pts = instance_j["pts"]
                for seg_idx in range(len(pts) - 1):
                    segment = pts[seg_idx : seg_idx + 2][:, :2]
                    status, distance, projection_pts = point_to_line_distance(
                        middle_pt[:2], segment[0], segment[1]
                    )

                    if (
                        status == 1
                        and distance
                        < assign_near_instance_cfg["double_lane_dist_thresh"]
                    ):
                        lane_x_vcs_list = pts[:, 0]
                        lane_y_vcs_list = pts[:, 1]
                        lane_x_vcs_min = np.min(lane_x_vcs_list)
                        lane_x_vcs_max = np.max(lane_x_vcs_list)
                        lane_y_vcs_min = np.min(lane_y_vcs_list)
                        lane_y_vcs_max = np.max(lane_y_vcs_list)
                        lane_y_vcs_range = lane_y_vcs_max - lane_y_vcs_min
                        lane_x_vcs_range = lane_x_vcs_max - lane_x_vcs_min

                        # 1 means ch2, 0 means ch1
                        if lane_x_vcs_range >= lane_y_vcs_range:
                            if middle_pt[1] > projection_pts[1]:
                                instance_j["assign_channel"] = 1
                            else:
                                instance_i["assign_channel"] = 1
                        else:
                            if middle_pt[0] > projection_pts[0]:
                                instance_j["assign_channel"] = 1
                            else:
                                instance_i["assign_channel"] = 1
                        break

    return online_mapping_gt


def merge_lane_instance_id(online_mapping_gt, merge_instance_cfg, head_groups):
    """Merge lane instance.

    More details refer to:
    https://horizonrobotics.feishu.cn/docx/XqAdd5kUKo7gjZx0Fkyc7DIKnog.

    Args:
        online_mapping_gt: Raw online mapping gt.
        merge_instance_cfg: Cfg of merge instances that are physically
            connected but have different properties. Keys as below:
            "normal_distance_thresh": Normal instances distance lower
                than this will be merge.
            "double_distance_thresh": Double instances distance lower
                than this will be merge.
            "direction_thresh": The angle difference between two merge
                instances must be smaller than this value.
        head_groups: Output head infos of each group.
    """

    assert (
        "lane_double" in head_groups and "lane_property" in head_groups
    ), "Merging operation rely on the double type of lane."
    double_idx = head_groups["lane_double"]["cls_list"].index("double")
    stay_idx = head_groups["lane_property"]["cls_remap"]["stay"]

    start_pts = []
    end_pts = []
    merge_flag_list = []
    instance_id_list = []
    black_list_flag = 0
    special_list_flag = 1
    white_list_flag = 2
    for category, instances in online_mapping_gt.items():
        if category != "solid_lanes":
            continue
        for instance in instances:
            # get in-region part of instance
            valid_idx = np.where(instance["pts"][:-1, -1] >= 1)[0]
            start_pts.append(instance["pts"][valid_idx[0]][:2])
            end_pts.append(instance["pts"][valid_idx[-1] + 1][:2])

            # stay type put in merge blacklist, double use smaller thresh.
            if instance["heads"]["lane_property"] == stay_idx:
                merge_flag_list.append(black_list_flag)
            elif instance["heads"]["lane_double"] == double_idx:
                merge_flag_list.append(special_list_flag)
            else:
                merge_flag_list.append(white_list_flag)

    if not start_pts:
        return online_mapping_gt

    start_pts = np.vstack(start_pts)
    end_pts = np.vstack(end_pts)
    merge_flag_list = np.array(merge_flag_list)
    start_non_visit_flag = np.ones_like(merge_flag_list)
    end_non_visit_flag = np.ones_like(merge_flag_list)
    instance_id_list = np.arange(start_pts.shape[0])

    for idx, end_pt in enumerate(end_pts):
        if end_non_visit_flag[idx] == 0:
            continue
        if merge_flag_list[idx] == special_list_flag:
            dist_delta = merge_instance_cfg["double_distance_thresh"]
        else:
            dist_delta = merge_instance_cfg["normal_distance_thresh"]

        # find near start and end point
        end_non_visit_flag[idx] = 0
        start_pts_diff = start_pts - end_pt
        end_pts_diff = end_pts - end_pt
        start_near_pts_id = np.where(
            (np.linalg.norm(start_pts_diff, axis=1) < dist_delta)
            * start_non_visit_flag
            * merge_flag_list
            > 0
        )[0].tolist()
        end_near_pts_id = np.where(
            (np.linalg.norm(end_pts_diff, axis=1) < dist_delta)
            * end_non_visit_flag
            * merge_flag_list
            > 0
        )[0].tolist()
        if idx in start_near_pts_id:
            start_near_pts_id.remove(idx)
        start_non_visit_flag[start_near_pts_id] = 0
        end_non_visit_flag[end_near_pts_id] = 0

        # if num of near endpoint > 1, thought as merge/split case
        near_pts_num = len(start_near_pts_id) + len(end_near_pts_id)
        if near_pts_num == 1:
            merge_pts_id = None
            if len(start_near_pts_id) == 1:
                id_to_be_merge = instance_id_list[start_near_pts_id]
                merge_pts_id = start_near_pts_id
            else:
                id_to_be_merge = instance_id_list[end_near_pts_id]
                merge_pts_id = end_near_pts_id

            # only merge instances have similar direction
            merge_line_pts = np.vstack(
                [start_pts[merge_pts_id], end_pts[merge_pts_id]]
            )
            (
                merge_line_x_min,
                merge_line_x_max,
                merge_line_y_min,
                merge_line_y_max,
            ) = cal_xy_minmax(merge_line_pts)
            cur_line_pts = np.vstack([start_pts[idx], end_pts[idx]])
            (
                cur_line_x_min,
                cur_line_x_max,
                cur_line_y_min,
                cur_line_y_max,
            ) = cal_xy_minmax(cur_line_pts)
            merge_direction = np.arctan(
                (merge_line_x_max - merge_line_x_min)
                / np.maximum(merge_line_y_max - merge_line_y_min, 1e-5)
            )
            cur_direction = np.arctan(
                (cur_line_x_max - cur_line_x_min)
                / np.maximum(cur_line_y_max - cur_line_y_min, 1e-5)
            )
            if abs(merge_direction - cur_direction) > np.deg2rad(
                merge_instance_cfg["direction_thresh"]
            ):
                continue

            instance_id_list[
                instance_id_list == id_to_be_merge
            ] = instance_id_list[idx]

    for idx, instance in enumerate(online_mapping_gt["solid_lanes"]):
        instance["merged_ins_id"] = instance_id_list[idx]
    return online_mapping_gt


def process_solid_dash_line(
    online_mapping_gt, head_groups, merge_solid_dash_cfg
):
    """Process solid dash line.

    More details refer to:
    https://horizonrobotics.feishu.cn/wiki/RTASwJIeximljIkqWn2cC0j5nYt.

    Args:
        online_mapping_gt: Raw online mapping gt.
        head_groups: Output head infos of each group.
        merge_solid_dash_cfg: Merge solid dash line cfg. Keys as below:
            "dist_thresh": Distance of two lines below thresh may be
                solid dash line.
            "length_thresh": Lines shorter than this length will not
                be processed.
            "split_half_thresh": Endpoints distance below this will not
                be split.
            "split_diff_thresh": Total length diff below this will not
                be split.
            "split_ratio": Solid dash lines with a length ratio smalle
                than this proportion will not be split.
    """

    dist_thresh = merge_solid_dash_cfg["dist_thresh"]

    double_idx = head_groups["lane_double"]["cls_list"].index("double")
    solid_idx = head_groups["lane_dashed"]["cls_list"].index("solid")
    dashed_idx = head_groups["lane_dashed"]["cls_list"].index("dashed")
    left_solid_right_dash_idx = head_groups["lane_dashed"]["cls_list"].index(
        "left_solid_right_dash"
    )
    left_dash_right_solid_idx = head_groups["lane_dashed"]["cls_list"].index(
        "left_dash_right_solid"
    )
    bidirection_idx = head_groups["lane_direction"]["cls_list"].index(
        "bidirection"
    )

    for category, instances in online_mapping_gt.items():
        if category != "solid_lanes":
            continue

        del_dashed_list = []
        for i, instance_i in enumerate(instances):
            if instance_i["heads"]["lane_dashed"] != dashed_idx:
                continue
            # get in-region part of instance
            valid_idx_i = np.where(instance_i["pts"][:-1, -1] >= 1)[0].tolist()
            valid_idx_i.append(max(valid_idx_i) + 1)
            valid_pts_i = instance_i["pts"][valid_idx_i]

            # skip horizontal and short line
            (
                line_x_vcs_min_i,
                line_x_vcs_max_i,
                line_y_vcs_min_i,
                line_y_vcs_max_i,
                line_x_vcs_range_i,
                line_y_vcs_range_i,
            ) = cal_xy_range(valid_pts_i, output_minmax=True)
            if line_x_vcs_range_i < line_y_vcs_range_i:
                continue
            length_i = np.linalg.norm(
                np.array(
                    [
                        line_x_vcs_max_i - line_x_vcs_min_i,
                        line_y_vcs_max_i - line_y_vcs_min_i,
                    ]
                )
            )
            if length_i < merge_solid_dash_cfg["length_thresh"]:
                continue

            for j, instance_j in enumerate(instances):
                if i == j or (instance_j["heads"]["lane_dashed"] != solid_idx):
                    continue
                # skip single but different direciton line
                if (
                    instance_j["heads"]["lane_direction"]
                    != instance_i["heads"]["lane_direction"]
                    and instance_j["heads"]["lane_direction"]
                    != bidirection_idx
                    and instance_i["heads"]["lane_direction"]
                    != bidirection_idx
                ):
                    continue

                # skip horizontal and short line
                valid_idx_j = np.where(instance_j["pts"][:-1, -1] >= 1)[
                    0
                ].tolist()
                valid_idx_j.append(max(valid_idx_j) + 1)
                valid_pts_j = instance_j["pts"][valid_idx_j]
                (
                    line_x_vcs_min_j,
                    line_x_vcs_max_j,
                    line_y_vcs_min_j,
                    line_y_vcs_max_j,
                    line_x_vcs_range_j,
                    line_y_vcs_range_j,
                ) = cal_xy_range(valid_pts_j, output_minmax=True)
                if line_x_vcs_range_j < line_y_vcs_range_j:
                    continue
                length_j = np.linalg.norm(
                    np.array(
                        [
                            line_x_vcs_max_j - line_x_vcs_min_j,
                            line_y_vcs_max_j - line_y_vcs_min_j,
                        ]
                    )
                )
                if length_j < merge_solid_dash_cfg["length_thresh"]:
                    continue

                start_project_id = -1
                end_project_id = -1
                mid_project_id = -1
                start_project_pt = None
                end_project_pt = None
                mid_project_pt = None
                same_direction_flag = False

                # calculate start/end/mid point distances of two lines
                if (
                    abs(line_y_vcs_min_i - line_y_vcs_min_j) < dist_thresh
                    or abs(line_y_vcs_max_i - line_y_vcs_max_j) < dist_thresh
                    or (
                        line_x_vcs_max_j >= line_x_vcs_max_i
                        and line_x_vcs_min_j <= line_x_vcs_min_i
                    )
                ):
                    start_pt_i = valid_pts_i[0][:2]
                    end_pt_i = valid_pts_i[-1][:2]
                    pts_num_minus = valid_pts_i.shape[0] - 1
                    middle_segment = instance_i["pts"][
                        pts_num_minus // 2 : pts_num_minus // 2 + 2
                    ]
                    mid_pt_i = np.mean(middle_segment, axis=0)[:2]

                    start_pt_j = valid_pts_j[0][:2]
                    end_pt_j = valid_pts_j[-1][:2]
                    for pt_id in range(len(instance_j["pts"]) - 1):
                        (
                            _,
                            distance_start,
                            projection_pts_start,
                        ) = point_to_line_distance(
                            start_pt_i,
                            instance_j["pts"][pt_id][:2],
                            instance_j["pts"][pt_id + 1][:2],
                        )
                        (
                            _,
                            distance_end,
                            projection_pts_end,
                        ) = point_to_line_distance(
                            end_pt_i,
                            instance_j["pts"][pt_id][:2],
                            instance_j["pts"][pt_id + 1][:2],
                        )
                        (
                            _,
                            distance_mid,
                            projection_pts_mid,
                        ) = point_to_line_distance(
                            mid_pt_i,
                            instance_j["pts"][pt_id][:2],
                            instance_j["pts"][pt_id + 1][:2],
                        )
                        if (
                            start_project_id == -1
                            and projection_pts_start is not None
                            and distance_start < dist_thresh
                        ):
                            start_project_pt = projection_pts_start
                            start_project_id = pt_id
                        if (
                            end_project_id == -1
                            and projection_pts_end is not None
                            and distance_end < dist_thresh
                        ):
                            end_project_pt = projection_pts_end
                            end_project_id = pt_id
                        if (
                            mid_project_id == -1
                            and projection_pts_mid is not None
                            and distance_mid < dist_thresh
                        ):
                            mid_project_pt = projection_pts_mid
                            mid_project_id = pt_id
                    same_direction_flag = (end_pt_i[0] - start_pt_i[0]) * (
                        end_pt_j[0] - start_pt_j[0]
                    ) > 0

                # split lines with satisfied distance according to length
                if (
                    start_project_pt is not None
                    and end_project_pt is not None
                    and mid_project_pt is not None
                    and same_direction_flag
                ):
                    if (
                        length_i / length_j
                        < merge_solid_dash_cfg["split_ratio"]
                        or length_j - length_i
                        > merge_solid_dash_cfg["split_diff_thresh"]
                    ):
                        new_instance_start = copy.deepcopy(instance_j)
                        new_instance_end = copy.deepcopy(instance_j)
                        if (
                            np.linalg.norm(end_pt_i - end_pt_j)
                            > merge_solid_dash_cfg["split_half_thresh"]
                        ):
                            new_instance_end["pts"][end_project_id][
                                :2
                            ] = end_project_pt
                            new_instance_end["pts"] = new_instance_end["pts"][
                                end_project_id:
                            ]
                            instance_j["pts"][end_project_id + 1][
                                :2
                            ] = end_project_pt
                            instance_j["pts"][end_project_id + 1][2] = 0
                            instance_j["pts"] = instance_j["pts"][
                                : end_project_id + 2
                            ]
                            ins_end_valid_idx = np.where(
                                new_instance_end["pts"][:-1, -1] >= 1
                            )[0].tolist()
                            if ins_end_valid_idx:
                                instances.append(new_instance_end)
                        if (
                            np.linalg.norm(start_pt_i - start_pt_j)
                            > merge_solid_dash_cfg["split_half_thresh"]
                        ):
                            new_instance_start["pts"][start_project_id + 1][
                                :2
                            ] = start_project_pt
                            new_instance_start["pts"][start_project_id + 1][
                                2
                            ] = 0
                            new_instance_start["pts"] = new_instance_start[
                                "pts"
                            ][: start_project_id + 2]
                            instance_j["pts"][start_project_id][
                                :2
                            ] = start_project_pt
                            instance_j["pts"] = instance_j["pts"][
                                start_project_id:
                            ]
                            ins_start_valid_idx = np.where(
                                new_instance_start["pts"][:-1, -1] >= 1
                            )[0].tolist()
                            if ins_start_valid_idx:
                                instances.append(new_instance_start)

                    # judge solid dash line type
                    del_dashed_list.append(i)
                    if (start_pt_i[1] + end_pt_i[1]) < (
                        start_project_pt[1] + end_project_pt[1]
                    ):
                        instance_j["heads"][
                            "lane_dashed"
                        ] = left_solid_right_dash_idx
                    else:
                        instance_j["heads"][
                            "lane_dashed"
                        ] = left_dash_right_solid_idx
                    instance_j["heads"]["lane_double"] = double_idx
                    if "assign_channel" in instance_j:
                        instance_j.pop("assign_channel")
                    new_valid_idx_j = np.where(
                        instance_j["pts"][:-1, -1] >= 1
                    )[0].tolist()
                    if not new_valid_idx_j:
                        del_dashed_list.append(j)
                    if i in del_dashed_list:
                        break

        # del dash line
        new_instances = []
        for i, instance_i in enumerate(instances):
            if i in del_dashed_list:
                continue
            new_instances.append(instance_i)
        online_mapping_gt[category] = new_instances

    return online_mapping_gt


def get_gt_online_mapping(
    gt_online_mapping_ori: dict,
    head_groups: dict,
    out_size: tuple,
    roi_weight_cfg: dict = None,
    roadedge_occ_cfg: dict = None,
    origin_imgs: list = None,
    meta_info: dict = None,
    map_dilate: int = 0,
    dilate_weight: dict = None,
    view_bev_size: tuple = (512, 512),
    vcs_range: tuple = (-30, -51.2, 72.4, 51.2),
    split_close_roadedge: bool = False,
    image_files: list = None,
    image_size: tuple = (320, 512),
    view_cols: int = 3,
    view_sub_head: bool = True,
    visualize_output_dir: str = None,
    global_ignore_index: int = -1,
    assign_near_instance_cfg: dict = None,
    merge_instance_cfg: dict = None,
    block_warp_padding: dict = None,
    merge_crosspoint: bool = False,
    om_short_filter_cfg: dict = None,
    om_horizontal_cfg: dict = None,
    merge_solid_dash_cfg: dict = None,
):
    """Generate online mapping target.

    Args:
        gt_online_mapping_ori: origin om gt data.
        imgs: origin input img list
        meta_info: dict contains each view's calibration param
        head_groups: output head infos of each group.
        roi_weight_cfg: all configurations related to ROI weights.
        roadedge_occ_cfg: config of filting gt with roadedge.
        map_dilate: the dialte radius of generating om gt.
        dilate_weight: weight dict of different dilate region.
        out_size: model output size.
        view_bev_size: the size of view bev results.
        vcs_range: vcs range, in meter.(order is
            (bottom,right,top,left), e.g.(-30, -51.2, 72.4, 51.2))
        split_close_roadedge: whether split U-shape or close roadedge.
        image_files: origin image paths.
        image_size: image size, in pixel.(order is (h,w))
        view_cols: the column of view img.
        view_sub_head: if view sub task head.
        visualize_output_dir: gt visualization dir.
        global_ignore_index: global ignore class index.
        assign_near_instance_cfg: near instance gt assign config.
        merge_instance_cfg: cfg of merge instances that are physically
            connected but have different properties.
        block_warp_padding: order is (left,right,up,bottom).
        merge_crosspoint: If True, transport online mapping gt to crosspoint.
        om_short_filter_cfg: filter short instance config.
        om_horizontal_cfg: horizontal instance process cfg.
        merge_solid_dash_cfg: process solid dash line cfg.
    """
    out_h, out_w = out_size
    bottom, right, top, left = vcs_range
    if dilate_weight is None:
        map_dilate = 0
    scope_h = top - bottom
    scope_w = left - right
    region = (top, bottom, left, right)
    res_h = scope_h / out_h
    res_w = scope_w / out_w
    mat_vcs2out = get_vcs2bev_img_mat(vcs_range, (out_h, out_w))
    # generate init gt array map data
    gt_stats = get_init_gt(
        head_groups, out_h, out_w, global_ignore_index, roi_weight_cfg
    )

    # parse origin om data according head groups info
    online_mapping_gt = parse_total_element_origin_data(
        gt_online_mapping_ori, head_groups
    )

    # split U-shape or close roadedge
    if split_close_roadedge:
        online_mapping_gt = split_roadedge(online_mapping_gt)

    # split om gt with vcs region box
    online_mapping_gt = split_by_region(online_mapping_gt, vcs_range)

    # remove distant horizontal instance
    if om_horizontal_cfg is not None:
        online_mapping_gt = process_horizontal_line(
            online_mapping_gt, om_horizontal_cfg
        )

    # clear om gt with closed curb
    online_mapping_gt, occ_segments = process_roadedges_occ(
        online_mapping_gt,
        roadedge_occ_cfg,
    )

    # split om gt to new instance
    online_mapping_gt = split_instance_with_space(online_mapping_gt)

    # filter short instance
    online_mapping_gt = filter_short_instance(
        online_mapping_gt, om_short_filter_cfg
    )
    online_mapping_gt = remove_outlier_instance(online_mapping_gt)

    if assign_near_instance_cfg is not None:
        # four lane processing must before double lane
        online_mapping_gt = process_four_line(
            online_mapping_gt, head_groups, assign_near_instance_cfg
        )
        online_mapping_gt = process_double_line(
            online_mapping_gt, head_groups, assign_near_instance_cfg
        )

    if merge_solid_dash_cfg is not None:
        online_mapping_gt = process_solid_dash_line(
            online_mapping_gt, head_groups, merge_solid_dash_cfg
        )

    if merge_instance_cfg is not None:
        online_mapping_gt = merge_lane_instance_id(
            online_mapping_gt, merge_instance_cfg, head_groups
        )

    # generate valid region mask to improve the speed of gt generating
    valid_mask = get_valid_mask(mat_vcs2out, online_mapping_gt, out_h, out_w)
    if not np.any(valid_mask):
        warnings.warn("Empty om gt, timestamp: {}".format(image_files[0]))
        return gt_stats

    # sort instances
    online_mapping_gt = sort_by_dist2ego(online_mapping_gt)

    # convert om gt dict to stacked array
    line_segments, line_segments_infos = get_line_segment(online_mapping_gt)
    if len(line_segments) == 0:
        warnings.warn(
            "Empty om gt, error timestamp: {}".format(image_files[0])
        )
        return gt_stats

    # generate grids center position values of output map
    grids = get_grids(region, res_h, res_w, out_h, out_w)

    # update gt map
    gt_stats = update_gt_stats(
        gt_stats,
        head_groups,
        grids,
        valid_mask,
        line_segments,
        line_segments_infos,
        out_h,
        out_w,
        res_h,
        res_w,
        map_dilate,
        dilate_weight,
        vcs_range,
        roi_weight_cfg=roi_weight_cfg,
        global_ignore_index=global_ignore_index,
        assign_near_instance_cfg=assign_near_instance_cfg,
        om_horizontal_cfg=om_horizontal_cfg,
    )

    if visualize_output_dir is not None:
        view_bev_h, view_bev_w = view_bev_size
        view_mat_vcs2bev = get_vcs2bev_img_mat(vcs_range, view_bev_size)
        tmp = image_files[0].split("/")
        pack_name = tmp[-3]
        image_name = tmp[-1]
        image_rel_path = os.path.join(pack_name, image_name)
        os.makedirs(visualize_output_dir, exist_ok=True)
        tic = time.time()
        # plot raw img as n * 3 arrangements
        view_raw_img = draw_raw_img(
            image_files, origin_imgs[0], image_size, view_cols
        )

        # draw origin gt
        target_categories = ["ignores"]
        for _, head_infos in head_groups.items():
            main_key = head_infos["key"].split("#")[0]
            if main_key not in target_categories:
                target_categories.append(main_key)
        view_ori_gts = draw_raw_gt(
            gt_online_mapping_ori,
            view_bev_size,
            view_mat_vcs2bev,
            target_categories,
        )
        if roadedge_occ_cfg is not None:
            view_ori_gts = view_ori_gts[:2]
        # plot gt in ipm images
        homo_offset = meta_info.get("homo_offset", None)
        ipm_img_sizes = meta_info.get("ipm_img_sizes", None)
        if (origin_imgs is not None) and (homo_offset is not None):
            origin_imgs_bk = copy.deepcopy(origin_imgs)
            homo_offset_bk = copy.deepcopy(homo_offset)
            view_ipm_img = draw_ipm_img(
                gt_online_mapping_ori,
                origin_imgs_bk,
                ipm_img_sizes,
                homo_offset_bk,
                view_bev_size,
                view_mat_vcs2bev,
                target_categories,
                block_warp_padding,
            )
        else:
            view_ipm_img = []
        # draw cliped gy
        view_cliped_gts = draw_clip_raw_gt(
            online_mapping_gt,
            view_bev_size,
            view_mat_vcs2bev,
            target_categories,
        )
        # draw mask
        gt_stats_copy = copy.deepcopy(gt_stats)
        view_mask = draw_dilate_mask(
            gt_stats_copy, view_bev_size, view_mat_vcs2bev
        )
        # draw labels
        group_lanes = parse_gt_pts(gt_stats_copy, top, left, res_h, res_w)
        view_head_imgs = (
            view_ipm_img + view_ori_gts + view_cliped_gts + view_mask
        )
        for head, head_infos in head_groups.items():
            multi_head = head_infos.get("multi", False)
            group = head_infos["group"]
            color_table = get_cls_color_table(head_infos)
            lanes = group_lanes[group]
            if multi_head:
                view_head = draw_multi_head(
                    lanes, head, view_bev_size, view_mat_vcs2bev, color_table
                )
                view_head_imgs += view_head
            elif view_sub_head:
                label_map = gt_stats_copy[group][head]
                view_head = draw_single_head(
                    lanes,
                    head,
                    label_map,
                    view_bev_size,
                    view_mat_vcs2bev,
                    color_table,
                )
                view_head_imgs += view_head

        # gather view images
        base_w = view_raw_img.shape[1]
        fuse_view_imgs = [view_raw_img]
        new_h = view_bev_h * base_w // (view_bev_w * view_cols)
        head_img = arrange_imgs(
            view_head_imgs, view_cols, new_h, base_w // view_cols
        )
        fuse_view_imgs.append(head_img)
        # save view img
        save_path = os.path.join(visualize_output_dir, image_rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, np.vstack(fuse_view_imgs))

        print("visualize gt time cost:", time.time() - tic)

    if merge_crosspoint:
        gt_stats["occ_segments"] = occ_segments

    return gt_stats


def point_to_line_distance(point, line_start, line_end):
    """Calculate the distance from a point to a line segment.

    Args:
        point: Coordinates of the point to calculate distance from,
            a tuple or list of shape (2,).
        line_start: Coordinates of the starting point of the line segment,
            a tuple or list of shape (2,).
        line_end: Coordinates of the ending point of the line segment,
            a tuple or list of shape (2,).
    """

    status = -1
    distance = -1

    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # if the line segment has zero length, return the distance
    # from the point to the starting point
    if line_length == 0:
        status = 0
        distance = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        projection_pts = None
        return status, distance, projection_pts

    # calculate the projection ratio of the point onto the line segment
    t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_length ** 2)

    # calculate the coordinates of the projected point
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)

    # if the projected point is on the line segment,
    # return the distance from the point to the projected point
    if t >= 0 and t <= 1:
        status = 1
        distance = math.sqrt((x - projection_x) ** 2 + (y - projection_y) ** 2)
    else:
        status = 0
        # return the minimum distance from the point to the
        # starting and ending points of the line segment
        distance_start = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        distance_end = math.sqrt((x - x2) ** 2 + (y - y2) ** 2)
        distance = min(distance_start, distance_end)

    return status, distance, (projection_x, projection_y)


def get_roi_vcs_range_box(bev_size, vcs_range, roi_vcs_range):
    """Get roi box when roi_vcs_range smaller than vcs_range.

    Args:
        bev_size: (h, w) input bev image size.
        vcs_range: (b, r, t, l) vcs range of input image.
        roi_vcs_range: (b, r, t, l) vcs range of trained model.

    Returns: (x, y, h, w) roi box.

    """
    spatial_resolution = (
        abs(vcs_range[2] - vcs_range[0]) / bev_size[0],
        abs(vcs_range[1] - vcs_range[3]) / bev_size[1],
    )

    top = int(
        decimal_div(vcs_range[2] - roi_vcs_range[2], spatial_resolution[0])
    )  # v
    left = int(
        decimal_div(vcs_range[3] - roi_vcs_range[3], spatial_resolution[1])
    )  # u
    bottom = int(
        decimal_div(vcs_range[2] - roi_vcs_range[0], spatial_resolution[0])
    )  # v
    right = int(
        decimal_div(vcs_range[3] - roi_vcs_range[1], spatial_resolution[1])
    )  # u

    return (top, left, bottom, right)
