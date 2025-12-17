# Copyright (c) Horizon Robotics. All rights reserved.

import math

import numpy as np
import torch
from sklearn.cluster import DBSCAN

try:
    from horizon_plugin_pytorch import om_ogc
except ImportError:
    om_ogc = None

__all__ = [
    "pred_preprocess",
    "convert_pred",
    "get_offset",
]


def ogc_diversity(
    pixel1: np.array,
    pixel2: np.array,
    pose_weight: float = 0.1,
    split_channel: bool = False,
) -> float:
    """
    Compute the diversity between two om pixels.

    Args:
        pixel1: [prob, row, col, channel, r, sin, cos, embedding]
        pixel2: [prob, row, col, channel, r, sin, cos, embedding]
        pose_weight: the weight of pose diversity
        split_channel: if running cluster on different channel

    Returns:
        diversity: float value
    """

    if split_channel and pixel1[3] != pixel2[3]:
        return 1e8

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
    pose_weight: float = 0.1,
    cluster_thr: float = 0.9,
    split_channel: bool = False,
) -> np.array:
    """
    OGC (Offset Growth Cluster) for Online Mapping Post Process.

    Notice: Generally, the channel C means set num, the default value is 2

    Args:
        pred_cls: [C, H, W], class label
        pred_prob: [C, H, W], class confidence
        pred_r: [C, H, W], offset radius
        pred_sin: [C, H, W], offset sin value
        pred_cos: [C, H, W], offset cos value
        pred_embedding: [C, H, W, D], embedding features for instance
        cls_thr: used to select valid pixels from pred_probs
        radius_l: the radius of longitudinal searching
        radius_t: the radius of transverse searching
        min_num: the minimum number of clustering points
        pose_weight: the weight of computing pose diversity
        cluster_thr: the threshold of point similarly
        split_channel: if running cluster on different channel

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
                        d = ogc_diversity(
                            anchor_pixel, pixel, pose_weight, split_channel
                        )
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


def sgc_search_local(
    key_points,
    channel_map,
    index_map,
    root_index,
    search_space,
    split_channel,
    cluster_thr,
    cluster_buffer_thr=0,
    aux_fea=None,
):
    """Find points that satisfy the requirement in local space.

    Args:
        key_points: all valid points
        channel_map: the map recording channel occupancy
        index_map: the map recording indexes of points
        root_index: the index of root point
        search_space: the local space of searching
        split_channel: bool, if running cluster on different channel
        cluster_thr: float, the threshold of point similarly
        cluster_buffer_thr: float, the redundant threshold of point similarly
        aux_fea: the auxiliary feature of computing points similarly

    Returns:
        searched_indexs: the indexes of searched points
    """

    searched_indexs = []
    root_pt = key_points[root_index]
    root_row = root_pt["row"]
    root_col = root_pt["col"]
    root_ch = root_pt["channel"]
    root_fea = root_pt["fea"]
    rows, cols = channel_map.shape
    channel_check = 1 << root_ch if split_channel else (1 << 0) + (1 << 1)
    channels_list = {1: [0], 2: [1], 3: [0, 1]}
    if split_channel:
        channels_list[3] = [root_ch]
    for grid in search_space:
        offset_x, offset_y = grid
        row = root_row - offset_x
        col = root_col - offset_y
        if row < 0 or col < 0 or row >= rows or col >= cols:
            continue
        channel_type = channel_map[row, col]
        if channel_type & channel_check == 0:
            continue
        for channel in channels_list[channel_type]:
            # check if this channel still has valid point
            if (channel_type & (1 << channel)) == 0:
                continue
            index = index_map[row, col, channel] - 1
            pt = key_points[index]
            # embedding feature distance
            fea_dist = np.linalg.norm(root_fea - pt["fea"])
            if fea_dist >= cluster_thr and aux_fea is not None:
                fea_dist = np.linalg.norm(aux_fea - pt["fea"])
            if fea_dist < cluster_thr:
                searched_indexs.append([index, True])
            elif fea_dist < cluster_buffer_thr:
                searched_indexs.append([index, False])

    return searched_indexs


def sgc_get_root_direction(
    key_points,
    channel_map,
    index_map,
    root_index,
    search_space,
    split_channel,
    cluster_thr,
):
    """Find the distribution direction of points in local space.

    Args:
        key_points: all valid points
        channel_map: the map recording channel occupancy
        index_map: the map recording indexes of points
        root_index: the index of root point
        search_space: the space of searching
        split_channel: bool, if running cluster on different channel
        cluster_thr: float, the threshold of point similarly

    Returns:
        forward_direction: the direction of searching forward
        backward_direction: the direction of searching backward
    """

    cluster_buffer_thr = min(cluster_thr * 1.5, cluster_thr + 0.5)
    searched_indexs = sgc_search_local(
        key_points,
        channel_map,
        index_map,
        root_index,
        search_space,
        split_channel,
        cluster_thr,
        cluster_buffer_thr,
    )
    direction_sum = 0
    direction_weights_sum = 1e-6
    root_pt = key_points[root_index]
    for index, inner_flag in searched_indexs:
        pt = key_points[index]
        dir = np.arctan2(pt["y"] - root_pt["y"], pt["x"] - root_pt["x"])
        if abs(dir) > np.pi * 0.5:
            dir = dir - np.pi if dir > 0 else dir + np.pi
        weight = 1 if inner_flag else 0.5
        direction_weights_sum += weight
        direction_sum += weight * dir
    # compute main direction
    main_direction = direction_sum / direction_weights_sum
    if main_direction > 0:
        forward_direction = main_direction
        backward_direction = main_direction + np.pi
    else:
        forward_direction = main_direction + 2 * np.pi
        backward_direction = main_direction + np.pi
    dir_num = len(searched_indexs)
    return forward_direction, backward_direction, dir_num


def sgc_recurrent_search(
    key_points,
    channel_map,
    index_map,
    root_index,
    inner_search_bin_spaces,
    outer_search_bin_spaces,
    split_channel,
    cluster_thr,
    inner_thr,
    root_direction,
    is_forward,
    cluster_indexes,
):
    """Search in one direction.

    Args:
        key_points: all valid points
        channel_map: the map recording channel occupancy
        index_map: the map recording indexes of points
        root_index: the index of root point
        inner_search_bin_spaces: pre-built search space of inner range
        outer_search_bin_spaces: pre-built search space of outer range
        split_channel: bool, if running cluster on different channel
        cluster_thr: float, the threshold of point similarly
        inner_thr: float, the distance threshold of find inlier points
        root_direction: the init searching direction
        is_forward: search forward or backward
        cluster_indexes: the points indexes set of a valid cluster

    Returns:
        cluster_indexes: list, the points indexes set of a valid cluster
    """

    bins = len(inner_search_bin_spaces)
    angle_stride = np.pi * 2 / bins

    start_index = root_index
    start_direction = root_direction
    start_pt = key_points[start_index]
    fea_count = 1.0
    fea_sum = start_pt["fea"].copy()
    aux_fea = None
    cluster_buffer_thr = min(cluster_thr * 1.5, cluster_thr + 0.5)
    seq_length = 0
    last_pos = np.array([start_pt["x"], start_pt["y"]])
    while True:
        # compute bin
        bin = min(max(int(start_direction / angle_stride), 0), bins - 1)
        # local search
        search_space = inner_search_bin_spaces[bin]
        searched_indexs = sgc_search_local(
            key_points,
            channel_map,
            index_map,
            start_index,
            search_space,
            split_channel,
            cluster_thr,
            cluster_buffer_thr,
            aux_fea,
        )
        if len(searched_indexs) == 0:
            search_space = outer_search_bin_spaces[bin]
            searched_indexs = sgc_search_local(
                key_points,
                channel_map,
                index_map,
                start_index,
                search_space,
                split_channel,
                cluster_thr,
                0,
                aux_fea,
            )
        # find the farthest point in the same direction
        next_index = -1
        next_direction = 0
        next_priority = -1e4
        start_pt = key_points[start_index]
        for index_pair in searched_indexs:
            index, inner_flag = index_pair
            if not inner_flag:
                continue
            pt = key_points[index]
            dx = pt["x"] - start_pt["x"]
            dy = pt["y"] - start_pt["y"]
            dist = np.hypot(dx, dy)
            if dist < 1e-2:
                continue
            direction = np.arctan2(dy, dx)
            if direction < 0:
                direction += 2 * np.pi
            direction_diff = np.abs(direction - start_direction)
            direction_diff = np.mod(direction_diff, 2 * np.pi)
            if direction_diff > np.pi:
                direction_diff = 2 * np.pi - direction_diff
            priority = dist - direction_diff
            if priority > next_priority:
                next_priority = priority
                next_index = index
                next_direction = direction
        if next_index < 0:
            break
        # filter outlier points
        next_pt = key_points[next_index]
        A = next_pt["y"] - start_pt["y"]
        B = start_pt["x"] - next_pt["x"]
        C = next_pt["x"] * start_pt["y"] - start_pt["x"] * next_pt["y"]
        square_d = A * A + B * B + 1e-8
        norm_d = np.sqrt(square_d)
        inner_info = []
        for index_pair in searched_indexs:
            index = index_pair[0]
            pt = key_points[index]
            dx = pt["x"] - start_pt["x"]
            dy = pt["y"] - start_pt["y"]
            dist = np.abs((A * pt["x"] + B * pt["y"] + C) / norm_d)
            if dist > inner_thr:
                continue
            alpha = (A * dy - B * dx) / square_d
            inner_info.append([index, alpha])
        if len(inner_info) == 0:
            break
        inner_info = sorted(inner_info, key=lambda x: x[1])
        # update cluster status
        for inner_item in inner_info:
            index = inner_item[0]
            inner_pt = key_points[index]
            row, col = inner_pt["row"], inner_pt["col"]
            channel = inner_pt["channel"]
            channel_map[row, col] -= 1 << channel
            if is_forward:
                cluster_indexes.append(index)
            else:
                cluster_indexes.insert(0, index)
            # update feature sum
            fea_sum += inner_pt["fea"]
            fea_count += 1
            # update sequence length
            new_pos = np.array([inner_pt["x"], inner_pt["y"]])
            seq_length += np.linalg.norm(last_pos - new_pos)
            last_pos = new_pos
        # prepare for next search
        start_index = inner_info[-1][0]
        start_direction = next_direction
        aux_fea = fea_sum / fea_count

    return cluster_indexes, seq_length


def sgc_cluster(
    pred_cls: np.array,
    pred_prob: np.array,
    pred_r: np.array,
    pred_sin: np.array,
    pred_cos: np.array,
    pred_embedding: np.array,
    cls_thr: float,
    vcs_range: list,
    radius_l: int = 5,
    radius_t: int = 2,
    direction_range: int = 90,
    min_length: float = 2.5,
    inner_thr: float = 0.5,
    cluster_thr: float = 0.9,
    split_channel: bool = False,
):
    """
    SGC (Sequence Growth Cluster) for Online Mapping Post Process.

    Notice: Generally, the channel C means set num, the default value is 2

    Args:
        pred_cls: [C, H, W], class label
        pred_prob: [C, H, W], class confidence
        pred_r: [C, H, W], offset radius
        pred_sin: [C, H, W], offset sin value
        pred_cos: [C, H, W], offset cos value
        pred_embedding: [C, H, W, D], embedding features for instance
        cls_thr: used to select valid pixels from pred_probs
        vcs_range: vcs range, in meter.(order is
            (bottom,right,top,left), e.g.(-30, -51.2, 72.4, 51.2))
        radius_l: the radius of longitudinal searching
        radius_t: the radius of transverse searching
        direction_range: the angle range of searching
        min_length: the minimum length of lane
        inner_thr: the distance threshold of find inlier points
        cluster_thr: the threshold of point similarly
        split_channel: if running cluster on different channel

    Returns:
        cluster_result: [C, H, W], instance id (0: means background or noise)
        sequence_result: [C, H, W], sequence order of each lane
    """

    # init: default output
    C, H, W = pred_cls.shape
    assert C <= 2, "out channels must <= 2!"
    cluster_result = np.zeros((C, H, W), dtype=np.int32)
    sequence_result = np.zeros((C, H, W), dtype=np.int32)

    # init: compute local search space with direction
    bins = 36
    outer_radius_l = radius_l * 2 - 1
    coarse_map_x, coarse_map_y = np.meshgrid(
        np.linspace(-outer_radius_l, outer_radius_l, outer_radius_l * 2 + 1),
        np.linspace(-outer_radius_l, outer_radius_l, outer_radius_l * 2 + 1),
    )
    coarse_grids = np.concatenate(
        [coarse_map_x[:, :, np.newaxis], coarse_map_y[:, :, np.newaxis]],
        axis=2,
    ).astype(np.int)
    coarse_map_direction = np.arctan2(coarse_map_y, coarse_map_x)
    coarse_map_direction[coarse_map_direction < 0] += 2 * np.pi
    inner_direction_range = (direction_range + 0.5) / 180.0 * np.pi
    outer_direction_range = (direction_range * 0.5 + 0.5) / 180.0 * np.pi
    local_x_mask = np.abs(coarse_map_x) <= radius_l
    local_y_mask = np.abs(coarse_map_y) <= radius_l
    local_inner_mask = np.logical_and(local_x_mask, local_y_mask)
    local_outer_mask = ~local_inner_mask
    local_search_inner_grids = []
    local_search_outer_grids = []
    for b in range(bins):
        direction = 2 * np.pi / bins * (b + 0.5)
        direction_diff = np.abs(coarse_map_direction - direction)
        direction_diff = np.minimum(2 * np.pi - direction_diff, direction_diff)
        inner_direction_mask = direction_diff < inner_direction_range
        outer_direction_mask = direction_diff < outer_direction_range
        sin_v = np.sin(direction)
        cos_v = np.cos(direction)
        dist = np.abs(sin_v * coarse_map_x - cos_v * coarse_map_y)
        dist_mask = dist < radius_t + 0.5
        dist_mask[outer_radius_l, outer_radius_l] = False
        inner_filter_mask = np.logical_and(inner_direction_mask, dist_mask)
        inner_filter_mask = np.logical_and(inner_filter_mask, local_inner_mask)
        inner_filter_grids = coarse_grids[inner_filter_mask, :].tolist()
        local_search_inner_grids.append(inner_filter_grids)
        outer_filter_mask = np.logical_and(outer_direction_mask, dist_mask)
        outer_filter_mask = np.logical_and(outer_filter_mask, local_outer_mask)
        outer_filter_grids = coarse_grids[outer_filter_mask, :].tolist()
        local_search_outer_grids.append(outer_filter_grids)
    local_inner_mask[outer_radius_l, outer_radius_l] = False
    root_seqrch_grids = coarse_grids[local_inner_mask, :].tolist()

    # get vcs range
    bottom, right, top, left = vcs_range
    res_h = (top - bottom) / H
    res_w = (left - right) / W
    unit_res = max(res_h, res_w)
    min_num = max(int(min_length / unit_res + 0.5) + 1, 4)
    # get foreground class
    pos = pred_prob > cls_thr
    cls_labels = np.unique(pred_cls).tolist()
    cls_labels.sort()
    # core: run cluster with direction sequence
    cluster_id = 0
    for cls_label in cls_labels:
        # generate valid pixels mask
        mask = np.logical_and(pos, pred_cls == cls_label)
        valid_num = mask.astype(np.int32).sum()
        if valid_num <= min_num:
            continue
        key_points = []
        # extract all valid points
        valid_pos = np.where(mask)
        for index in range(valid_num):
            channel = valid_pos[0][index]
            row = valid_pos[1][index]
            col = valid_pos[2][index]
            key_pt = {}
            key_pt["row"] = row
            key_pt["col"] = col
            key_pt["channel"] = channel
            key_pt["prob"] = pred_prob[channel, row, col]
            key_pt["r"] = pred_r[channel, row, col]
            key_pt["sin"] = pred_sin[channel, row, col]
            key_pt["cos"] = pred_cos[channel, row, col]
            x_origin = top - row * res_h - res_h / 2
            y_origin = left - col * res_w - res_w / 2
            key_pt["x"] = x_origin - key_pt["r"] * key_pt["cos"] * unit_res
            key_pt["y"] = y_origin - key_pt["r"] * key_pt["sin"] * unit_res
            key_pt["dist"] = np.hypot(key_pt["x"], key_pt["y"])
            key_pt["fea"] = pred_embedding[channel, row, col, :]
            key_points.append(key_pt)
        valid_num = len(key_points)
        # sort key_points with distance from ego car
        key_points = sorted(key_points, key=lambda x: x["dist"])
        # build memory map to store forground points information
        channel_map = np.zeros((H, W), dtype=np.uint8)
        index_map = np.zeros((H, W, C), dtype=np.int32)
        # build memory map
        for index in range(valid_num):
            row = key_points[index]["row"]
            col = key_points[index]["col"]
            channel = key_points[index]["channel"]
            index_map[row, col, channel] = index + 1
            channel_map[row, col] += 1 << channel
        # core search loop
        # init flags as -1: UNCLASSIFIED; -2: NOISE, >=0: CLUSTER_ID
        cluster_flags = [-1] * valid_num
        for index in range(valid_num):
            if cluster_flags[index] != -1:
                continue
            root_index = index
            # clear root point
            root_pt = key_points[root_index]
            root_channel = root_pt["channel"]
            channel_map[root_pt["row"], root_pt["col"]] -= 1 << root_channel
            cluster_index = [root_index]
            # find local init direction
            forward_dir, backward_dir, dir_num = sgc_get_root_direction(
                key_points,
                channel_map,
                index_map,
                root_index,
                root_seqrch_grids,
                split_channel,
                cluster_thr,
            )
            if dir_num < 2:
                continue
            # recurrent search
            cluster_index, forward_length = sgc_recurrent_search(
                key_points,
                channel_map,
                index_map,
                root_index,
                local_search_inner_grids,
                local_search_outer_grids,
                split_channel,
                cluster_thr,
                inner_thr,
                forward_dir,
                True,
                cluster_index,
            )
            cluster_index, backward_length = sgc_recurrent_search(
                key_points,
                channel_map,
                index_map,
                root_index,
                local_search_inner_grids,
                local_search_outer_grids,
                split_channel,
                cluster_thr,
                inner_thr,
                backward_dir,
                False,
                cluster_index,
            )
            # remove noise or shorte line
            length = forward_length + backward_length
            if len(cluster_index) < min_num or length < min_length:
                for seq_index in cluster_index:
                    cluster_flags[seq_index] = -2
                continue
            # update cluster map
            cluster_id += 1
            for seq_id, seq_index in enumerate(cluster_index):
                cluster_flags[seq_index] = cluster_id
                pt = key_points[seq_index]
                row, col, channel = pt["row"], pt["col"], pt["channel"]
                cluster_result[channel, row, col] = cluster_id
                sequence_result[channel, row, col] = seq_id + 1

    return cluster_result, sequence_result


def embedding_post_process(
    embedding: np.array,
    bin_seg: np.array,
    cluster_alg: str = "dbscan",
    band_width: float = 1.5,
    max_num_lane=None,
) -> np.array:
    """
    First use mean shift to find dense cluster center.

    Args:
        embedding: [H, W, embed_dim]
        bin_seg: [H, W], each pixel is 0 or 1, 0 for background pixel
        delta_v: coordinates within distance of 2*delta_v to cluster center are

    Returns:
        cluster_result: [H, W], index of different lanes on each pixel
    """
    cluster_result = np.zeros(bin_seg.shape, dtype=np.int32)
    cluster_list = embedding[bin_seg > 0]  # 64*64*2, 4
    if len(cluster_list) == 0:
        return cluster_result
    if cluster_alg == "dbscan":
        alg = DBSCAN(eps=band_width, min_samples=1)
    else:
        raise NotImplementedError
    alg.fit(cluster_list)
    labels = alg.labels_
    cluster_result[bin_seg > 0] = labels + 1
    return cluster_result


def get_offset(stats: dict, eps: float = 1e-5) -> dict:
    """Get x, y offset related to gird center.

    Args:
        stats: Gt or pred stats contatin cls, embedding, r,
            cos, sin of each branch.
        eps: Avoid zero div. Defaults to 1e-5.

    Returns:
        Gt or pred stats with x, y offset, actan(theta)
            and elements above.
    """
    for group in stats:
        pred_x = -stats[group]["r"] * stats[group]["cos"]
        pred_y = -stats[group]["r"] * stats[group]["sin"]
        stats[group]["offset"] = np.concatenate(
            (pred_x[..., np.newaxis], pred_y[..., np.newaxis]), axis=-1
        )
        stats[group]["direction"] = (
            np.arctan(stats[group]["sin"] / (stats[group]["cos"] + eps))
            / np.pi
            + 0.5
        )
    return stats


def get_cls_prob(pred_cls, loss_type, max_patch, is_sub_task=False):
    """Process cls head to get class label prob according to loss type.

    Args:
        pred_cls: Pred cls data map.
        loss_type: the type of loss.
        max_patch: the max patch segment num.
        is_sub_task: if True, there is no background cls.
    """
    pred_data_list = [pred_cls]
    if max_patch == 2:
        pred_data_list = torch.chunk(pred_cls, 2, dim=1)

    pred_cls_list = []
    pred_prob_list = []
    for pred_data in pred_data_list:
        if loss_type == "ce_loss":
            pred_prob = torch.softmax(pred_data, dim=1).cpu().detach().numpy()
            if not is_sub_task:
                pred_prob = pred_prob[:, 1:, ...]
        elif loss_type == "focal_loss":
            pred_prob = torch.sigmoid(pred_data).cpu().detach().numpy()
        else:
            raise NotImplementedError
        prob_data = np.max(pred_prob, axis=1)
        cls_data = np.argmax(pred_prob, axis=1).astype(np.int) + 1
        pred_cls_list.append(cls_data)
        pred_prob_list.append(prob_data)

    pred_cls = np.concatenate(pred_cls_list, axis=0)
    pred_prob = np.concatenate(pred_prob_list, axis=0)
    return pred_cls, pred_prob


def pred_preprocess(batch_id, pred, head_groups):
    """Process cls head to get class label and corresponding score.

    Args:
        pred_stats: Pred stats contatin cls, embedding, r,
            cos, sin of each branch.
        head_groups: OM heads configs.
    """
    # get group loss type
    group_loss_type = {}
    group_max_patch = {}
    group_sub_heads = {}
    for head, head_infos in head_groups.items():
        group = head_infos["group"]
        group_head = head_groups[group]
        loss_type = group_head.get("loss_type", "ce_loss")
        max_patch_segment = group_head.get("max_patch_segment", 1)
        multi_head = head_infos.get("multi", False)
        if group not in group_sub_heads:
            group_sub_heads[group] = []
        if multi_head:
            group_loss_type[group] = loss_type
            group_max_patch[group] = max_patch_segment
        else:
            group_sub_heads[group].append(head)

    pred_stats = {}
    for group in group_loss_type:
        pred_stats[group] = {}
        pred_cls = pred[f"pred_online_mapping_cls_{group}_frame0"][0][
            batch_id : batch_id + 1
        ]
        pred_instance = pred[f"pred_online_mapping_instance_{group}_frame0"][
            0
        ][batch_id : batch_id + 1]
        pred_r = pred[f"pred_online_mapping_r_{group}_frame0"][0][
            batch_id : batch_id + 1
        ]
        pred_sin = pred[f"pred_online_mapping_sin_{group}_frame0"][0][
            batch_id : batch_id + 1
        ]
        pred_cos = pred[f"pred_online_mapping_cos_{group}_frame0"][0][
            batch_id : batch_id + 1
        ]
        # preprocess cls and corresponding prob
        loss_type = group_loss_type[group]
        max_patch = group_max_patch[group]
        pred_cls, pred_prob = get_cls_prob(pred_cls, loss_type, max_patch)
        # process instance
        if max_patch == 2:
            slice_a, slice_b = torch.chunk(pred_instance, 2, dim=1)
            # 1,8,64,64 -> 2,4,64,64
            pred_instance = torch.cat((slice_a, slice_b))
        pred_instance = (
            pred_instance.cpu().detach().numpy().transpose((0, 2, 3, 1))
        )

        # update main group info
        pred_stats[group]["prob"] = pred_prob
        pred_stats[group]["cls"] = pred_cls
        # preprocess embedding: set * h * w * embedding_size
        pred_stats[group]["instance"] = pred_instance
        # prcess regression output
        pred_stats[group]["r"] = pred_r[0].cpu().detach().numpy()
        pred_stats[group]["sin"] = pred_sin[0].cpu().detach().numpy()
        pred_stats[group]["cos"] = pred_cos[0].cpu().detach().numpy()

        # update sub heads
        for head in group_sub_heads[group]:
            head_key = f"pred_online_mapping_{head}_{group}_frame0"
            head_cls = pred[head_key][0][batch_id : batch_id + 1]
            head_cls, head_prob = get_cls_prob(
                head_cls,
                loss_type,
                max_patch,
                True,
            )
            pred_stats[group][head + "_cls"] = head_cls
            pred_stats[group][head + "_prob"] = head_prob
    return pred_stats


def convert_pred(
    pred_stats: dict,
    head_groups: dict,
    vcs_range: list,
    cluster_alg: str,
    cluster_cfg: dict,
    cluster_split_channel: bool,
) -> dict:
    """Cluster lane instance and transform cls_id.

    Args:
        pred_stats: Pred stats contatin cls, embedding, r,
            cos, sin of each branch.
        head_groups: OM heads configs.
        vcs_range: vcs range, in meter.(order is
            (bottom,right,top,left), e.g.(-30, -51.2, 72.4, 51.2))
        cluster_alg: Cluster algorithm, such as ogc, dbscan.
        cluster_cfg: Cluster cofig.
        cluster_split_channel: if run cluster on different channel.

    Returns:
        Pred stats with instance result and
            elements above.
    """
    # get group threshold and cls num
    group_thrs = {}
    group_cls_dims = {}
    for _, head_infos in head_groups.items():
        multi_head = head_infos.get("multi", False)
        group = head_infos["group"]
        if multi_head:
            cls_thr = head_infos.get("cls_thr", "-1")
            group_thrs[group] = cls_thr
            num_class = max([v for _, v in head_infos["cls_remap"].items()])
            group_cls_dims[group] = num_class

    cls_offset = 0
    global_instance_offset = 0
    global_instance_id = 0
    for group in pred_stats:
        pred_stats[group]["embedding"] = pred_stats[group]["instance"].copy()
        pred_stats[group]["instance"] = np.zeros_like(pred_stats[group]["cls"])
        if cluster_alg == "ogc":
            if om_ogc is None:
                pred_stats[group]["instance"] = ogc_cluster(
                    pred_stats[group]["cls"],
                    pred_stats[group]["prob"],
                    pred_stats[group]["r"],
                    pred_stats[group]["sin"],
                    pred_stats[group]["cos"],
                    pred_stats[group]["embedding"],
                    cls_thr=group_thrs[group],
                    radius_l=9,
                    radius_t=2,
                    min_num=1,
                    pose_weight=cluster_cfg["pose_weights"][group],
                    cluster_thr=cluster_cfg["cluster_bw"],
                    split_channel=cluster_split_channel,
                )
            else:
                cls_tensor = torch.tensor(
                    pred_stats[group]["cls"],
                    dtype=torch.int32,
                )
                prob_tensor = torch.tensor(pred_stats[group]["prob"])
                r_tensor = torch.tensor(pred_stats[group]["r"])
                sin_tensor = torch.tensor(pred_stats[group]["sin"])
                cos_tensor = torch.tensor(pred_stats[group]["cos"])
                ins_tensor = torch.tensor(pred_stats[group]["embedding"])
                pred_stats[group]["instance"] = om_ogc(
                    cls_tensor,
                    prob_tensor,
                    r_tensor,
                    sin_tensor,
                    cos_tensor,
                    ins_tensor,
                    cls_thr=group_thrs[group],
                    cls_num=group_cls_dims[group],
                    radius_l=9,
                    radius_t=2,
                    min_num=1,
                    pose_weight=cluster_cfg["pose_weights"][group],
                    cluster_thr=cluster_cfg["cluster_bw"],
                    merge=True,
                    split_channel=cluster_split_channel,
                ).numpy()
                # update instance id with global instance offset
                mask = pred_stats[group]["instance"] > 0
                instance_num = np.max(pred_stats[group]["instance"])
                pred_stats[group]["instance"][mask] += global_instance_offset
                global_instance_offset += instance_num
        elif cluster_alg == "sgc":
            min_length = 2.5 if "lane" in group else 2.0
            # Notice: if top range > 40, treat it as driving
            if vcs_range[2] > 40:
                min_length *= 2
            cluster_result, sequence_result = sgc_cluster(
                pred_stats[group]["cls"],
                pred_stats[group]["prob"],
                pred_stats[group]["r"],
                pred_stats[group]["sin"],
                pred_stats[group]["cos"],
                pred_stats[group]["embedding"],
                cls_thr=group_thrs[group],
                vcs_range=vcs_range,
                radius_l=5,
                radius_t=2,
                direction_range=90,
                min_length=min_length,
                inner_thr=0.5,
                cluster_thr=cluster_cfg["cluster_bw"],
                split_channel=cluster_split_channel,
            )
            pred_stats[group]["instance"] = cluster_result
            pred_stats[group]["sequence"] = sequence_result
            # update instance id with global instance offset
            mask = pred_stats[group]["instance"] > 0
            instance_num = np.max(pred_stats[group]["instance"])
            pred_stats[group]["instance"][mask] += global_instance_offset
            global_instance_offset += instance_num
        else:
            channels = pred_stats[group]["embedding"].shape[0]
            channel_indexes = range(channels)
            if not cluster_split_channel:
                channel_indexes = [channel_indexes]
            for ch in channel_indexes:
                pred_cls = pred_stats[group]["cls"][ch]
                pred_prob = pred_stats[group]["prob"][ch]
                pred_embedding = pred_stats[group]["embedding"][ch]
                # make sure the instance memory keep contiguous
                pred_instance = pred_stats[group]["instance"]
                if cluster_split_channel:
                    pred_instance = pred_instance[ch]

                if cluster_alg == "dbscan":
                    pos = pred_prob >= group_thrs[group]
                    cls_labels = np.unique(pred_cls).tolist()
                    cls_labels.sort()
                    for cls_label in cls_labels:
                        bin_seg = np.logical_and(
                            pos, pred_cls == cls_label
                        ).astype(np.int)
                        sub_instance = embedding_post_process(
                            pred_embedding,
                            bin_seg,
                            band_width=cluster_cfg["cluster_bw"],
                        )
                        for sub_instance_id in np.unique(sub_instance):
                            ins = np.logical_and(
                                bin_seg, sub_instance == sub_instance_id
                            )
                            if ins.sum() == 0:
                                continue
                            global_instance_id += 1
                            pred_instance[ins] = global_instance_id
                else:
                    raise NotImplementedError
        # convert cls with group offset
        pred_stats[group]["cls"] += cls_offset
        cls_offset += group_cls_dims[group]
    return pred_stats
