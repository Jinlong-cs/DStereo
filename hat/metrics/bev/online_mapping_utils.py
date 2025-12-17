# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

try:
    from horizon_plugin_pytorch import om_extract
except ImportError:
    om_extract = None

__all__ = [
    "chamfer_distance",
    "convert_gt",
    "get_offset",
    "get_smoothing",
    "get_vcs_lane",
    "pts_format_dict",
]


def chamfer_distance(x, y, metric="l2", direction="bi", dist_thresh=10):
    r"""Chamfer distance between two point clouds.

    Args:
        x: numpy array [n_points_x, n_dims]
            first point cloud
        y: numpy array [n_points_y, n_dims]
            second point cloud
        metric: string or callable, default 'l2'
            metric to use for distance computation.
            Any metric from scikit-learn or scipy.spatial.distance can be used.
        direction: str
            direction of Chamfer distance.
                "y_to_x": computes average minimal distance
                          from every point in y to x
                "x_to_y": computes average minimal distance
                          from every point in x to y
                "bi": compute both
    Returns:
        chamfer_dist: float
            computed bidirectional Chamfer distance:
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


def convert_gt(gt_stats: dict, head_groups: dict) -> dict:
    """Convert gt stats to npy array and transform cls_id \
    inside branch to global.

    Args:
        stats: Gt stats contatin cls, embedding, r,
            cos, sin of each branch.
        head_groups: Output head configurations.

    Returns:
        Gt stats with prob, transformed cls_id and elements above.
    """

    group_cls_dims = {}
    group_sub_heads = {}
    for head, head_infos in head_groups.items():
        multi_head = head_infos.get("multi", False)
        group = head_infos["group"]
        if group not in group_sub_heads:
            group_sub_heads[group] = []
        if multi_head:
            num_class = max([v for _, v in head_infos["cls_remap"].items()])
            group_cls_dims[group] = num_class
        else:
            group_sub_heads[group].append(head)
    cls_offset = 0
    for group in gt_stats:
        sub_add_attrs = {}
        for attr in gt_stats[group]:
            if attr == "cls":
                positive_mask = gt_stats[group][attr] > 0
                gt_stats[group][attr][positive_mask] += cls_offset
                cls_offset += group_cls_dims[group]
            if isinstance(gt_stats[group][attr], torch.Tensor):
                gt_stats[group][attr] = gt_stats[group][attr].cpu().numpy()
            if attr in group_sub_heads[group]:
                sub_add_attrs[attr + "_cls"] = gt_stats[group][attr] + 1
                sub_add_attrs[attr + "_prob"] = (
                    gt_stats[group][attr] >= 0
                ).astype(np.float)
        if sub_add_attrs:
            gt_stats[group].update(sub_add_attrs)
        gt_stats[group]["prob"] = (gt_stats[group]["cls"] > 0).astype(np.float)
    return gt_stats


def get_target_categorys(head_groups: dict) -> dict:
    """Get taregt global categories from head groups info.

    Args:
        head_groups: Output head configurations.

    Returns:
        Global categories
    """
    target_categorys = {}
    for head, head_infos in head_groups.items():
        multi_head = head_infos.get("multi", False)
        if not multi_head:
            continue
        # get cls list
        cls_list = head_infos.get("cls_list", None)
        if cls_list is None:
            cls_list = head_infos["cls_remap"].keys()
        cls_list = list(cls_list)
        cls_offset = len(target_categorys)
        for i in range(len(cls_list)):
            cls_name = cls_list[i]
            target_categorys[f"{head}_{cls_name}"] = cls_offset + i + 1
    return target_categorys


def get_online_mapping_dict(lanes, cls_id_2_category, lanes_cls=None):
    # return x_vcs, y_vcs, cls, prob, sin, cos
    online_mapping_dict = {}
    for i in range(len(lanes)):
        lane = lanes[i]
        if len(lane) <= 2:
            continue
        lane = np.array(lane)
        if lanes_cls is not None and lanes_cls[i] is not None:
            cls_id = lanes_cls[i]
        else:
            counts = np.bincount(lane[:, 4].astype(np.int))
            cls_id = np.argmax(counts)
        category = cls_id_2_category[cls_id]
        if category not in online_mapping_dict:
            online_mapping_dict.update({category: []})
        online_mapping_dict[category].append(lane)
    return online_mapping_dict


def get_seq_lane(lane, res_h, res_w):
    """Get sequence lane."""
    step = max(res_h, res_w)
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


def seq_recurrent_search(
    lane,
    sub_seq,
    search_indexes,
    start_index,
    res_h,
    res_w,
    forward,
):
    """Search next valid sequence point.

    Args:
        lane: raw lane points set.
        sub_seq: generated sub indexes sequence.
        search_indexes: indexes lists to be searched.
        start_index: the index of start search point.
        res_h: the resolution of height grid.
        res_w: the resolution of width grid.
        forward: forward search or backward search.

    """
    step = max(res_h, res_w)
    near_del_thr = step * 0.75
    last_index = -1
    search_range_max = 8.0
    search_range_min = 3.0
    current_index = start_index
    if forward:
        sub_seq.clear()
        sub_seq.append(current_index)
    elif len(sub_seq) > 1:
        last_index = sub_seq[1]

    alpha = 5
    is_not_lane = False
    if lane[0][4] != 1:
        alpha = 2.0
        is_not_lane = True
    while search_indexes:
        # find next point
        next_index = -1
        x = lane[current_index][0]
        y = lane[current_index][1]
        # determine the search direction
        if last_index < 0:
            current_cos = lane[current_index][7]
            current_sin = lane[current_index][6]
            if forward ^ (current_sin > 0):
                direction = [current_cos, -current_sin]
            else:
                direction = [-current_cos, current_sin]
        else:
            last_pt = lane[last_index]
            direction = [y - last_pt[1], x - last_pt[0]]
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        target_x = x + direction[1] * step
        target_y = y + direction[0] * step
        # search nearest with direction
        min_dist = 1e8
        min_index = -1
        min_cross = -1
        near_indexes = []
        for i in range(len(search_indexes)):
            if search_indexes[i] < 0:
                continue
            pt = lane[search_indexes[i]]
            dx = pt[0] - x
            dy = pt[1] - y
            if abs(dx) > search_range_max or abs(dy) >= search_range_max:
                continue
            eul_dist = (dx ** 2 + dy ** 2) ** 0.5 + 1e-8
            if eul_dist < search_range_max:
                near_indexes.append(i)
            # check same direction: [0-180 degree] -> [1, -1] -> [1, 3]
            cross = (direction[0] * dy + direction[1] * dx) / eul_dist
            if cross < 0 and eul_dist > search_range_min:
                continue
            direction_dist = alpha * (2.0 - cross) ** 2
            eul_dist = (
                (pt[0] - target_x) ** 2 + (pt[1] - target_y) ** 2
            ) ** 0.5 + direction_dist
            if min_dist > eul_dist:
                min_dist = eul_dist
                min_index = i
                min_cross = cross
        # update search list
        if min_index >= 0 and (is_not_lane or min_cross > 0):
            target_inedx = search_indexes[min_index]
            # check border split
            next_index = target_inedx
            search_indexes[min_index] = -1
            for near_index in near_indexes:
                if near_index == min_index:
                    continue
                near_pt = lane[search_indexes[near_index]]
                near_dist = np.linalg.norm(near_pt[:2] - np.array([x, y]))
                if near_dist < near_del_thr:
                    search_indexes[near_index] = -1
        search_indexes = [x for x in search_indexes if x >= 0]
        if next_index >= 0:
            if forward:
                sub_seq.append(next_index)
            else:
                sub_seq.insert(0, next_index)
            last_index = current_index
            current_index = next_index
        else:
            break
    return search_indexes, sub_seq


def vcs_sequence(lane, res_h, res_w):
    """Lane sequence on vcs recurrently.

    Args:
        lane: raw lane points set.
        res_h: the resolution of height grid.
        res_w: the resolution of width grid.

    """
    pt_num = len(lane)
    if pt_num < 3:
        return [lane]

    # recursively search in one direction
    search_indexes = list(range(pt_num))
    lanes = []
    while search_indexes:
        start_index = search_indexes.pop()
        sub_seq = []
        # forward search
        search_indexes, sub_seq = seq_recurrent_search(
            lane,
            sub_seq,
            search_indexes,
            start_index,
            res_h,
            res_w,
            True,
        )
        # backward search
        search_indexes, sub_seq = seq_recurrent_search(
            lane,
            sub_seq,
            search_indexes,
            start_index,
            res_h,
            res_w,
            False,
        )
        # extract
        if len(sub_seq) > 1:
            sub_lane = []
            for index in sub_seq:
                sub_lane.append(lane[index])
            lanes.append(sub_lane)

    return lanes


def sequence_postprocess(lanes, res_h, res_w):
    """Lane sequence.

    Args:
        lanes: all lanes with points set.
        res_h: the resolution of height grid.
        res_w: the resolution of width grid.

    """
    lane_results = []
    for lane in lanes:
        post_lanes = vcs_sequence(lane, res_h, res_w)
        max_size = 3
        max_index = -1
        for i in range(len(post_lanes)):
            if len(post_lanes[i]) > max_size:
                max_size = len(post_lanes[i])
                max_index = i
        if max_index >= 0:
            lane_results.append(post_lanes[max_index])
    return lane_results


def get_vcs_lane(
    gt_stats,
    head_groups,
    out_h,
    out_w,
    top,
    left,
    res_h,
    res_w,
    target_categorys,
    pred_stats=None,
    embedding_dim=4,
    use_seq=False,
    use_seq_post=False,
):
    group_thrs = {}
    sub_heads = []
    for head, head_infos in head_groups.items():
        multi_head = head_infos.get("multi", False)
        group = head_infos["group"]
        if multi_head:
            group_thrs[group] = head_infos.get("cls_thr", "-1")
        else:
            sub_heads.append(head)
    sub_head_num = len(sub_heads)

    unique_id = []
    for group in gt_stats:
        unique_id += np.unique(gt_stats[group]["instance"]).tolist()
    instance_id_dict = {
        instance_id: i for i, instance_id in enumerate(np.unique(unique_id))
    }
    num_pred_lane = len(instance_id_dict)

    # check if has sequence order from sgc cluster
    has_seq_order = True
    for group in gt_stats:
        if "sequence" not in gt_stats[group]:
            has_seq_order = False

    cls_id_2_category = {v: k for k, v in target_categorys.items()}

    lanes = [[] for _ in range(num_pred_lane)]
    lanes_cls_info = [None for _ in range(num_pred_lane)]
    # sequence order from sgc cluster
    lanes_seq_order = [[] for _ in range(num_pred_lane)]
    unit_res = max(res_h, res_w)
    # the base info of a pt:
    # x_vcs, y_vcs, x_vcs_origin, y_vcs_origin, cls, prob, sin, cos
    base_pt_dim = 8
    # h, w, norm(offset)
    base_pos_dim = 3
    for group in gt_stats:
        cls_threshold = group_thrs[group]
        if om_extract is not None:
            # valid_pixels: [N, 11]
            #   (instance_id, c, h, w, cls, prob, x_vcs, y_vcs,
            #    x_vcs_origin, y_vcs_origin, offset_dist)
            # instance_info: [M, 4] (instance_id, instance_num, cls, prob)
            valid_pixels, instance_info = om_extract(
                gt_stats[group]["cls"],
                gt_stats[group]["prob"],
                gt_stats[group]["offset"],
                gt_stats[group]["instance"],
                top,
                left,
                res_h,
                res_w,
                cls_threshold,
            )
            # init lane instance
            pts_dim = (
                base_pt_dim + embedding_dim + base_pos_dim + sub_head_num * 2
            )
            current_lane = {}
            current_lane_pts_id = {}
            current_lane_seq_order = {}
            for index in range(instance_info.shape[0]):
                instance_id = int(instance_info[index][0] + 0.1)
                instance_id = instance_id_dict[instance_id]
                instance_size = int(instance_info[index][1] + 0.1)
                instance_cls = int(instance_info[index][2] + 0.1)
                current_lane[instance_id] = np.zeros((instance_size, pts_dim))
                current_lane_pts_id[instance_id] = 0
                current_lane_seq_order[instance_id] = [-1] * instance_size
                lanes_cls_info[instance_id] = instance_cls
            for index in range(valid_pixels.shape[0]):
                valid_pt = valid_pixels[index]
                instance_id = instance_id_dict[int(valid_pt[0] + 0.1)]
                pts_id = current_lane_pts_id[instance_id]
                seq_order_i = current_lane_seq_order[instance_id]
                current_lane_pts_id[instance_id] += 1
                lane_i = current_lane[instance_id]
                ch = int(valid_pt[1] + 0.1)
                h = int(valid_pt[2] + 0.1)
                w = int(valid_pt[3] + 0.1)
                lane_i[pts_id, 0] = valid_pt[6]  # x_vcs
                lane_i[pts_id, 1] = valid_pt[7]  # y_vcs
                lane_i[pts_id, 2] = valid_pt[8]  # x_vcs_origin
                lane_i[pts_id, 3] = valid_pt[9]  # y_vcs_origin
                lane_i[pts_id, 4] = valid_pt[4]  # cls
                lane_i[pts_id, 5] = valid_pt[5]  # prob
                lane_i[pts_id, 6] = gt_stats[group]["sin"][ch, h, w]  # sin
                lane_i[pts_id, 7] = gt_stats[group]["cos"][ch, h, w]  # cos
                offset = base_pt_dim
                if "embedding" in gt_stats[group]:
                    gt_emb = gt_stats[group]["embedding"][ch, h, w, :]
                    lane_i[pts_id, offset : offset + embedding_dim] = gt_emb
                if "sequence" in gt_stats[group]:
                    seq_order_i[pts_id] = gt_stats[group]["sequence"][ch, h, w]
                offset = base_pt_dim + embedding_dim
                lane_i[pts_id, offset] = h  # h
                lane_i[pts_id, offset + 1] = w  # w
                lane_i[pts_id, offset + 2] = valid_pt[-1]  # offset length
                offset = base_pt_dim + embedding_dim + base_pos_dim
                # process sub heads
                for k in range(sub_head_num):
                    sub_key = sub_heads[k] + "_cls"
                    if sub_key not in gt_stats[group]:
                        continue
                    label = gt_stats[group][sub_key][ch, h, w]
                    lane_i[pts_id, offset + k * 2] = label
                    if pred_stats:
                        label = pred_stats[group][sub_key][ch, h, w]
                        lane_i[pts_id, offset + k * 2 + 1] = label
            for inst_id in current_lane:
                lanes[inst_id] = current_lane[inst_id].tolist()
                lanes_seq_order[inst_id] = current_lane_seq_order[inst_id]
        else:
            for h in np.arange(0, out_h):
                for w in np.arange(0, out_w):
                    for ch in np.arange(0, gt_stats[group]["cls"].shape[0]):
                        if not (
                            gt_stats[group]["cls"][ch, h, w] > 0
                            and gt_stats[group]["prob"][ch, h, w]
                            >= cls_threshold
                        ):
                            continue
                        instance_id = instance_id_dict[
                            gt_stats[group]["instance"][ch, h, w]
                        ]
                        if instance_id <= 0:
                            continue
                        x_vcs_origin = top - h * res_h - res_h / 2
                        y_vcs_origin = left - w * res_w - res_w / 2
                        x_vcs = (
                            x_vcs_origin
                            + gt_stats[group]["offset"][ch, h, w, 0] * unit_res
                        )
                        y_vcs = (
                            y_vcs_origin
                            + gt_stats[group]["offset"][ch, h, w, 1] * unit_res
                        )
                        pt = [
                            x_vcs,
                            y_vcs,
                            x_vcs_origin,
                            y_vcs_origin,
                            gt_stats[group]["cls"][ch, h, w],
                            gt_stats[group]["prob"][ch, h, w],
                            gt_stats[group]["sin"][ch, h, w],
                            gt_stats[group]["cos"][ch, h, w],
                        ]
                        if "embedding" in gt_stats[group]:
                            embedding = gt_stats[group]["embedding"][
                                ch, h, w, :
                            ]
                            pt += embedding.tolist()
                        else:
                            pt += [-1] * embedding_dim
                        offset = gt_stats[group]["offset"][ch, h, w, :]
                        pt += [h, w, np.linalg.norm(offset)]
                        # process sub heads
                        sub_labels = [0] * sub_head_num * 2
                        for k in range(sub_head_num):
                            sub_key = sub_heads[k] + "_cls"
                            if sub_key not in gt_stats[group]:
                                continue
                            sub_labels[k * 2] = gt_stats[group][sub_key][
                                ch, h, w
                            ]
                            if pred_stats:
                                sub_labels[k * 2 + 1] = pred_stats[group][
                                    sub_key
                                ][ch, h, w]
                        pt += sub_labels
                        lanes[instance_id].append(pt)
                        seq_order = -1
                        if "sequence" in gt_stats[group]:
                            seq_order = gt_stats[group]["sequence"][ch, h, w]
                        lanes_seq_order[instance_id].append(seq_order)

    # get raw result
    online_mapping_dict = get_online_mapping_dict(
        lanes,
        cls_id_2_category,
        lanes_cls_info,
    )

    online_mapping_dict_seq = None

    if use_seq and not has_seq_order:
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
        if use_seq_post:
            seq_lanes = sequence_postprocess(sorted_lanes, res_h, res_w)
        else:
            for lane in sorted_lanes:
                seq_lane = get_seq_lane(lane, res_h, res_w)
                seq_lanes.append(seq_lane)
        # convert
        online_mapping_dict_seq = get_online_mapping_dict(
            seq_lanes, cls_id_2_category
        )
    elif has_seq_order:
        # sort lanes
        seq_lanes = []
        for lane, lane_seq_order in zip(lanes, lanes_seq_order):
            if len(lane) == 0:
                continue
            sorted_lane = [x for _, x in sorted(zip(lane_seq_order, lane))]
            seq_lanes.append(sorted_lane)
        # convert
        online_mapping_dict_seq = get_online_mapping_dict(
            seq_lanes, cls_id_2_category
        )

    return online_mapping_dict, online_mapping_dict_seq, group_thrs


def get_smoothing(online_mapping, deg=3):
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


def pts_format_dict(lane, lane_pts_attr_start_end, sub_head):
    formatted_list = []
    lane_list = lane.tolist()
    for pt in lane_list:
        pt_dict = {}
        for key, (start, end) in lane_pts_attr_start_end.items():
            if key in sub_head or key in ["cls", "bev"]:
                pt_dict[key] = [int(val) for val in pt[start:end]]
            else:
                pt_dict[key] = pt[start:end]
        formatted_list.append(pt_dict)
    return formatted_list
