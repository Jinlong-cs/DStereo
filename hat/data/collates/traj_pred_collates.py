# Copyright (c) Horizon Robotics. All rights reserved.

import os
from typing import Dict, List

import numpy as np
import torch

from hat.core.traj_pred_typing import SeqIndex

__all__ = [
    "collate_multipath",
    "collate_multipath_viz",
    "collate_uniformpath_viz",
    "collate_vectornet",
    "collate_vectornet_viz",
    "bev_gt_path_function",
    "gt_path_func_for_data_pipeline_v3",
    "CatStackCollator",
    "lmdb_path_replace_func_for_data_pipeline_v3",
    "lmdb_path_replace_func_for_data_pipeline_v5",
    "collate_SGNet",
]


def collate_multipath(batch: List[Dict]) -> Dict:
    """Collate function for Multipath trajectory prediction model.

    Args:
        batch: a list of dataloader output.

    Returns:
        batch_data: the dictionary that contains necessary
            information for trajectory prediction.
    """
    verify_key = "valid_track_ids"
    required_key = [
        "dataset_index",
        "seq_center",
        "valid_track_ids",
        "valid_img_coords",
        "valid_masks",
        "valid_class",
        "state_vectors",
        "resize_ratio",
        "road_map",
        "rendered_obs",
        "future_trajectories",
        "high_freq_ctx_trajectories",
    ]
    optional_key = [
        "img_homographys",
        "affine_transforms",
        "head_mask",
        "drivable_road_map",
        "selected_anchors",
        "selected_anchors_mask",
        "sampled_anchors",
        "sampled_anchors_mask",
        "ctx_masks",
        "filtered_obs_ids",
        "filtered_obs_trajs",
        "filtered_obs_gt",
        "filtered_obs_gt_masks",
        "filtered_obs_class",
        "lat_behaviors",
        "lon_behaviors",
        "vcs_coords",
        "valid_behav_track_ids",
        "agent_classes",
    ]
    batch_data = {i: [] for i in required_key + optional_key}

    # 1. Unpack the batch data.
    for item in batch:
        if len(item[verify_key]) == 0:
            continue
        for key in required_key:
            if key in item:
                batch_data[key].append(item[key])
            else:
                raise ValueError(
                    f"The batch data do not have the required key {key}."
                )
        for key in optional_key:
            if key in item:
                batch_data[key].append(item[key])

    if not len(batch_data[verify_key]):
        return None

    # 2. Tensor packing and type conversion.
    # 2.1. Road map:
    #   [[H, W]] -> [batch_size, 1, H, W]
    #   [[H, W, C]] -> [batch_size, C, H, W]
    batch_road_maps = torch.FloatTensor(np.stack(batch_data["road_map"], 0))
    if len(batch_road_maps.shape) == 3:
        batch_data["road_map"] = torch.unsqueeze(batch_road_maps, 1)
    elif len(batch_road_maps.shape) == 4:
        batch_data["road_map"] = batch_road_maps.permute(
            (0, 3, 1, 2)
        ).contiguous()
    else:
        raise ValueError(
            "The shape of road maps should be either [H, W] or [H, W, C]."
        )

    # 2.2. Rendered obstacle OG.
    #   [[H, W, C]] -> [batch_size, C, H, W]
    batch_rendered_obs = torch.FloatTensor(
        np.stack(batch_data["rendered_obs"], 0)
    )
    batch_data["rendered_obs"] = batch_rendered_obs.permute(
        (0, 3, 1, 2)
    ).contiguous()

    # 2.3. Ground truth trajectories: [num_obj, traj_len, 2]
    batch_gts = [
        torch.FloatTensor(gt) for gt in batch_data["future_trajectories"]
    ]
    batch_data["future_trajectories"] = torch.cat(batch_gts, dim=0)

    # 2.4. Masks: [num_obj, traj_len]
    batch_masks = [
        torch.tensor(masks, dtype=bool) for masks in batch_data["valid_masks"]
    ]
    batch_data["valid_masks"] = torch.cat(batch_masks, dim=0)

    # 2.5. State vectors: [num_obj, 3 + num_extend_state_vectors, 1, 1]
    batch_state_vectors = [
        torch.FloatTensor(state_vector)
        for state_vector in batch_data["state_vectors"]
    ]
    batch_state_vectors = torch.cat(batch_state_vectors, dim=0)
    batch_state_vectors = batch_state_vectors.view(
        batch_state_vectors.shape + (1, 1)
    )
    batch_data["state_vectors"] = batch_state_vectors

    # 2.6. Resize ratio: [num_obj, 2]
    batch_resize_ratio = [
        torch.FloatTensor(item) for item in batch_data["resize_ratio"]
    ]
    batch_data["resize_ratio"] = torch.cat(batch_resize_ratio, dim=0)

    # 2.7. Homography matrix: [num_obj, 3, 3]
    if len(batch_data["img_homographys"]):
        batch_homographys = [
            torch.FloatTensor(homographys)
            for homographys in batch_data["img_homographys"]
        ]
        batch_data["img_homographys"] = torch.cat(batch_homographys, dim=0)

    # 2.8. Affline transforms: [num_obj, 3, 3]
    if len(batch_data["affine_transforms"]):
        batch_affine_transforms = [
            torch.FloatTensor(transform)
            for transform in batch_data["affine_transforms"]
        ]
        batch_data["affine_transforms"] = torch.stack(
            batch_affine_transforms, dim=0
        )

    # 2.9. Head mask.
    if len(batch_data["head_mask"]):
        batch_head_mask = [
            torch.FloatTensor(head_mask)
            for head_mask in batch_data["head_mask"]
        ]
        batch_data["head_mask"] = torch.cat(batch_head_mask, dim=0).transpose(
            1, 0
        )

    # 2.10. Sequence center and dataset index.
    batch_data["dataset_index"] = torch.tensor(batch_data["dataset_index"])
    batch_data["seq_center"] = [
        torch.tensor(list(i)) for i in batch_data["seq_center"]
    ]

    # 2.11. Valid class
    #   [[H, W, C]] -> [batch_size, C, H, W]
    batch_valid_class = []
    for i in batch_data["valid_class"]:
        batch_valid_class += i
    batch_valid_class = torch.FloatTensor(np.array(batch_valid_class))
    batch_data["batch_valid_class"] = batch_valid_class

    # 2.12. Drivable area road map:
    #   [[H, W]] -> [batch_size, 1, H, W]
    #   [[H, W, C]] -> [batch_size, C, H, W]
    if len(batch_data["drivable_road_map"]):
        batch_road_maps = torch.FloatTensor(
            np.stack(batch_data["drivable_road_map"], 0)
        )
        if len(batch_road_maps.shape) == 3:
            batch_data["drivable_road_map"] = torch.unsqueeze(
                batch_road_maps, 1
            )
        elif len(batch_road_maps.shape) == 4:
            batch_data["drivable_road_map"] = batch_road_maps.permute(
                (0, 3, 1, 2)
            ).contiguous()
        else:
            raise ValueError(
                "The shape of road maps should be either [H, W] or [H, W, C]."
            )

    # 2.13. Selected anchors:
    #   [num_obj, num_anchors, traj_len, 2]
    if len(batch_data["selected_anchors"]):
        anchors = np.concatenate(batch_data["selected_anchors"], axis=0)
        num_obj, num_anchors = anchors.shape[0:2]
        ex_anchors = np.concatenate(
            [np.zeros([num_obj, num_anchors, 1, 2]), anchors], axis=2
        )

        anchors = torch.FloatTensor(anchors)
        batch_data["selected_anchors"] = anchors

        ex_anchor_diff = np.diff(ex_anchors, axis=2)
        normed_diff_x = np.clip(ex_anchor_diff[:, :, :, 0], 0, 8)
        normed_diff_x = (normed_diff_x - 4) * 1.25
        normed_diff_y = np.clip(ex_anchor_diff[:, :, :, 1], -1.5, 1.5)
        normed_diff_y *= 10 / 3
        batch_data["selected_anchors_diff_x"] = torch.FloatTensor(
            normed_diff_x
        )
        batch_data["selected_anchors_diff_y"] = torch.FloatTensor(
            normed_diff_y
        )

    # 2.14. Selected anchors mask:
    #   [num_obj, num_anchors]
    if len(batch_data["selected_anchors_mask"]):
        anc_mask = np.concatenate(batch_data["selected_anchors_mask"], axis=0)
        anc_mask = torch.FloatTensor(anc_mask)
        batch_data["selected_anchors_mask"] = anc_mask

    # 2.15. Sampled anchors mask:
    #   [num_obj, num_anchors]
    if len(batch_data["sampled_anchors_mask"]):
        anc_mask = np.concatenate(batch_data["sampled_anchors_mask"], axis=0)
        anc_mask = torch.FloatTensor(anc_mask)
        batch_data["sampled_anchors_mask"] = anc_mask

    # 2.16. Sampled anchors:
    #   [num_obj, num_anchors, traj_len, 2]
    if len(batch_data["sampled_anchors"]):
        anchors = np.concatenate(batch_data["sampled_anchors"], axis=0)
        num_obj, num_anchors = anchors.shape[0:2]
        ex_anchors = np.concatenate(
            [np.zeros([num_obj, num_anchors, 1, 2]), anchors], axis=2
        )

        anchors = torch.FloatTensor(anchors)
        batch_data["sampled_anchors"] = anchors

        ex_anchor_diff = np.diff(ex_anchors, axis=2)
        normed_diff_x = np.clip(ex_anchor_diff[:, :, :, 0], 0, 8)
        normed_diff_x = (normed_diff_x - 4) * 1.25
        normed_diff_y = np.clip(ex_anchor_diff[:, :, :, 1], -1.5, 1.5)
        normed_diff_y *= 10 / 3
        batch_data["sampled_anchors_diff_x"] = torch.FloatTensor(normed_diff_x)
        batch_data["sampled_anchors_diff_y"] = torch.FloatTensor(normed_diff_y)

    # 2.17. Fuse the sampled anchors and selected anchors:
    #   [num_obj, num_anchors, traj_len, 2]
    if len(batch_data["selected_anchors"]) and len(
        batch_data["sampled_anchors"]
    ):
        batch_data["fuse_anchors_diff_x"] = torch.where(
            batch_data["sampled_anchors_mask"][:, :, None] > 0,
            batch_data["sampled_anchors_diff_x"],
            batch_data["selected_anchors_diff_x"],
        )
        batch_data["fuse_anchors_diff_y"] = torch.where(
            batch_data["sampled_anchors_mask"][:, :, None] > 0,
            batch_data["sampled_anchors_diff_y"],
            batch_data["selected_anchors_diff_y"],
        )
        batch_data["fuse_anchors"] = torch.where(
            batch_data["sampled_anchors_mask"][:, :, None, None] > 0,
            batch_data["sampled_anchors"],
            batch_data["selected_anchors"],
        )
        batch_data["fuse_anchors_mask"] = torch.where(
            batch_data["sampled_anchors_mask"][:, :] > 0,
            batch_data["sampled_anchors_mask"],
            batch_data["selected_anchors_mask"],
        )

    # 2.18. context trajectories: [num_obj, traj_len, 2]
    if len(batch_data["high_freq_ctx_trajectories"]):
        batch_ctx_trajs = [
            torch.FloatTensor(traj)
            for traj in batch_data["high_freq_ctx_trajectories"]
        ]
        batch_data["high_freq_ctx_trajectories"] = torch.cat(
            batch_ctx_trajs, dim=0
        )

    # 2.19. context masks: [num_obj, traj_len]
    if len(batch_data["ctx_masks"]):
        batch_ctx_masks = [
            torch.tensor(ctx_mask, dtype=bool)
            for ctx_mask in batch_data["ctx_masks"]
        ]
        batch_data["ctx_masks"] = torch.cat(batch_ctx_masks, dim=0)

    # 2.20. filtered_obs_trajs: [num_obj, traj_len, 2]
    if len(batch_data["filtered_obs_trajs"]):
        batch_filtered_obs_trajs = [
            torch.FloatTensor(filtered_obs_trajs)
            for filtered_obs_trajs in batch_data["filtered_obs_trajs"]
        ]
        batch_data["filtered_obs_trajs"] = torch.cat(
            batch_filtered_obs_trajs, dim=0
        )

    # 2.21. filtered_obs_gt: [num_obs_too_far_obj, traj_len, 2]
    if len(batch_data["filtered_obs_gt"]):
        batch_filtered_obs_gt = [
            torch.FloatTensor(gt) for gt in batch_data["filtered_obs_gt"]
        ]
        batch_data["filtered_obs_gt"] = torch.cat(batch_filtered_obs_gt, dim=0)

    # 2.22. filtered_obs_gt_masks: [num_obs_too_far_obj, traj_len]
    if len(batch_data["filtered_obs_gt_masks"]):
        batch_filtered_obs_gt_masks = [
            torch.tensor(masks, dtype=bool)
            for masks in batch_data["filtered_obs_gt_masks"]
        ]
        batch_data["filtered_obs_gt_masks"] = torch.cat(
            batch_filtered_obs_gt_masks, dim=0
        )

    # 2.23 context trajectories of valid obs
    if "high_freq_ctx_trajectories" in batch_data:
        batch_data["tcn_ctx_trajectories"] = (
            batch_data["high_freq_ctx_trajectories"]
            .unsqueeze(-1)
            .permute(0, 2, 3, 1)
        )

    # 2.24 Lateral behavior label: [num_obj]
    if len(batch_data["lat_behaviors"]):
        batch_lat_behaviors = [
            torch.LongTensor(lat_behavior)
            for lat_behavior in batch_data["lat_behaviors"]
        ]
        batch_data["lat_behaviors"] = torch.cat(batch_lat_behaviors, dim=0)

    # 2.25 Longitudinal behavior label: [num_obj]
    if len(batch_data["lon_behaviors"]):
        batch_lon_behaviors = [
            torch.LongTensor(lon_behavior)
            for lon_behavior in batch_data["lon_behaviors"]
        ]
        batch_data["lon_behaviors"] = torch.cat(batch_lon_behaviors, dim=0)

    # 2.26 vcs coordinates: [num_obj, 2]
    if len(batch_data["vcs_coords"]):
        vcs_coords = [
            torch.FloatTensor(vcs_coord)
            for vcs_coord in batch_data["vcs_coords"]
        ]
        batch_data["vcs_coords"] = torch.cat(vcs_coords, dim=0)

    # 3. del unused optional keys
    for key in optional_key:
        if not len(batch_data[key]):
            batch_data.pop(key)

    return batch_data


def collate_multipath_viz(batch):
    batch_data = collate_multipath(batch)
    if batch_data is None:
        return batch_data
    batch_seq_df = []
    batch_seq_index = []
    batch_track_stat_dict = []
    batch_track_yaw_dict = []
    batch_safe_area = []
    for item in batch:
        if len(item["valid_track_ids"]) == 0:
            continue
        batch_seq_df.append(item["seq_df"])
        i_index = list(item["seq_index"].traj_group_index)
        i_slice = item["seq_index"].frame_slice
        batch_seq_index.append([i_index, i_slice])
        batch_track_stat_dict.append(item["track_stat_dict"])
        batch_track_yaw_dict.append(item["track_yaw_dict"])
        if "safe_area" in item:
            batch_safe_area.append(item["safe_area"])

    batch_data["seq_index"] = batch_seq_index
    batch_data["seq_df"] = batch_seq_df
    batch_data["track_stat_dict"] = batch_track_stat_dict
    batch_data["track_yaw_dict"] = batch_track_yaw_dict
    batch_data["safe_area"] = batch_safe_area
    return batch_data


def collate_uniformpath_viz(batch):
    batch_data = collate_multipath_viz(batch)
    batch_data["valid_track_ids"] = batch_data["filtered_obs_ids"]
    batch_data["valid_class"] = batch_data["filtered_obs_class"]
    return batch_data


def bev_gt_path_function(
    image_dir, bev_bucket_path, plate, date_token, timestamp, image_suffix
):
    date = date_token.split("-")[0]
    bev_map_path = os.path.join(
        image_dir,
        bev_bucket_path[0],
        plate + date + "_D",
        date_token,
        bev_bucket_path[1],
        timestamp + image_suffix,
    )
    return bev_map_path


def gt_path_func_for_data_pipeline_v3(
    image_dir, bev_bucket_path, plate, date_token, timestamp, image_suffix
):
    date = date_token.split("-")[0]
    if type(bev_bucket_path[0]) is str:
        bev_bucket_path[0] = [bev_bucket_path[0]]
    bev_map_path = None
    for bev_path in bev_bucket_path[0]:
        bev_map_path_prefix = os.path.join(
            image_dir, bev_path, plate + date + "_D"
        )
        date_token_dir = date_token
        path_list = os.listdir(bev_map_path_prefix)
        date_token_dir = None
        for dirname in path_list:
            if dirname.startswith(date_token):
                date_token_suffix = dirname.split(date_token)[-1]
                date_token_dir = f"{date_token_suffix}/{bev_bucket_path[1]}"
                break
        if date_token_dir is None:
            continue
        else:
            bev_map_path = os.path.join(
                bev_map_path_prefix,
                date_token + date_token_dir,
                timestamp + image_suffix,
            )
    return bev_map_path


def lmdb_path_replace_func_for_data_pipeline_v3(instance):
    for ds in instance.loaded_ds.datasets:
        ds.navi_file_dir = "/bucket/input/J5FSD/"
        root_dir = "11_perception_prediction/Navi_data_Apr2022/separate_day/"
        ds.navi_lmdb_path = (
            root_dir + ds.navi_lmdb_path.split("superdrive_dataset_navi/")[-1]
        )
        ds.struct_road_lmdb_path = (
            root_dir
            + ds.struct_road_lmdb_path.split("superdrive_dataset_navi/")[-1]
        )


def lmdb_path_replace_func_for_data_pipeline_v5(instance):
    for ds in instance.loaded_ds.datasets:
        if "J5FSD" in ds.navi_file_dir:
            ds.navi_file_dir = (
                "/bucket/input/J5FSD/" + ds.navi_file_dir.split("J5FSD/")[-1]
            )
        elif "SD_Algorithm" in ds.navi_file_dir:
            ds.navi_file_dir = (
                "/bucket/input/SD_Algorithm/"
                + ds.navi_file_dir.split("SD_Algorithm/")[-1]
            )


def collate_vectornet(batch: List[Dict]) -> Dict:
    """Collate function for scene-based VectorNet prediction model.

    Args:
        batch: a list of dataloader output.

    Returns:
        batch_data: the dictionary that contains necessary
            information for trajectory and behavior prediction.
    """
    verify_key = "valid_track_ids"
    required_key = [
        "dataset_index",
        "seq_center",
        "valid_track_ids",
        "valid_img_coords",
        "valid_masks",
        "valid_class",
        "future_trajectories",
        "struct_road_feats",
        "struct_road_masks",
        "struct_traj_feats",
        "struct_traj_masks",
        "itp_fut_obs_trajs",
        "itp_fut_obs_masks",
        "road_feat_scale",
        "traj_feat_scale",
        "struct_num_pad_track_ids",
    ]
    optional_key = [
        "state_vectors",
        "behav_state_vectors",
        "head_mask",
        "road_map",
        "selected_anchors",
        "selected_anchors_mask",
        "sampled_anchors",
        "sampled_anchors_mask",
        "ctx_masks",
        "filtered_obs_ids",
        "filtered_obs_trajs",
        "filtered_obs_gt",
        "filtered_obs_gt_masks",
        "filtered_obs_class",
        "agent_classes",
        "lat_behaviors",
        "lon_behaviors",
        "vcs_coords",
        "valid_behav_track_ids",
        "track_ids",
        "context_states",
        "vis_lanes",
        "file_names",
        "ctx_trajectories",
        "end_points",
        "diff_fut_trajs",
        "goal_coords",
        "nearest_goal_idxs",
        "nearest_road_idxs",
        "goal_coords_scale",
        "goal_masks",
    ]

    # split the batch input if there is splitted dataflow
    batch_data = {i: [] for i in required_key + optional_key}
    # 1. Unpack the batch data.
    for item in batch:
        if len(item[verify_key]) == 0:
            continue
        for key in required_key:
            if key in item:
                batch_data[key].append(item[key])
            else:
                raise ValueError(
                    f"The batch data do not have the required key {key}."
                )
        for key in optional_key:
            if key in item:
                batch_data[key].append(item[key])

    if not len(batch_data[verify_key]):
        return None

    # 2. Tensor packing and type conversion.
    # 2.1. Road map:
    #   [[H, W]] -> [batch_size, 1, H, W]
    #   [[H, W, C]] -> [batch_size, C, H, W]
    if len(batch_data["road_map"]):
        batch_road_maps = torch.FloatTensor(
            np.stack(batch_data["road_map"], 0)
        )
        if len(batch_road_maps.shape) == 3:
            batch_data["road_map"] = torch.unsqueeze(batch_road_maps, 1)
        elif len(batch_road_maps.shape) == 4:
            batch_data["road_map"] = batch_road_maps.permute(
                (0, 3, 1, 2)
            ).contiguous()
        else:
            raise ValueError(
                "The shape of road maps should be either [H, W] or [H, W, C]."
            )

    # 2.2. Rendered obstacle OG.
    #   [[H, W, C]] -> [batch_size, C, H, W]
    # None

    # 2.3. Ground truth trajectories: [num_obj, traj_len, 2]
    batch_gts = [
        torch.FloatTensor(np.array(gt))
        for gt in batch_data["future_trajectories"]
    ]
    batch_data["future_trajectories"] = torch.cat(batch_gts, dim=0)

    # 2.4. Masks: [num_obj, traj_len]
    batch_masks = [
        torch.tensor(np.array(masks), dtype=bool)
        for masks in batch_data["valid_masks"]
    ]
    batch_data["valid_masks"] = torch.cat(batch_masks, dim=0)

    # 2.5. State vectors: [num_obj, 3, 1, 1]
    if len(batch_data["state_vectors"]):
        batch_state_vectors = [
            torch.FloatTensor(np.array(state_vector))
            for state_vector in batch_data["state_vectors"]
        ]
        batch_state_vectors = torch.cat(batch_state_vectors, dim=0)
        batch_state_vectors = batch_state_vectors.view(
            batch_state_vectors.shape + (1, 1)
        )
        batch_data["state_vectors"] = batch_state_vectors

    if len(batch_data["behav_state_vectors"]):
        batch_behav_state_vectors = [
            torch.FloatTensor(np.array(behav_state_vector))
            for behav_state_vector in batch_data["behav_state_vectors"]
        ]
        batch_behav_state_vectors = torch.cat(batch_behav_state_vectors, dim=0)
        batch_behav_state_vectors = batch_behav_state_vectors.view(
            batch_behav_state_vectors.shape + (1, 1)
        )
        batch_data["behav_state_vectors"] = batch_behav_state_vectors
    # 2.6. Resize ratio: [num_obj, 2]
    # None

    # 2.7. Homography matrix: [num_obj, 3, 3]
    # None

    # 2.8. Affline transforms: [num_obj, 3, 3]
    # None

    # 2.9. Head mask.
    if len(batch_data["head_mask"]):
        batch_head_mask = [
            torch.FloatTensor(np.array(head_mask))
            for head_mask in batch_data["head_mask"]
        ]
        batch_data["head_mask"] = torch.cat(batch_head_mask, dim=0).transpose(
            1, 0
        )

    # 2.10. Sequence center and dataset index.
    batch_data["dataset_index"] = torch.tensor(batch_data["dataset_index"])
    batch_data["seq_center"] = [
        torch.tensor(list(i)) for i in batch_data["seq_center"]
    ]

    # 2.11. Valid class
    #   [[H, W, C]] -> [batch_size, C, H, W]
    batch_valid_class = []
    for i in batch_data["valid_class"]:
        batch_valid_class += i
    batch_valid_class = torch.FloatTensor(np.array(batch_valid_class))
    batch_data["batch_valid_class"] = batch_valid_class

    # 2.12. Drivable area road map:
    #   [[H, W]] -> [batch_size, 1, H, W]
    #   [[H, W, C]] -> [batch_size, C, H, W]
    # None

    # 2.13. Selected anchors:
    #   [num_obj, num_anchors, traj_len, 2]
    if len(batch_data["selected_anchors"]):
        anchors = np.concatenate(batch_data["selected_anchors"], axis=0)
        num_obj, num_anchors = anchors.shape[0:2]
        ex_anchors = np.concatenate(
            [np.zeros([num_obj, num_anchors, 1, 2]), anchors], axis=2
        )

        anchors = torch.FloatTensor(anchors)
        batch_data["selected_anchors"] = anchors

        ex_anchor_diff = np.diff(ex_anchors, axis=2)
        normed_diff_x = np.clip(ex_anchor_diff[:, :, :, 0], 0, 8)
        normed_diff_x = (normed_diff_x - 4) * 1.25
        normed_diff_y = np.clip(ex_anchor_diff[:, :, :, 1], -1.5, 1.5)
        normed_diff_y *= 10 / 3
        batch_data["selected_anchors_diff_x"] = torch.FloatTensor(
            normed_diff_x
        )
        batch_data["selected_anchors_diff_y"] = torch.FloatTensor(
            normed_diff_y
        )

    # 2.14. Selected anchors mask:
    #   [num_obj, num_anchors]
    if len(batch_data["selected_anchors_mask"]):
        anc_mask = np.concatenate(batch_data["selected_anchors_mask"], axis=0)
        anc_mask = torch.FloatTensor(anc_mask)
        batch_data["selected_anchors_mask"] = anc_mask

    # 2.15. Sampled anchors mask:
    #   [num_obj, num_anchors]
    if len(batch_data["sampled_anchors_mask"]):
        anc_mask = np.concatenate(batch_data["sampled_anchors_mask"], axis=0)
        anc_mask = torch.FloatTensor(anc_mask)
        batch_data["sampled_anchors_mask"] = anc_mask

    # 2.16. Sampled anchors:
    #   [num_obj, num_anchors, traj_len, 2]
    if len(batch_data["sampled_anchors"]):
        anchors = np.concatenate(batch_data["sampled_anchors"], axis=0)
        num_obj, num_anchors = anchors.shape[0:2]
        ex_anchors = np.concatenate(
            [np.zeros([num_obj, num_anchors, 1, 2]), anchors], axis=2
        )

        anchors = torch.FloatTensor(anchors)
        batch_data["sampled_anchors"] = anchors

        ex_anchor_diff = np.diff(ex_anchors, axis=2)
        normed_diff_x = np.clip(ex_anchor_diff[:, :, :, 0], 0, 8)
        normed_diff_x = (normed_diff_x - 4) * 1.25
        normed_diff_y = np.clip(ex_anchor_diff[:, :, :, 1], -1.5, 1.5)
        normed_diff_y *= 10 / 3
        batch_data["sampled_anchors_diff_x"] = torch.FloatTensor(normed_diff_x)
        batch_data["sampled_anchors_diff_y"] = torch.FloatTensor(normed_diff_y)

    # 2.17. Fuse the sampled anchors and selected anchors:
    #   [num_obj, num_anchors, traj_len, 2]
    if len(batch_data["selected_anchors"]) and len(
        batch_data["sampled_anchors"]
    ):
        batch_data["fuse_anchors_diff_x"] = torch.where(
            batch_data["sampled_anchors_mask"][:, :, None] > 0,
            batch_data["sampled_anchors_diff_x"],
            batch_data["selected_anchors_diff_x"],
        )
        batch_data["fuse_anchors_diff_y"] = torch.where(
            batch_data["sampled_anchors_mask"][:, :, None] > 0,
            batch_data["sampled_anchors_diff_y"],
            batch_data["selected_anchors_diff_y"],
        )
        batch_data["fuse_anchors"] = torch.where(
            batch_data["sampled_anchors_mask"][:, :, None, None] > 0,
            batch_data["sampled_anchors"],
            batch_data["selected_anchors"],
        )
        batch_data["fuse_anchors_mask"] = torch.where(
            batch_data["sampled_anchors_mask"][:, :] > 0,
            batch_data["sampled_anchors_mask"],
            batch_data["selected_anchors_mask"],
        )

    # 2.18. context trajectories: [num_obj, traj_len, 2]
    # None

    # 2.19. context masks: [num_obj, traj_len]
    if len(batch_data["ctx_masks"]):
        batch_ctx_masks = [
            torch.tensor(ctx_mask, dtype=bool)
            for ctx_mask in batch_data["ctx_masks"]
        ]
        batch_data["ctx_masks"] = torch.cat(batch_ctx_masks, dim=0)

    # 2.20. filtered_obs_trajs: [num_obj, traj_len, 2]
    if len(batch_data["filtered_obs_trajs"]):
        batch_filtered_obs_trajs = [
            torch.FloatTensor(np.array(filtered_obs_trajs))
            for filtered_obs_trajs in batch_data["filtered_obs_trajs"]
        ]
        batch_data["filtered_obs_trajs"] = torch.cat(
            batch_filtered_obs_trajs, dim=0
        )

    # 2.21. filtered_obs_gt: [num_obs_too_far_obj, traj_len, 2]
    if len(batch_data["filtered_obs_gt"]):
        batch_filtered_obs_gt = [
            torch.FloatTensor(np.array(gt))
            for gt in batch_data["filtered_obs_gt"]
        ]
        batch_data["filtered_obs_gt"] = torch.cat(batch_filtered_obs_gt, dim=0)

    # 2.22. filtered_obs_gt_masks: [num_obs_too_far_obj, traj_len]
    if len(batch_data["filtered_obs_gt_masks"]):
        batch_filtered_obs_gt_masks = [
            torch.tensor(masks, dtype=bool)
            for masks in batch_data["filtered_obs_gt_masks"]
        ]
        batch_data["filtered_obs_gt_masks"] = torch.cat(
            batch_filtered_obs_gt_masks, dim=0
        )

    # 2.23 History valid homography matrix: [num_obj, 3, 3]
    # None

    # 2.24 context trajectories of valid obs
    if "high_freq_ctx_trajectories" in batch_data:
        batch_data["tcn_ctx_trajectories"] = (
            batch_data["high_freq_ctx_trajectories"]
            .unsqueeze(-1)
            .permute(0, 2, 3, 1)
        )

    # 2.25 Obstacle trajectories features.
    # [num_obj, max_obs_num, num_traj_pts, num_traj_feats]
    batch_traj_feats = [
        torch.FloatTensor(np.array(traj_feats))
        for traj_feats in batch_data["struct_traj_feats"]
    ]
    batch_traj_feats = torch.cat(batch_traj_feats, dim=0)
    traj_scale = np.stack(batch_data["traj_feat_scale"])
    traj_scale = np.max(traj_scale, axis=0)
    traj_scale = torch.FloatTensor(traj_scale)[None, None, None, :]
    batch_traj_feats = batch_traj_feats * traj_scale
    batch_data["struct_traj_feats"] = batch_traj_feats.contiguous()

    # 2.26 Road segment features surrounding obstacles.
    # [num_obj, max_seg_num, num_seg_pts, num_seg_feats]
    batch_seg_feats = [
        torch.FloatTensor(np.array(seg_feats))
        for seg_feats in batch_data["struct_road_feats"]
    ]
    batch_seg_feats = torch.cat(batch_seg_feats, dim=0)
    road_scale = np.stack(batch_data["road_feat_scale"])
    road_scale = np.max(road_scale, axis=0)
    road_scale = torch.FloatTensor(road_scale)[None, None, None, :]
    batch_seg_feats = batch_seg_feats * road_scale
    batch_data["struct_road_feats"] = batch_seg_feats.contiguous()

    # 2.27. Obstacle trajectory masks. [num_obj, max_obs_num]
    batch_traj_masks = [
        torch.FloatTensor(np.array(traj_masks))
        for traj_masks in batch_data["struct_traj_masks"]
    ]
    batch_data["struct_traj_masks"] = torch.cat(batch_traj_masks, dim=0)

    # 2.28. Obstacle road element masks. [num_obj, max_seg_num]
    batch_road_masks = [
        torch.FloatTensor(np.array(road_masks))
        for road_masks in batch_data["struct_road_masks"]
    ]
    batch_data["struct_road_masks"] = torch.cat(batch_road_masks, dim=0)

    # 2.29. Interpolated future trajectories. [num_obj, itp_traj_len, 2].
    batch_itp_gts = [
        torch.FloatTensor(np.array(gt))
        for gt in batch_data["itp_fut_obs_trajs"]
    ]
    batch_data["itp_fut_obs_trajs"] = torch.cat(batch_itp_gts, dim=0)

    # 2.30. Interpolated future masks: [num_obj, itp_traj_len]
    batch_itp_masks = [
        torch.tensor(masks, dtype=bool)
        for masks in batch_data["itp_fut_obs_masks"]
    ]
    batch_data["itp_fut_obs_masks"] = torch.cat(batch_itp_masks, dim=0)

    # 2.31. Lateral behavior label: [num_obj]
    if len(batch_data["lat_behaviors"]):
        batch_lat_behaviors = [
            torch.LongTensor(lat_behavior)
            for lat_behavior in batch_data["lat_behaviors"]
        ]
        batch_data["lat_behaviors"] = torch.cat(batch_lat_behaviors, dim=0)

    # 2.32 Longitudinal behavior label: [num_obj]
    if len(batch_data["lon_behaviors"]):
        batch_lon_behaviors = [
            torch.LongTensor(lon_behavior)
            for lon_behavior in batch_data["lon_behaviors"]
        ]
        batch_data["lon_behaviors"] = torch.cat(batch_lon_behaviors, dim=0)

    # 2.33 vcs coordinates: [num_obj, 2]
    if len(batch_data["vcs_coords"]):
        vcs_coords = [
            torch.FloatTensor(vcs_coord)
            for vcs_coord in batch_data["vcs_coords"]
        ]
        batch_data["vcs_coords"] = torch.cat(vcs_coords, dim=0)

    # 2.34. vis_lanes: List[List[array.shape=(10,2)],...]
    if len(batch_data["vis_lanes"]):
        tmp = []
        for vis_lanes in batch_data["vis_lanes"]:
            tmp.extend(vis_lanes)
        batch_data["vis_lanes"] = tmp

    # 2.35. sample["file_names"]:List[str,...],len=num_valid_obs
    if len(batch_data["file_names"]):
        tmp = []
        for file_names in batch_data["file_names"]:
            tmp.extend(file_names)
        batch_data["file_names"] = tmp

    # 2.36. sample["ctx_trajectories"]:(num_valid_obs,num_his_frames,2)
    if len(batch_data["ctx_trajectories"]):
        ctx_trajectories = [
            torch.FloatTensor(np.array(history_traj))
            for history_traj in batch_data["ctx_trajectories"]
        ]
        batch_data["ctx_trajectories"] = torch.cat(ctx_trajectories, dim=0)

    # 2.37.
    # 采样点坐标数值的缩放比例
    if len(batch_data["goal_coords_scale"]):
        goal_scale = np.stack(batch_data["goal_coords_scale"])
        goal_scale = np.max(goal_scale, axis=0)

    # 2.38. sample["end_points"]:(num_valid_obs, 2, 1, 1),未来轨迹的终点
    if len(batch_data["end_points"]):
        end_points = [
            torch.FloatTensor(future_end_points)
            for future_end_points in batch_data["end_points"]
        ]
        end_points = torch.cat(end_points, dim=0)
        end_points_scale = torch.FloatTensor(goal_scale)[None, :]
        end_points = end_points * end_points_scale
        batch_data["end_points"] = end_points.contiguous()
        batch_data["end_points"] = (
            batch_data["end_points"].unsqueeze(-1).unsqueeze(-1)
        )

    # 2.39. goal_coords: [num_obs, 2, 1, num_goals]
    if len(batch_data["goal_coords"]):
        goal_coords = [
            torch.FloatTensor(obs_goal)
            for obs_goal in batch_data["goal_coords"]
        ]
        goal_coords = torch.cat(goal_coords, dim=0)
        goal_coords_scale = torch.FloatTensor(goal_scale)[None, None, :]
        goal_coords = goal_coords * goal_coords_scale
        batch_data["goal_coords"] = goal_coords.contiguous()
        batch_data["goal_coords"] = (
            batch_data["goal_coords"].unsqueeze(1).permute(0, 3, 1, 2)
        )

    # 2.40. Ground truth trajectories's difference: [num_obj, 1, traj_len, 2]
    if len(batch_data["diff_fut_trajs"]):
        batch_gts = [
            torch.FloatTensor(np.array(gt))
            for gt in batch_data["diff_fut_trajs"]
        ]
        if len(batch_gts) > 0:
            batch_data["diff_fut_trajs"] = torch.cat(batch_gts, dim=0)
            batch_data["diff_fut_trajs"] = batch_data[
                "diff_fut_trajs"
            ].unsqueeze(1)
        else:
            batch_data["diff_fut_trajs"] = []

    # 2.41. nearest_goal_idxs: [num_obs,1,1,1]
    if len(batch_data["nearest_goal_idxs"]):
        batch_nearest_goal_idxs = [
            torch.FloatTensor(nearest_goal_idxs)
            for nearest_goal_idxs in batch_data["nearest_goal_idxs"]
        ]
        batch_data["nearest_goal_idxs"] = (
            torch.cat(batch_nearest_goal_idxs, dim=0)
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
        )

    # 2.42. nearest_road_idxs: [num_obs,1,1,1]
    if len(batch_data["nearest_road_idxs"]):
        batch_nearest_road_idxs = [
            torch.FloatTensor(nearest_road_idxs)
            for nearest_road_idxs in batch_data["nearest_road_idxs"]
        ]
        batch_data["nearest_road_idxs"] = (
            torch.cat(batch_nearest_road_idxs, dim=0)
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
        )

    # 2.43. goal_masks: [num_obs, 1, 1, num_goals]
    if len(batch_data["goal_masks"]):
        goal_masks = [
            torch.FloatTensor(obs_goal)
            for obs_goal in batch_data["goal_masks"]
        ]
        goal_masks = torch.cat(goal_masks, dim=0)
        batch_data["goal_masks"] = goal_masks.contiguous()
        batch_data["goal_masks"] = (
            batch_data["goal_masks"].unsqueeze(1).unsqueeze(1)
        )

    # 2.44 Track ids of all traj feats.
    # [num_obj, max_obs_num]
    if len(batch_data["struct_num_pad_track_ids"]):
        struct_num_pad_track_ids = [
            torch.FloatTensor(np.array(traj_feats))
            for traj_feats in batch_data["struct_num_pad_track_ids"]
        ]
        struct_num_pad_track_ids = torch.cat(struct_num_pad_track_ids, dim=0)
        batch_data[
            "struct_num_pad_track_ids"
        ] = struct_num_pad_track_ids.contiguous()

    # 3. del unused optional keys
    for key in optional_key:
        if not len(batch_data[key]):
            batch_data.pop(key)

    return batch_data


label_colors = {
    "others": [0, 0, 0],
    "roadedge": [0, 0, 255],
    "roadarrow": [47, 79, 79],
    "solid_lanes": [200, 200, 200],
    "stopline": [192, 0, 64],
    "crosswalk": [255, 127, 80],
    "sections": [0, 0, 0],
    "junctions": [0, 0, 0],
    "virtuallanes": [127, 127, 127],
}
vectornet_color_map = torch.stack(
    [torch.tensor(v) for _, v in label_colors.items()]
)


def collate_vectornet_viz(batch: List[Dict]) -> Dict:
    """Collate function for VectorNet during visualization process.

    Args:
        batch: a list of dataloader output.

    Returns:
        batch_data: the dictionary that contains necessary
            information for trajectory prediction.
    """
    global vectornet_color_map
    batch_data = collate_vectornet(batch)

    batch_seq_df = []
    batch_seq_index = []
    batch_track_stat_dict = []
    batch_track_yaw_dict = []
    batch_safe_area = []
    for item in batch:
        if len(item["valid_track_ids"]) == 0:
            continue
        batch_seq_df.append(item["seq_df"])
        if isinstance(item["seq_index"], List):
            i_index = item["seq_index"]
            batch_seq_index.append(i_index)
        elif isinstance(item["seq_index"], SeqIndex):
            i_index = list(item["seq_index"].traj_group_index)
            i_slice = item["seq_index"].frame_slice
            batch_seq_index.append([i_index, i_slice])
        else:
            raise ValueError("Unsupported type of item['seq_index'].")
        batch_track_stat_dict.append(item["track_stat_dict"])
        batch_track_yaw_dict.append(item["track_yaw_dict"])
        if "safe_area" in item:
            batch_safe_area.append(item["safe_area"])

    batch_data["seq_index"] = batch_seq_index
    batch_data["seq_df"] = batch_seq_df
    batch_data["track_stat_dict"] = batch_track_stat_dict
    batch_data["track_yaw_dict"] = batch_track_yaw_dict
    batch_data["safe_area"] = batch_safe_area

    if "road_map" in batch_data:
        if batch_data["road_map"].shape[1] == 1:
            road_map = batch_data["road_map"].squeeze(1)
            if len(road_map.shape) == 4:
                batch_data["rendered_frames"] = road_map / 128 - 1
            elif len(road_map.shape) == 3:
                color_map = vectornet_color_map.float()
                num_elements = len(color_map)
                colored_road_map = torch.zeros(
                    list(road_map.shape) + [num_elements]
                )
                for i in range(num_elements):
                    colored_road_map[:, :, :, i] = road_map == i
                colored_road_map = torch.matmul(colored_road_map, color_map)
                colored_road_map = colored_road_map / 128 - 1
                batch_data["rendered_frames"] = colored_road_map.permute(
                    0, 3, 1, 2
                )
            else:
                raise ValueError(
                    "The ndim of road maps should be either 3 or 4."
                )
        elif batch_data["road_map"].shape[1] == 3:
            batch_data["rendered_frames"] = batch_data["road_map"] / 128 - 1
        else:
            raise ValueError(
                "The ndim of road map's shape[1] should be either 3 or 1."
            )
    else:
        batch_data["rendered_frames"] = None

    return batch_data


class CatStackCollator(object):
    """
    The collator to choose which items need to be concatenate or stack.

    Args:
        keys_to_stack: keys of the items to be stack.
        keys_to_concat: keys of the items to be concat.
        max_agent_num: max value of mean agent number in a single batch.
    """

    def __init__(
        self,
        keys_to_stack: List[str],
        keys_to_concat: List[str],
        max_agent_num: int = 5,
    ):
        self.keys_to_stack = keys_to_stack
        self.keys_to_concat = keys_to_concat
        self.max_agent_num = int(max_agent_num)

    def __call__(self, batch: List[Dict]):
        batch_output = {}

        for key in self.keys_to_stack:
            if key not in batch[0]:
                continue
            batch_output[key] = torch.stack(
                [torch.Tensor(sample[key]) for sample in batch]
            )

        valid_key = None
        agent_num = None
        for key in self.keys_to_concat:
            if key not in batch[0]:
                continue
            if valid_key is None:
                valid_key = key
                agent_num = [len(sample[key]) for sample in batch]

            extra_num = sum(agent_num) - self.max_agent_num * len(agent_num)
            for i in range(len(agent_num)):
                if extra_num > 0 and agent_num[i] > self.max_agent_num:
                    tmp = min(extra_num, agent_num[i] - self.max_agent_num)
                    agent_num[i] -= tmp
                    extra_num -= tmp
                elif extra_num <= 0:
                    break

            batch_output[key] = torch.cat(
                [
                    torch.Tensor(sample[key][: agent_num[i]])
                    for i, sample in enumerate(batch)
                ]
            )

        if valid_key is not None:
            key = valid_key
            batch_output["agent_num"] = torch.Tensor(agent_num)

            batch_index = []
            for i, n in enumerate(agent_num):
                batch_index.extend([i] * n)
            batch_output["batch_index"] = torch.Tensor(batch_index)
        return batch_output


def collate_SGNet(batch: List) -> Dict:
    """Collate function for SGNet trajectory prediction model.

    Args:
        batch: a list of dataloader output.

    Returns:
        batch_data: the dictionary that contains necessary \
            information for trajectory prediction.
    """

    required_key = [
        "input_traj",
        "raw_input_traj",
        "rand_cvae_seed",
    ]
    optional_key = [
        "target_traj",
        "raw_target_traj",
        "stamp",
        "date_token",
        "id",
        "type",
        "stage",
        "enable_relative",
        "position",
        "height",
        "vcs_vel",
        "global_vel",
    ]

    batch_data = {i: [] for i in required_key + optional_key}

    # 1. Unpack the batch data.
    for item in batch:
        for key in required_key:
            if key in item:
                batch_data[key].append(item[key])
        for key in optional_key:
            if key in item:
                batch_data[key].append(item[key])

    # 2. Tensor packing and type conversion.
    # 2.1. input_traj:
    # [dim, 1, enc_steps] -> [batch_size, dim, 1, enc_steps]
    batch_input_traj = [
        torch.FloatTensor(input_traj).unsqueeze(0)
        for input_traj in batch_data["input_traj"]
    ]
    batch_data["input_traj"] = torch.cat(batch_input_traj, dim=0)

    # 2.2. raw_input_traj:
    # [dim, 1, enc_steps] -> [batch_size, dim, 1, enc_steps]
    raw_batch_raw_raw_input_traj = [
        torch.FloatTensor(raw_input_traj).unsqueeze(0)
        for raw_input_traj in batch_data["raw_input_traj"]
    ]
    batch_data["raw_input_traj"] = torch.cat(
        raw_batch_raw_raw_input_traj, dim=0
    )

    # 2.3. rand_cvae_seed:
    # [dim, 1, enc_steps] -> [batch_size, dim, 1, enc_steps]
    batch_rand_cvae_seed = [
        torch.FloatTensor(rand_cvae_seed).unsqueeze(0)
        for rand_cvae_seed in batch_data["rand_cvae_seed"]
    ]
    batch_data["rand_cvae_seed"] = torch.cat(batch_rand_cvae_seed, dim=0)

    # 2.4. target_traj.
    # [dim, 1, dec_steps] -> [batch_size, dim, 1, dec_steps]
    batch_target_traj = [
        torch.FloatTensor(target_traj).unsqueeze(0)
        for target_traj in batch_data["target_traj"]
    ]
    batch_data["target_traj"] = torch.cat(batch_target_traj, dim=0)

    # 2.5. raw_target_traj.
    # [predict_dim, 1, dec_steps] -> [batch_size, predict_dim, 1, dec_steps]
    batch_raw_target_traj = [
        torch.FloatTensor(raw_target_traj).unsqueeze(0)
        for raw_target_traj in batch_data["raw_target_traj"]
    ]
    batch_data["raw_target_traj"] = torch.cat(batch_raw_target_traj, dim=0)

    # 2.6. position and current position
    batch_data["position"] = np.array(batch_data["position"])
    batch_data["height"] = np.array(batch_data["height"])
    batch_data["vcs_vel"] = np.array(batch_data["vcs_vel"])
    batch_data["global_vel"] = np.array(batch_data["global_vel"])

    # 3. del unused optional keys
    for key in optional_key:
        if not len(batch_data[key]):
            batch_data.pop(key)

    return batch_data
