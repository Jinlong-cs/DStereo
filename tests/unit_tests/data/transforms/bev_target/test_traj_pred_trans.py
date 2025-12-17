# -*- coding: utf-8 -*-
# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import pandas as pd
import torch

from hat.data.transforms.traj_pred_trans import (
    ANCObtainHomographyTemporal,
    TrajPredMultiTransform,
)

params = {
    "bev_size": (448, 512),
    "vcs_range": (-31.8, -76.8, 102.6, 76.8),
    "ego_as_obs_flag": False,
    "default_ego_id": -42,
    "only_keep_vehicle": False,
    "num_frames_per_clip": 4,
    "num_frames_for_pred": 6,
    "bounce_thr": 3.1415926,
    "static_thr": {"vehicle": 1, "cyclist": 0.7, "pedestrain": 0.2},
    "static_obs_compensate_prob": 0.6,
    "fill_inverse_traj_flag": True,
    "pred_traj_num": 5,
    "ego_vcs_trans_flag": True,
    "obs_lcf_position_trans_flag": True,
    "obs_lcf_yaw_rotate_flag": False,
    "max_his_odo_len": 12,
    "save_memory_flag": False,
}

obs_columns = [
    "frame_id",
    "obs_x",
    "obs_y",
    "obs_z",
    "obs_height",
    "obs_width",
    "obs_length",
    "obs_yaw",
    "track_id",
    "classification",
    "obs_vx",
    "obs_vy",
    "obs_vz",
    "obs_wy",
]

ego_columns = [
    "ego_x",
    "ego_y",
    "ego_z",
    "ego_height",
    "ego_width",
    "ego_length",
    "ego_yaw",
    "ego_vx",
    "ego_vy",
    "ego_wy",
    "frame_id",
]


def get_fake_data():
    rows_num = 100
    allFrmLen = params["num_frames_per_clip"] + params["num_frames_for_pred"]
    his_ego_len = params["num_frames_per_clip"] + params["max_his_odo_len"] - 1
    np.random.seed(123456)
    target_raws = np.random.randn(rows_num, len(obs_columns)).astype(
        np.float32
    )
    frame_id = np.sort(np.random.randint(0, allFrmLen, (rows_num,)))
    track_id = np.random.randint(0, 6, (rows_num,))
    classification = np.random.randint(0, 3, (rows_num,))
    width = np.random.rand(rows_num) + 0.1
    length = np.random.rand(rows_num) + 0.1
    target_raws[:, obs_columns.index("frame_id")] = frame_id
    target_raws[:, obs_columns.index("track_id")] = track_id
    target_raws[:, obs_columns.index("classification")] = classification
    target_raws[:, obs_columns.index("obs_width")] = width
    target_raws[:, obs_columns.index("obs_length")] = length

    ego_raws = np.random.randn(allFrmLen, len(ego_columns)).astype(np.float32)
    frame_id = np.arange(allFrmLen)
    ego_raws[:, ego_columns.index("frame_id")] = frame_id

    ego_his_arrs = np.random.randn(his_ego_len, 3).astype(np.float32)

    target_data = pd.DataFrame(
        data=target_raws,
        columns=obs_columns,
    )
    target_data.drop_duplicates(
        subset=["track_id", "frame_id"],
        keep="first",
        inplace=True,
    )
    target_data = target_data.values
    obj_clmn = obs_columns.index("track_id")
    lbl_clmn = obs_columns.index("classification")
    yaw_clmn = obs_columns.index("obs_yaw")
    hz_clmn = obs_columns.index("obs_height")
    xclmn = obs_columns.index("obs_x")
    yclmn = obs_columns.index("obs_y")
    ww = obs_columns.index("obs_width")
    ll = obs_columns.index("obs_length")
    vxclmn = obs_columns.index("obs_vx")
    vyclmn = obs_columns.index("obs_vy")
    vzclmn = obs_columns.index("obs_vz")
    targets_list_tmp = []
    for frm_id in range(params["num_frames_per_clip"]):
        mask = target_data[:, obs_columns.index("frame_id")] == frm_id
        targets_pframe = torch.tensor(target_data[mask, :].copy())
        yaws = targets_pframe[:, [yaw_clmn]]
        t_instances = {
            "obj_idxes": targets_pframe[:, obj_clmn].long(),
            "labels": targets_pframe[:, lbl_clmn].long(),
            "boxes": targets_pframe[:, [yclmn, xclmn, ll, ww]],
            "yaws": torch.cat([torch.cos(yaws), torch.sin(yaws)], dim=-1),
            "bev_loc_z": targets_pframe[:, hz_clmn],
            "heights": targets_pframe[:, hz_clmn],
            "scores": torch.ones_like(targets_pframe[:, 0]),
            "velocities": targets_pframe[:, [vxclmn, vyclmn, vzclmn]],
        }
        targets_list_tmp.append(t_instances)

    his_egoinfo_clip = []
    for frm_id in range(params["num_frames_per_clip"]):
        ego_iter = ego_his_arrs[frm_id : frm_id + params["max_his_odo_len"]]
        his_egoinfo_clip.append(torch.from_numpy(ego_iter))
    his_egoinfo_clip = torch.stack(his_egoinfo_clip, dim=0)

    clip_data = {
        "motr_targets": {
            "bev_tracking": targets_list_tmp,
            "trajectory_pred": {
                "targets_raw": pd.DataFrame(
                    data=target_raws,
                    columns=obs_columns,
                ),
                "ego_raw": pd.DataFrame(data=ego_raws, columns=ego_columns),
                "sample_interval": 10,
                "clip_idx": 149,
                "num_frames_per_clip": params["num_frames_per_clip"],
            },
        },
        "odo_info": his_egoinfo_clip,
        "sample_split_index": 0,
    }
    return clip_data


def get_fake_homo_data():
    his_ego_len = params["num_frames_per_clip"] + params["max_his_odo_len"] - 1
    np.random.seed(123456)
    ego_his_arrs = np.random.randn(his_ego_len, 3).astype(np.float32)

    his_egoinfo_clip = []
    for frm_id in range(params["num_frames_per_clip"]):
        ego_iter = ego_his_arrs[frm_id : frm_id + params["max_his_odo_len"]]
        his_egoinfo_clip.append(torch.from_numpy(ego_iter))
    his_egoinfo_clip = torch.stack(his_egoinfo_clip, dim=0)

    clip_data = {
        "odo_info": his_egoinfo_clip,
    }
    return clip_data


def test_trajpred_multi_transform():
    clip_data = get_fake_data()

    transforms = TrajPredMultiTransform(
        use_fut_info_filter=True,
        vcs_range=params["vcs_range"],
        car_ped_cyc_type_id={"vehicle": 0, "pedestrain": 2, "cyclist": 1},
        static_thr=params["static_thr"],
        num_sample_per_clip=params["num_frames_per_clip"],
        min_num_his_frame_thr=2,
        min_num_fut_frame_thr=1,
        max_context_frame_num=4,
        num_frames_for_pred=params["num_frames_for_pred"],
        pred_traj_modals_num=params["pred_traj_num"],
        max_his_odo_len=params["max_his_odo_len"],
        yaw_diff_bounce_thr=params["bounce_thr"],
        static_obs_compensate_prob=params["static_obs_compensate_prob"],
        ego_vcs_trans_flag=params["ego_vcs_trans_flag"],
        obs_lcf_position_trans_flag=params["obs_lcf_position_trans_flag"],
        obs_lcf_yaw_rotate_flag=params["obs_lcf_yaw_rotate_flag"],
        future_traj_only=True,
        filter_unstable_class=True,
        fill_inverse_traj_flag=params["fill_inverse_traj_flag"],
        save_memory_flag=params["save_memory_flag"],
        only_keep_vehicle=params["only_keep_vehicle"],
    )
    clip_data = transforms(clip_data)
    new_bev_tracking0 = clip_data["motr_targets"]["bev_tracking"][0]
    assert (
        "gt_traj_regs" in list(new_bev_tracking0.keys())
        and "gt_traj_masks" in list(new_bev_tracking0.keys())
        and "gt_traj_modals" in list(new_bev_tracking0.keys())
    )


def test_obtain_homography_temporal():
    clip_data = get_fake_homo_data()

    trans_homo_temporal = ANCObtainHomographyTemporal(
        bev_size=params["bev_size"], vcs_range=params["vcs_range"]
    )
    clip_data = trans_homo_temporal(clip_data)
    assert clip_data["homography_temporal"].shape == (
        1,
        params["num_frames_per_clip"],
        3,
        3,
    )
