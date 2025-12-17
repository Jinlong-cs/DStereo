from enum import Enum
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R

from hat.core.traj_pred_utils import Affine2D
from hat.data.transforms.auto_3dv import ANCTemporalHomo
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "ANCObtainHomographyTemporal",
    "TrajPredMultiTransform",
]

DEFAULT_GT_MODAL_INDEX = -1
VALID_CLASS_NUM = 0


class FilterFlag(Enum):
    normal = 0
    vehicle_only = 1
    his_frame_not_enough = 2
    fut_frame_not_enough = 3
    ped_label_unstable = 4
    obs_lcf_out_of_bev = 5
    obs_stationary = 6
    obs_yaw_bouncing = 7
    filter_by_json = 8
    token_idx_less_than_zero = 9
    visibility = 10


class DropDuplicatedRecords:
    """Drop duplicates records with same values of ["track_id", "frame_id"].

    Only keep the first one.
    """

    def __init__(self):
        pass

    def __call__(self, clip_data: Dict):
        assert "motr_targets" in clip_data
        trajectory_pred = clip_data["motr_targets"]["trajectory_pred"]
        targets_raw = trajectory_pred["targets_raw"]
        targets_raw = targets_raw.drop_duplicates(
            subset=["track_id", "frame_id"], keep="first"
        )
        trajectory_pred["targets_raw"] = targets_raw
        clip_data["motr_targets"]["trajectory_pred"] = trajectory_pred
        return clip_data


class AddEgoInfosToGt:
    """
    Treat ego as a common obstacle, and add ego info to target and instance.

    Args:
        vcs_range: range of vehicle coordinage system.(bottom, right, top,
            left).
        ego_as_obs_flag: Whether treat ego as a commom obstacle or not.
        default_ego_id:If ego_as_obs_flag, give ego a track-id.Default to -1,
            this is optional.
        default_ego_label: if ego_as_obs_flag is true, give ego a class-
            Label. Default to 0, 0 stand for vehicle class in most case.
    """

    def __init__(
        self,
        vcs_range: Tuple,
        ego_as_obs_flag: bool = False,
        default_ego_id: int = -1,
        default_ego_label: int = 0,
    ):
        self.vcs_range = vcs_range

        self.ego_as_obs_flag = ego_as_obs_flag
        self.default_ego_id = default_ego_id
        self.default_ego_label = default_ego_label
        self.const_col_name = [
            "track_id",
            "classification",
            "comb_token",
            "visibility",
        ]

    def _add_ego_info_to_target(
        self, cur_target_df: pd.DataFrame, cur_ego_df: pd.DataFrame
    ):
        ego_values = cur_ego_df.values

        const_col = np.array(
            [self.default_ego_id, self.default_ego_label, -1, 4]
        )[np.newaxis, :]
        const_col = const_col.repeat(ego_values.shape[0], 0)
        cur_ego_values = np.concatenate(
            [
                ego_values,
                const_col,
            ],
            0,
        )
        ego_columns = list(cur_ego_df.keys()) + self.const_col_name

        obs_columns = list(cur_target_df.keys())

        ego_index = []
        for obscolname in obs_columns:
            if "obs_" == obscolname[:4]:
                egocolname = "ego_" + obscolname[4:]
                ego_index.append(ego_columns.index(egocolname))
            else:
                ego_index.append(ego_columns.index(obscolname))

        cur_ego_raws = cur_ego_values[:, ego_index]
        cur_target_data = np.concatenate(
            [cur_ego_raws, cur_target_df.values], axis=0
        )
        return pd.DataFrame(data=cur_target_data, columns=obs_columns)

    def _add_ego_info_to_instance(
        self, bev_tracking: Dict, ego_raws: pd.DataFrame
    ):
        ego_columns = list(ego_raws.keys())
        ego_raws_data = ego_raws.values
        ego_wid_col = ego_columns.index("ego_width")
        ego_len_col = ego_columns.index("ego_length")
        ego_hei_col = ego_columns.index("ego_height")
        ego_yaw_col = ego_columns.index("ego_yaw")
        frame_id_col = ego_columns.index("frame_id")
        for idx in range(len(bev_tracking)):
            t_instances = {}
            cur_instance = bev_tracking[idx]
            mask = ego_raws_data[:, frame_id_col] == idx
            cur_ego_raw = ego_raws_data[mask, :][0]
            t_instances["obj_idxes"] = torch.cat(
                [
                    cur_instance["obj_idxes"],
                    torch.tensor([self.default_ego_id]).to(
                        cur_instance["obj_idxes"]
                    ),
                ],
                0,
            )

            t_instances["labels"] = torch.cat(
                [
                    cur_instance["labels"],
                    torch.tensor([self.default_ego_label]).to(
                        cur_instance["labels"]
                    ),
                ],
                0,
            )

            bev_norm_y = (self.vcs_range[3] - 0) / (
                self.vcs_range[3] - self.vcs_range[1]
            )
            bev_norm_x = (self.vcs_range[2] - 0) / (
                self.vcs_range[2] - self.vcs_range[0]
            )
            bev_norm_width = cur_ego_raw[ego_wid_col] / (
                self.vcs_range[3] - self.vcs_range[1]
            )
            bev_norm_height = cur_ego_raw[ego_len_col] / (
                self.vcs_range[2] - self.vcs_range[0]
            )
            ts_boxes = torch.tensor(
                [
                    bev_norm_y,
                    bev_norm_x,
                    bev_norm_width,
                    bev_norm_height,
                ]
            ).to(cur_instance["boxes"])
            t_instances["boxes"] = torch.cat(
                [cur_instance["boxes"], ts_boxes.unsqueeze(0)], 0
            )

            t_yaw = cur_ego_raw[ego_yaw_col] - cur_ego_raw[ego_yaw_col]
            ts_cossinyaw = torch.tensor([np.cos(t_yaw), np.sin(t_yaw)]).to(
                cur_instance["yaws"]
            )
            t_instances["yaws"] = torch.cat(
                [cur_instance["yaws"], ts_cossinyaw.unsqueeze(0)], 0
            )

            t_instances["bev_loc_z"] = torch.cat(
                [
                    cur_instance["bev_loc_z"],
                    torch.tensor([cur_ego_raw[ego_hei_col]]).to(
                        cur_instance["bev_loc_z"]
                    ),
                ],
                0,
            )
            t_instances["heights"] = torch.cat(
                [
                    cur_instance["heights"],
                    torch.tensor([cur_ego_raw[ego_hei_col]]).to(
                        cur_instance["heights"]
                    ),
                ],
                0,
            )
            t_instances["scores"] = torch.cat(
                [
                    cur_instance["scores"],
                    torch.tensor([1.0]).to(cur_instance["scores"]),
                ],
                0,
            )
            if "velocities" in bev_tracking[0]:
                ego_vx_col = ego_columns.index("ego_vx")
                ego_vy_col = ego_columns.index("ego_vy")
                tmp_yaw = cur_ego_raw[ego_yaw_col]
                vx = cur_ego_raw[ego_vx_col]
                vy = cur_ego_raw[ego_vy_col]
                vx_1 = vx * np.cos(tmp_yaw) + vy * np.sin(tmp_yaw)
                vy_1 = -vx * np.sin(tmp_yaw) + vy * np.cos(tmp_yaw)
                cur_velo = torch.tensor([vx_1, vy_1, 0.0]).to(
                    cur_instance["velocities"]
                )
                t_instances["velocities"] = torch.cat(
                    [cur_instance["velocities"], cur_velo.unsqueeze(0)], 0
                )

            if "visibility" in bev_tracking[0]:
                t_instances["visibility"] = torch.cat(
                    [
                        cur_instance["visibility"],
                        torch.tensor([4]).to(cur_instance["visibility"]),
                    ],
                    0,
                )

            bev_tracking[idx] = t_instances
        return bev_tracking

    def __call__(self, clip_data: Dict):
        assert "motr_targets" in clip_data
        trajectory_pred = clip_data["motr_targets"]["trajectory_pred"]

        ego_raw = trajectory_pred["ego_raw"]

        targets_raw = trajectory_pred["targets_raw"]

        if self.ego_as_obs_flag:
            bev_tracking = clip_data["motr_targets"]["bev_tracking"]
            bev_tracking = self._add_ego_info_to_instance(
                bev_tracking, ego_raw
            )
            clip_data["motr_targets"]["bev_tracking"] = bev_tracking

            # add one targets that key infomations comes from ego_raw.
            targets_raw = self._add_ego_info_to_target(
                targets_raw,
                ego_raw,
            )

        trajectory_pred["targets_raw"] = targets_raw
        clip_data["motr_targets"]["trajectory_pred"] = trajectory_pred
        return clip_data


class PhyToBEVCoord:
    """
    Transform physical coordinate of data to BEV coordinate within a clip.

    Args:
        vcs_range: range of vehicle coordinage system.(bottom, right, top,
            left).
    """

    def __init__(self, vcs_range: Tuple):
        self.vcs_range = vcs_range

    def __call__(self, clip_data: Dict):
        assert "motr_targets" in clip_data
        trajectory_pred = clip_data["motr_targets"]["trajectory_pred"]
        ego_df = trajectory_pred["ego_raw"]
        targets_df = trajectory_pred["targets_raw"]

        # trans to ego-VCS, then to BEV coordinate
        wcs_obs_xyyaw = targets_df.loc[
            :, ["frame_id", "obs_x", "obs_y", "obs_yaw"]
        ]
        wcs_ego_xyyaw = ego_df.loc[
            :, ["frame_id", "ego_x", "ego_y", "ego_yaw"]
        ]

        merged_xyyaw = pd.merge(wcs_obs_xyyaw, wcs_ego_xyyaw, on="frame_id")

        obs_relative_xy = (
            merged_xyyaw.loc[:, ["obs_x", "obs_y"]].values
            - merged_xyyaw.loc[:, ["ego_x", "ego_y"]].values
        )
        obs_relative_xy = obs_relative_xy[:, :, np.newaxis]  # [num_obs, 2, 1]

        ego_yaw = merged_xyyaw.loc[:, ["ego_yaw"]].values
        cosyaw = np.cos(ego_yaw).astype(np.float64)
        sinyaw = np.sin(ego_yaw).astype(np.float64)
        rotation_mat = np.concatenate(
            [cosyaw, sinyaw, -sinyaw, cosyaw], axis=-1
        ).reshape(
            -1, 2, 2
        )  # [num_obs, 2, 2]

        tmp = np.matmul(rotation_mat, obs_relative_xy)[..., 0]
        obs_vcs_x, obs_vcs_y = tmp[..., 0], tmp[..., 1]
        obs_vcs_yaw = merged_xyyaw.loc[:, "obs_yaw"].values - ego_yaw[..., 0]
        targets_df["lcf_vcs_x"] = obs_vcs_x
        targets_df["lcf_vcs_y"] = obs_vcs_y
        targets_df["lcf_vcs_yaw"] = obs_vcs_yaw

        # vcs to bev and normalization.
        bev_norm_x = (self.vcs_range[2] - obs_vcs_x) / (
            self.vcs_range[2] - self.vcs_range[0]
        )
        bev_norm_y = (self.vcs_range[3] - obs_vcs_y) / (
            self.vcs_range[3] - self.vcs_range[1]
        )
        bev_yaw = obs_vcs_yaw + np.pi

        targets_df["norm_bev_x"] = bev_norm_x
        targets_df["norm_bev_y"] = bev_norm_y
        targets_df["bev_yaw"] = bev_yaw
        trajectory_pred["targets_raw"] = targets_df
        clip_data["motr_targets"]["trajectory_pred"] = trajectory_pred
        return clip_data


class FilterObsStacles:
    """
    Filter out obstacles based on a set of rules.

    Rules include:
        1. lcf_position within temp-BEV
        2. track_id's classification matches leaving mode
        3. enough number of history frame
        4. enough number of future frame
        5. yaw_bouncing
        6. stationary
        7. visibility
        8. ped_label_unstable

    This transform method requires the clip-data to have the keys 'bev_...'.
    Therefore, it requires dataflow has passed `PhyToBEVCoord`.

    For some filter conditions, different agent class have different params.
    The input classes of data only support 3, names pedestrain, cyclist and
    car.

    Args:
        use_fut_info_filter: if False(not use future info for filtering):
            only considerate history frame in yaw_bouncing, stationary
            filter and frame number filter. (one can't obtain future info
            in current frame in practise.)
        car_ped_cyc_type_id:  the classification id of the car, the pedestrain
            and the cyclist.(e.g. {"vehicle": 0, "cyclist": 1,
            "pedestrain": 2].)
        static_thr:  if displace of adjacent frames is smaller than threshold,
            then filter out this obstacle[m].Each agent class should give one
            value.(e.g. {"vehicle": 1.0, "cyclist": 0.5, "pedestrain": 0.2].).
            It shold be share the same keys in car_ped_cyc_type_id.
        num_sample_per_clip:  the number of frames of one clip.
        min_num_fut_frame_thr: the minimum future frame number of agents.
        min_num_his_frame_thr:  the minimum historical frame number of agents.
        max_context_frame_num:  the maximum historical frame number of agents,
            larger than this number is discarded.
        max_future_frame_num:the maximum future frame number of agents, larger
            than this number is discarded.
        only_keep_vehicle: Default to false. If true, obstacles being not
            vehicle will be filtered out.
        yaw_diff_bounce_thr: the yaw-diff thr between two adjacent frames in
            Rad after transforming to [0,2*pi). Default to 2*pi, whitch means
            no bound.
        static_obs_compensate_prob:with a 1-prob probability random add static
            agents to train set.Default to 1.0, means no compensate.
        filter_unstable_class: if True, filter ids according to whose class is
            stable or not.Only enable on cyclist and pedestrain. If true, here
            use_fut_info_filter shold be ture, too.
        default_ego_id:If ego_as_obs_flag, give ego a track-id.Default to -1,
            this is optional.
        visibility: the level of visibility.Agents' visibility blow it will be
            discard.Default to 2(50% occulusion)
    """

    def __init__(
        self,
        use_fut_info_filter: bool,
        car_ped_cyc_type_id: Dict,
        static_thr: Dict,
        num_sample_per_clip: int,
        min_num_his_frame_thr: int,
        min_num_fut_frame_thr: int,
        max_context_frame_num: int,
        max_future_frame_num: int,
        only_keep_vehicle: bool = False,
        yaw_diff_bounce_thr: float = 6.284,
        static_obs_compensate_prob: float = 1.0,
        filter_unstable_class: bool = False,
        default_ego_id: int = -1,
        visibility: int = 2,
    ):
        self.use_fut_info_filter = use_fut_info_filter
        self.car_ped_cyc_type_id = car_ped_cyc_type_id
        self.only_keep_vehicle = only_keep_vehicle
        self.num_sample_per_clip = num_sample_per_clip
        self.min_num_fut_frame_thr = min_num_fut_frame_thr
        if self.min_num_fut_frame_thr > 0:
            assert self.use_fut_info_filter
        self.min_num_his_frame_thr = min_num_his_frame_thr
        self.max_context_frame_num = max_context_frame_num
        self.max_future_frame_num = max_future_frame_num
        self.yaw_diff_bounce_thr = yaw_diff_bounce_thr
        assert len(static_thr) == len(car_ped_cyc_type_id)
        self.speed_static_thr = {
            car_ped_cyc_type_id[key]: static_thr[key]
            for key, _ in static_thr.items()
        }
        self.filter_unstable_class = filter_unstable_class
        if filter_unstable_class:
            assert self.use_fut_info_filter
        self.default_ego_id = default_ego_id
        self.static_obs_compensate_prob = static_obs_compensate_prob
        self.visibility = visibility

    def __call__(self, clip_data: Dict):
        assert "motr_targets" in clip_data
        trajectory_pred = clip_data["motr_targets"]["trajectory_pred"]
        target_df = trajectory_pred["targets_raw"]
        obs_columns = list(target_df.keys())
        target_vals = target_df.values

        frame_id_col = obs_columns.index("frame_id")
        track_id_col = obs_columns.index("track_id")
        cls_id_col = obs_columns.index("classification")
        bev_x_col = obs_columns.index("norm_bev_x")
        bev_y_col = obs_columns.index("norm_bev_y")
        org_yaw_col = obs_columns.index("obs_yaw")
        obs_vx_col = obs_columns.index("obs_vx")
        obs_vy_col = obs_columns.index("obs_vy")
        obs_wy_col = obs_columns.index("obs_wy")

        if "visibility" in obs_columns:
            vis_col = obs_columns.index("visibility")

        lcf_track_id_arr = []
        ids_state_flag = []
        valid_ids = []
        valid_classes = []
        target_raw = []
        for lcf_id in range(self.num_sample_per_clip):
            min_context_frame_id = lcf_id - self.max_context_frame_num + 1
            max_future_frame_id = lcf_id + self.max_future_frame_num
            # frame id range filter
            mask = (target_vals[:, frame_id_col] >= min_context_frame_id) & (
                target_vals[:, frame_id_col] <= max_future_frame_id
            )
            targets_info = target_vals[mask]

            if 0 == len(targets_info):
                # if filter out all agent, return empty
                target_raw.append({})
                lcf_track_id_arr.append(np.array([], dtype=np.int))
                state_flag = np.array([], dtype=np.int)
                cur_validids = np.array([], dtype=np.int)
                cur_classes = {}
            else:
                lcf_valid_info = targets_info[
                    lcf_id == targets_info[:, frame_id_col]
                ]  # last context frame data
                last_context_frame_track_ids = lcf_valid_info[
                    :, track_id_col
                ].astype(np.int)
                last_context_frame_classes = lcf_valid_info[
                    :, cls_id_col
                ].astype(np.int)
                cur_classes = {
                    hhkey: hhvalue
                    for hhkey, hhvalue in zip(
                        last_context_frame_track_ids,
                        last_context_frame_classes,
                    )
                }
                # 1. filter ids according to leaving mode
                lcf_track_id_arr.append(last_context_frame_track_ids)
                state_flag = np.zeros(
                    shape=[len(last_context_frame_track_ids)], dtype=np.int
                )
                if self.only_keep_vehicle:
                    mask = (
                        self.car_ped_cyc_type_id["vehicle"]
                        != lcf_valid_info[:, cls_id_col]
                    )
                    state_flag[mask] = FilterFlag.vehicle_only.value
                # 2. filter ids according to whether last context frame
                # position is within BEV area or not.
                # NOTE: compare consistance should replace with old code here.
                mask = (
                    (lcf_valid_info[:, bev_x_col] < 0)
                    | (lcf_valid_info[:, bev_x_col] >= 1)
                    | (lcf_valid_info[:, bev_y_col] < 0)
                    | (lcf_valid_info[:, bev_y_col] >= 1)
                )
                mask = mask & (0 == state_flag)
                state_flag[mask] = FilterFlag.obs_lcf_out_of_bev.value
                if "visibility" in obs_columns:
                    mask = (lcf_valid_info[:, vis_col] < self.visibility) & (
                        0 == state_flag
                    )
                    state_flag[mask] = FilterFlag.visibility.value

                valid_data_out = {}
                if np.any(
                    0 == state_flag
                ):  # if does not filter out all agent,
                    for idx, flag, track_id in zip(
                        np.arange(len(state_flag)),
                        state_flag,
                        last_context_frame_track_ids,
                    ):
                        if flag > 0:
                            continue

                        tmp_track_id_data = targets_info[
                            track_id == targets_info[:, track_id_col]
                        ]
                        cls_id = cur_classes[track_id]

                        # 3. filter id according to the number of history and
                        # future frame
                        valid_context_frame_num = np.sum(
                            tmp_track_id_data[:, frame_id_col] <= lcf_id
                        )
                        valid_fut_frame_num = np.sum(
                            tmp_track_id_data[:, frame_id_col] > lcf_id
                        )
                        if (
                            valid_context_frame_num
                            < self.min_num_his_frame_thr
                        ):
                            state_flag[
                                idx
                            ] = FilterFlag.his_frame_not_enough.value
                            continue
                        if valid_fut_frame_num < self.min_num_fut_frame_thr:
                            state_flag[
                                idx
                            ] = FilterFlag.fut_frame_not_enough.value
                            continue

                        frameids = tmp_track_id_data[:, frame_id_col]
                        orgyaws = tmp_track_id_data[:, org_yaw_col]
                        class_ids = tmp_track_id_data[:, cls_id_col]
                        org_vx, org_vy, org_wy = (
                            tmp_track_id_data[:, obs_vx_col],
                            tmp_track_id_data[:, obs_vy_col],
                            tmp_track_id_data[:, obs_wy_col],
                        )
                        if not self.use_fut_info_filter:
                            org_vx = org_vx[frameids <= lcf_id]
                            org_vy = org_vy[frameids <= lcf_id]
                            org_wy = org_wy[frameids <= lcf_id]

                            orgyaws = orgyaws[frameids <= lcf_id]
                            class_ids = class_ids[frameids <= lcf_id]
                            frameids = frameids[frameids <= lcf_id]

                        # 4. filter id according to yaw
                        if self.yaw_diff_bounce_thr < np.pi and (
                            self.default_ego_id != track_id
                        ):
                            speed = np.sqrt(org_vx ** 2 + org_vy ** 2)
                            yaw = np.arctan2(org_vy, org_vx)
                            tmp_yaw = np.unwrap(
                                yaw[
                                    speed >= self.speed_static_thr[int(cls_id)]
                                ]
                            )
                            if len(tmp_yaw) >= 2:
                                if (
                                    np.max(np.abs(tmp_yaw[1:] - tmp_yaw[:-1]))
                                    >= self.yaw_diff_bounce_thr
                                ):
                                    state_flag[
                                        idx
                                    ] = FilterFlag.obs_yaw_bouncing.value
                                    continue

                        # 5. Filter by classification.
                        # As pedestrain or cyclist, it must be always the
                        # same classification in the context frames.Otherwise,
                        # the perception or tracking may be wrong, skip it.
                        if (
                            self.filter_unstable_class
                            and cls_id != self.car_ped_cyc_type_id["vehicle"]
                            and (self.default_ego_id != track_id)
                        ):
                            if np.any(cls_id != class_ids):
                                state_flag[
                                    idx
                                ] = FilterFlag.ped_label_unstable.value
                                continue

                        # 6. filter id according to speed.
                        avg_speed = np.mean(np.sqrt(org_vx ** 2 + org_vy ** 2))
                        if avg_speed < self.speed_static_thr[int(cls_id)] and (
                            self.default_ego_id != track_id
                        ):
                            if (
                                np.random.rand()
                                <= self.static_obs_compensate_prob
                            ):  # if is in trainning, static_pass_prob pass it.
                                state_flag[
                                    idx
                                ] = FilterFlag.obs_stationary.value
                                continue

                        valid_data_out[int(track_id)] = pd.DataFrame(
                            data=tmp_track_id_data,
                            columns=obs_columns,
                        )
                target_raw.append(valid_data_out)
                cur_validids = lcf_track_id_arr[-1][0 == state_flag]
            ids_state_flag.append(state_flag)
            valid_ids.append(cur_validids)
            valid_classes.append(cur_classes)
        trajectory_pred["targets_raw"] = target_raw
        trajectory_pred["ids_list"] = lcf_track_id_arr
        trajectory_pred["state_flag_list"] = ids_state_flag
        trajectory_pred["valid_ids"] = valid_ids
        trajectory_pred["valid_classes"] = valid_classes
        clip_data["motr_targets"]["trajectory_pred"] = trajectory_pred
        return clip_data


class GeneTrajMask:
    """
    Generate mask for each trajectory.

    This transform function requires data have passed `FilterObsStacles`.

    Args:
        max_context_frame_num:  the maximum historical frame number of agents,
            default to 4, more than this number is discarded.
        max_future_frame_num:the maximum future frame number of agents,default
            to 12, more than this number is discarded.
        ego_vcs_trans_flag: whether rotate obstacle's coordinates according to
            ego's yaw and position in the last context frame. Default to True.
        obs_lcf_position_trans_flag:  whether translate obstacle's coordinates
            according to its position in the last context frame.    Default to
            True.
        obs_lcf_yaw_rotate_flag:if true,rotate obstacle's coordinate according
            to its yaw in the last context frame.   Default to False. when not
            translating, there does not rotate, too.
        future_traj_only: If True, only return future trajectories, whose last
            dim is max_future_frame_num.   If False, return history and future
            trajectories together,    whose last dim is max_future_frame_num +
            max_context_frame_num.
        fill_inverse_traj_flag:    If True, fill the GT-trajectories of agents
            moving by higt speed and opposite to ego motion direction in
            training mode.
        fill_inverse_traj_prob: Prob to fill the GT-trajectories of agents.
            Default to 0.4.
    """

    def __init__(
        self,
        max_context_frame_num: int = 4,
        max_future_frame_num: int = 12,
        ego_vcs_trans_flag: bool = True,
        obs_lcf_position_trans_flag: bool = True,
        obs_lcf_yaw_rotate_flag: bool = False,
        future_traj_only: bool = True,
        fill_inverse_traj_flag: bool = False,
        fill_inverse_traj_prob: float = 0.4,
    ):
        self.max_context_frame_num = max_context_frame_num
        self.max_future_frame_num = max_future_frame_num
        self.total_frame_num = max_context_frame_num + max_future_frame_num
        self.ego_vcs_trans_flag = ego_vcs_trans_flag
        self.obs_lcf_position_trans_flag = (
            obs_lcf_position_trans_flag & ego_vcs_trans_flag
        )
        self.obs_lcf_yaw_rotate_flag = (
            obs_lcf_yaw_rotate_flag & obs_lcf_position_trans_flag
        )
        self.future_traj_only = future_traj_only
        self.fill_inverse_traj_flag = fill_inverse_traj_flag
        self.fill_inverse_traj_prob = fill_inverse_traj_prob
        self.select_keys = [
            "frame_id",
            "obs_x",
            "obs_y",
            "obs_yaw",
            "obs_vx",
            "obs_vy",
            "obs_wy",
        ]

    def _rotate_or_translate_coordinate(
        self,
        coordxy,
        yaw: float,
        pos_x: float,
        pos_y: float,
        position_trans_flag: bool,
        rotate_flag: bool,
    ):
        tmp_x, tmp_y = coordxy[:, 0], coordxy[:, 1]
        if position_trans_flag:
            tmp_x, tmp_y = Affine2D.coord_translate(tmp_x, tmp_y, pos_x, pos_y)
        if rotate_flag:
            tmp_x, tmp_y = Affine2D.coord_rotate(tmp_x, tmp_y, yaw)

        return np.array([tmp_x, tmp_y]).T  # N * 2

    def _fill_invs_trajs_by_constantv(self, cur_traj, cur_mask, prob=0.3):
        ratio = np.mean(cur_mask)
        # 如果已经有足够多的轨迹点，就不填补了 ##
        if ratio >= 0.95 or np.random.rand() >= prob:
            return cur_traj, cur_mask

        # 如果有效点个数小于4不填补，有效点数少，填补结果不精确##
        traj_len = cur_traj.shape[0]
        valid_len = int(np.sum(cur_mask[:, 0]))
        if valid_len < 4:
            return cur_traj, cur_mask

        # 因为用了最后四个点计算速度和平均位置，所以这四个点必须全部有效,##
        # 否则不做填补 ##
        last_valid_pos = np.arange(traj_len)[cur_mask[:, 0] > 0.5][-1]
        if np.any(cur_mask[last_valid_pos - 3 : last_valid_pos + 1, 0] == 0):
            return cur_traj, cur_mask

        valid_traj = cur_traj[cur_mask[:, 0] > 0.5]
        diff = valid_traj[1:, :] - valid_traj[:-1, :]
        mean_diff = np.mean(diff[-3:], 0).reshape([-1, 2])
        # last valid pos
        mean_pos = np.mean(valid_traj[-3:], 0).reshape([-1, 2]) + mean_diff

        # 填补时仅填补未来帧的轨迹点，注意LCF帧必然是有效点，##
        # 故last valid pos最小从Lcf开始 ##
        comp_data = (np.arange(traj_len - last_valid_pos - 1) + 1).reshape(
            [-1, 1]
        ) * mean_diff + mean_pos
        cur_traj[last_valid_pos + 1 :] = comp_data
        cur_mask[last_valid_pos + 1 :] = 1
        return cur_traj, cur_mask

    def __call__(self, clip_data: Dict):
        assert "motr_targets" in clip_data
        trajectory_pred = clip_data["motr_targets"]["trajectory_pred"]
        valid_ids = trajectory_pred["valid_ids"]
        targets_df = trajectory_pred["targets_raw"]

        ego_df = trajectory_pred["ego_raw"]
        ego_raw = ego_df.loc[
            :, ["ego_x", "ego_y", "ego_yaw", "ego_vx", "ego_vy"]
        ].values

        traj_list = []
        mask_list = []
        vxvywy_list = []
        for lcfid, frm_valid_ids in enumerate(valid_ids):
            frm_targets_dict = targets_df[
                lcfid
            ]  # temp valid frame targets_raw
            traj_tmp_frame = {}
            mask_tmp_frame = {}
            vxvywy_tmp_frame = {}
            if 0 < len(frm_targets_dict):
                if self.ego_vcs_trans_flag:
                    ego_x, ego_y, ego_yaw, ego_vx, ego_vy = ego_raw[lcfid]
                for trackid in frm_valid_ids:
                    trackid_df = frm_targets_dict[int(trackid)]
                    track_id_data = trackid_df.loc[:, self.select_keys].values
                    tmp_frame = track_id_data[:, 0].astype(np.int)
                    cur_traj_xy = track_id_data[:, [1, 2]]
                    lcfyaw = track_id_data[lcfid == tmp_frame, 3][0]
                    lcfx, lcfy = cur_traj_xy[lcfid == tmp_frame].reshape([-1])

                    cur_traj_vxvywy = track_id_data[:, [4, 5, 6]]
                    lcf_vx, lcf_vy = cur_traj_vxvywy[
                        lcfid == tmp_frame
                    ].reshape([-1])[:2]

                    # trans to ego-VCS
                    cur_ego_vcs_traj_xy = self._rotate_or_translate_coordinate(
                        cur_traj_xy, ego_yaw, ego_x, ego_y, True, True
                    )
                    Lcf_vcs_x, Lcf_vcs_y = cur_ego_vcs_traj_xy[
                        lcfid == tmp_frame
                    ].reshape([-1])
                    lcf_vcs_yaw = lcfyaw - ego_yaw
                    if self.ego_vcs_trans_flag:
                        cur_traj_xy = cur_ego_vcs_traj_xy
                        lcfx, lcfy = cur_traj_xy[lcfid == tmp_frame].reshape(
                            [-1]
                        )
                        lcfyaw = lcfyaw - ego_yaw

                    # trans to obstacle VCS
                    if self.obs_lcf_position_trans_flag:
                        cur_traj_xy = self._rotate_or_translate_coordinate(
                            cur_traj_xy,
                            lcfyaw,
                            lcfx,
                            lcfy,
                            self.obs_lcf_position_trans_flag,
                            self.obs_lcf_yaw_rotate_flag,
                        )

                    # trajs initialize to max-pred-traj-length.
                    traj_tmp = np.zeros(
                        shape=[self.total_frame_num, 2],
                        dtype=track_id_data.dtype,
                    )
                    mask_tmp = np.zeros(
                        shape=[self.total_frame_num, 2],
                        dtype=track_id_data.dtype,
                    )
                    vxvywy_tmp = np.zeros(
                        shape=[self.total_frame_num, 3],
                        dtype=track_id_data.dtype,
                    )
                    frameList = (
                        tmp_frame - (lcfid - self.max_context_frame_num + 1)
                    ).astype(np.int)
                    traj_tmp[frameList] = cur_traj_xy
                    mask_tmp[frameList] = 1.0
                    vxvywy_tmp[frameList] = cur_traj_vxvywy

                    if (
                        self.future_traj_only
                    ):  # only return future trajectories or return all
                        if self.fill_inverse_traj_flag:
                            # here is a data-argumentation. Deal with the lack
                            # of future GT-trajs of obstacle leaving out of
                            # bev range and disappeared in few seconds later.
                            speed_ego = np.sqrt(ego_vx ** 2 + ego_vy ** 2)
                            speed_obs = np.sqrt(lcf_vx ** 2 + lcf_vy ** 2)
                            # 如果与自车方向夹角小于2/3*pi,则不填补##
                            lcf_vcs_yaw_diff_pi = np.abs(
                                np.mod(lcf_vcs_yaw, np.pi * 2) - np.pi
                            )
                            if ((speed_ego + speed_obs) > 7) and (
                                lcf_vcs_yaw_diff_pi < np.pi / 3
                            ):
                                (
                                    traj_tmp,
                                    mask_tmp,
                                ) = self._fill_invs_trajs_by_constantv(
                                    traj_tmp,
                                    mask_tmp,
                                    prob=self.fill_inverse_traj_prob,
                                )
                        traj_tmp = traj_tmp[self.max_context_frame_num :, :]
                        mask_tmp = mask_tmp[self.max_context_frame_num :, :]
                        vxvywy_tmp = vxvywy_tmp[
                            self.max_context_frame_num :, :
                        ]

                    traj_tmp_frame[int(trackid)] = traj_tmp
                    mask_tmp_frame[int(trackid)] = mask_tmp
                    vxvywy_tmp_frame[int(trackid)] = vxvywy_tmp
            traj_list.append(traj_tmp_frame)
            mask_list.append(mask_tmp_frame)
            vxvywy_list.append(vxvywy_tmp_frame)

        trajectory_pred["gt_traj_list"] = traj_list
        trajectory_pred["gt_mask_list"] = mask_list
        trajectory_pred["gt_vxvywy_list"] = vxvywy_list
        clip_data["motr_targets"]["trajectory_pred"] = trajectory_pred
        return clip_data


class MatchModalIndex:
    """
    Generate GT-modal for each trajectory according to yaw-diff .

    Yaw-diff is yaw difference between end-point and current frame.This
    transform function requires data have passed `GeneTrajMask`.
    Args:
        num_modals:number of predicting trajecotry model, only when num_modals
            in [3,5,7], given agents GT-traj modal.
        speed_static_thr: agents' speed below threshold don't given GT-modal.
    """

    def __init__(
        self,
        num_modals: int,
        speed_static_thr: float = 0.2,
    ):
        self.num_modals = num_modals
        # split thresholds of modal according to yaw-diff between its last
        # valid future traj-positon and LCF positon.
        # here is threshold between [Straight-travel, left-cutin, left-turn,
        # left-U-turn, right-U-turn, right-turn, right-cutin]
        turn_yaw_split_thr = np.array(
            [
                0.0,
                np.pi / 40,
                4.2 * np.pi / 36,
                13 * np.pi / 36,
                np.pi,
                59 * np.pi / 36,
                67.8 * np.pi / 36,
                79 * np.pi / 40,
            ]
        )
        if 7 == num_modals:
            # Straight-travel, left-cutin, left-turn, left-U-turn,
            # right-U-turn, right-turn, right-cutin
            split_thr = turn_yaw_split_thr
        elif 5 == num_modals:
            # Straight-travel, left-cutin, left-turn, right-turn, right-cutin
            split_thr = turn_yaw_split_thr[[0, 1, 2, 4, 6, 7]]
        elif 3 == num_modals:
            # Straight-travel, left-turn, right-turn
            split_thr = turn_yaw_split_thr[[0, 1, 4, 7]]
        else:
            assert num_modals >= 1
            split_thr = None
        self.split_thr = split_thr
        self.speed_static_thr = speed_static_thr

    def __call__(self, clip_data: Dict):
        if self.split_thr is None:
            # no assigned GT-modals
            return clip_data

        assert "motr_targets" in clip_data
        trajectory_pred = clip_data["motr_targets"]["trajectory_pred"]
        valid_ids = trajectory_pred["valid_ids"]
        traj_list = trajectory_pred["gt_traj_list"]
        mask_list = trajectory_pred["gt_mask_list"]
        vxvywy_list = trajectory_pred["gt_vxvywy_list"]

        modal_labels = []
        for cur_valid_ids, cur_gt_trajs, cur_gt_masks, cur_gt_vxvywys in zip(
            valid_ids, traj_list, mask_list, vxvywy_list
        ):
            cur_label_dict = {}
            for cur_id in cur_valid_ids:
                cur_label = DEFAULT_GT_MODAL_INDEX
                cur_traj = cur_gt_trajs[cur_id]
                cur_mask = cur_gt_masks[cur_id]
                cur_vxvywy = cur_gt_vxvywys[cur_id]
                if np.sum(cur_mask[:, 0]) >= 4:
                    # agents' valid number of frame less than 4 is discard.
                    cur_traj = cur_traj[cur_mask[:, 0] >= 0.5]
                    cur_vxvywy = cur_vxvywy[cur_mask[:, 0] >= 0.5]

                    cur_v = np.sqrt((cur_vxvywy[:, :2] ** 2).sum(-1))
                    if (
                        np.mean(cur_v) > 0.1
                    ):  # speed less than 0.1 stand for stationary
                        yaw = np.arctan2(cur_vxvywy[:, 1], cur_vxvywy[:, 0])
                        tmp_yaw = yaw[cur_v >= self.speed_static_thr]
                        yaw_list = np.unwrap(tmp_yaw)
                        if len(yaw_list) >= 2:  # valid yaw list length >=2
                            if (
                                np.max(np.abs(yaw_list[1:] - yaw_list[:-1]))
                                < np.pi / 2
                            ):  # adjacent yaw diff less than thr
                                cur_yaw = np.mod(
                                    np.mean(yaw_list - yaw_list[0]), np.pi * 2
                                )
                                if cur_yaw < self.split_thr[-1]:
                                    tmp = np.nonzero(cur_yaw >= self.split_thr)
                                    cur_label = tmp[0][-1]
                                else:
                                    cur_label = 0

                cur_label_dict[cur_id] = cur_label
            modal_labels.append(cur_label_dict)

        trajectory_pred["modal_labels"] = modal_labels
        clip_data["motr_targets"]["trajectory_pred"] = trajectory_pred
        return clip_data


class InterploteToInstance:
    """
    Generate GT-modal for each trajectory according to its end-point.

    This transform function requires data have passed `GeneTrajMask`.
    Args:
        pred_frame_num: number frames of trajectory prediction.
        save_memory_flag: if true, remove most of items in trajectory_pred.
        max_his_odo_len: maximum length of history odometry.
    """

    def __init__(
        self,
        pred_frame_num: int,
        max_his_odo_len: int,
        save_memory_flag: bool = False,
    ):
        self.pred_frame_num = pred_frame_num
        self.save_memory_flag = save_memory_flag
        self.max_his_odo_len = max_his_odo_len

    def __call__(self, clip_data: Dict):
        assert "motr_targets" in clip_data
        trajectory_pred = clip_data["motr_targets"].pop("trajectory_pred")
        valid_ids = trajectory_pred["valid_ids"]
        traj_list = trajectory_pred["gt_traj_list"]  # list[{key:[frames,2]}]
        mask_list = trajectory_pred["gt_mask_list"]  # list[{key:[frames,2]}]
        modal_labels = (
            trajectory_pred["modal_labels"]
            if "modal_labels" in trajectory_pred
            else None
        )

        bev_tracking = clip_data["motr_targets"]["bev_tracking"]

        trajectory_pred_list = [None] * len(bev_tracking)
        for idx in range(len(bev_tracking)):
            # calculate the index in current clip.
            clip_idx = idx + clip_data["sample_split_index"] * len(
                bev_tracking
            )
            gt_bev = bev_tracking[idx]
            obj_arr = gt_bev["obj_idxes"].clone().cpu().numpy().astype(np.int)
            len_obj = obj_arr.shape[0]

            cur_gt_traj = traj_list[clip_idx]
            cur_gt_mask = mask_list[clip_idx]
            cur_valid_ids = valid_ids[clip_idx]
            cur_modals = (
                modal_labels[clip_idx] if modal_labels is not None else {}
            )

            data_traj_init = torch.zeros(
                [len_obj, self.pred_frame_num, 2], dtype=torch.float
            )
            data_mask_init = torch.zeros(
                [len_obj, self.pred_frame_num, 2], dtype=torch.float
            )
            data_modal_init = DEFAULT_GT_MODAL_INDEX * torch.ones(
                [len_obj], dtype=torch.long
            )  # DEFAULT_GT_MODAL_INDEX means no GT-modal label

            # insert GT-trajs, GT-masks, GT-modals to bevtracking
            for layer_idx, track_id in enumerate(obj_arr):
                if track_id in cur_valid_ids:
                    data_traj_init[layer_idx] = torch.tensor(
                        cur_gt_traj[track_id], dtype=torch.float
                    )
                    data_mask_init[layer_idx] = torch.tensor(
                        cur_gt_mask[track_id], dtype=torch.float
                    )
                    if track_id in cur_modals:
                        data_modal_init[layer_idx] = cur_modals[track_id]

            bev_tracking[idx]["gt_traj_regs"] = data_traj_init
            bev_tracking[idx]["gt_traj_masks"] = data_mask_init
            bev_tracking[idx]["gt_traj_modals"] = data_modal_init

            # change trajectory_pred from {key: list[{}]} to list[key:{}]
            traj_info_tmp = {}
            if not self.save_memory_flag:
                traj_info_tmp.update(
                    {
                        "ego_raw": trajectory_pred["ego_raw"][clip_idx:],
                        "targets_raw": trajectory_pred["targets_raw"][
                            clip_idx
                        ],
                        "ids_list": trajectory_pred["ids_list"][clip_idx],
                        "state_flag_list": trajectory_pred["state_flag_list"][
                            clip_idx
                        ],
                        "valid_ids": trajectory_pred["valid_ids"][clip_idx],
                        "valid_classes": trajectory_pred["valid_classes"][
                            clip_idx
                        ],
                        "gt_traj_list": trajectory_pred["gt_traj_list"][
                            clip_idx
                        ],
                        "gt_mask_list": trajectory_pred["gt_mask_list"][
                            clip_idx
                        ],
                        "gt_vxvywy_list": trajectory_pred["gt_vxvywy_list"][
                            clip_idx
                        ],
                        "modal_labels": trajectory_pred["modal_labels"][
                            clip_idx
                        ],
                    }
                )
            trajectory_pred_list[idx] = traj_info_tmp

        clip_data["motr_targets"]["bev_tracking"] = bev_tracking
        clip_data["motr_targets"]["trajectory_pred"] = trajectory_pred_list
        return clip_data


@OBJECT_REGISTRY.register
class TrajPredMultiTransform:
    """Combine a series of relevant traj-tranfroms in this class.

    Some transforms reqiure pass through some other transform,and relationship
    make trajpred transform a series.The series of transforms are as follows:
    (1) `DropDuplicatedRecords`
    (2) `AddEgoInfosToGt`
    (3) `PhyToBEVCoord`
    (4) `FilterObsStacles`
    (5) `GeneTrajMask`
    (6) `MatchModalIndex`
    (7) `InterploteToInstance`

    Args:
        use_fut_info_filter: if False(not use future info for filtering):
            only considerate history frame in yaw_bouncing, stationary
            filter and frame number filter. (one can't obtain future info
            in current frame in practise.)
        vcs_range: range of vehicle coordinage system.(bottom, right, top,
            left).
        car_ped_cyc_type_id:  the classification id of the car, the pedestrain
            and the cyclist.(e.g. {"vehicle": 0, "cyclist": 1,
            "pedestrain": 2].)
        static_thr:  if displace of adjacent frames is smaller than threshold,
            then filter out this obstacle[m].Each agent class should give one
            value.(e.g. {"vehicle": 1.0, "cyclist": 0.5, "pedestrain": 0.2].).
            It shold be share the same keys in car_ped_cyc_type_id.
        num_sample_per_clip:  the number of frames of one clip.
        min_num_his_frame_thr:  the minimum historical frame number of agents.
        min_num_fut_frame_thr: the minimum future frame number of agents.
        max_context_frame_num:  the maximum historical frame number of agents,
            larger than this number is discarded.
        num_frames_for_pred: number of trajectory prediction frames, and also
            equal to max_future_frame_num in `FilterObsStacles`.
        pred_traj_modals_num:number of predicting trajecotry model, only when
            in [3,5,7], given agents GT-traj modal.
        max_his_odo_len: maximum length of history odometry.
        yaw_diff_bounce_thr: the yaw-diff thr between two adjacent frames in
            Rad after transforming to [0,2*pi). Default to 2*pi, whitch means
            no bound.
        static_obs_compensate_prob:with a 1-prob probability random add static
            agents to train set.Default to 1.0, means no compensate.
        filter_unstable_class: if true, filter ids according to whose class is
            stable or not.Only enable on cyclist and pedestrain.
        visibility: the level of visibility.Agents' visibility blow it will be
            discard.Default to 2(50% occulusion)
        ego_vcs_trans_flag: whether rotate obstacle's coordinates according to
            ego's yaw and position in the last context frame. Default to True.
        obs_lcf_position_trans_flag:  whether translate obstacle's coordinates
            according to its position in the last context frame.    Default to
            True.
        obs_lcf_yaw_rotate_flag:if true,rotate obstacle's coordinate according
            to its yaw in the last context frame.   Default to False. when not
            translating, there does not rotate, too.
        future_traj_only: If true, only return future trajectories, whose last
            dim is max_future_frame_num.   If false, return history and future
            trajectories together,    whose last dim is max_future_frame_num +
            max_context_frame_num.
        fill_inverse_traj_flag:    If True, fill the GT-trajectories of agents
            moving by high speed and opposite to ego motion direction.
        fill_inverse_traj_prob: Prob to fill the GT-trajectories of agents.
            Defaut to 0.4(experience value).
        only_keep_vehicle: Default to false. If true, obstacles being not
            vehicle will be filtered out.
        ego_as_obs_flag: Whether treat ego as a commom obstacle or not.
        default_ego_id: If ego_as_obs_flag, give ego a track-id.Default to -1,
            this is optional.
        default_ego_label: if ego_as_obs_flag is true, give ego a class-
            Label. Default to 0, 0 stand for vehicle class in most case.
        save_memory_flag: if true, remove most of items in `trajectory_pred`
            to save memory.
    """

    def __init__(
        self,
        use_fut_info_filter: bool,
        vcs_range: tuple,
        car_ped_cyc_type_id: Dict,
        static_thr: Dict,
        num_sample_per_clip: int,
        min_num_his_frame_thr: int,
        min_num_fut_frame_thr: int,
        max_context_frame_num: int,
        num_frames_for_pred: int,
        pred_traj_modals_num: int,
        max_his_odo_len: int,
        yaw_diff_bounce_thr: float = 6.284,
        static_obs_compensate_prob: float = 1.0,
        filter_unstable_class: bool = False,
        visibility: int = 2,
        ego_vcs_trans_flag: bool = True,
        obs_lcf_position_trans_flag: bool = True,
        obs_lcf_yaw_rotate_flag: bool = False,
        future_traj_only: bool = True,
        fill_inverse_traj_flag: bool = False,
        fill_inverse_traj_prob: float = 0.4,
        only_keep_vehicle: bool = False,
        ego_as_obs_flag: bool = False,
        default_ego_id: int = -1,
        default_ego_label: int = 0,
        save_memory_flag: bool = False,
    ):
        rmv_dup = DropDuplicatedRecords()
        trans_addego = AddEgoInfosToGt(
            vcs_range=vcs_range,
            ego_as_obs_flag=ego_as_obs_flag,
            default_ego_id=default_ego_id,
            default_ego_label=default_ego_label,
        )

        trans_phy2bev = PhyToBEVCoord(
            vcs_range=vcs_range,
        )

        trans_filter_obs = FilterObsStacles(
            use_fut_info_filter=use_fut_info_filter,
            car_ped_cyc_type_id=car_ped_cyc_type_id,
            only_keep_vehicle=only_keep_vehicle,
            num_sample_per_clip=num_sample_per_clip,
            min_num_his_frame_thr=min_num_his_frame_thr,
            min_num_fut_frame_thr=min_num_fut_frame_thr,
            max_context_frame_num=max_context_frame_num,
            max_future_frame_num=num_frames_for_pred,
            yaw_diff_bounce_thr=yaw_diff_bounce_thr,
            static_thr=static_thr,
            static_obs_compensate_prob=static_obs_compensate_prob,
            filter_unstable_class=filter_unstable_class,
            default_ego_id=default_ego_id,
            visibility=visibility,
        )

        trans_gene_gttrajs = GeneTrajMask(
            max_context_frame_num=max_context_frame_num,
            max_future_frame_num=num_frames_for_pred,
            ego_vcs_trans_flag=ego_vcs_trans_flag,
            obs_lcf_position_trans_flag=obs_lcf_position_trans_flag,
            obs_lcf_yaw_rotate_flag=obs_lcf_yaw_rotate_flag,
            future_traj_only=future_traj_only,
            fill_inverse_traj_flag=fill_inverse_traj_flag,
            fill_inverse_traj_prob=fill_inverse_traj_prob,
        )

        obtain_gt_modal = MatchModalIndex(
            num_modals=pred_traj_modals_num,
            speed_static_thr=np.min(list(static_thr.values())),
        )

        trans_interp_instance = InterploteToInstance(
            pred_frame_num=num_frames_for_pred,
            save_memory_flag=save_memory_flag,
            max_his_odo_len=max_his_odo_len,
        )

        self.transform_list = [
            rmv_dup,
            trans_addego,
            trans_phy2bev,
            trans_filter_obs,
            trans_gene_gttrajs,
            obtain_gt_modal,
            trans_interp_instance,
        ]

    def __call__(self, clip_data: Dict):
        assert "motr_targets" in clip_data
        targets_raw = clip_data["motr_targets"]["trajectory_pred"][
            "targets_raw"
        ]
        mask = targets_raw["classification"] >= VALID_CLASS_NUM
        clip_data["motr_targets"]["trajectory_pred"][
            "targets_raw"
        ] = targets_raw[mask]
        for transform in self.transform_list:
            clip_data = transform(clip_data)
        return clip_data


@OBJECT_REGISTRY.register
class ANCObtainHomographyTemporal:
    """
    calculate homography matrix using in temporal fusion.

    Args:
        bev_size: size of bird's eye view.(w, h)
        vcs_range: vcs range corresponding to bev-size.(bottom, right, top,
            left).
        return_relative: if true, return relative HomographyTemporal(t->t-1,
            t-1->t-2, t-2->t-3, ...). If false, return absolute
            HomographyTemporal( t->t-1, t->t-2, t->t-3, ...).
    """

    def __init__(
        self,
        bev_size: tuple,
        vcs_range: tuple,
        return_relative: bool = True,
    ):
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.return_relative = return_relative

        self.temporal_homo = ANCTemporalHomo(
            bev_size, vcs_range, return_relative=return_relative
        )

    def __call__(self, clip_data: Dict):
        assert "odo_info" in clip_data

        odo_info = clip_data["odo_info"]

        ego_xyyaws = [his_ego_xyyaw[-2] for his_ego_xyyaw in odo_info]
        ego_xyyaws.append(odo_info[-1][-1])

        # obtain pose frame by frame
        poses = []
        for cur_xyyaw in ego_xyyaws[::-1]:  # inverse order to t,t-1,t-2,...
            x, y, yaw = cur_xyyaw.numpy()
            # pose = [R T
            #         0 1]. Because only x, y, and yaw input in odo_info, we
            # here supplement z=0, roll=0, pitch=0.
            euler = [0.0, 0.0, yaw]  # angle order is (roll, pitch, yaw)
            rtt_mat = R.from_euler("xyz", euler, degrees=False).as_matrix()
            cur_pose = np.zeros(shape=(4, 4), dtype=np.float32)
            cur_pose[:3, :3] = rtt_mat
            cur_pose[0, 3] = x
            cur_pose[1, 3] = y
            cur_pose[3, 3] = 1
            poses.append(cur_pose)

        data = {"pose": poses}
        out_homo_mat = self.temporal_homo(data)
        out_homo_mat["homography_temporal"] = torch.tensor(
            out_homo_mat["homography_temporal"]
        ).unsqueeze(0)
        clip_data.update(out_homo_mat)
        return clip_data


@OBJECT_REGISTRY.register
class ObtainHomoOffsetTemporal:
    """
    get exist homo offset using in temporal fusion.

    Args:
        homooffset_temporal_url: url of homooffset.
    """

    def __init__(
        self,
        homooffset_temporal_url: str,
        bev_size: Tuple[int],
        grid_quant_scale: float,
    ):
        self.homooffset_temporal_url = homooffset_temporal_url
        self.bev_size = bev_size
        self.grid_quant_scale = grid_quant_scale

    def __call__(self, clip_data: Dict):
        timestamp = int(clip_data["timestamp"] * 1000)
        homooffset_temporal_file = self.homooffset_temporal_url.replace(
            "$timestamp", str(timestamp)
        )
        homooffset_temporal = np.fromfile(
            homooffset_temporal_file, dtype=np.int16
        )
        # homooffset_temporal = homooffset_temporal.reshape((1, self.bev_size[0], self.bev_size[1], 2))  # noqa
        homooffset_temporal = homooffset_temporal.reshape(
            (1, 2, self.bev_size[0], self.bev_size[1])
        )
        homooffset_temporal = homooffset_temporal.transpose((0, 2, 3, 1))
        homooffset_temporal = homooffset_temporal.astype(np.float64)
        homooffset_temporal *= self.grid_quant_scale
        clip_data.update(
            {"homo_offset_temporal": torch.from_numpy(homooffset_temporal)}
        )

        return clip_data
