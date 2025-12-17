# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import logging
import math
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from hat.core.traj_pred_typing import (
    FILTER_FLAG,
    LANE_MARK_TYPES,
    PedestrainSafeArea,
    SeqCenter,
    StaticSafeArea,
    VehicleSafeArea,
)
from hat.core.traj_pred_utils import Affine2D
from hat.core.traj_pred_utils import TdtCoordHelper as TCH
from hat.core.traj_pred_utils import (
    is_point_in_convex_polygon,
    normalize_yaw,
    normalize_yaw_array,
)
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "GetLcfTimeStamp",
    "GenFutureTrackids",
    "RemapObsCls",
    "GetSeqDataFrameMask",
    "SelectYawArray",
    "FilterObstacles",
    "FilterObstaclesByFuture",
    "BehaviorSampling",
    "GenStatesAndMask",
    "GenHighFreqTraj",
    "GetTrajPredObjectsInfo",
    "GetBehavPredObjectsInfo",
    "VectorNetTrajExtractor",
    "GetNavinetmapInfo",
    "GetObstacleSafeArea",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class GetLcfTimeStamp:
    """Get last context frame timestamp.

    The timestamp is used as the token to get the corresponding image for
    trajectory prediction. \

    To use, the user should construct a `GetLcfTimeStamp`. After instantiation,
    the callable function will return a new dict that contains the following
    changes: \
        1. the `lcf_timestamp` was added.
    """

    def __init__(self):
        pass

    def __call__(self, sample: dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'last_context_frame_id'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'frame_id', 'timestamp'.

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample with added 'lcf_timestamp'.
        """
        seq_df = sample["seq_df"]
        last_context_frame_id = sample["last_context_frame_id"]
        seq_lcf_mask = seq_df["frame_id"] == last_context_frame_id
        lcf_timestamp = seq_df.loc[seq_lcf_mask, "timestamp"].tolist()[0]
        sample["lcf_timestamp"] = lcf_timestamp

        return sample


@OBJECT_REGISTRY.register
class GenFutureTrackids:
    """Generate a list of all valid `track_id` for trajectory prediction.

    Each valid `track_id` should appear in the last context frame. \

    Note that, the selected track ids are not necessarily be used in
    trajectory prediction. \

    To use, the user should construct a `GenFutureTrackids` instance with the
    required initialization parameters. After instantiation, the callable
    function will return a new dict that contains the following changes: \
        1. the list of `track_ids` was added.
    """

    def __init__(
        self, shuffle: bool = False, max_obs_num: Optional[int] = None
    ):
        """Initialize method.

        Args:
            shuffle: whether to shuffle `track_ids`.
            max_obs_num: maximum number of objects in `trajectories` and
                `img_coords` of one sequence.
        """
        self.shuffle = shuffle
        self.max_obs_num = max_obs_num

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'last_context_frame_id'

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'track_id'.

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample with added 'track_ids' (list).
        """
        df = sample["seq_df"]
        last_context_frame_id = sample["last_context_frame_id"]
        vals = df.values
        cols = df.columns
        col_track_id = cols.get_loc("track_id")
        last_frame_mask = df.frame_id == last_context_frame_id
        track_ids = vals[last_frame_mask, col_track_id]

        if self.max_obs_num is not None:
            if self.shuffle:
                np.random.shuffle(track_ids)
            track_ids = track_ids[: self.max_obs_num]

        sample["track_ids"] = track_ids.tolist()
        sample["track_ids"].sort()

        return sample


@OBJECT_REGISTRY.register
class RemapObsCls:
    """Remap obstacle class from given type id to unified type id.

    The unified id will be used in subsequent transform. Since we
    need to remap obstacle class one by one, the input dictionary
    must have the key 'track_ids'. This means that this transform
    must be used after 'GenFutureTrackids'.

    To use, the user should construct a `RemapObsCls`. After instantiation,
    the callable function will return a new dict that contains the following
    changes: \
        1. the `agent_classes` was added.
    """

    def __init__(
        self,
        veh_type_id: int,
        ped_cyc_type_id: Optional[List] = None,
        detect_peds_by_shape: bool = False,
        ped_shape_thr: float = 1,
    ):
        """Initialize method.

        Args:
            veh_type_id: the classification id of the vehicle. It can
                also be a list if more than one categories are mapped
                to vehicle.
            ped_cyc_type_id: the classification id of the pedestrian
                and cyclist.
            detect_peds_by_shape: the mode to classify pedestrains.
                If true, means to classify by the length and width of
                obstacles. else, directly use the class in the df.
            ped_shape_thr: the maximum length if corresponding
                `detect_peds_by_shape` is True.
        """
        self.label_remapping = {}
        if type(veh_type_id) is int:
            self.label_remapping[veh_type_id] = 0
        else:
            for veh_id in veh_type_id:
                self.label_remapping[veh_id] = 0

        ped_type_id, cyc_type_id = -1, -2
        if (ped_cyc_type_id is not None) and len(ped_cyc_type_id):
            (ped_type_id, cyc_type_id) = ped_cyc_type_id
            self.label_remapping[ped_type_id] = 1
            self.label_remapping[cyc_type_id] = 2

        self.ped_type_id = ped_type_id
        self.cyc_type_id = cyc_type_id
        self.detect_peds_by_shape = detect_peds_by_shape
        self.ped_shape_thr = ped_shape_thr

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in 'sample':
            1. 'seq_df'.
            2. 'last_context_frame_id'.

        Args:
            sample (Dict): input priginal sample.

        Returns:
            sample (Dict): the sample includes new keys "agent_classes"
        """
        seq_df = sample["seq_df"].copy()
        last_context_frame_id = sample["last_context_frame_id"]

        seq_df_vals = seq_df.values
        seq_df_cols = seq_df.columns
        frame_id_col = seq_df_cols.get_loc("frame_id")
        track_id_col = seq_df_cols.get_loc("track_id")
        class_col = seq_df_cols.get_loc("classification")
        obs_length_col = seq_df_cols.get_loc("obs_length")
        obs_width_col = seq_df_cols.get_loc("obs_width")

        lcf_mask = seq_df_vals[:, frame_id_col] == last_context_frame_id

        track_ids = sample["track_ids"]
        agent_classes = []

        for track_id in track_ids:
            track_id_mask = seq_df_vals[:, track_id_col] == track_id
            cur_mask = lcf_mask & track_id_mask

            lcf_df_vals = seq_df_vals[cur_mask, :]
            agent_class = lcf_df_vals[:, class_col][0]

            if self.detect_peds_by_shape:
                agent_width = seq_df_vals[cur_mask, obs_width_col].astype(
                    "float"
                )
                agent_length = seq_df_vals[cur_mask, obs_length_col].astype(
                    "float"
                )
                track_is_ped = (agent_width <= self.ped_shape_thr) & (
                    agent_length <= self.ped_shape_thr
                )
                if track_is_ped:
                    agent_class = self.ped_type_id
                else:
                    # The agent is not small enough, but the perception result
                    # says that it is pedestrain. We give it a cyclist type.
                    if agent_class == self.ped_type_id:
                        agent_class = self.cyc_type_id

            # if current class doesn't belong to vehicle, pedestrian and
            # cyclist, we remap it as vehicle id.
            remap_class = 0
            if agent_class in self.label_remapping:
                remap_class = self.label_remapping[agent_class]
            agent_classes.append(remap_class)

        sample["agent_classes"] = agent_classes
        return sample


@OBJECT_REGISTRY.register
class GetSeqDataFrameMask:
    def __init__(
        self,
        freq_ratio: int = 1,
    ):
        """Initialize method.

        Args:
            freq_ratio: ratio of data frequency compared to 2Hz.
        """
        self.freq_ratio = freq_ratio
        self.need_downsample = True if freq_ratio > 1 else False

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'last_context_frame_id'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'frame_id'

        Args:
            sample (dict): input original sample.
            sample (Dict): the input sample dictionary with additional
                items: `lctx_mask`, `ctx_mask`, `fut_mask`, `ctx_frame_id`,
                `fut_frame_id`, `sampled_ctx_mask`(optional),
                `sampled_ctx_frame_id`(optional), `sampled_fut_mask`(optional),
                `sampled_fut_frame_id`(optional).

        Returns:
            sample (dict):
        """
        df = sample["seq_df"]
        last_context_frame_id = sample["last_context_frame_id"]
        vals = df.values
        cols = df.columns
        frame_id_col = cols.get_loc("frame_id")

        last_frame_mask = df.frame_id == last_context_frame_id
        ctx_frame_mask = df.frame_id <= last_context_frame_id
        future_frame_mask = df.frame_id > last_context_frame_id

        sample["lctx_mask"] = last_frame_mask
        sample["ctx_mask"] = ctx_frame_mask
        sample["fut_mask"] = future_frame_mask

        all_ctx_frame = np.unique(vals[ctx_frame_mask, frame_id_col])
        all_ctx_frame.sort()
        all_fut_frame = np.unique(vals[future_frame_mask, frame_id_col])
        all_fut_frame.sort()
        sample["ctx_frame_id"] = all_ctx_frame
        sample["fut_frame_id"] = all_fut_frame

        if self.need_downsample:
            # Sample context frames
            sampled_ctx_frame = all_ctx_frame[
                np.where(
                    (all_ctx_frame - last_context_frame_id) % self.freq_ratio
                    == 0
                )[0]
            ]
            sampled_ctx_frame_mask = df.frame_id == last_context_frame_id
            for frame in sampled_ctx_frame:
                sampled_ctx_frame_mask |= df.frame_id == frame
            sample["sampled_ctx_mask"] = sampled_ctx_frame_mask
            sample["sampled_ctx_frame_id"] = sampled_ctx_frame

            # Sample future frames
            sampled_fut_frame = all_fut_frame[
                np.where(
                    (all_fut_frame - last_context_frame_id) % self.freq_ratio
                    == 0
                )[0]
            ]
            for idx, frame in enumerate(sampled_fut_frame):
                if idx == 0:
                    sampled_fut_frame_mask = df.frame_id == frame
                else:
                    sampled_fut_frame_mask |= df.frame_id == frame
            sample["sampled_fut_mask"] = sampled_fut_frame_mask
            sample["sampled_fut_frame_id"] = sampled_fut_frame

        return sample


@OBJECT_REGISTRY.register
class SelectYawArray:
    """Select yaw array for the subsequent calculation.

    We use type flag to represent the method for yaw selection: \
    0 - just use historical calcuated yaw \
    1 - use historical calcuated yaw and the last perception yaw. \
    2 - just use perception yaw \
    3 - fuse, if the last perception yaw has large difference from \
        calculated result, use calculated. Otherwise, use perception. \
    4 - fuse, a very complex logic \
    5 - just use perception yaw, handle the yaw direction reverse \
        issue \
    6 - based on 5, solve the issue that the yaw sudden change in \
        the yaw array \
    7 - use perception velocity to calculate the yaw \

    This transform should be used after `GenFutureTrackids`,
    `RemapObsCls` and `GetSeqDataFrameMask`.
    """

    def __init__(
        self,
        yaw_select_type: List = (3, 3, 3),
        ego_track_id: int = -42,
        yaw_thr: float = 1.571,
        enable_none: bool = True,
    ):
        """Initialize method.

        Args:
            yaw_select_type: the yaw selection type respectively for
                vehicles, pedestrains and cyclists.
            ego_track_id: the ego track id.
            yaw_thr: the threshold to select given yaw or calcualted
                yaw. Defaults to np.pi / 2.
            enable_none: whether to return None when the function is not
                sure if the yaw is right. Default to True. If the user
                always need to use the obstacle yaw (not just the valid
                obs), please set it as False.
        """
        self.yaw_select_type = yaw_select_type
        self.ego_track_id = ego_track_id
        self.yaw_thr = yaw_thr
        self.enable_none = enable_none

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'track_ids'.
            3. 'agent_classes'.
            4. 'lctx_mask'
            5. 'ctx_mask' or 'sampled_ctx_mask'

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'track_id', 'x', 'y', 'obs_yaw', 'pos_x', 'pos_y', 'yaw'
                'classification', 'obs_length', 'obs_width', 'timestamp'.

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample with added 'track_yaw_dict'.
        """
        df = sample["seq_df"]
        track_ids = sample["track_ids"]
        agent_classes = sample["agent_classes"]

        vals = df.values
        cols = df.columns
        phy_x_col = cols.get_loc("x")
        phy_y_col = cols.get_loc("y")
        phy_yaw_col = cols.get_loc("obs_yaw")
        vcs_abs_vx_col = cols.get_loc("abs_vx_global")
        vcs_abs_vy_col = cols.get_loc("abs_vy_global")

        last_frame_mask = sample["lctx_mask"]
        if "sampled_ctx_mask" in sample:
            ctx_frame_mask = sample["sampled_ctx_mask"]
        else:
            ctx_frame_mask = sample["ctx_mask"]

        ego_mask = df["track_id"] == self.ego_track_id
        lcf_ego_mask = last_frame_mask & ego_mask
        lcf_ego_yaw = vals[lcf_ego_mask, phy_yaw_col].astype("float64")

        track_yaw_dict = {}
        df_yaw_array = vals[:, phy_yaw_col]
        for (track_id, agent_class) in zip(track_ids, agent_classes):
            agent_mask = df["track_id"] == track_id
            lcf_agent_mask = last_frame_mask & agent_mask
            ctx_agent_mask = ctx_frame_mask & agent_mask

            # -- position
            his_x = vals[ctx_agent_mask, phy_x_col].astype("float64")
            his_y = vals[ctx_agent_mask, phy_y_col].astype("float64")
            # -- yaw
            lcf_yaw = vals[lcf_agent_mask, phy_yaw_col].astype("float64")
            all_yaw = vals[ctx_agent_mask, phy_yaw_col].astype("float64")
            # -- velocity
            vcs_vel_x = vals[ctx_agent_mask, vcs_abs_vx_col].astype("float64")
            vcs_vel_y = vals[ctx_agent_mask, vcs_abs_vy_col].astype("float64")

            # 5.1. Calcuate the selected yaw array.
            select_yaw = self._select_yaw_array_for_tomorrow(
                his_x,
                his_y,
                lcf_yaw,
                all_yaw,
                vcs_vel_x,
                vcs_vel_y,
                lcf_ego_yaw,
                self.yaw_select_type[agent_class],
                track_id == self.ego_track_id,
                thr=self.yaw_thr,
                enable_none=self.enable_none,
            )
            if select_yaw is not None:
                select_yaw = np.unwrap(select_yaw)
            track_yaw_dict[track_id] = select_yaw
            if select_yaw is None:
                continue
            if len(select_yaw) == 1:
                df_yaw_array[ctx_agent_mask] = select_yaw[0]
            else:
                if len(select_yaw) < len(all_yaw):
                    tmp_step = (select_yaw[-1] - select_yaw[0]) / (
                        len(all_yaw) - 1
                    )
                    select_yaw = (
                        np.arange(len(all_yaw)) * tmp_step + select_yaw[0]
                    )
                df_yaw_array[ctx_agent_mask] = select_yaw

        sample["track_yaw_dict"] = track_yaw_dict
        sample["seq_df"]["obs_yaw"] = df_yaw_array
        sample["seq_df"]["obs_yaw"] = sample["seq_df"]["obs_yaw"].astype(
            "float64"
        )
        return sample

    @staticmethod
    def _select_yaw_array_for_tomorrow(
        his_x: np.array,
        his_y: np.array,
        lcf_yaw: np.array,
        all_yaw: np.array,
        vel_x: np.array,
        vel_y: np.array,
        lcf_ego_yaw: float,
        type: int,
        is_ego: bool = False,
        thr: float = 1.571,
        enable_none: bool = True,
    ):
        """Select yaw array for the subsequent processes.

        Args:
            his_x: the historical x coordinates.
            his_y: the historical y coordinates.
            lcf_yaw: the perception yaw in the last context frame.
            all_yaw: the perception yaw array.
            vel_x: the historical perception x velocity.
            vel_y: the historical perception y velocity.
            lcf_ego_yaw: the perception ego yaw in the last context frame.
            type: the yaw selection method.
            is_ego: whether the obstacle is ego vehicle.
            thr: threshold to select given yaw or calcualted yaw.
            enable_none: whether to return None when the function is
                not sure if the yaw is right. Default to True. If the user
                always need to use the obstacle yaw (not just the valid obs),
                please set it as False.

        Returns:
            yaw_array (np.array): the selected yaw array.
        """
        if is_ego:
            return all_yaw
        if len(his_y) < 2:
            if enable_none:
                return None
            else:
                return all_yaw
        cal_yaw_array = np.arctan2(np.diff(his_y), np.diff(his_x))
        if type == 0:
            yaw_array = cal_yaw_array
        elif type == 1:
            yaw_array = [cal_yaw_array.reshape([-1]), lcf_yaw.reshape([-1])]
            yaw_array = np.concatenate(yaw_array, axis=0)
        elif type == 2:
            yaw_array = all_yaw
        elif type == 3:
            yaw_diff = cal_yaw_array[-1] - lcf_yaw
            yaw_diff = yaw_diff % (2 * np.pi)
            if (yaw_diff < thr) or (yaw_diff > 2 * np.pi - thr):
                # This means the perception yaw is correct.
                yaw_array = all_yaw
            else:
                yaw_array = cal_yaw_array
        elif type == 4:
            yaw_reverse_thr = 5 * np.pi / 6
            yaw_diff = cal_yaw_array[-1] - lcf_yaw
            yaw_diff = yaw_diff % (2 * np.pi)
            given_yaw_bounce = False
            if len(all_yaw) > 1:
                given_yaw_diff = np.diff(all_yaw) % (2 * np.pi)
                if np.any(
                    np.bitwise_and(
                        given_yaw_diff < thr, given_yaw_diff > 2 * np.pi - thr
                    )
                ):
                    given_yaw_bounce = True
            if given_yaw_bounce:
                yaw_array = cal_yaw_array
            else:
                if (yaw_diff >= yaw_reverse_thr) and (
                    yaw_diff <= 2 * np.pi - yaw_reverse_thr
                ):
                    yaw_array = all_yaw + np.pi
                elif (yaw_diff < thr) or (yaw_diff > 2 * np.pi - thr):
                    # This means the perception yaw is correct.
                    yaw_array = all_yaw
                else:
                    yaw_array = cal_yaw_array
        elif type == 5:
            yaw_reverse_thr = 5 * np.pi / 6
            yaw_diff = cal_yaw_array[-1] - lcf_yaw
            yaw_diff = yaw_diff % (2 * np.pi)
            if (yaw_diff >= yaw_reverse_thr) and (
                yaw_diff <= 2 * np.pi - yaw_reverse_thr
            ):
                yaw_array = all_yaw + np.pi
            else:
                yaw_array = all_yaw
        elif type == 6:
            yaw_reverse_thr = 5 * np.pi / 6

            if len(all_yaw) >= 2:
                all_yaw_diff = np.diff(all_yaw) % (2 * np.pi)
                sudden_change_flag = np.any(
                    np.logical_and(
                        all_yaw_diff >= yaw_reverse_thr,
                        all_yaw_diff <= 2 * np.pi - yaw_reverse_thr,
                    )
                )
                if sudden_change_flag:
                    all_yaw = all_yaw % np.pi

            # Below are the same as the condition type == 5.
            yaw_diff = cal_yaw_array[-1] - all_yaw[-1]
            yaw_diff = yaw_diff % (2 * np.pi)
            if (yaw_diff >= yaw_reverse_thr) and (
                yaw_diff <= 2 * np.pi - yaw_reverse_thr
            ):
                yaw_array = all_yaw + np.pi
            else:
                yaw_array = all_yaw
        elif type == 7:
            vcs_yaw = np.arctan2(vel_y, vel_x)
            yaw_array = vcs_yaw + lcf_ego_yaw
        else:
            raise ValueError(f"Undefined yaw array selection type {type}")
        return yaw_array


@OBJECT_REGISTRY.register
class FilterObstacles:  # noqa: D205,D400
    """Filter out obstacles based on rules we set, is used for multi agent
    pipeline to select valid track id to predict. \

    For some filter conditions, different agent class have different params.
    We use remaped class as index to get current condition of the class.
    Thus this transform should be used after `GenFutureTrackids`,
    `RemapObsCls` and `GetSeqDataFrameMask`. \

    To use, the user should construct a `FilterObstacles`. After instantiation,
    the callable function will return a new dict that contains the following
    changes: \
        1. the `valid_track_ids` and `drift_ids` was added.
    """

    def __init__(
        self,
        is_training: bool,
        is_multiagent: bool,
        ego_track_id: int,
        ped_cyc_type_id: List,
        leaving_mode: str,
        valid_distance: list,
        num_frame_thr: int,
        bounce_thr: int,
        speed_drift_thr: List,
        veh_lateral_drift_thr: float,
        static_thr: List = (1, 0.2, 0.2),
        if_train_filter_bounce: List = (True, True),  # all space, surrounding
        if_val_filter_bounce: List = (False, False),  # all space, surrounding
        classify_by_shape: bool = False,
        ped_shape_thr: float = 1,
        use_ego_track_train: bool = True,
        resolution: float = 0.2,
    ):
        """Initialize method.

        Args:
            is_training (bool): whether the filtered track id list is for
                model training. The filter for training is more strict.
            is_multiagent (bool): whether the sequence is for multiagent
                trajectory prediction. If the parameter is True, this class
                will filter the obstacles based on the configurations.
            ego_track_id (int): the ego vehicle track id.
            ped_cyc_type_id (list): the classification id of the pedestrain
                and cyclist.
            leaving_mode (str): choose which agents to leave, should
                be one of ['ego', 'vehicles', 'all'].
            valid_distance (list): the valid distance threshold of ego car and
                obstacles [m], the form is [veh[x_max/min, y_max/min], ped[],
                cycli[]], represents the range in each direction.
            num_frame_thr (int): the minimum (historical and future) frame
                number of valid tracks.
            if_val_filter_bounce (bool): whether use yaw bounce detection in
                validation.
            bounce_thr (float): the maximum yaw diff between two adjacent
                frames. [Rad]
            speed_drift_thr (float): the threshold to check whether the
                obstacle is drifting. If the obstacle moves too fast,
                maybe it is a drifting perception. [m]
            veh_lateral_drift_thr (float): the threshold to check whether the
                vehicle is lateral drifting.
            static_thr (List, optional): if the distance between two frames is
                smaller than the threshold, then filter out this obstacle [m].
                Each agent class should have one value. Default [1, 0.2, 0.2].
            if_train_filter_bounce (List, optional): whether to filter yaw
                bouncing obstacles during training. [all space, surrounding].
            if_val_filter_bounce (List, optional): whether to filter yaw
                bouncing obstacles during validation. [all space, surrounding].
            classify_by_shape (bool, optional): the mode to classify
                pedestrains. If true, means to classify by the
                length and width of obstacles. else, directly use the
                classification in the df.
            ped_shape_thr (float, optional): the maximum length and width of
                pedestrains.
            use_ego_track_train (bool): if certainly all ego tracks in
                training. Defaults to False.
            resolution (float): query resolution of the map.
        """
        self.is_training = is_training
        self.is_multiagent = is_multiagent
        self.ego_track_id = ego_track_id
        self.ped_cyc_type_id = ped_cyc_type_id
        self.leaving_mode = leaving_mode
        assert leaving_mode in [
            "ego",
            "vehicles",
            "all",
        ], f"Undefined type of leaving mode {leaving_mode}."
        self.valid_distance = valid_distance
        self.static_thr = static_thr
        self.num_frame_thr = num_frame_thr
        self.bounce_thr = bounce_thr
        self.speed_drift_thr = speed_drift_thr
        self.veh_lateral_drift_thr = veh_lateral_drift_thr
        self.classify_by_shape = classify_by_shape
        self.ped_shape_thr = ped_shape_thr
        self.use_ego_track_train = use_ego_track_train
        self.internel_count = [0, 0]
        self.resolution = resolution

        # Params about yaw bouncing filter.
        bounce_flag = (
            if_train_filter_bounce if is_training else if_val_filter_bounce
        )
        if bounce_thr is None:
            bounce_flag = [False, False]
        self.if_filter_bounce, self.if_surrounding_filter_bounce = bounce_flag

        self.if_filter_distance = (
            True if (valid_distance is not None) else False
        )
        self.if_filter_num_ctx = True if (num_frame_thr is not None) else False
        self.if_filter_stationary = True if (static_thr is not None) else False

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'track_ids'.
            3. 'agent_classes'.
            4. 'track_yaw_dict'.
            5. 'lctx_mask'.
            6. 'ctx_mask' or 'sampled_ctx_mask'.
            7. 'fut_mask' or 'sampled_fut_mask'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'track_id', 'x', 'y', 'obs_yaw', 'pos_x', 'pos_y', 'yaw'
                'classification', 'obs_length', 'obs_width', 'timestamp'.

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample with added 'valid_track_ids'
                'track_stat_dict', and 'drift_ids'.
        """
        # Generate valid track ids.
        df = sample["seq_df"]
        vals = df.values
        cols = df.columns
        track_ids = sample["track_ids"]
        agent_classes = sample["agent_classes"]
        track_yaw_dict = sample["track_yaw_dict"]
        last_frame_mask = sample["lctx_mask"]
        if "sampled_ctx_mask" in sample:
            ctx_frame_mask = sample["sampled_ctx_mask"]
        else:
            ctx_frame_mask = sample["ctx_mask"]

        if not self.is_multiagent:
            sample["valid_track_ids"] = [self.ego_track_id]
            sample["drift_ids"] = []
            return sample

        phy_x_col = cols.get_loc("x")
        phy_y_col = cols.get_loc("y")
        phy_yaw_col = cols.get_loc("obs_yaw")
        class_col = cols.get_loc("classification")

        # Filter the valid track ids.
        valid_track_ids = []
        drift_ids = []
        track_stat_dict = {}

        assert len(track_ids) == len(
            agent_classes
        ), f"{len(track_ids)} is different from {len(agent_classes)}"

        for (track_id, agent_class) in zip(track_ids, agent_classes):
            agent_mask = df["track_id"] == track_id
            lcf_agent_mask = last_frame_mask & agent_mask
            ctx_agent_mask = ctx_frame_mask & agent_mask
            self.internel_count[0] += 1

            # 0.0. Extract the trajectory information from the dataframe.
            # -- position
            his_x = vals[ctx_agent_mask, phy_x_col].astype("float64")
            his_y = vals[ctx_agent_mask, phy_y_col].astype("float64")
            lcf_x = vals[lcf_agent_mask, phy_x_col].astype("float64")
            lcf_y = vals[lcf_agent_mask, phy_y_col].astype("float64")
            # -- yaw
            lcf_yaw = vals[lcf_agent_mask, phy_yaw_col].astype("float64")
            # -- shape and class
            his_class = vals[ctx_agent_mask, class_col].astype("int")

            # filter nan sample
            if np.isnan(lcf_x) or np.isnan(lcf_y) or np.isnan(lcf_yaw):
                continue

            # Choose leaving agents.
            if self.leaving_mode == "ego":
                if track_id != self.ego_track_id:
                    track_stat_dict[track_id] = FILTER_FLAG["ego_only"]
                    continue
            elif self.leaving_mode == "vehicles":
                filter_flag = False
                for type_id in self.ped_cyc_type_id:
                    if type_id in his_class:
                        filter_flag = True
                if filter_flag:
                    track_stat_dict[track_id] = FILTER_FLAG["vehicle_only"]
                    continue

            # 0. If needed, skip the filter of the ego vehicle.
            if track_id == self.ego_track_id and (
                not self.is_training or self.use_ego_track_train
            ):
                valid_track_ids.append(track_id)
                track_stat_dict[track_id] = FILTER_FLAG["normal"]
                self.internel_count[1] += 1
                continue

            # 1. Filter by frame numbers in context and future frames.
            # -- The obstacle that appears in very few historical frames cannot
            #    provide enough information to describe the trajectory.
            # -- The obstacle that appears in very few future frames cannot
            #    provide a long-enough ground-truth for training.
            if self.if_filter_num_ctx or self.if_filter_bounce:
                if his_x.shape[0] < self.num_frame_thr:
                    track_stat_dict[track_id] = FILTER_FLAG[
                        "his_frame_not_enough"
                    ]
                    continue

            # 2. Filter the stationary obstacles by displacement.
            # -- Filter out the stationary obstacles. If the mode is training,
            #    we use the entire trajectory, otherwise, we just use the
            #    historical trajectory.
            if self.if_filter_stationary:
                sta_x = his_x
                sta_y = his_y
                sta_thr = self.static_thr[agent_class]
                if sta_x.shape[0] <= 1:
                    continue
                displace = np.sqrt(
                    (sta_x[1:] - sta_x[:-1]) ** 2
                    + (sta_y[1:] - sta_y[:-1]) ** 2
                )
                mean_displace = np.mean(displace)
                if (mean_displace < sta_thr) and (
                    track_id != self.ego_track_id
                ):
                    track_stat_dict[track_id] = FILTER_FLAG["obs_stationary"]
                    continue

            # Body check of the agent:
            # -- whether current agent is pedestrian. Note in RemapObsCls,
            # we already remap pedestrian type id is 1.
            track_is_veh = agent_class == 0
            track_is_ped = agent_class == 1

            # TODO (shengzhe.dai): the following parameters are newly added
            # to the real vehicle software. I need more time to sync those
            # logics
            # -- whether the agent is in blind spot of perception sensor.
            # track_in_blind_spot = False
            # -- whether the agent is in a special roi (a small area
            # surrounding the ego vehicle) and be cruial for PNC.
            track_in_roi = False

            # 3. Filter the yaw bounce trajectories.
            # -- Here, we filter the trajectories that have too large yaw
            #    difference. According to the perception, the used `yaw`
            #    can be derived directly from perception or by manually
            #    calculation.
            select_yaw = track_yaw_dict[track_id]
            if select_yaw is None:
                track_stat_dict[track_id] = FILTER_FLAG["his_frame_not_enough"]
                continue
            agent_filter_bounce = (track_id != self.ego_track_id) and (
                (track_in_roi and self.if_surrounding_filter_bounce)
                or (not track_in_roi and self.if_filter_bounce)
            )
            if (
                agent_filter_bounce
                and len(select_yaw) > 1
                and (not track_is_ped)
            ):
                yaw_array_diff = np.diff(select_yaw)
                yaw_array_diff = np.abs(normalize_yaw_array(yaw_array_diff))
                max_yaw_diff = np.max(yaw_array_diff)
                if max_yaw_diff > self.bounce_thr:
                    track_stat_dict[track_id] = FILTER_FLAG["obs_yaw_bouncing"]
                    continue

            # 4. Filter or project the drifting obstacles.
            # -- The obstacles at the edge of the perception area may have
            #    position bounce. It seems that the obstacle is moving at
            #    a very fast velocity (i.e., drifting), which is unreasonable.
            #    If the obstacle is drifting, skip.
            if self.speed_drift_thr and len(self.speed_drift_thr) == 3:
                drift_thr = self.speed_drift_thr[int(agent_class)]
                is_drifting = self._detect_drift_velo(
                    drift_thr,
                    his_x,
                    his_y,
                )
                if is_drifting:
                    track_stat_dict[track_id] = FILTER_FLAG["speed_drifting"]
                    drift_ids.append(track_id)
                    continue

            # 5. Filter the lateral drifting vehicle tracks.
            if track_is_veh:
                if len(his_x) <= 1:
                    track_stat_dict[track_id] = FILTER_FLAG[
                        "his_frame_not_enough"
                    ]
                    continue
                is_veh_drifting = self._detect_lateral_drift_vehs(
                    his_x, his_y, select_yaw, self.veh_lateral_drift_thr
                )
                if is_veh_drifting:
                    track_stat_dict[track_id] = FILTER_FLAG[
                        "vehicle_lateral_drifting"
                    ]
                    drift_ids.append(track_id)
                    continue

            # 6. Filter by distance.
            # -- Obstacles that are too far away usually have large perception
            #    errors. This distance threshold should be set according to
            #    the perception equipment.
            if self.if_filter_distance:
                obs_key = ["x", "y"]
                out_df = sample["seq_df"].copy()
                local_obs_loc = TCH.global_phy_to_local_phy(
                    out_df[obs_key].values, sample["seq_center"]
                )
                local_obs_loc = local_obs_loc[lcf_agent_mask, :]
                local_obs_loc = local_obs_loc.astype("float64")
                lcf_obs_x = local_obs_loc[0, 0]
                lcf_obs_y = local_obs_loc[0, 1]
                dis_range = self.valid_distance[agent_class]
                if (
                    (lcf_obs_x > dis_range[0])
                    or (lcf_obs_x < dis_range[1])
                    or (lcf_obs_y > dis_range[2])
                    or (lcf_obs_y < dis_range[3])
                ):
                    track_stat_dict[track_id] = FILTER_FLAG["obs_too_far"]
                    continue

            track_stat_dict[track_id] = FILTER_FLAG["normal"]
            self.internel_count[1] += 1
            valid_track_ids.append(track_id)

        if len(valid_track_ids):
            valid_track_ids.sort()
        else:
            valid_track_ids.append(self.ego_track_id)
            self.internel_count[1] += 1

        if len(drift_ids):
            drift_ids.sort()
        sample["valid_track_ids"] = valid_track_ids
        sample["drift_ids"] = drift_ids
        sample["track_stat_dict"] = track_stat_dict
        return sample

    @staticmethod
    def _detect_unstable_obs_class(
        classify_by_shape, ped_shape_thr, width, length, obs_class
    ):
        """Detect whether the obstacle has unstable classification results.

        Once an obstacle is detected as pedestrain or cyclist, it must be
        always pedestrain or cyclist in the given frames. Otherwise, the
        perception or tracking may be wrong.

        Args:
            classify_by_shape (bool): if classify by shape or given class.
            ped_shape_thr (float): the threshold to classify a pedestrain.
            his_width (np.array): the obstacle width.
            his_length (np.array): the obstacle length.
            obs_class (np.array): the obstacle classification.

        Returns:
            unstable_class (bool): whether the obstacle has unstable
                classifications.
            track_is_ped (bool): whether this obstacle is pedestrain or
                cyclist.
        """
        unstable_class = False
        if classify_by_shape:
            is_ped = (width <= ped_shape_thr) & (length <= ped_shape_thr)
            if True in is_ped:
                if np.sum(is_ped) != len(width):
                    unstable_class = True
        else:
            if np.any((obs_class[:-1] - obs_class[1:]) != 0):
                unstable_class = True
        return unstable_class

    @staticmethod
    def _detect_lateral_drift_vehs(
        his_x,
        his_y,
        obs_yaw,
        veh_lateral_drift_threshold=1.0,
    ):
        """Detect whether the vehicle is drifting in its lateral direction.

        Args:
            his_x (np.array): the historical x coordinates.
            his_y (np.array): the historical y coordinates.
            obs_yaw (np.array): the obstacle yaw array.
            veh_drift_threshold (float): the threshold of drifting.

        Returns:
            is_veh_drifting (bool): whether the obstacle is drifting along its
                lateral direction.
        """
        assert len(his_x) == len(his_y), (
            "The length of the given x and y coordinate lists should"
            f"be equal, but {len(his_x)} vs {len(his_y)}."
        )
        # The `velo` variables here are not strict velocities, because they
        # are not divided by delta t.
        his_velo = np.sqrt(np.diff(his_x) ** 2 + np.diff(his_y) ** 2)
        his_cal_yaw = np.arctan2(np.diff(his_y), np.diff(his_x))
        his_given_yaw = obs_yaw[(len(obs_yaw) - len(his_velo)) :]
        delta_yaw = his_cal_yaw - his_given_yaw
        his_lateral_velo = np.sin(delta_yaw) * his_velo
        his_lateral_velo = np.abs(his_lateral_velo)
        if np.max(his_lateral_velo) > veh_lateral_drift_threshold:
            is_veh_drifting = True
        else:
            is_veh_drifting = False
        return is_veh_drifting

    @staticmethod
    def _detect_drift_velo(
        drift_thr,
        his_x,
        his_y,
    ):
        """Detect whether the pedestrain is drifting.

        The pedestrains at the edge of the perception area may have position
        bounce. It seems that the pedestrain is moving at a very fast velocity
        (i.e., drifting), which is unreasonable. If the obstacle is drifting
        along the ego velocity direction, it will not influence the ego
        vehicle, thus we can skip this pedestrain during trajectory prediction.

        Args:
            drift_thr (np.array): the threshold of drifting.
            his_x (np.array): the historical x coordinates.
            his_y (np.array): the historical y coordinates.

        Returns:
            is_drifting (bool): whether the obstacle is drifting pedestrain.
        """
        assert len(his_x) == len(his_y), (
            "The length of the given x and y coordinate lists should"
            f"be equal, but {len(his_x)} vs {len(his_y)}."
        )
        is_drifting = False
        if len(his_x) <= 1:
            is_drifting = True
        # The `velo` variables here are not strict velocities, because they
        # are not divided by delta t.
        his_velo = np.sqrt(np.diff(his_x) ** 2 + np.diff(his_y) ** 2)
        max_velo = np.max(his_velo)
        if max_velo >= drift_thr:
            is_drifting = True
        return is_drifting


@OBJECT_REGISTRY.register
class FilterObstaclesByFuture:  # noqa: D205,D400
    """Filter out obstacles based on future states, is used for multi agent
    training pipeline to select valid track id to predict and train valid
    head. \

    For some filter conditions, different agent class have different params.
    We use remaped class as index to get current condition of the class.
    Thus this transform should be used after "GenFutureTrackids" and
    "RemapObsCls" and "FilterObstacles". \

    To use, the user should construct a `FilterObstaclesByFuture`. After
    instantiation, the callable function will return a new dict that
    contains the following changes: \
        1. the `valid_track_ids` and `drift_ids` was added.
    """

    def __init__(
        self,
        is_multiagent,
        ego_track_id,
        ped_cyc_type_id,
        leaving_mode: str,
        num_frame_thr: list,
        bounce_thr: int,
        speed_drift_thr: List,
        veh_lateral_drift_thr: float,
        static_thr: List = (1, 0.2, 0.2),
        if_filter_bounce: List = (True, True),  # all space, surrounding
        classify_by_shape: bool = False,
        ped_shape_thr: float = 1,
        use_ego_track_train: bool = True,
        enable_incomplete_gts: bool = False,
        key_name: str = "valid_future_track_ids",
        if_filter_traj_reverse: bool = True,
    ):
        """Initialize method.

        Args:
            is_multiagent (bool): whether the sequence is for multiagent
                trajectory prediction. If the parameter is True, this class
                will filter the obstacles based on the configurations.
            ego_track_id (int): the ego vehicle track id.
            ped_cyc_type_id (list): the classification id of the pedestrain
                and cyclist.
            leaving_mode (str): choose which agents to leave, should
                be one of ['ego', 'vehicles', 'all'].
            num_frame_thr (list): the minimum frame number of valid
                tracks.
            if_val_filter_bounce (bool): whether use yaw bounce detection in
                validation.
            bounce_thr (float): the maximum yaw diff between two adjacent
                frames. [Rad]
            speed_drift_thr (List): the threshold to check whether the
                obstacle is drifting. If the obstacle moves too fast,
                maybe it is an drifting perception. [m]
            veh_lateral_drift_thr (float): the threshold to check whether the
                vehicle is drifting.
            static_thr (List, optional): if the distance between two frames is
                smaller than the threshold, then filter out this obstacle [m].
                Each agent class should have one value. Default [1, 0.2, 0.2].
            if_train_filter_bounce (List, optional): whether to filter yaw
                bouncing obstacles during training. [all space, surrounding].
            if_val_filter_bounce (List, optional): whether to filter yaw
                bouncing obstacles during validation. [all space, surrounding].
            classify_by_shape (bool, optional): the mode to classify
                pedestrains. If true, means to classify by the
                length and width of obstacles. else, directly use the
                classification in the df.
            ped_shape_thr (float, optional): the maximum length and width of
                pedestrains.
            use_ego_track_train (bool): if certainly all ego tracks in
                training. Defaults to False.
            enable_incomplete_gts (bool, optional): whether to predict cases
                that do not have complete ground-truth future trajectories
                (i.e., the number of frames is smaller than `traj_len`).
                If the `data_source` is 'nuscenes' and the current stage is
                    validation, this parameter must be False. Because the
                    metric calcuator in NuScenes api does not support the
                    comparison between a complete prediction result and an
                    incomplete ground-truth trajectory.
                Otherwise, it is okay to set this parameter as True.
            if_filter_traj_reverse (bool, optional): whether filter the reverse
                futrue trajectory.
        """
        self.is_multiagent = is_multiagent
        self.ego_track_id = ego_track_id
        self.ped_cyc_type_id = ped_cyc_type_id
        self.leaving_mode = leaving_mode
        assert leaving_mode in [
            "ego",
            "vehicles",
            "all",
        ], f"Undefined type of leaving mode {leaving_mode}."
        self.static_thr = static_thr
        self.num_frame_thr = num_frame_thr
        self.bounce_thr = bounce_thr
        self.speed_drift_thr = speed_drift_thr
        self.veh_lateral_drift_thr = veh_lateral_drift_thr
        self.classify_by_shape = classify_by_shape
        self.ped_shape_thr = ped_shape_thr
        self.use_ego_track_train = use_ego_track_train
        self.internel_count = [0, 0]
        self.enable_incomplete_gts = enable_incomplete_gts
        self.key_name = key_name
        self.if_filter_traj_reverse = if_filter_traj_reverse

        # Params about yaw bouncing filter.
        bounce_flag = if_filter_bounce
        if bounce_thr is None:
            bounce_flag = [False, False]
        self.if_filter_bounce, self.if_surrounding_filter_bounce = bounce_flag

        self.if_filter_num_ctx = True if (num_frame_thr is not None) else False
        self.if_filter_stationary = True if (static_thr is not None) else False

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'last_context_frame_id'.
            3. 'track_ids'.
            4. 'agent_classes'.
            5. 'track_yaw_dict'.
            6. 'valid_track_ids'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'track_id', 'x', 'y', 'obs_yaw', 'pos_x', 'pos_y', 'yaw'
                'classification', 'obs_length', 'obs_width', 'timestamp'.

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample with added 'valid_future_track_ids'.
        """
        # Generate valid track ids.
        df = sample["seq_df"]
        vals = df.values
        cols = df.columns
        track_ids = sample["track_ids"]
        agent_classes = sample["agent_classes"]
        last_frame_mask = sample["lctx_mask"]
        if "sampled_fut_mask" in sample:
            future_frame_mask = sample["sampled_fut_mask"]
        else:
            future_frame_mask = sample["fut_mask"]

        # "valid_future_track_ids" is subset of "valid_track_ids"
        if "valid_track_ids" in sample:
            valid_track_ids = sample["valid_track_ids"]
        else:
            valid_track_ids = sample["track_ids"]

        phy_x_col = cols.get_loc("x")
        phy_y_col = cols.get_loc("y")
        phy_yaw_col = cols.get_loc("obs_yaw")

        # Filter the valid future track ids.
        valid_future_track_ids = []
        track_stat_dict = {}

        assert len(track_ids) == len(
            agent_classes
        ), f"{len(track_ids)} is different from {len(agent_classes)}"

        for (track_id, agent_class) in zip(track_ids, agent_classes):

            # "valid_future_track_ids" is subset of "valid_track_ids"
            if track_id not in valid_track_ids:
                continue

            agent_mask = df["track_id"] == track_id
            lcf_agent_mask = last_frame_mask & agent_mask
            future_agent_mask = future_frame_mask & agent_mask

            # 0.0. Extract the trajectory information from the dataframe.
            # -- position
            lcf_x = vals[lcf_agent_mask, phy_x_col].astype("float64")
            lcf_y = vals[lcf_agent_mask, phy_y_col].astype("float64")
            future_x = vals[future_agent_mask, phy_x_col].astype("float64")
            future_y = vals[future_agent_mask, phy_y_col].astype("float64")
            # -- yaw
            lcf_yaw = vals[lcf_agent_mask, phy_yaw_col].astype("float64")
            future_yaw = vals[future_agent_mask, phy_yaw_col].astype("float64")

            # filter nan sample
            if np.isnan(lcf_x) or np.isnan(lcf_y) or np.isnan(lcf_yaw):
                continue

            # 0. If needed, skip the filter of the ego vehicle.
            if track_id == self.ego_track_id and self.use_ego_track_train:
                valid_future_track_ids.append(track_id)
                track_stat_dict[track_id] = FILTER_FLAG["normal"]
                continue

            # 1. Filter by frame numbers in future frames.
            # -- The obstacle that appears in very few future frames cannot
            #    provide a long-enough ground-truth for training.
            if self.if_filter_num_ctx:
                if future_x.shape[0] < self.num_frame_thr[int(agent_class)]:
                    track_stat_dict[track_id] = FILTER_FLAG[
                        "fut_frame_not_enough"
                    ]
                    continue

            # 2. Filter or project the drifting obstacles.
            # -- The obstacles at the edge of the perception area may have
            #    position bounce. It seems that the obstacle is moving at
            #    a very fast velocity (i.e., drifting), which is unreasonable.
            #    If the obstacle is drifting, skip.
            if self.speed_drift_thr:
                drift_thr = self.speed_drift_thr[int(agent_class)]
                is_drifting = FilterObstacles._detect_drift_velo(
                    drift_thr,
                    future_x,
                    future_y,
                )
                if is_drifting:
                    track_stat_dict[track_id] = FILTER_FLAG["speed_drifting"]
                    continue

            # 3. Filter the lateral drifting vehicle tracks.
            if agent_class == 0:  # the agent_class of vehicle is 0
                if len(future_x) <= 1:
                    track_stat_dict[track_id] = FILTER_FLAG[
                        "fut_frame_not_enough"
                    ]
                    continue
                is_veh_drifting = FilterObstacles._detect_lateral_drift_vehs(
                    future_x, future_y, future_yaw, self.veh_lateral_drift_thr
                )
                if is_veh_drifting:
                    track_stat_dict[track_id] = FILTER_FLAG[
                        "vehicle_lateral_drifting"
                    ]
                    continue

            # 4. Filter the trajs not have correct direction
            if self.if_filter_traj_reverse:
                # future trajs have correct direction
                obs_future_x = sample["states"][
                    track_ids.index(track_id), :, 0
                ]
                obs_future_y = sample["states"][
                    track_ids.index(track_id), :, 1
                ]
                x_in_reverse_area = obs_future_x < 0
                y_in_reverse_area = abs(obs_future_y) < 1
                xy_in_reverse_area = x_in_reverse_area & y_in_reverse_area
                reverse_node_num = np.sum(xy_in_reverse_area)
                if reverse_node_num > len(obs_future_x) * 0.3:
                    track_stat_dict[track_id] = FILTER_FLAG[
                        "vehicle_fut_traj_reverse"
                    ]
                    continue

            track_stat_dict[track_id] = FILTER_FLAG["normal"]
            valid_future_track_ids.append(track_id)

        if len(valid_future_track_ids) or len(valid_future_track_ids) == len(
            valid_track_ids
        ):
            valid_future_track_ids.sort()
        else:
            valid_future_track_ids.append(self.ego_track_id)
            track_stat_dict[self.ego_track_id] = FILTER_FLAG["normal"]

        sample[self.key_name] = valid_future_track_ids
        sample[self.key_name + "_stat_dict"] = track_stat_dict
        return sample


@OBJECT_REGISTRY.register
class BehaviorSampling:
    """Filter invalid behavior and do random down-sampling.

    This transform method requires the sample to have the keys \
    `seq_df`, `last_context_frame_id`, `valid_track_ids`.\
    Therefore, it must be called after `FilterObstacles`.
    """

    def __init__(
        self,
        is_training: bool,
        ego_track_id: int,
        down_sampling: bool = False,
        down_ratio: float = 0.05,
        use_ego: bool = True,
        valid_track_ids_key: str = None,
        dist_thresh_ratio: int = 1 / 4,
    ):
        """Initialize method.

        Args:
            is_training: whether the filtered track id list is for
                model training. The filter for training is more strict.
            ego_track_id: the ego vehicle track id.
            down_sampling: whether to down sample the major class objects.
            down_ratio: the ratio of down sampling.
            use_ego: whether to use track id of the ego car in training.
            valid_track_ids_key: the key name of valid track ids.
            dis_thresh_ratio: the ratio of distance thresh from obs to
                laneline.
        """
        self.is_training = is_training
        self.ego_track_id = ego_track_id
        self.use_ego = use_ego
        self.down_sampling = down_sampling
        self.down_ratio = down_ratio
        self.valid_track_ids_key = valid_track_ids_key
        self.dist_thresh_ratio = dist_thresh_ratio
        self.default_lane_width = 3.75

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'last_context_frame_id'.
            3. 'valid_track_ids'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'track_id', 'frame_id', 'lat_behav_label'.

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample with added 'valid_behav_track_ids'
                and updated "seq_df"
        """
        df = sample["seq_df"]
        vals = df.values
        cols = df.columns
        ts_col = cols.get_loc("timestamp")
        global_x_col = cols.get_loc("x")
        global_y_col = cols.get_loc("y")
        global_yaw_col = cols.get_loc("obs_yaw")
        obs_width_col = cols.get_loc("obs_width")
        lat_label_col = cols.get_loc("lat_behav_label")
        overline_col = cols.get_loc("is_over_laneline")
        lat_dist_cnt_to_overline_col = cols.get_loc(
            "lat_dist_obs_cnt_to_overline_for_dist_restrict"
        )

        last_frame_mask = sample["lctx_mask"]
        ctx_mask = sample["ctx_mask"]
        lcf_id = sample["last_context_frame_id"]
        valid_track_ids = sample[self.valid_track_ids_key]

        down_sampled_track_ids = []
        for track_id in valid_track_ids:
            # We need ego car to avoid empty loss input.
            if track_id == self.ego_track_id:
                if self.use_ego:
                    down_sampled_track_ids.append(track_id)
                continue
            agent_mask = df["track_id"] == track_id
            lcf_agent_mask = last_frame_mask & agent_mask
            ctx_agent_mask = ctx_mask & agent_mask
            obs_lat_label = vals[lcf_agent_mask, lat_label_col].item()
            ctx_df_vals = vals[ctx_agent_mask, :]
            lctx_df_vals = vals[lcf_agent_mask, :]
            # Extract historical center points
            global_cnt_cols = [global_x_col, global_y_col, global_yaw_col]
            his_global_coords = ctx_df_vals[:, global_cnt_cols].astype(
                "float64"
            )
            # Convertion from global coordinates to centric coordinates
            obs_seq_center = SeqCenter(
                his_global_coords[0, 0],
                his_global_coords[0, 1],
                his_global_coords[0, 2],
            )
            his_local_coords = TCH.global_phy_to_local_phy(
                his_global_coords, obs_seq_center
            )
            his_ts = ctx_df_vals[:, ts_col] / 1000  # convert ms to s.
            his_time_diff = np.diff(his_ts)
            # Restrict the lat_label based on the lateral speed and the
            # distance from obs to laneline.
            obs_width = lctx_df_vals[0, obs_width_col]
            lat_dist_obs_cnt_to_overline = lctx_df_vals[
                :, lat_dist_cnt_to_overline_col
            ][0]
            is_over_laneline = lctx_df_vals[:, overline_col][0]
            dist_threshold = self.dist_thresh_ratio * (
                self.default_lane_width - obs_width
            )
            # his_lat_velo = (
            #     his_local_coords[-1, 1] - his_local_coords[0, 1]
            # ) / (his_ts[-1] - his_ts[0])
            his_lat_velo = np.mean(
                np.abs(np.diff(his_local_coords[:, 1])) / his_time_diff
            )

            # del invalid label like " " or nan.
            try:
                obs_lat_label = float(obs_lat_label)
            except ValueError:
                obs_lat_label = -1
            if np.isnan(obs_lat_label):
                obs_lat_label = -1
            # Check if lat_label is consistent with moving direction
            if obs_lat_label == 1.0 and not np.all(
                np.diff(his_local_coords[:, 1]) >= 0
            ):
                obs_lat_label = -1.0
                continue
            if obs_lat_label == 2.0 and not np.all(
                np.diff(his_local_coords[:, 1]) <= 0
            ):
                obs_lat_label = -1.0
                continue

            # if obs_width is too large, set as invalid
            if obs_width > 3.0:
                obs_lat_label = -1.0
                continue

            # Adjust lat_label dynamically based on his_lat_velo and obs_width
            if obs_lat_label == 1 or obs_lat_label == 2:
                # Only lane-change samples which are close enough to laneline
                # are allowed
                if (
                    lat_dist_obs_cnt_to_overline > self.default_lane_width
                    or lat_dist_obs_cnt_to_overline < 0
                ):
                    obs_lat_label = -1.0
                    continue
                # Only the obstacles whose width exceeds 2m are limited to
                # overline status
                if obs_width < 2:
                    obs_edge_to_laneline_dist = (
                        lat_dist_obs_cnt_to_overline - obs_width / 2
                    )
                    if his_lat_velo >= 0 and his_lat_velo < 0.5:
                        compensator_dist = (
                            2 * dist_threshold / 0.5 * his_lat_velo
                            - dist_threshold
                        )
                        if obs_edge_to_laneline_dist > (
                            dist_threshold + compensator_dist
                        ):
                            obs_lat_label = 0.0
                else:
                    if not is_over_laneline:
                        obs_lat_label = 0.0
            # update the lat_label of the current track_id
            # in the last context frame.
            index = df[
                (df.track_id == track_id) & (df.frame_id == lcf_id)
            ].index[0]
            df.loc[index, "lat_behav_label"] = obs_lat_label

            # filter obstacles with invalid behav label
            if obs_lat_label < 0:
                continue
            # random down-sampling obstacles with label 0.
            elif (
                self.is_training and obs_lat_label == 0 and self.down_sampling
            ):
                if np.random.rand() > self.down_ratio:
                    continue
            down_sampled_track_ids.append(track_id)

        sample["seq_df"] = df
        sample["valid_behav_track_ids"] = down_sampled_track_ids

        return sample


@OBJECT_REGISTRY.register
class GenStatesAndMask:
    """(Sample and) Generate structured states and masks for the obstacles.

    This transform method requires the sample to have the keys `ctx_mask`,\
    `future_mask`,`ctx_frames`, `future_frames`, `track_ids`.\
    Therefore, it must be defined after GetSeqDataFrameMasks,\
    `GenFutureTrackids` or`FilterObstacles`.

    To use, the user should construct a `GenStatesAndMask` instance with the
    required initialization parameters. After instantiation, the callable
    function will return a updated dict that contains the five new elements as
    below: \
        1. `states` (np.array, [num_obj, traj_len, num_states]): the \
            ground-truth states of objects within the future time frames. \
            Here, `traj_len` is the length of the future trajectory. \
        2. `masks` (np.array, [num_obj, traj_len]): valid object indicator \
            for `states`. \
        3. `context_states` (np.array, [num_obj, num_ctx, num_states]): the \
            ground-truth states of objects within the historical context time \
            frames. Here, `num_ctx` is the length of the historical \
            trajectory. \
        4. `context_masks` (np.array, [num_obj, num_ctx]): valid object \
            indicator for `context_states`. \
        5. `classification` (np.array, [num_obj]: classification type ID of \
            the objects. \
        6. `state_complete_track_ids` (list): the track ids that the \
            obstacles have complete state information. \
        7. `direction_correct_track_ids` (list): the track ids that the \
            obstacles move forward. \

    If you want to use the `states` in `MultiPathTransform`, please put the
    required multipath x&y in the first two columns in `state_col_name`.
    """

    def __init__(
        self,
        state_col_name: tuple = ("object_centric_x", "object_centric_y"),
        enable_incomplete_gts: bool = False,
        ego_track_id: int = -42,
        is_training: bool = True,
        get_sampled_frames: bool = True,
    ):
        """Initialize method.

        Args:
            state_col_name (list): the column names related to the structured
                states. The user must ensure that these columns are exist in
                the `seq_df` of the input sample.
            enable_incomplete_gts (bool): whether to predict cases
                that do not have complete ground-truth future trajectories
                (i.e., the number of frames is smaller than `traj_len`).
                If the `data_source` is 'nuscenes' and the current stage is
                    validation, this parameter must be False. Because the
                    metric calcuator in NuScenes api does not support the
                    comparison between a complete prediction result and an
                    incomplete ground-truth trajectory.
                Otherwise, it is okay to set this parameter as True.
            ego_track_id (int): the ego vehicle track id.
            is_training (bool): whether the filtered track id list
                is for model training. The filter for training is
                more strict.
            get_sampled_frames (bool): Whether to use the context
                and future frame mask of sampled data.
                If freq_ratio > 0, need to sample.
        """
        self.state_col_name = state_col_name
        self.enable_incomplete_gts = enable_incomplete_gts or (not is_training)
        self.ego_track_id = ego_track_id
        self.is_training = is_training
        self.get_sampled_frames = get_sampled_frames

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'track_ids'
            3. 'ctx_mask'
            4. 'future_mask'
            5. 'ctx_frames'
            6. 'future_frames'
            7. (optional) 'sampled_ctx_mask', 'sampled_fut_mask',
                          'sampled_ctx_frame_id', 'sampled_fut_frame_id'
        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                `track_id`, `frame_id` and all the keys in
                `self.state_col_name`.

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the input sample dictionary with additional
                items: `states`, `masks`, `context_states`, `context_masks`,
                `classification`, `state_complete_track_ids`(optional),
                `direction_correct_track_ids`(optional).
        """
        df = sample["seq_df"]
        track_ids = sample["track_ids"]

        df_cols = df.columns
        df_values = df.values
        col_frame_id = df_cols.get_loc("frame_id")
        col_track_id = df_cols.get_loc("track_id")
        col_class = df_cols.get_loc("classification")
        col_states = list(
            map(lambda x: df_cols.get_loc(x), self.state_col_name)
        )

        # 0. Get the masks for context and future frames. If freq_ratio > 1,
        # need to sample.
        if self.get_sampled_frames and "sampled_ctx_mask" in sample:
            future_mask = sample["sampled_fut_mask"]
            future_frames = sample["sampled_fut_frame_id"]
        else:
            future_mask = sample["fut_mask"]
            future_frames = sample["fut_frame_id"]
        ctx_mask = sample["ctx_mask"]
        ctx_frames = sample["ctx_frame_id"]

        # 1. Initialize result value arrays.
        # -- future states: [num_obj, traj_len, num_states]
        # -- future masks: [num_obj, traj_len]
        # -- context states: [num_obj, num_ctx, num_states]
        # -- context masks: [num_obj, num_ctx]
        # -- classification: [num_obj]
        len_track_ids = len(track_ids)
        len_state_cols = len(self.state_col_name)
        len_ctx = len(ctx_frames)
        len_fut = len(future_frames)
        states = np.zeros(
            shape=[len_track_ids, len_fut, len_state_cols],
            dtype=np.float64,
        )
        masks = np.zeros(shape=[len_track_ids, len_fut], dtype=np.int64)
        ctx_states = np.zeros(
            shape=[len_track_ids, len_ctx, len_state_cols],
            dtype=np.float64,
        )
        ctx_masks = np.zeros(shape=[len_track_ids, len_ctx], dtype=np.int64)
        classification = np.zeros(shape=[len_track_ids], dtype=np.int64)

        # 2. Fill values for the result value arrays. If needed, save
        # the track_id of the valid obstacles that have nice state values.

        for t_idx, track_id in enumerate(track_ids):
            # -- DataFrame row mask for current track id
            track_id_mask = df_values[:, col_track_id] == track_id
            # -- frame id values of future and context
            obj_future_frame_ids = df_values[
                future_mask & track_id_mask, col_frame_id
            ]
            obj_ctx_frame_ids = df_values[
                ctx_mask & track_id_mask, col_frame_id
            ]
            # -- index of frame_ids in the result arrays.
            #    Not all trajectories contain a full range of frames, therefore
            #    the number of resulting frame_ids may be less than the length
            #    of the result arrays. We explicitly compute the index of each
            #    frame_id within the result array and use fancy indexing to
            #    assign values.
            obj_future_arr_idx = np.in1d(future_frames, obj_future_frame_ids)
            obj_ctx_arr_idx = np.in1d(ctx_frames, obj_ctx_frame_ids)
            # -- extract states and classification. Classification should be
            #    unique, so we take the first value of the unique value list
            obj_future_vals = df_values[future_mask & track_id_mask, :]
            obj_context_vals = df_values[ctx_mask & track_id_mask, :]
            obj_future_states = obj_future_vals[:, col_states]
            obj_ctx_states = obj_context_vals[:, col_states]
            obj_class = np.unique(obj_context_vals[:, col_class])[0]

            states[t_idx, obj_future_arr_idx, :] = obj_future_states
            masks[t_idx, obj_future_arr_idx] = 1
            ctx_states[t_idx, obj_ctx_arr_idx, :] = obj_ctx_states
            ctx_masks[t_idx, obj_ctx_arr_idx] = 1
            classification[t_idx] = obj_class

        sample["states"] = states
        sample["masks"] = masks
        sample["context_states"] = ctx_states
        sample["context_masks"] = ctx_masks
        sample["classification"] = classification
        return sample


@OBJECT_REGISTRY.register
class GenHighFreqTraj:
    """Get the high frequency trajectories for TCN module.

    This transform method requires the sample to have the following keys,
    which means that is must be used after some transform methods. \

    +------------------------------+----------------------------------------+
    |    requires                  |  needed transforms                     |
    +==============================+========================================+
    | track_ids,valid_track_ids    |   GenFutureTrackids, FilterObstacles   |
    +------------------------------+----------------------------------------+

    """

    def __init__(
        self,
        get_sampled_frames: bool = False,
        enable_incomplete_gts: bool = False,
        is_training: bool = True,
    ):
        """Initialize method.

        Args:
            get_sampled_frames (bool): Whether to use the context
            and future frame mask of sampled data.
            If freq_ratio > 0, need to sample.
            enable_incomplete_gts (bool, optional): whether to predict cases
                that do not have complete ground-truth future trajectories
                (i.e., the number of frames is smaller than `traj_len`).
                If the `data_source` is 'nuscenes' and the current stage is
                    validation, this parameter must be False. Because the
                    metric calcuator in NuScenes api does not support the
                    comparison between a complete prediction result and an
                    incomplete ground-truth trajectory.
                Otherwise, it is okay to set this parameter as True.
            is_training (bool): whether the filtered track id list is for
                model training. The filter for training is more strict.
        """
        self.get_sampled_frames = get_sampled_frames
        self.enable_incomplete_gts = enable_incomplete_gts
        self.is_training = is_training
        self.gen_states_and_masks_object = GenStatesAndMask(
            enable_incomplete_gts=self.enable_incomplete_gts,
            is_training=self.is_training,
            get_sampled_frames=self.get_sampled_frames,
        )

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'track_ids'
            3. 'lctx_mask'.
            4. 'ctx_mask'
            5. 'future_mask'
            6. 'ctx_frames'
            7. 'future_frames'
            8. (optional) 'sampled_ctx_mask', 'sampled_fut_mask',
                          'sampled_ctx_frame_id', 'sampled_fut_frame_id'
        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                `track_id`, `frame_id` and all the keys in
                `self.state_col_name`.

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the input sample dictionary with additional
                items: `high_freq_ctx_trajectories`, `high_freq_ctx_masks`.
        """
        df = sample["seq_df"]
        df_cols = df.columns
        df_values = df.values
        col_track_id = df_cols.get_loc("track_id")
        lctx_mask = sample["lctx_mask"]
        high_freq_sample = sample.copy()
        high_freq_sample = self.gen_states_and_masks_object(high_freq_sample)
        full_ctx_trajs = high_freq_sample["context_states"]
        full_ctx_masks = high_freq_sample["context_masks"]
        track_yaw_dict = sample["track_yaw_dict"]
        full_track_ids = sample["track_ids"]
        center_ids = sample["valid_track_ids"]
        if "valid_future_track_ids" in sample:
            valid_future_track_ids = sample["valid_future_track_ids"]
            center_ids = list(set(center_ids) & set(valid_future_track_ids))
        center_ids.sort()

        # Extract the high frequency context trajectories for TCN module.
        valid_ctx_trajs = []
        valid_ctx_masks = []
        for track_id in center_ids:
            track_id_mask = df_values[:, col_track_id] == track_id
            track_id_index = full_track_ids.index(track_id)
            agent_ctx_traj = full_ctx_trajs[track_id_index]
            agent_ctx_mask = full_ctx_masks[track_id_index]
            cur_mask = lctx_mask & track_id_mask
            if not np.any(cur_mask):
                continue
            if track_yaw_dict[track_id] is None:
                continue
            valid_ctx_trajs.append(agent_ctx_traj)
            valid_ctx_masks.append(agent_ctx_mask)
        sample["high_freq_ctx_trajectories"] = valid_ctx_trajs
        sample["high_freq_ctx_masks"] = valid_ctx_masks
        return sample


@OBJECT_REGISTRY.register
class GetTrajPredObjectsInfo:
    """Get the information of all the valid objects for trajectory prediction.

    This transform method requires the sample to have the following keys,
    which means that is must be used after some transform methods. \

    +------------------------------+----------------------------------------+
    |    requires                  |  needed transforms                     |
    +==============================+========================================+
    | seq_center                   |   GenSeqCenter                         |
    | track_ids, valid_track_ids   |   GenFutureTrackids, RemapObsCls       |
    | track_yaw_dict               |   SelectYawArray                       |
    | states, masks, agent_classes |   FilterObstacles, GenStatesAndMask    |
    | ctx_mask                     |   GenSeqDataFrameMask
    +------------------------------+----------------------------------------+

    """

    def __init__(
        self,
        scene_img_height: int,
        scene_img_width: int,
        ego_track_id: int,
        ped_cyc_type_id: List,
        timestamp_unit: str = "ms",
        use_state_vectors: bool = True,
        use_instant_state_vectors: bool = False,
        num_state_vectors: int = 3,
        if_clip_state_vectors: bool = True,
        state_vectors_log: List = (True, False, False),
        state_vectors_clip_bound: List = (5, 5, 0.5),
        state_vectors_scales: List = (1, 1, 10),
        assign_trajs_for_filtered_obs: bool = False,
        traj_len: int = 12,
        reverse: bool = False,
        filter_invalid_end: bool = False,
    ):
        """Initialize method.

        Args:
            scene_img_height (int): rendered map height.
            scene_img_width (int): rendered map width.
            ego_track_id (int): the ego vehicle track id.
            timestamp_unit (str, optional): the unit of time stamp. Default
                to "ms". It should be in ["s", "ms"].
            use_state_vectors (bool, optional): whether to use some extra
                state vectors of the objects. The state vectors will be
                directly input into the model, and be concatenated with the
                feature from backbone. The state vectors include some
                acceleration information, etc.
                Note: The use of state vectors could improve the model metrics,
                but for real road scenarios, these kinds of information is not
                easy to obtain. The user can turn off this option according to
                the actual situation.
            use_instant_state_vectors (bool, optional): whether the state
                vector use the instantaneous values in the last context frame
                or the mean values within the context frames. Default to False.
            if_clip_state_vectors (bool, optioal): whether to adjust the state
                vectors to the same range by log, clip and scale. Default to
                True. We recommend to open it to avoid some quantization error.
                If this switch is open, the parameter `state_vectors_log`,
                `state_vectors_clip_bound` and `state_vectors_scales` is
                enabled. Here we give a set of recommended values, which are
                empirical values calculated from a large batch of trajectory
                data.
            assign_trajs_for_filtered_obs (bool, optional): whether to assign
                rule trajectories for filtered obstacles. If set as true,
                obstacles that are too far will be assigned uniform speed
                trajectories, and other filtered obstacles will be assigned
                static trajectories. Default to false.
            traj_len (int, optional): the length of the future trajectory.
            reverse (bool, optional): if the direction of VCS and image
                coordinate system is on the contrary (like in the BEV
                scenario), reverse should be True.
            get_sampled_frames (bool): Whether to use the context
                and future frame mask of sampled data.
                If freq_ratio > 0, need to sample.
            filter_invalid_end (bool, optional): Whether to filter obs with
                invalid future end point.
        """
        self.scene_img_height = scene_img_height
        self.scene_img_width = scene_img_width
        self.ego_track_id = ego_track_id
        # Parameters about state vectors.
        self.ped_cyc_type_id = ped_cyc_type_id
        self.use_state_vectors = use_state_vectors
        self.use_instant_state_vectors = use_instant_state_vectors
        self.num_state_vectors = num_state_vectors
        self.if_clip_state_vectors = if_clip_state_vectors
        self.state_vectors_log = state_vectors_log
        self.state_vectors_clip_bound = state_vectors_clip_bound
        self.state_vectors_scales = state_vectors_scales
        self.assign_trajs_for_filtered_obs = assign_trajs_for_filtered_obs
        self.traj_len = traj_len
        self.reverse = reverse
        self.filter_invalid_end = filter_invalid_end
        assert (
            len(state_vectors_log)
            == len(state_vectors_clip_bound)
            == len(state_vectors_scales)
            == num_state_vectors
        )

        if timestamp_unit == "ms":
            self.stamp_scale = 1000
        elif timestamp_unit == "s":
            self.stamp_scale = 1
        else:
            raise ValueError("The timestamp_unit should be ms or s.")

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'
            2. 'last_context_frame_id'
            3. 'track_ids'
            4. 'states'
            5. 'masks'
            6. 'rendered_obs'
            7. 'agent_classes'
            8. 'track_yaw_dict'
            9. 'track_stat_dict'
            10. 'ctx_mask' or 'sampled_ctx_mask'.
            11. (optional) 'valid_track_ids', 'state_complete_track_ids',
               'direction_correct_track_ids'

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'img_x', 'img_y', 'yaw', 'img_obs_yaw', 'track_id', 'frame_id',
                'x', 'y', 'obs_yaw', 'timestamp', 'classification',
                'obs_length', 'obs_width'

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample includes new keys
            'ori_seq_df', 'valid_class',
                'valid_img_coords', 'valid_masks', 'future_trajectories',
                'resize_ratio', 'state_vectors',
                'filtered_obs_ids'(optional), 'filtered_obs_trajs'(optional),
                'filtered_obs_gt'(optional), 'filtered_obs_gt_masks'(optional),
                'filtered_obs_class'(optional) and update keys 'seq_df',
                'rendered_obs', 'valid_track_ids'.
        """
        # Extract information from sample.
        seq_df = sample["seq_df"].copy()
        last_context_frame_id = sample["last_context_frame_id"]
        # -- from GenSeqCenter
        seq_center = sample["seq_center"]
        # -- from FilterObstacles or GenFutureTrackids
        full_track_ids = sample["track_ids"]
        if "valid_track_ids" in sample:
            center_ids = sample["valid_track_ids"]
        else:
            center_ids = sample["track_ids"]
        # -- from GenStatesAndMask
        # TODO (shiqi.tan): add the states of filtered obstacles
        # and delete the sort function.
        full_future_trajs = sample["states"]
        full_future_masks = sample["masks"]
        full_ctx_trajs = sample["context_states"]
        full_ctx_masks = sample["context_masks"]
        agent_classes = sample["agent_classes"]
        if "valid_future_track_ids" in sample:
            valid_future_track_ids = sample["valid_future_track_ids"]
            center_ids = list(set(center_ids) & set(valid_future_track_ids))
        center_ids.sort()
        # -- from SelectYawArray
        track_yaw_dict = sample["track_yaw_dict"]
        seq_df["last_context_frame_id"] = last_context_frame_id
        lctx_mask = sample["lctx_mask"]
        # need to sample.
        if "sampled_ctx_mask" in sample:
            ctx_mask = sample["sampled_ctx_mask"]
        else:
            ctx_mask = sample["ctx_mask"]

        # # Filter the out-of-scene rows according to img coords.
        (
            valid_cls,
            valid_track_ids,
            valid_fut_trajs,
            valid_traj_masks,
            valid_img_coords,
            valid_his_phy_coords,
            _,
            valid_ctx_masks,
        ) = self.extract_obs_img_coords(
            seq_df,
            seq_center,
            full_future_trajs,
            full_future_masks,
            full_ctx_trajs,
            full_ctx_masks,
            center_ids,
            full_track_ids,
            agent_classes,
            ctx_mask,
            lctx_mask,
            track_yaw_dict,
            self.ego_track_id,
            self.reverse,
            self.filter_invalid_end,
        )

        if self.use_state_vectors:
            state_vectors, ori_state_vectors = self._extract_state_vectors(
                valid_track_ids,
                valid_his_phy_coords,
                valid_cls,
                track_yaw_dict,
            )
            sample["state_vectors"] = state_vectors
            sample["ori_state_vectors"] = ori_state_vectors

        if self.assign_trajs_for_filtered_obs:
            (
                filtered_obs_ids,
                filtered_obs_trajs,
                filtered_obs_gt,
                filtered_obs_gt_masks,
                filtered_obs_class,
            ) = self._gen_traj_for_filtered_obstacles(
                sample,
                seq_df,
                full_future_trajs,
                full_future_masks,
                full_track_ids,
                agent_classes,
                lctx_mask,
                ctx_mask,
                track_yaw_dict,
            )

        # Update the sample dict.
        sample["ori_seq_df"] = sample["seq_df"]
        sample["seq_df"] = seq_df
        sample["valid_class"] = valid_cls
        sample["valid_track_ids"] = valid_track_ids
        sample["valid_img_coords"] = valid_img_coords
        sample["valid_masks"] = valid_traj_masks
        sample["future_trajectories"] = valid_fut_trajs
        if self.assign_trajs_for_filtered_obs:
            sample["filtered_obs_ids"] = filtered_obs_ids
            sample["filtered_obs_trajs"] = filtered_obs_trajs
            sample["filtered_obs_gt"] = filtered_obs_gt
            sample["filtered_obs_gt_masks"] = filtered_obs_gt_masks
            sample["filtered_obs_class"] = filtered_obs_class
        num_agents = len(valid_track_ids)
        sample["resize_ratio"] = np.array([[1.0, 1.0]] * num_agents)

        return sample

    @staticmethod
    def extract_obs_img_coords(
        seq_df,
        seq_center,
        full_future_trajs,
        full_future_masks,
        full_ctx_trajs,
        full_ctx_masks,
        center_ids: List,
        full_track_ids: List,
        agent_classes: List,
        ctx_mask: int,
        lctx_mask: int,
        track_yaw_dict: Dict,
        ego_track_id: int = -42,
        reverse: bool = False,
        filter_invalid_end: bool = False,
    ):  # noqa: D208
        """Extract obstacle image coordinates and trajectory info.

        Args:
            seq_df (pd.DataFrame): the orginal data frame.
            seq_center (ArrayLike): SeqCenter of the sequence dataframe in
                the physical coordinates.
            full_future_trajs ([type]): the future trajectories of all the
                obstacles.
            full_future_masks ([type]): the future trajectorie masks of all the
                obstacles.
            center_ids (List): the track ids to calculate image coordinates.
            full_track_ids (List): all track ids.
            agent_classes (List): the agent class of all obstacles.
            ctx_mask (int): The mask of context frame id.
            lctx_mask (): The mask of last context frame id.
            track_yaw_dict (Dict): the mapping between track_id and yaw array.
            ego_track_id (int, optional): the ego vehicle track id.
            reverse (bool, optional): if the direction of VCS and image
                coordinate system is on the contrary (like in the BEV
                scenario), reverse should be True.
            filter_invalid_end (bool, optional): Whether to filter obs with
                invalid future end point.

        Returns:
            valid_cls (List): the agent classes of all the valid obstacles.
            valid_track_ids (List): the track ids of all the valid obstacles.
            valid_fut_trajs (List): the future trajectories.
            valid_traj_masks (List): the future trajectory masks. \
            valid_img_coords (List): the image coordinates of the valid \
                obstacles in the last context frame. \
            valid_his_phy_coords (List): the historical physical coordinates \
                of all the valid obstacles. \
        """

        # Get column indices.
        seq_df_vals = seq_df.values
        seq_df_cols = seq_df.columns
        track_id_col = seq_df_cols.get_loc("track_id")
        phy_x_col = seq_df_cols.get_loc("x")
        phy_y_col = seq_df_cols.get_loc("y")
        phy_yaw_col = seq_df_cols.get_loc("obs_yaw")
        img_x_col = seq_df_cols.get_loc("img_x")
        img_y_col = seq_df_cols.get_loc("img_y")
        img_yaw_col = seq_df_cols.get_loc("img_obs_yaw")
        stamp_col = seq_df_cols.get_loc("timestamp")

        # Extract the information for valid obstacles (objects for prediction).
        valid_cls = []
        valid_track_ids = []
        valid_fut_trajs = []
        valid_traj_masks = []
        valid_ctx_trajs = []
        valid_ctx_masks = []
        valid_img_coords = []
        valid_his_phy_coords = []
        for track_id in center_ids:
            track_id_mask = seq_df_vals[:, track_id_col] == track_id
            track_id_index = full_track_ids.index(track_id)
            agent_fut_traj = full_future_trajs[track_id_index]
            agent_fut_mask = full_future_masks[track_id_index]
            agent_ctx_traj = full_ctx_trajs[track_id_index]
            agent_ctx_mask = full_ctx_masks[track_id_index]
            agent_class = agent_classes[track_id_index]
            cur_mask = lctx_mask & track_id_mask
            cur_ctx_mask = ctx_mask & track_id_mask
            if not np.any(cur_mask):
                continue
            if filter_invalid_end and agent_fut_mask[-1] == 0:
                if track_id != ego_track_id:
                    continue
            lcf_df_vals = seq_df_vals[cur_mask, :]
            img_coords_cols = [img_x_col, img_y_col, img_yaw_col]
            agent_img_coords = lcf_df_vals[:, img_coords_cols].reshape(-1)

            ctx_df_vals = seq_df_vals[cur_ctx_mask, :]
            used_phy_cols = [phy_x_col, phy_y_col, phy_yaw_col, stamp_col]
            his_phy_coords = ctx_df_vals[:, used_phy_cols].astype("float64")

            if track_yaw_dict[track_id] is None:
                continue
            select_obs_yaw = track_yaw_dict[track_id][-1]
            select_img_yaw = select_obs_yaw - seq_center.yaw
            if reverse:
                select_img_yaw += np.pi
            agent_img_coords[2] = select_img_yaw

            valid_cls.append(agent_class)
            valid_track_ids.append(track_id)
            valid_fut_trajs.append(agent_fut_traj)
            valid_traj_masks.append(agent_fut_mask)
            valid_ctx_trajs.append(agent_ctx_traj)
            valid_ctx_masks.append(agent_ctx_mask)
            valid_img_coords.append(agent_img_coords)
            valid_his_phy_coords.append(his_phy_coords)
        return (
            valid_cls,
            valid_track_ids,
            valid_fut_trajs,
            valid_traj_masks,
            valid_img_coords,
            valid_his_phy_coords,
            valid_ctx_trajs,
            valid_ctx_masks,
        )

    def _extract_state_vectors(
        self,
        valid_track_ids: List,
        valid_his_phy_coords: List,
        valid_cls: List,
        track_yaw_dict: Dict,
    ):
        """Get agent state vectors (velocity, acceleration and yaw rate).

        Args:
            valid_track_ids (List): the valid track id list.
            valid_his_phy_coords (List): the historical coornates of the valid
                obstacles.
            valid_cls (List): the classification of the valid obstacles.
            track_yaw_dict (Dict): the mapping between track_id and yaw array.

        Returns:
            state_vectors (np.array, [num_valid_obj, num_state_vectors]): the
                agent state vectors.
        """
        state_vectors = np.zeros(
            [len(valid_track_ids), self.num_state_vectors]
        )
        for idx, (track_id, his_phy, track_cls) in enumerate(
            zip(valid_track_ids, valid_his_phy_coords, valid_cls)
        ):
            his_x = his_phy[:, 0]
            his_y = his_phy[:, 1]
            his_yaw = track_yaw_dict[track_id]
            his_stamp = his_phy[:, 3] / self.stamp_scale
            len_ctx = len(his_x)
            len_yaw = len(his_yaw)
            # The first-order variable requires a length of at least 2.
            # -- Calculate velocity.
            his_time_diff = his_stamp[1:] - his_stamp[:-1]
            his_velo = np.zeros([len(his_yaw)])
            if len_ctx > 1:
                his_velo = (
                    np.sqrt(
                        (his_x[1:] - his_x[:-1]) ** 2
                        + (his_y[1:] - his_y[:-1]) ** 2
                    )
                    / his_time_diff
                )
            # -- Calculate yaw rate.
            if len_yaw > 1:
                his_yaw_diff = his_yaw[1:] - his_yaw[:-1]
                assert len_yaw <= len_ctx, (
                    "The length of the obs yaw cannot be bigger than"
                    "the length of the all time stamps."
                    f"{len_yaw} vs {len_ctx}, {his_yaw} {his_phy} {track_id}"
                )
                yaw_s_idx = len(his_time_diff) - len(his_yaw_diff)
                his_yaw_rate = his_yaw_diff / his_time_diff[yaw_s_idx:]
                if self.use_instant_state_vectors and (
                    track_cls not in self.ped_cyc_type_id
                ):
                    state_vectors[idx, 0] = his_velo[-1]
                    state_vectors[idx, 2] = his_yaw_rate[-1]
                else:
                    state_vectors[idx, 0] = np.mean(his_velo)
                    state_vectors[idx, 2] = np.mean(his_yaw_rate)

            # The Second-order variable requires a length of at least 3.
            if len_ctx > 2:
                # -- Calculate acceleration.
                his_acc = (his_velo[1:] - his_velo[:-1]) / his_time_diff[1:]
                if self.use_instant_state_vectors:
                    state_vectors[idx, 1] = his_acc[-1]
                else:
                    state_vectors[idx, 1] = np.mean(his_acc)

        origin_state_vectors = copy.deepcopy(state_vectors)
        if self.if_clip_state_vectors:
            for i, (if_log, clip_b, scale) in enumerate(
                zip(
                    self.state_vectors_log,
                    self.state_vectors_clip_bound,
                    self.state_vectors_scales,
                )
            ):
                tmp_sv = state_vectors[:, i]
                if if_log:
                    tmp_sv = np.log(np.maximum(tmp_sv, 1e-2))
                tmp_sv = np.clip(tmp_sv, -clip_b, clip_b)
                tmp_sv *= scale
                state_vectors[:, i] = tmp_sv
        return state_vectors, origin_state_vectors

    def _assign_uniform_trajectories(
        self,
        seq_df,
        full_future_trajs,
        full_future_masks,
        full_track_ids: List,
        agent_classes: List,
        lctx_mask,
        ctx_mask,
        track_yaw_dict,
        obs_too_far_ids,
    ):
        """Assign uniform trajectories to obstacles that are too far.

        Args:
            seq_df (pd.DataFrame): the orginal data frame.
            full_future_trajs ([type]): the future trajectories of all the
                obstacles.
            full_future_masks ([type]): the future trajectorie masks of all the
                obstacles.
            full_track_ids (List): all track ids.
            agent_classes (List): the agent class of all obstacles.
            lctx_mask (List): The mask of last context frame id.
            ctx_mask (List): The mask of context frame id.
            track_yaw_dict (Dict): the mapping between track_id and yaw array.
            obs_too_far_ids (List): track ids of too far obstacles.

        Returns:
            obs_too_far_trajs (np.array): the uniform trajectories for too far
                obstacles.
            valid_fut_trajs (List): the future trajectories.
            valid_traj_masks (List): the future trajectory masks.
        """
        seq_df_vals = seq_df.values
        seq_df_cols = seq_df.columns
        track_id_col = seq_df_cols.get_loc("track_id")
        phy_x_col = seq_df_cols.get_loc("x")
        phy_y_col = seq_df_cols.get_loc("y")
        phy_yaw_col = seq_df_cols.get_loc("obs_yaw")
        stamp_col = seq_df_cols.get_loc("timestamp")

        valid_fut_trajs = []
        valid_traj_masks = []
        valid_agent_class = []
        obs_too_far_trajs = np.zeros([len(obs_too_far_ids), self.traj_len, 2])
        for idx, track_id in enumerate(obs_too_far_ids):
            track_id_mask = seq_df_vals[:, track_id_col] == track_id
            track_id_index = full_track_ids.index(track_id)
            agent_fut_traj = full_future_trajs[track_id_index]
            agent_fut_mask = full_future_masks[track_id_index]
            agent_class = agent_classes[track_id_index]
            cur_mask = lctx_mask & track_id_mask
            cur_ctx_mask = ctx_mask & track_id_mask
            obs_too_far_traj = np.zeros([self.traj_len, 2])
            if not np.any(cur_mask):
                obs_too_far_trajs[idx] = obs_too_far_traj
                continue
            ctx_df_vals = seq_df_vals[cur_ctx_mask, :]
            used_phy_cols = [phy_x_col, phy_y_col, phy_yaw_col, stamp_col]
            his_phy_coords = ctx_df_vals[:, used_phy_cols].astype("float64")
            his_x = his_phy_coords[:, 0]
            his_y = his_phy_coords[:, 1]
            his_yaw = track_yaw_dict[track_id]
            his_stamp = his_phy_coords[:, 3] / self.stamp_scale
            len_ctx = len(his_x)
            his_time_diff = his_stamp[1:] - his_stamp[:-1]
            his_displace = np.zeros([len(his_yaw)])
            if len_ctx > 1:
                his_displace = np.sqrt(
                    (his_x[1:] - his_x[:-1]) ** 2
                    + (his_y[1:] - his_y[:-1]) ** 2
                )
            obs_too_far_dis = (
                his_displace[-1] / his_time_diff[-1] * 0.5
                if len(his_displace)
                else 0
            )
            for traj_idx in range(self.traj_len):
                obs_too_far_traj[traj_idx, 0] = (
                    traj_idx + 1
                ) * obs_too_far_dis
            obs_too_far_trajs[idx] = obs_too_far_traj
            valid_fut_trajs.append(agent_fut_traj)
            valid_traj_masks.append(agent_fut_mask)
            valid_agent_class.append(agent_class)
        return (
            obs_too_far_trajs,
            valid_fut_trajs,
            valid_traj_masks,
            valid_agent_class,
        )

    def _assign_static_trajectories(
        self,
        full_future_trajs,
        full_future_masks,
        full_track_ids: List,
        agent_classes: List,
        filtered_obs_ids: List,
    ):
        """Assign static trajs to obstacles that are filtered except too far obs.

        Args:
            full_future_trajs ([type]): the future trajectories of all the
                obstacles.
            full_future_masks ([type]): the future trajectorie masks of all the
                obstacles.
            full_track_ids (List): all track ids.
            agent_classes (List): the agent class of all obstacles.
            filtered_obs_ids (List): track ids of filtered obstacles.

        Returns:
            filtered_obs_trajs (np.array): the static trajectory for each
                filtered obstacle.
            valid_fut_trajs (List): the future trajectories.
            valid_traj_masks (List): the future trajectory masks.
        """
        valid_fut_trajs = []
        valid_traj_masks = []
        valid_agent_class = []
        for track_id in filtered_obs_ids:
            track_id_index = full_track_ids.index(track_id)
            agent_fut_traj = full_future_trajs[track_id_index]
            agent_fut_mask = full_future_masks[track_id_index]
            agent_class = agent_classes[track_id_index]
            valid_fut_trajs.append(agent_fut_traj)
            valid_traj_masks.append(agent_fut_mask)
            valid_agent_class.append(agent_class)
        filtered_obs_trajs = np.zeros(
            [len(filtered_obs_ids), self.traj_len, 2]
        )
        return (
            filtered_obs_trajs,
            valid_fut_trajs,
            valid_traj_masks,
            valid_agent_class,
        )

    def _gen_traj_for_filtered_obstacles(
        self,
        sample,
        seq_df,
        full_future_trajs,
        full_future_masks,
        full_track_ids,
        agent_classes,
        lctx_mask,
        ctx_mask,
        track_yaw_dict,
    ):
        """Assign ruled trajectories to obstacles that are filtered.

        Args:
            sample (Dict): input original sample.
            seq_df (pd.DataFrame): the orginal data frame.
            full_future_trajs ([type]): the future trajectories of all the
                obstacles.
            full_future_masks ([type]): the future trajectorie masks of all the
                obstacles.
            full_track_ids (List): all track ids.
            agent_classes (List): the agent class of all obstacles.
            lctx_mask (): The mask of last context frame id.
            ctx_mask (int): The mask of context frame id.
            track_yaw_dict (Dict): the mapping between track_id and yaw array.

        Returns:
            filtered_obs_ids (List): filtered obstacles' id.
            filtered_obs_trajs (np.array): the ruled trajectories for filtered
                obstacles.
            filtered_obs_gt (List): the future trajectories.
            filtered_obs_gt_masks (List): the future trajectory masks.
        """
        obs_too_far_ids = []
        for track_id in sample["track_ids"]:
            if (
                sample["track_stat_dict"][track_id]
                == FILTER_FLAG["obs_too_far"]
            ):
                obs_too_far_ids.append(track_id)
        (
            obs_too_far_trajs,
            obs_too_far_gt,
            obs_too_far_gt_masks,
            obs_too_far_class,
        ) = self._assign_uniform_trajectories(
            seq_df,
            full_future_trajs,
            full_future_masks,
            full_track_ids,
            agent_classes,
            lctx_mask,
            ctx_mask,
            track_yaw_dict,
            obs_too_far_ids,
        )
        other_filtered_obs_ids = list(
            set(full_track_ids)
            - set(obs_too_far_ids)
            - set(sample["valid_track_ids"])
        )
        (
            other_filtered_obs_trajs,
            other_filtered_obs_gt,
            other_filtered_obs_gt_masks,
            other_filtered_obs_class,
        ) = self._assign_static_trajectories(
            full_future_trajs,
            full_future_masks,
            full_track_ids,
            agent_classes,
            other_filtered_obs_ids,
        )
        # If there are no obstacles that needs to be filtered, there will be
        # problems running Uniformpath structure.
        filtered_obs_ids = [-42]
        filtered_obs_trajs = np.zeros([1, self.traj_len, 2])
        filtered_obs_gt = [np.zeros([self.traj_len, 2])]
        filtered_obs_gt_masks = [np.ones([self.traj_len])]
        filtered_obs_class = [np.zeros([self.traj_len])]
        if len(obs_too_far_ids):
            filtered_obs_ids += obs_too_far_ids
            filtered_obs_trajs = np.concatenate(
                (filtered_obs_trajs, obs_too_far_trajs)
            )
            filtered_obs_gt += obs_too_far_gt
            filtered_obs_gt_masks += obs_too_far_gt_masks
            filtered_obs_class += obs_too_far_class
        if len(other_filtered_obs_ids):
            filtered_obs_ids += other_filtered_obs_ids
            filtered_obs_trajs = np.concatenate(
                (filtered_obs_trajs, other_filtered_obs_trajs)
            )
            filtered_obs_gt += other_filtered_obs_gt
            filtered_obs_gt_masks += other_filtered_obs_gt_masks
            filtered_obs_class += other_filtered_obs_class
        return (
            filtered_obs_ids,
            filtered_obs_trajs,
            filtered_obs_gt,
            filtered_obs_gt_masks,
            filtered_obs_class,
        )


@OBJECT_REGISTRY.register
class GetBehavPredObjectsInfo:
    """Get the information of all the valid objects for behavior prediction.

    This transform method requires the sample to have the following keys,
    which means that is must be used after some transform methods.

    +------------------------------+----------------------------------------+
    |    requires                  |  needed transforms                     |
    +==============================+========================================+
    | seq_df, track_ids            |   GenSeqCenter                         |
    | valid_track_ids              |   GenFutureTrackids, RemapObsCls       |
    | lctx_mask                    |   SelectYawArray                       |
    | last_context_frame_id        |   FilterObstacles, GenStatesAndMask    |
    +------------------------------+----------------------------------------+

    """

    def __init__(
        self,
        ego_track_id: int,
        use_behav_state_vectors: bool = False,
        num_behav_state_vectors: int = 20,
        use_obs_interaction: bool = False,
        use_self_boxes: bool = False,
        use_self_states: bool = False,
        if_norm: bool = True,
        norm_scale: int = 5,
        mode: str = "scene",
    ):
        """Initialize method.

        Args:
            ego_track_id (int): the ego vehicle track id.
            use_behav_state_vectors (bool, optional): whether to use behav
                state vectors of the objects. The state vectors include
                historical features.
                Note: The use of state vectors could improve the model metrics,
                but for real road scenarios, these kinds of information is not
                easy to obtain. The user can turn off this option according to
                the actual situation.
            num_behav_state_vectors (int, optional): length of behav state
                vectors.
            use_obs_interaction (bool): whether to use interation information
                between target obstacle and the others.
            use_self_boxes (bool): whether to use historical bboxes of target
                obstacle.
            use_self_states (bool): whether to use historical states of target
                obstacle.
            if_norm (bool, optional): whether to normalize the state vectors.
            norm_scale (bool, optional): value range after normalization.
            mode (str): sample is a scene or a obstacle.
        """
        self.ego_track_id = ego_track_id
        # Parameters about state vectors.
        self.use_behav_state_vectors = use_behav_state_vectors
        self.num_behav_state_vectors = num_behav_state_vectors
        self.use_obs_interaction = use_obs_interaction
        self.use_obs_selfboxes = use_self_boxes
        self.use_obs_selfstates = use_self_states
        self.if_norm = if_norm
        self.norm_scale = norm_scale
        self.mode = mode

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'
            2. 'lctx_mask'
            3. 'valid_track_ids'
            4. 'last_context_frame_id'

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'lat_behav_label', 'lon_behav_label', 'is_over_laneline',
                'dist_to_egocar',  "lat_dist_obs_cnt_to_overline",
                "lat_dist_obs_b1_to_overline", "lat_dist_obs_b2_to_overline",
                "lat_dist_obs_b3_to_overline", "lat_dist_obs_b4_to_overline",

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample includes new keys 'lat_behaviors',
                'lon_behaviors' and update keys 'state_vectors'.
        """
        # Extract information from sample.
        seq_df = sample["seq_df"].copy()
        # seq_center = sample["seq_center"]
        lcf_id = sample["last_context_frame_id"]
        if self.mode == "scene":
            if "valid_track_ids" in sample:
                center_ids = sample["valid_track_ids"]
            else:
                center_ids = sample["track_ids"]
            if len(center_ids) == 0:
                return sample
            (
                valid_lat_behavs,
                valid_lon_behavs,
                valid_over_lanelines,
                valid_dist_to_ego,
                valid_lat_dist_to_overline,
                valid_centric_feats,
            ) = self.collect_all_obs_info(
                seq_df,
                center_ids,
                lcf_id,
            )

            if self.use_behav_state_vectors:
                behav_state_vectors = self._extend_state_vectors(
                    center_ids,
                    valid_over_lanelines,
                    valid_dist_to_ego,
                    valid_lat_dist_to_overline,
                    valid_centric_feats,
                )
                sample["behav_state_vectors"] = np.concatenate(
                    [sample["state_vectors"], behav_state_vectors], axis=1
                )

            # Update the sample dict.
            # lat_behaviors and lon_behaviors: [num_obj]
            sample["lat_behaviors"] = np.array(valid_lat_behavs)
            sample["lon_behaviors"] = np.array(valid_lon_behavs)
        elif self.mode == "obstacle":
            if self.use_behav_state_vectors:
                track_id = sample["track_id"]
                (
                    his_centric_feats,
                    over_laneline,
                    _,
                    _,
                    _,
                    _,
                ) = self._extract_obs_behav_info(seq_df, track_id, lcf_id)
                state_vectors = np.zeros(self.num_behav_state_vectors)
                state_vectors[:1] = over_laneline
                # state_vectors[1] = dist_to_ego
                # state_vectors[2:7] = lat_dist_to_overline
                state_vectors[1:] = his_centric_feats
                state_vectors = (state_vectors.reshape(-1, 1, 1)).astype(
                    np.float32
                )
                sample["behav_state_vectors"] = np.concatenate(
                    [sample["behav_state_vectors"][:3], state_vectors], axis=0
                )
            else:
                sample["behav_state_vectors"] = sample["behav_state_vectors"][
                    :3
                ]
        else:
            raise ValueError(
                f"Behav transform has not supported {self.mode} mode yet!"
            )

        return sample

    def collect_all_obs_info(
        self,
        seq_df,
        center_ids: List,
        lcf_id: int,
    ):
        """Collect all obs info of each frame.

        Args:
            seq_df (pd.DataFrame): the orginal data frame.
            center_ids (list[int]): the track ids to calculate image \
                coordinates.
            lcf_id (int): the last context frame id.

        Returns:
            valid_track_ids (list[int]): the track ids of all valid obstacles.
            valid_lat_behavs (list[float]): the groundtruth lateral behaviors.
            valid_lon_behavs (list[float]): the groundtruth longitudinal \
                behaviors.
            valid_over_lanelines (list[bool]): if object cutting the laneline.
            valid_dist_to_ego (list[float]): distances between objects and \
                the ego car.
            valid_lat_dist_to_laneline (list[numpy.ndarray]): distances \
                between objects and the laneline they are cutting.
            valid_centric_feats (list[numpy.ndarray]): centric hand-crafted \
                features.
        """
        # Extract the behav gt and features for valid obstacles.
        valid_lat_behavs = []
        valid_lon_behavs = []
        valid_over_lanelines = []
        valid_dist_to_ego = []
        valid_lat_dist_to_overline = []
        valid_centric_feats = []
        for track_id in center_ids:
            (
                his_centric_feats,
                over_laneline,
                dist_to_ego,
                lat_dist_to_overline,
                lat_label,
                lon_label,
            ) = self._extract_obs_behav_info(seq_df, track_id, lcf_id)

            valid_lat_behavs.append(lat_label)
            valid_lon_behavs.append(lon_label)
            valid_over_lanelines.append(over_laneline)
            valid_dist_to_ego.append(dist_to_ego)
            valid_lat_dist_to_overline.append(lat_dist_to_overline)
            valid_centric_feats.append(his_centric_feats)

        return (
            valid_lat_behavs,
            valid_lon_behavs,
            valid_over_lanelines,
            valid_dist_to_ego,
            valid_lat_dist_to_overline,
            valid_centric_feats,
        )

    def _extract_obs_behav_info(
        self,
        seq_df,
        track_id: int,
        lcf_id: int,
    ):
        """Extract historical handcrafted features and behavior label.

        Args:
            seq_df (pd.DataFrame): the orginal data frame.
            track_id (int): track id of the obstacle.
            lcf_id (int): the last context frame id.

        Returns:
            his_centric_feats (numpy.ndarray): centric hand-crafted \
                features.
            over_laneline (bool): if object cutting the laneline.
            dist_to_ego (float): distances between objects and \
                the ego car.
            lat_dist_to_laneline (numpy.ndarray): distances between \
                objects and the laneline they are cutting.
            lat_label (int): lateral behavior label of the obs.
            lon_label (int): longitudinal behavior label of the obs.
        """
        seq_df_vals = seq_df.values
        seq_df_cols = seq_df.columns
        stamp_col = seq_df_cols.get_loc("timestamp")
        frame_id_col = seq_df_cols.get_loc("frame_id")
        track_id_col = seq_df_cols.get_loc("track_id")
        # Coords cols
        global_x_col = seq_df_cols.get_loc("x")
        global_y_col = seq_df_cols.get_loc("y")
        global_yaw_col = seq_df_cols.get_loc("obs_yaw")
        global_x0_col = seq_df_cols.get_loc("x0")
        global_x1_col = seq_df_cols.get_loc("x1")
        global_x2_col = seq_df_cols.get_loc("x2")
        global_x3_col = seq_df_cols.get_loc("x3")
        global_y0_col = seq_df_cols.get_loc("y0")
        global_y1_col = seq_df_cols.get_loc("y1")
        global_y2_col = seq_df_cols.get_loc("y2")
        global_y3_col = seq_df_cols.get_loc("y3")
        # Behavior label
        lat_behav_col = seq_df_cols.get_loc("lat_behav_label")
        lon_behav_col = seq_df_cols.get_loc("lon_behav_label")
        # Feat cols
        overline_col = seq_df_cols.get_loc("is_over_laneline")
        dist_to_ego_col = seq_df_cols.get_loc("dist_to_egocar")
        lat_dist_cnt_to_overline_col = seq_df_cols.get_loc(
            "lat_dist_obs_cnt_to_overline"
        )
        lat_dist_b1_to_overline_col = seq_df_cols.get_loc(
            "lat_dist_obs_b1_to_overline"
        )
        lat_dist_b2_to_overline_col = seq_df_cols.get_loc(
            "lat_dist_obs_b2_to_overline"
        )
        lat_dist_b3_to_overline_col = seq_df_cols.get_loc(
            "lat_dist_obs_b3_to_overline"
        )
        lat_dist_b4_to_overline_col = seq_df_cols.get_loc(
            "lat_dist_obs_b4_to_overline"
        )

        # Generating mask.
        track_id_mask = seq_df_vals[:, track_id_col] == track_id
        lcf_mask = seq_df_vals[:, frame_id_col] == lcf_id
        cur_mask = lcf_mask & track_id_mask
        his_mask = seq_df_vals[:, frame_id_col] <= lcf_id

        # Hand-crafted feature engineering
        concate_feats = []
        his_df_vals = seq_df_vals[his_mask]
        ctx_frame_ids = np.sort(np.unique(his_df_vals[:, frame_id_col]))
        his_self_mask = his_mask & track_id_mask
        his_self_vals = seq_df_vals[his_self_mask, :]
        # Complete frames to align historical dimensions
        if his_self_vals.shape[0] < len(ctx_frame_ids):
            miss_frames = len(ctx_frame_ids) - his_self_vals.shape[0]
            add_rows = np.tile(his_self_vals[-1], (miss_frames, 1))
            his_self_vals = np.vstack([his_self_vals, add_rows])

        his_phy_cols = [global_x_col, global_y_col, global_yaw_col]
        his_agent_pos = his_self_vals[:, his_phy_cols].astype("float64")

        if self.use_obs_interaction:
            his_box_cols = [
                global_x0_col,
                global_y0_col,
                global_x2_col,
                global_y2_col,
            ]
            his_phy_pos = his_df_vals[:, his_phy_cols].astype("float64")
            his_phy_box = his_df_vals[:, his_box_cols].astype("float64")
            his_phy_box = his_phy_box.reshape(-1, 2, 2)
            # Extract orthogonal distance from 3 nearest obstacles ahead and
            # 3 nearest obstacles from the rear
            his_head_nearest_box = np.zeros(len(ctx_frame_ids) * 3 * 4)
            his_rear_nearest_box = np.zeros(len(ctx_frame_ids) * 3 * 4)
            # Note: len(ctx_frame_ids) != his_self_vals.shape[0]
            for i, ctx_fid in enumerate(ctx_frame_ids):
                each_frame_mask = his_df_vals[:, frame_id_col] == ctx_fid
                seq_center = SeqCenter(
                    his_agent_pos[i, 0],  # global x
                    his_agent_pos[i, 1],  # global y
                    his_agent_pos[i, 2],  # global yaw
                )
                local_coords = TCH.global_phy_to_local_phy(
                    his_phy_pos[each_frame_mask, :2], seq_center
                )
                local_boxes = TCH.global_phy_to_local_phy(
                    his_phy_box[each_frame_mask].reshape(-1, 2), seq_center
                )
                local_boxes = local_boxes.reshape(-1, 4)
                head_mask = local_coords[:, 0] > 0
                rear_mask = local_coords[:, 0] < 0
                for mask, part_nearest_box in zip(
                    [head_mask, rear_mask],
                    [his_head_nearest_box, his_rear_nearest_box],
                ):
                    part_coords = local_coords[mask]
                    part_dist = np.sqrt(
                        part_coords[:, 0] ** 2 + part_coords[:, 1] ** 2
                    )
                    part_num = min(3, part_dist.shape[0])
                    closest_obs_idx = np.argsort(part_dist)[:part_num]
                    part_nearest_box[
                        i * 12 : i * 12 + part_num * 4
                    ] = local_boxes[closest_obs_idx].reshape(-1)
            his_nearest_box = np.hstack(
                [his_head_nearest_box, his_rear_nearest_box]
            )
            concate_feats.append(his_nearest_box)  # len += 96

        if self.use_obs_selfboxes:
            # Extract corner points of historical bounding boxes
            global_poly0_cols = [global_x0_col, global_y0_col]
            his_global_poly0 = (his_self_vals[:, global_poly0_cols]).astype(
                "float64"
            )
            global_poly1_cols = [global_x1_col, global_y1_col]
            his_global_poly1 = (his_self_vals[:, global_poly1_cols]).astype(
                "float64"
            )
            global_poly2_cols = [global_x2_col, global_y2_col]
            his_global_poly2 = (his_self_vals[:, global_poly2_cols]).astype(
                "float64"
            )
            global_poly3_cols = [global_x3_col, global_y3_col]
            his_global_poly3 = (his_self_vals[:, global_poly3_cols]).astype(
                "float64"
            )
            # Convertion from global coordinates to centric coordinates
            obs_seq_center = SeqCenter(
                his_agent_pos[-1, 0],
                his_agent_pos[-1, 1],
                his_agent_pos[-1, 2],
            )
            his_centric_bboxes = []
            for his_global_poly in [
                his_global_poly0,
                his_global_poly1,
                his_global_poly2,
                his_global_poly3,
            ]:
                his_global_poly = TCH.global_phy_to_local_phy(
                    his_global_poly, obs_seq_center
                )
                his_centric_bboxes.append(his_global_poly)
            his_centric_bboxes = np.hstack(his_centric_bboxes)
            his_centric_bboxes = his_centric_bboxes.reshape(-1)
            concate_feats.append(his_centric_bboxes)  # len += 32

        if self.use_obs_selfstates:
            obs_seq_center = SeqCenter(
                his_agent_pos[-1, 0],
                his_agent_pos[-1, 1],
                his_agent_pos[-1, 2],
            )
            his_local_coords = TCH.global_phy_to_local_phy(
                his_agent_pos, obs_seq_center
            )
            his_centric_yaws = his_agent_pos[:, 2] - his_agent_pos[0, 2]
            # Extract historical sequence of obs info
            his_ts = his_self_vals[:, stamp_col] / 1000  # convert ms to s.
            his_time_diff = np.diff(his_ts)
            first_itp_idx = np.nonzero(his_time_diff)[0].max() + 1
            # Calculate and interpolate lateral velocity
            his_lat_vels = np.ones_like(his_time_diff)
            his_lat_vels[:first_itp_idx] = (
                np.diff(his_local_coords[:, 1])[:first_itp_idx]
                / his_time_diff[:first_itp_idx]
            )
            if first_itp_idx < len(his_time_diff):
                his_lat_vels[first_itp_idx:] *= his_lat_vels[first_itp_idx - 1]
            # Calculate and interpolate longitudinal velocity
            his_lon_vels = np.ones_like(his_time_diff)
            his_lon_vels[:first_itp_idx] = (
                np.diff(his_local_coords[:, 0])[:first_itp_idx]
                / his_time_diff[:first_itp_idx]
            )
            if first_itp_idx < len(his_time_diff):
                his_lon_vels[first_itp_idx:] *= his_lon_vels[first_itp_idx - 1]
            # Calculate and interpolate yaw rate
            his_yaw_rates = np.ones_like(his_time_diff)
            his_yaw_rates[:first_itp_idx] = (
                np.diff(his_centric_yaws)[:first_itp_idx]
                / his_time_diff[:first_itp_idx]
            )
            if first_itp_idx < len(his_time_diff):
                his_yaw_rates[first_itp_idx:] *= his_yaw_rates[
                    first_itp_idx - 1
                ]
            concate_feats += [
                his_lat_vels,  # 3
                his_lon_vels,  # 3
                his_centric_yaws,  # 4
                his_yaw_rates,  # 3
            ]  # len += 13

        if len(concate_feats):
            his_centric_feats = np.hstack(concate_feats)
        else:
            his_centric_feats = None

        lcf_df_vals = seq_df_vals[cur_mask, :]
        if track_id != self.ego_track_id:
            # Extract behavior label
            lat_label = lcf_df_vals[:, lat_behav_col].item()
            lon_label = lcf_df_vals[:, lon_behav_col].item()
            try:
                lat_label = float(lat_label)
            except ValueError:
                lat_label = -1.0
            try:
                lon_label = float(lon_label)
            except ValueError:
                lon_label = -1.0
            over_laneline = lcf_df_vals[:, overline_col].item()
            dist_to_ego = lcf_df_vals[:, dist_to_ego_col].item()
            assert not np.isnan(dist_to_ego), f"{lcf_id}-{track_id}"

            lat_dist_cols = [
                lat_dist_cnt_to_overline_col,
                lat_dist_b1_to_overline_col,
                lat_dist_b2_to_overline_col,
                lat_dist_b3_to_overline_col,
                lat_dist_b4_to_overline_col,
            ]
            lat_dist_to_overline = lcf_df_vals[:, lat_dist_cols].reshape(-1)
            if np.isnan(lat_dist_to_overline.sum()):
                lat_dist_to_overline = np.full(
                    lat_dist_to_overline.shape, -1.0
                )
        else:
            lat_label, lon_label, over_laneline, dist_to_ego = (
                0.0,
                0.0,
                0.0,
                0.0,
            )
            lat_dist_to_overline = np.full(5, -1.0)

        return (
            his_centric_feats,
            over_laneline,
            dist_to_ego,
            lat_dist_to_overline,
            lat_label,
            lon_label,
        )

    def _extend_state_vectors(
        self,
        valid_track_ids: List,
        valid_over_lanelines: List,
        valid_dist_to_ego: List,
        valid_lat_dist_to_overline: List,
        valid_centric_feats: List,
    ):
        """Get agent state vectors (velocity, acceleration and yaw rate).

        Args:
            valid_track_ids (list[int]): list of the valid track ids.
            valid_over_lanelines (list[bool]): if object cutting the laneline.
            valid_dist_to_ego (list[float]): distances between objects and the
                ego car.
            valid_lat_dist_to_laneline (list[float]): distances between objects
                and the laneline they are cutting.
            valid_centric_feats (list[numpy.ndarray]): centric hand-crafted \
                features.

        Returns:
            state_vectors (np.array, [num_valid_obj, num_state_vectors]): the
                agent state vectors.
        """
        state_vectors = np.zeros(
            [len(valid_track_ids), self.num_behav_state_vectors]
        )
        for idx, (
            over_laneline,
            # dist_to_ego,
            # lat_dist_to_overline,
            centric_feats,
        ) in enumerate(
            zip(
                valid_over_lanelines,
                # valid_dist_to_ego,
                # valid_lat_dist_to_overline,
                valid_centric_feats,
            )
        ):
            state_vectors[idx, :1] = over_laneline
            # state_vectors[idx, 1] = dist_to_ego
            # state_vectors[idx, 2:7] = lat_dist_to_overline
            state_vectors[idx, 1:] = centric_feats

        if self.if_norm:
            for i in range(1, self.num_behav_state_vectors):
                tmp_sv = state_vectors[:, i]
                tmp_sv = tmp_sv / np.abs(tmp_sv).max()
                assert not np.isnan(tmp_sv.sum())
                tmp_sv *= self.norm_scale
                state_vectors[:, i] = tmp_sv

        return state_vectors


@OBJECT_REGISTRY.register
class VectorNetTrajExtractor:
    """The trajectory sequence feature extractor for VectorNet.

    This class should be used after the following transforms:
    "GenFutureTrackids", "GenStatesAndMask", "RemapObsCls",
    "GetTrajPredObjectsInfo", "SelectYawArray", "GenSeqCenter".
    """

    def __init__(
        self,
        map_origin_params: List,
        source_freq: int,
        target_freq: int,
        ego_track_id: int = -42,
        max_obs_num: int = 32,
        local_ele_seg_thr: int = 100,
        image_coordinates: str = "bev",
        reverse: bool = False,
        traj_feat_dim: int = 9,
        scale: Optional[List] = None,
        num_obs_type: Optional[int] = 3,
        traj_optional_feats: Optional[List] = None,
        valid_track_ids_key: str = "valid_track_ids",
        if_itp_traj: bool = False,
    ):
        """Initialize method.

        Args:
            map_origin_params: paramters to calculate the offset of the
                map center in the map coordinates.
            source_freq: the original trajectory point frequency of the
                used dataset.
            target_freq: the target trajectory point frequency of the
                target trajectory sequence during feature extracting.
                It cannot be smaller than `source_freq` and should be
                divisible by `source_freq`
            ego_track_id: the track id of the ego vehicle.
            max_obs_num: The maximum number of surrounding obstacle
                trajectory features.
            local_ele_seg_thr: the distance threshold of valid polyline
                segments (from the obstacle). Defaults to 100 [m].
            image_coordinates: the name of the image coordinates.
                It should be one of ["bev", "img]. The
                corresponding coordinate trans method is
                ["PhyToBEV", "PhyToImg"].
            reverse: if the direction of VCS and image coordinate system
                is on the contrary (like in the BEV scenario), reverse
                should be True.
            traj_feat_dim: the dim of trajectory feature. Including
                start_x, start_y, end_x, end_y, time_stamp, pid,
                and one-hot for type.
            scale: the scale of the extracted features. Default to None,
                which means all the scales are 1.
            num_obs_type: the number of obstacle types.
            traj_optional_feats: the optional features for obstacle
                trajectory.
            valid_track_ids_key: the key name of valid track ids.
            if_itp_traj: if interpolate context and
                future trajectories.
        """
        assert target_freq >= source_freq
        self.map_origin_params = map_origin_params
        self.ego_track_id = ego_track_id
        self.max_obs_num = max_obs_num
        self.local_ele_seg_thr = local_ele_seg_thr
        self.image_coordinates = image_coordinates
        self.sample_ratio = int(target_freq / source_freq)
        self.step_t = 1 / target_freq
        self.reverse = reverse
        self.num_obs_type = num_obs_type
        self.valid_track_ids_key = valid_track_ids_key
        self.if_itp_traj = if_itp_traj
        if traj_optional_feats is None:
            self.traj_optional_feats = []
        else:
            self.traj_optional_feats = traj_optional_feats

        if self.image_coordinates == "bev":
            self.reverse = True
            bev_ori_x, bev_ori_y, img_resolu = self.map_origin_params
            self.img_center_offset = [bev_ori_x, bev_ori_y]
            self.img_resolu = img_resolu
        elif self.image_coordinates == "img":
            self.reverse = False
            map_h, map_w, img_resolu = self.map_origin_params
            self.img_center_offset = [map_h / 2, map_w / 2]
            self.img_resolu = img_resolu
        else:
            raise ValueError(
                f"Undefined image coordinates {image_coordinates}"
            )

        # TODO (shengzhe.dai): the current traj_feat_dim is a constant value
        # because we only support basic vector features here. If we want to
        # add more features, wescale will change the corresponding code.
        self.traj_feat_dim = traj_feat_dim
        if scale is None:
            self.traj_feat_ = [1 for _ in range(self.traj_feat_dim)]
        else:
            self.traj_feat_scale = scale

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. "last_context_frame_id"
            2. "track_ids"
            3. "states", "context_states"
            4. "masks", "context_masks"
            5. "agent_classes".
            6. "ori_seq_df", "valid_track_ids"
            7. "seq_center"

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample includes new keys "struct_traj_feats",
                "struct_traj_masks", "struct_num_trajs", "itp_fut_obs_trajs",
                "itp_fut_obs_masks", struct_traj_pts_masks:
                - The shape of "struct_traj_feats" is:
                    [num_obs, max_obs_num, traj_len, num_feats].
                - The shape of "struct_traj_masks" is [num_obs, max_obs_num].
                - The shape of "struct_num_trajs" is [num_obs].
                - The shape of "itp_fut_obs_trajs" is
                    [num_valid_obs, itp_traj_len, 2].
                - The shape of "itp_fut_obs_masks" is
                    [num_valid_obs, itp_traj_len].
                - The shape of "struct_traj_pts_masks" is
                    [num_obs, max_obs_num, traj_len]
        """
        # 1. Get image coordinates of all obstacles.
        seq_center = sample["seq_center"]
        full_track_ids = sample["track_ids"]
        full_future_trajs = sample["states"]
        full_future_masks = sample["masks"]
        full_ctx_trajs = sample["context_states"]
        full_ctx_masks = sample["context_masks"]
        agent_classes = sample["agent_classes"]
        seq_df = sample["ori_seq_df"]
        track_yaw_dict = sample["track_yaw_dict"]
        lctx_mask = sample["lctx_mask"]
        obs_width = sample["obs_width"]
        obs_length = sample["obs_length"]
        # Get the masks for context and future frames. If freq_ratio > 1,
        # need to sample.
        if "sampled_ctx_mask" in sample:
            ctx_mask = sample["sampled_ctx_mask"]
        else:
            ctx_mask = sample["ctx_mask"]
        img_coor_info = GetTrajPredObjectsInfo.extract_obs_img_coords(
            seq_df,
            seq_center,
            full_future_trajs,
            full_future_masks,
            full_ctx_trajs,
            full_ctx_masks,
            full_track_ids,
            full_track_ids,
            agent_classes,
            ctx_mask,
            lctx_mask,
            track_yaw_dict,
            self.ego_track_id,
            self.reverse,
        )
        all_track_ids = img_coor_info[1]
        img_coords = img_coor_info[4]

        # 2. Get ctx trajectories of all obstacles.
        center_ids = sample[self.valid_track_ids_key]
        if self.if_itp_traj:
            all_ctx_trajs, all_ctx_masks = self._interpolate_trajectories(
                full_ctx_trajs,
                full_ctx_masks,
                full_track_ids,
                all_track_ids,
                self.sample_ratio,
                is_historical=True,
            )
        else:
            idx_all_track_ids = np.isin(full_track_ids, all_track_ids)
            all_ctx_trajs = full_ctx_trajs[idx_all_track_ids]
            all_ctx_masks = full_ctx_masks[idx_all_track_ids]

        assert (
            len(all_ctx_trajs)
            == len(all_ctx_masks)
            == len(all_track_ids)
            == len(img_coords)
        ), (
            "The lengths of the obstacle trajectory information are not "
            "equal, fail to extract structural traj segment features."
        )

        # 3. Get future trajectories of all target obstacles.
        if self.if_itp_traj:
            (
                all_obs_fut_trajs,
                all_obs_fut_masks,
            ) = self._interpolate_trajectories(
                full_future_trajs,
                full_future_masks,
                full_track_ids,
                center_ids,
                self.sample_ratio,
                is_historical=False,
            )
        else:
            idx_center_track_ids = np.isin(full_track_ids, center_ids)
            all_obs_fut_trajs = full_future_trajs[idx_center_track_ids]
            all_obs_fut_masks = full_future_masks[idx_center_track_ids]

        # 4. Trans each obstacle image coordinates to VCS coordinates.
        vcs_coords = {}
        vcs_trajs = np.zeros_like(all_ctx_trajs)
        offset_x, offset_y = self.img_center_offset
        img_resolu = self.img_resolu
        for idx, (traj, track_id, (img_x, img_y, img_yaw)) in enumerate(
            zip(all_ctx_trajs, all_track_ids, img_coords)
        ):
            if self.reverse:
                vcs_x = -(img_x - offset_x / img_resolu) * img_resolu
                vcs_y = -(img_y - offset_y / img_resolu) * img_resolu
                vcs_yaw = img_yaw - np.pi
            else:
                vcs_x = (img_x - offset_x / img_resolu) * img_resolu
                vcs_y = (img_y - offset_y / img_resolu) * img_resolu
                vcs_yaw = img_yaw
            vcs_coords[track_id] = [vcs_x, vcs_y, vcs_yaw]

            trans_x, trans_y = Affine2D.coord_rotate(
                traj[:, 0], traj[:, 1], -vcs_yaw
            )
            trans_x, trans_y = Affine2D.coord_translate(
                trans_x, trans_y, -vcs_x, -vcs_y
            )
            vcs_trajs[idx, :, 0] = trans_x
            vcs_trajs[idx, :, 1] = trans_y

        # 5. Extract traj structural features for each valid trajs.
        vec_mask = np.logical_and(all_ctx_masks[:, :-1], all_ctx_masks[:, 1:])
        vec_mask = vec_mask[:, :, None].repeat(self.traj_feat_dim, axis=2)
        num_obs, len_seg = vec_mask.shape[:2]
        assert (
            len(self.traj_optional_feats) > 0
        ), "optional feats cannot be empty"
        const_feats = []
        for feat_name in self.traj_optional_feats:
            if feat_name == "pid":
                pid = np.arange(1, num_obs + 1, 1)[:, None, None].repeat(
                    len_seg, axis=1
                )
                const_feats.append(pid)
            elif feat_name == "timestamp":
                timestamp = (
                    np.arange(1 - len_seg, 1, 1)[None, :, None].repeat(
                        num_obs, axis=0
                    )
                    * self.step_t
                )
                const_feats.append(timestamp)
            elif feat_name == "obstacle_class":
                obstacle_class = np.zeros(
                    (num_obs, len_seg, self.num_obs_type)
                )
                for i in range(num_obs):
                    obstacle_class[i, :, agent_classes[i]] = 1
                const_feats.append(obstacle_class)
            elif feat_name == "obs_width":
                obs_width = np.array(obs_width)[:, None, None].repeat(
                    len_seg, axis=1
                )
                const_feats.append(obs_width)
            elif feat_name == "obs_length":
                obs_length = np.array(obs_length)[:, None, None].repeat(
                    len_seg, axis=1
                )
                const_feats.append(obs_length)
            else:
                raise TypeError("unsupported traj feat_name")
        const_feats = np.concatenate(const_feats, axis=-1)

        all_traj_feats = []
        all_num_traj = []
        all_traj_masks = []
        all_traj_pts_masks = []
        all_pad_track_ids = []
        obs_coors = vcs_trajs.reshape([-1, 2])
        for track_id in center_ids:
            vcs_x, vcs_y, vcs_yaw = vcs_coords[track_id]
            trans_x, trans_y = Affine2D.coord_translate(
                obs_coors[:, 0], obs_coors[:, 1], vcs_x, vcs_y
            )
            trans_x, trans_y = Affine2D.coord_rotate(trans_x, trans_y, vcs_yaw)
            trans_xy = np.stack([trans_x, trans_y], axis=1).reshape(
                [num_obs, -1, 2]
            )
            trans_feats = np.concatenate(
                [trans_xy[:, :-1, :], trans_xy[:, 1:, :], const_feats], axis=-1
            )
            trans_feats[np.where(np.logical_not(vec_mask))] = 0

            # sort by distance
            obs_diff = np.sqrt(np.sum(trans_feats[:, -1, 2:3] ** 2, axis=-1))
            sort_idx = list(np.argsort(obs_diff))
            target_idx = all_track_ids.index(track_id)
            sort_idx.remove(target_idx)
            sort_idx = np.array([target_idx] + sort_idx)
            trans_feats = trans_feats[sort_idx, :, :]
            obs_diff = obs_diff[sort_idx]
            vec_mask_tmp = vec_mask[sort_idx]
            trans_feats = trans_feats[np.where(obs_diff < 100)[0], :, :]
            vec_mask_tmp = vec_mask_tmp[np.where(obs_diff < 100)[0], :, :]
            track_id_tmp = np.array(all_track_ids)[sort_idx]
            track_id_tmp = track_id_tmp[np.where(obs_diff < 100)[0]]

            # Perform padding or sampling.
            if len(trans_feats) < self.max_obs_num:
                num_valid = len(trans_feats)
                num_pad = self.max_obs_num - len(trans_feats)
                trans_feats = np.concatenate(
                    (
                        trans_feats,
                        np.zeros([num_pad, len_seg, self.traj_feat_dim]),
                    ),
                    axis=0,
                )
                valid_mask = np.concatenate(
                    [np.ones([num_valid]), np.zeros([num_pad])], axis=0
                )
                valid_pts_mask = np.concatenate(
                    (
                        vec_mask_tmp[:, :, 0],
                        np.zeros([num_pad, len_seg]),
                    ),
                    axis=0,
                )
                pad_track_ids = np.concatenate(
                    (
                        track_id_tmp,
                        np.zeros([num_pad]),
                    ),
                    axis=0,
                )
                all_num_traj.append(num_valid)
                all_traj_masks.append(valid_mask)
                all_traj_feats.append(trans_feats)
                all_traj_pts_masks.append(valid_pts_mask)
                all_pad_track_ids.append(pad_track_ids)
            else:
                all_num_traj.append(self.max_obs_num)
                all_traj_masks.append(np.ones([self.max_obs_num]))
                trans_feats = trans_feats[: self.max_obs_num, :, :]
                all_traj_feats.append(trans_feats)
                all_traj_pts_masks.append(
                    vec_mask_tmp[: self.max_obs_num, :, 0]
                )
                all_pad_track_ids.append(track_id_tmp[: self.max_obs_num])

        all_traj_feats = np.stack(all_traj_feats)
        all_traj_masks = np.stack(all_traj_masks)
        all_traj_pts_masks = np.stack(all_traj_pts_masks)
        all_pad_track_ids = np.stack(all_pad_track_ids)
        sample["struct_traj_feats"] = all_traj_feats
        sample["struct_traj_masks"] = all_traj_masks
        sample["struct_traj_pts_masks"] = all_traj_pts_masks
        sample["struct_num_trajs"] = all_num_traj
        sample["itp_fut_obs_trajs"] = all_obs_fut_trajs
        sample["itp_fut_obs_masks"] = all_obs_fut_masks
        sample["traj_feat_scale"] = self.traj_feat_scale
        sample["struct_num_pad_track_ids"] = all_pad_track_ids
        return sample

    @staticmethod
    def _interpolate_trajectories(
        full_trajs,
        full_masks,
        full_track_ids,
        valid_track_ids,
        sample_ratio: int,
        is_historical: bool = True,
    ):
        """Interpolate trajectories based on the given sampling ratio.

        Args:
            full_trajs (np.array, [num_obs, traj_len, 2]): the original trajs.
            full_masks (np.array, [num_obs, traj_len]): the original masks.
            full_track_ids (List, [num_obs]): all track ids.
            valid_track_ids (List, [num_valid_obs]): all track ids that needs
                to perform interpolation.
            sample_ratio (int): the sampling ratio. It should be an > 1 int.
            is_historical (bool, optional): whether the trajectories are
                historical trajectories.

        Returns:
            itp_trajs (np.array, [num_valid_obs, itp_traj_len, 2]): the inter-
                polated trajectories.
            itp_masks (np.array, [num_valid_obs, itp_traj_len]): the
                interpolated masks.
        """
        itp_trajs, itp_masks = [], []
        sr = sample_ratio
        for track_id in valid_track_ids:
            track_id_index = full_track_ids.index(track_id)
            agent_traj = full_trajs[track_id_index]
            agent_mask = full_masks[track_id_index]

            # If needed, interpolate trajectories to the goal frequency.
            # Note: all agent trajs are in the obstacle centric coordinates.
            if sr > 1:
                # - if interpolate future trajs, insert the origin point
                #   before interpolation.
                if not is_historical:
                    agent_traj = np.concatenate(
                        (np.array([[0, 0]]), agent_traj), axis=0
                    )
                    agent_mask = np.concatenate(
                        (agent_mask[0:1], agent_mask), axis=0
                    )
                # - perform interpolation.
                # -- trajectories.
                extend_traj, extend_mask = [], []
                for start, end in zip(agent_traj[:-1], agent_traj[1:]):
                    ext_seg = np.stack(
                        [
                            np.linspace(start[0], end[0], sr + 1),
                            np.linspace(start[1], end[1], sr + 1),
                        ],
                        axis=1,
                    )
                    if is_historical:
                        extend_traj.append(ext_seg[:sr])
                    else:
                        extend_traj.append(ext_seg[1:])
                # -- masks.
                for start, end in zip(agent_mask[:-1], agent_mask[1:]):
                    ext_mask = np.linspace(start, end, sr + 1)
                    if is_historical:
                        extend_mask.append(ext_mask[:sr])
                    else:
                        extend_mask.append(ext_mask[1:])
                # - if interpolate ctx trajs, insert the origin point after
                #   interpolation.
                if is_historical:
                    extend_traj.append(np.array([[0, 0]]))
                    extend_mask.append(np.array([end]))
                extend_traj = np.concatenate(extend_traj, axis=0)
                extend_mask = np.concatenate(extend_mask, axis=0)
                itp_trajs.append(extend_traj)
                itp_masks.append(extend_mask == 1)
            else:
                itp_trajs.append(agent_traj)
                itp_masks.append(agent_mask == 1)
        itp_trajs = np.stack(itp_trajs)
        itp_masks = np.stack(itp_masks)
        return itp_trajs, itp_masks


@OBJECT_REGISTRY.register
class GetNavinetmapInfo:
    """Get road elements relations from navigation infomation.

    The navigation infomation such as road arrows and stoplines are useful
    in predicting correct modal of an agent at the intersection. We save
    these kinds of road elements' infomation by this transform.

    This transform requires the sample from dataset to have the keys
    `navi_info`. Therefore, the dataset should be
    `AutoMultiAgentNaviDataset`.
    Besides, this class should be used after the following transforms:
    "GetTrajPredObjectsInfo", "SampleNaviTrajAnchor".

    To use, the user should construct a `GetNavinetmapInfo` instance with the
    required initialization parameters. After instantiation, the callable
    function will return a updated dict that contains one new element and one
    updated element as below: \
        1. `valid_block_marks_dict` (Dict): the bounding marks of each agent \
            in `valid_track_id`. \
        2. `valid_stopline_dis_dict` (Dict): distance to the bounding \
            stopline of each obstacle in `valid_track_id`. \
        3. `state_vectors` (np.array, [num_obj, num_state_vectors + \
            num_extend_state_vectors]): update road elements info to \
            `state_vectors` if `use_extend_state_vectors` is true.
    """

    # The marks type is defined in 'hat/core/traj_pred_typing.py'.
    MIXED_MARK_TO_SEPARATE_MARK = OrderedDict(
        {
            18: [1, 2, 3],
            19: [1, 2, 7],
            20: [3, 7],
            4: [1, 2],
            5: [2, 3],
            6: [1, 3],
            8: [2, 7],
            9: [1, 7],
        }
    )
    # Following types are useless for vehicle steering, we don't use them.
    DELETE_TYPE_IDS = [0, 2, 13, 14, 21, 22, 23, 24, 25, 26, 27]
    MARK_TYPE_IDS = list(set(LANE_MARK_TYPES.values()) - set(DELETE_TYPE_IDS))

    def __init__(
        self,
        ego_track_id: int = -42,
        use_extend_state_vectors: bool = False,
        if_norm_marks_label: bool = True,
        if_norm_distance: bool = True,
        norm_scale: int = 5,
        max_area_x: int = 100,
    ):
        """Initialize method.

        Args:
            ego_track_id: the track id of the ego vehicle.
            use_extend_state_vectors: whether to use navi features as
                extend state_vectors.
            if_norm_marks_label: whether to adjust the lane marks label to
                the same range by normalization. Default to True.
            if_norm_distance: whether to adjust the distance between obstacles
                and stoplines to the same range as `state_vectors`. Default
                to True.
            norm_scale: normalized range of the navi features. Default to 5
                (the same as original state_vectors).
            max_area_x: the furthest distance between the stopline and
                associated obstacle. Default to 100.
        """
        self.ego_track_id = ego_track_id
        self.use_extend_state_vectors = use_extend_state_vectors
        self.if_norm_marks_label = if_norm_marks_label
        self.if_norm_distance = if_norm_distance
        self.norm_scale = norm_scale
        self.max_area_x = max_area_x

    def __call__(self, sample: dict):
        """Callable function.

        The following keys are required in `sample`:
            1. `valid_track_ids`.
            2. `valid_lane_chain_dict`.
            3. `navi_info`.
            4. `struct_road`.
            5. `track_yaw_dict`.
            6. `seq_df`.
            7. `lcf_timestamp`.

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample with added 'valid_block_marks_dict'
                (corresponding to `valid_lane_chain_dict`, bind lane marks
                to each lane in the lane chain), 'valid_stopline_dis_dict',
                and updated `state_vectors`.
        """
        valid_track_ids = sample["valid_track_ids"]
        valid_lane_block_dict = sample["valid_lane_chain_dict"]
        navi_info = sample["navi_info"]
        struct_road = sample["struct_road"]
        track_yaw_dict = sample["track_yaw_dict"]
        seq_df = sample["seq_df"]
        lcf_timestamp = sample["lcf_timestamp"]

        if self.if_norm_marks_label:
            mark_types_dict = self._normalize_marks(
                self.MARK_TYPE_IDS, self.norm_scale
            )
        else:
            mark_types_dict = {
                mark_type: mark_type for mark_type in self.MARK_TYPE_IDS
            }

        phy_ego_yaw = track_yaw_dict[self.ego_track_id][-1]
        lcf_seq_df = seq_df.loc[seq_df["timestamp"] == lcf_timestamp]
        lcf_track_seq_df = lcf_seq_df.loc[
            lcf_seq_df["track_id"] == self.ego_track_id
        ]
        valid_block_marks_dict = {}
        valid_marks_dict = {}
        valid_stopline_dis_dict = {}
        stoplines = struct_road.get("stopline")

        for track_id in valid_track_ids:
            valid_mark, valid_block_marks = self._add_mark_info(
                track_id,
                mark_types_dict,
                navi_info,
                valid_lane_block_dict,
                valid_block_marks_dict,
                only_check_lane_of_obs=False,
            )
            valid_block_marks_dict[track_id] = valid_block_marks
            valid_marks_dict[track_id] = valid_mark

            lcf_track_seq_df = lcf_seq_df.loc[
                lcf_seq_df["track_id"] == track_id
            ]
            obs_center_x = lcf_track_seq_df["vcs_obs_x"].item()
            obs_center_y = lcf_track_seq_df["vcs_obs_y"].item()
            obs_vcs_yaw = normalize_yaw(
                track_yaw_dict[track_id][-1] - phy_ego_yaw
            )
            distance_to_stopline = self._add_stoplines_info(
                stoplines,
                obs_vcs_yaw,
                obs_center_x,
                obs_center_y,
            )
            if self.if_norm_distance:
                distance_to_stopline = np.log(
                    np.maximum(distance_to_stopline, 1e-2)
                )
                distance_to_stopline = np.clip(
                    distance_to_stopline, -self.norm_scale, self.norm_scale
                )
            valid_stopline_dis_dict[track_id] = distance_to_stopline

        extend_state_vectors = []
        for track_id in valid_track_ids:
            extend_state_vectors.append(
                [
                    valid_marks_dict[track_id],
                    valid_stopline_dis_dict[track_id],
                ]
            )
        if self.use_extend_state_vectors and len(valid_track_ids) > 0:
            sample["state_vectors"] = np.concatenate(
                (sample["state_vectors"], extend_state_vectors), axis=1
            )
        sample["valid_block_marks_dict"] = valid_block_marks_dict
        sample["valid_stopline_dis_dict"] = valid_stopline_dis_dict
        return sample

    def _merge_marks(
        self,
        lane_marks_list: List,
        mark_types: List,
    ):
        """Merge all possible marks in one lane together.

        Sometimes there are multiple road arrows (here we focus on lane marks)
        in a single lane. For an obstacle on the lane, it is ok to follow the
        directions of either marks in front of itself. And some kinds of lane
        marks can be merged to a composite mark. After merging these marks, we
        can easier bind the vehicle to a particular type of mark.

        Examples:
        1. types "arrow_left" + "arrow_forward"
            -> "arrow_left_and_forward"
        2. types "arrow_u_turn_and_left" + "arrow_left_and_forward"
            -> "arrow_forward_and_u_turn_and_left"
        3. type "arrow_left"
            -> "arrow_left"

        Args:
            lane_marks_list: all marks' type on a lane.
            mark_types: the marks type which can be merged.

        Returns:
            lane_marks_list (List): updated lane_marks_list.
        """
        need_merge = True
        for mark_type in mark_types:
            need_merge = need_merge and (lane_marks_list == mark_type).any()
        if need_merge:
            lane_marks_list = np.setdiff1d(lane_marks_list, mark_types)
            lane_marks_list = np.append(
                lane_marks_list,
                self._get_key(self.MIXED_MARK_TO_SEPARATE_MARK, mark_types),
            )
        return lane_marks_list

    def _split_mark(
        self,
        lane_marks: List,
        mark_types: List,
    ):
        """Split all possible marks in one lane together.

        As opposed to _merge_marks, one composite lane mark can be splited into
        multiple lane marks. Before merging marks, we can first split them.

        Examples:
        1. type "arrow_left_and_forward"
            -> "arrow_left" + "arrow_forward"
        2. type "arrow_forward_and_left_and_right"
            -> "arrow_forward" + "arrow_left" + "arrow_right"
        3. type "arrow_left"
            -> "arrow_left"

        Args:
            lane_marks: all marks' type on a lane.
            mark_types: the marks' type which can be splited.

        Returns:
            lane_marks (List): updated lane_marks.
        """
        for mark_type in mark_types:
            need_split = (lane_marks == mark_type).any()
            if need_split:
                lane_marks = np.setdiff1d(lane_marks, mark_type)
                lane_marks = np.append(
                    lane_marks, self.MIXED_MARK_TO_SEPARATE_MARK[mark_type]
                )
        return lane_marks

    def _update_lane_marks(
        self,
        lane_marks: List,
    ):
        """Update all possible types of mark in one lane.

        Args:
            lane_marks: all marks' type on a lane.

        Returns:
            lane_marks (List): updated lane_marks.
        """
        lane_marks = np.unique(lane_marks)
        lane_marks = np.setdiff1d(lane_marks, self.DELETE_TYPE_IDS)
        original_lane_marks_list = lane_marks

        if len(lane_marks) <= 1:
            return lane_marks

        lane_marks = self._split_mark(
            lane_marks, list(self.MIXED_MARK_TO_SEPARATE_MARK.keys())
        )
        for lanes in self.MIXED_MARK_TO_SEPARATE_MARK.values():
            lane_marks = self._merge_marks(lane_marks, lanes)

        lane_marks = np.setdiff1d(lane_marks, self.DELETE_TYPE_IDS)

        return lane_marks if len(lane_marks) <= 1 else original_lane_marks_list

    def _get_key(self, mark_dict, value):
        """Get key by the value of the dict.

        Args:
            mark_dict (Dict): a dict.
            value (List): split lane mark list.

        Returns:
            key (List): valid mixed mark type warpped into a list.
        """
        return [k for k, v in mark_dict.items() if v == value]

    def _normalize_marks(self, mark_types, scale):
        """Normalize mark types into the range we want.

        Args:
            mark_types (List): all marks' type on a lane.
            scale (int): adjust the mark type to the same range as the original
                state_vectors by scale.

        Returns:
            mark_norm_dict (Dict): normalized mark types.
        """
        d_min, d_max = np.min(mark_types), np.max(mark_types)
        norm_types = -scale + 2 * scale / (d_max - d_min) * (
            mark_types - d_min
        )
        mark_norm_dict = {}
        for mark, norm_mark in zip(self.MARK_TYPE_IDS, norm_types):
            mark_norm_dict[mark] = norm_mark
        return mark_norm_dict

    def _add_mark_info(
        self,
        track_id: float,
        mark_types_dict: dict,
        navi_info: dict,
        valid_lane_block_dict: dict,
        valid_block_marks_dict: dict,
        only_check_lane_of_obs: bool = False,
    ):
        """Add binding relationship between marks and obstacles.

        Args:
            track_id: obstacle track id.
            mark_types_dict: reasonable lane mark types.
            navi_info: navigation infomation.
            valid_lane_block_dict: valid lane blocks of all valid track.
            valid_block_marks_dict: mark types bound to all lanes of
                lane blocks.
            only_check_lane_of_obs: only bind mark type only on current lane
                or bind mark type on succ lanes of the lane block when
                current lane has no mark.

        Returns:
            mark_type (float): normalized mark type bound to target track.
            valid_block_marks_dict (Dict): updated valid_block_marks_dict.
        """
        lane_chains = valid_lane_block_dict.get(track_id)
        valid_block_marks_dict[track_id] = []
        if not lane_chains:
            return (
                mark_types_dict[LANE_MARK_TYPES["no_mark"]],
                valid_block_marks_dict[track_id],
            )
        for lane_chain in lane_chains:
            lane_chain_marks = []
            for lane in lane_chain:
                lane_marks = []
                lane_info = navi_info.get(lane)
                if lane_info:
                    lane_marks_info = lane_info.get("lane_mark")
                    if lane_marks_info:
                        lane_marks = [
                            mark_info.get("type")
                            for mark_info in lane_marks_info.values()
                        ]
                        lane_marks = self._update_lane_marks(lane_marks)
                        lane_marks = [
                            mark_types_dict[mark] for mark in lane_marks
                        ]

                lane_chain_marks.append(
                    list(lane_marks)
                    if len(lane_marks) > 0
                    else [mark_types_dict[LANE_MARK_TYPES["no_mark"]]]
                )
            valid_block_marks_dict[track_id].append(lane_chain_marks)

        # valid_block_marks_dict[track_id]: [num_block, num_lane, num_mark]
        if only_check_lane_of_obs:
            return (
                valid_block_marks_dict[track_id][0][0][0],
                valid_block_marks_dict[track_id],
            )
        else:  # also check succ lanes
            is_add = False
            for lane in valid_block_marks_dict[track_id][0]:
                mark = lane[0]
                if mark != mark_types_dict[LANE_MARK_TYPES["no_mark"]]:
                    is_add = True
                    return mark, valid_block_marks_dict[track_id]
            if not is_add:
                return (
                    mark_types_dict[LANE_MARK_TYPES["no_mark"]],
                    valid_block_marks_dict[track_id],
                )

    def _add_stoplines_info(
        self,
        stoplines,
        obs_yaw: float,
        obs_center_x: float,
        obs_center_y: float,
        obs_radius: int = 1,
        filtered_obs_angle: float = 1 / 18 * np.pi,
        filtered_stopline_angle: float = 1 / 9 * np.pi,
        filtered_around_stopline_obs_angle: float = 1 / 9 * np.pi,
    ):
        """Get min distance between the obstacle and the stoplines.

        By using stoplines' coordinate info extracted from `struct_road`
        and the obstacle's coordinate info, we can calculate whether some
        line segments intersect and get their distance. We take these
        infomation as new features.

        The details of bounding logic refer to:
        https://horizonrobotics.feishu.cn/docs/doccncaS2UKVZnZpyli31c4FXFe
        (2022.05.24 section.1)

        Args:
            stoplines: stoplines' coordinates in ego-agent's VCS system.
            obs_yaw: obstacle's yaw in ego-agent's VCS system.
            obs_center_x: obstacle's x-coordinate in ego-agent's VCS system.
            obs_center_y: obstacle's y-coordinate in ego-agent's VCS system.
            obs_radius: radius to determine whether the obstacle is in
                stopline area.
            filtered_obs_angle: the angle between the obstacle
                and ego-agent to determine whether the obstacle needs to
                be filtered. Default to 10°.
            filtered_stopline_angle: the angle between the stopline and
                vertical direction of the ego-agent to determine whether the
                stopline needs to be filtered. Default to 20°.
            filtered_around_stopline_obs_angle: the angle between the stoline
                and the obstacle to determine whether the obstacle needs to
                be filtered. Defalut to 20°.

        Returns:
            min_distance_to_stopline (float): distance from the center
            of obstacle to the center of the closest stopline.
        """
        min_distance_to_stopline = np.inf
        # Filter obstacles in the same direction as the ego-agent.
        if abs(obs_yaw) < filtered_obs_angle:
            return min_distance_to_stopline

        # Shape of stoplines [num_stoplines, num_polylines_seg, 3, 2]
        for stopline in stoplines:
            stopline_x1, stopline_y1 = stopline[0][0][0], stopline[0][1][0]
            stopline_x2, stopline_y2 = stopline[-1][0][0], stopline[-1][1][0]
            delta_x = stopline_x2 - stopline_x1
            delta_y = stopline_y2 - stopline_y1
            if delta_x == 0 or delta_y == 0:
                continue
            stopline_k = delta_y / delta_x

            stopline_yaw = math.atan(stopline_k)  # [-pi/2, pi/2]
            stopline_yaw = (
                stopline_yaw if stopline_yaw >= 0 else stopline_yaw + np.pi
            )
            # Filter stoplines perpendicular to the forward direction of ego
            # agent.
            if abs(stopline_yaw - np.pi / 2) < filtered_stopline_angle:
                continue

            stopline_area = self._get_effective_stopline_area(
                stopline_x1,
                stopline_y1,
                stopline_x2,
                stopline_y2,
                stopline_k,
                farthest_distance=self.max_area_x / 2,
            )

            obs_in_stopline_area = False
            for obs_radius_x in (-obs_radius, obs_radius):
                for obs_radius_y in (-obs_radius, obs_radius):
                    obs_in_stopline_area = (
                        obs_in_stopline_area
                        or is_point_in_convex_polygon(
                            obs_center_x + obs_radius_x,
                            obs_center_y + obs_radius_y,
                            stopline_area[:, 0],
                            stopline_area[:, 1],
                        )
                    )

            are_on_either_side = self.if_obs_and_ego_on_either_side(
                stopline_x1,
                stopline_y1,
                stopline_x2,
                stopline_y2,
                obs_center_x,
                obs_center_y,
            )

            if are_on_either_side and obs_in_stopline_area:
                obs_yaw_positive = obs_yaw if obs_yaw >= 0 else obs_yaw + np.pi
                # Filter obstacles running parallel to the stopline
                stopline_obs_angle = abs(stopline_yaw - obs_yaw_positive)
                if stopline_obs_angle < filtered_around_stopline_obs_angle:
                    continue
                stopline_center_x = (stopline_x1 + stopline_x2) / 2
                stopline_center_y = (stopline_y1 + stopline_y2) / 2
                distance_to_stopline_by_points = math.sqrt(
                    (stopline_center_x - obs_center_x) ** 2
                    + (stopline_center_y - obs_center_y) ** 2
                )
            else:
                distance_to_stopline_by_points = np.inf
            if distance_to_stopline_by_points < min_distance_to_stopline:
                min_distance_to_stopline = distance_to_stopline_by_points
        return min_distance_to_stopline

    def _get_effective_stopline_area(
        self,
        stopline_x1: float,
        stopline_y1: float,
        stopline_x2: float,
        stopline_y2: float,
        stopline_k: float,
        farthest_distance: float,
    ):
        """Cut an effective area for the stopline.

        When analysis the correlation between the target stopline and
        obstacles, we only focus on the obstacles around stoplines.

        Args:
            stopline_x1: the start x-coordinate of stopline segment.
            stopline_y1: the start y-coordinate of stopline segment.
            stopline_x2: the end x-coordinate of stopline segment.
            stopline_y2: the end y-coordinate of stopline segment.
            stopline_k: stopline's slope.
            farthest_distance: the distance to extend to each side of
                the stopline in x direction.

        Returns:
            stopline_area (np.array, [4, 2]): a rectangular area
                calculated by stopline segment's coords.
        """
        laneline_k = -1 / stopline_k
        laneline_b1 = stopline_y1 - laneline_k * stopline_x1
        laneline_b2 = stopline_y2 - laneline_k * stopline_x2
        stopline_area_x1 = stopline_x1 + farthest_distance
        stopline_area_y1 = laneline_k * stopline_area_x1 + laneline_b1
        stopline_area_x2 = stopline_x2 + farthest_distance
        stopline_area_y2 = laneline_k * stopline_area_x2 + laneline_b2
        stopline_area_x3 = stopline_x1 - farthest_distance
        stopline_area_y3 = laneline_k * stopline_area_x3 + laneline_b1
        stopline_area_x4 = stopline_x2 - farthest_distance
        stopline_area_y4 = laneline_k * stopline_area_x4 + laneline_b2
        stopline_area = np.array(
            [
                [stopline_area_x1, stopline_area_y1],
                [stopline_area_x2, stopline_area_y2],
                [stopline_area_x4, stopline_area_y4],
                [stopline_area_x3, stopline_area_y3],
            ]
        )
        return stopline_area

    def if_obs_and_ego_on_either_side(
        self,
        stopline_x1: float,
        stopline_y1: float,
        stopline_x2: float,
        stopline_y2: float,
        obs_center_x: float,
        obs_center_y: float,
    ):
        """Check if obs and ego-agent are on either side of stopline.

        Bring the center coord of ego-agent and obstacle into the analytical
        formula (Ax+By+C=0) of the stopline to get two values, then multiply
        two values. If the result is less than zero, they are on either side.

        Args:
            stopline_x1: the start x-coordinate of stopline segment.
            stopline_y1: the start y-coordinate of stopline segment.
            stopline_x2: the end x-coordinate of stopline segment.
            stopline_y2: the end y-coordinate of stopline segment.
            obs_center_x: obstacle's x-coordinate.
            obs_center_y: obstacle's y-coordinate.

        Returns:
            on_either_side (bool): the result of the positional relationship
                between the obstacle and ego-agent.
        """
        A = stopline_y2 - stopline_y1
        B = stopline_x1 - stopline_x2
        C = stopline_x2 * stopline_y1 - stopline_x1 * stopline_y2
        ego_result = C  # coord of ego-agent in VCS: (0, 0)
        obs_result = A * obs_center_x + B * obs_center_y + C
        on_either_side = ego_result * obs_result < 0
        return on_either_side


@OBJECT_REGISTRY.register
class GetObstacleSafeArea:
    """Get safe area of all obstacles.

    This transform should be used after `GenFutureTrackids`, `RemapObsCls`,
    `GetSeqDataFrameMask`, `SelectYawArray` and `GenSeqCenter`.
    """

    SUPPORTED_EXP_TRAJ_METHODS = ["CV"]

    def __init__(
        self,
        traj_len: int = 12,
        expand_traj_method: str = "CV",
        expand_base: float = 4,
        expand_ratio: float = 0.2,
        use_his_traj: bool = False,
        reverse: bool = True,
        img_params: tuple = (72.4, 51.2, 0.2),
    ):
        """Initialize method.

        Args:
            traj_len: the length of the future trajectory.
            expand_traj_method: the method to expand basic trajectories for
                safe areas.
            expand_base: the base width of the safe area. It is the width
                (perpendicular to the trajectory gradient) at the first
                future point.
            expand_ratio: the expanding parameters. The safe area width at
                the i-th future point is
                [1 + (i-1) * expand_ratio] * expand_base
            use_his_traj: whether to use historical trajectories in the safe
                areas.
            reverse: if the direction of VCS and image coordinate system
                is on the contrary (like in the BEV scenario), reverse
                should be True.
            img_params: the parameters to trans coordinates from the VCS
                coordinates to the image coordinates. It contains three
                values (origin_x, origin_y, resolution).
        """
        self.traj_len = traj_len
        self.expand_base = expand_base
        self.expand_ratio = expand_ratio
        self.use_his_traj = use_his_traj
        assert (
            expand_traj_method in self.SUPPORTED_EXP_TRAJ_METHODS
        ), f"Unsupported trajectory expanding method {expand_traj_method}."
        self.expand_traj_method = expand_traj_method
        self.reverse = reverse
        self.bev_origin_x, self.bev_origin_y, self.resolution = img_params
        bev_image_origin_x = int(self.bev_origin_x / self.resolution)
        bev_image_origin_y = int(self.bev_origin_y / self.resolution)
        self.seq_center_img_offset = np.array(
            [bev_image_origin_x, bev_image_origin_y]
        )

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. "seq_df"
            2. "track_ids"
            3. "agent_classes"
            4. "sampled_ctx_mask" or "ctx_mask"
            5. "track_yaw_dict".
            6. "seq_center"

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample includes new keys "safe_area",
        """
        track_ids = sample["track_ids"]
        agent_classes = sample["agent_classes"]
        seq_df = sample["seq_df"]
        if "sampled_ctx_mask" in sample:
            ctx_frame_mask = sample["sampled_ctx_mask"]
        else:
            ctx_frame_mask = sample["ctx_mask"]
        track_yaw_dict = sample["track_yaw_dict"]
        seq_center = sample["seq_center"]

        vals = seq_df.values
        cols = seq_df.columns
        track_id_col = cols.get_loc("track_id")
        phy_x_col = cols.get_loc("x")
        phy_y_col = cols.get_loc("y")

        safe_area = {}
        for track_id, agent_cls in zip(track_ids, agent_classes):
            track_id_mask = vals[:, track_id_col] == track_id
            ctx_agent_mask = ctx_frame_mask & track_id_mask
            # Get the current position and yaw.
            his_x = vals[ctx_agent_mask, phy_x_col].astype("float64")
            his_y = vals[ctx_agent_mask, phy_y_col].astype("float64")
            his_yaw = track_yaw_dict[track_id]

            # Trans historical trajectories to vcs
            his_x_vcs, his_y_vcs = Affine2D.coord_translate(
                his_x, his_y, seq_center.pos_x, seq_center.pos_y
            )
            his_x_vcs, his_y_vcs = Affine2D.coord_rotate(
                his_x_vcs, his_y_vcs, seq_center.yaw
            )
            his_traj_vcs = np.stack([his_x_vcs, his_y_vcs], axis=1)
            lctx_x = his_traj_vcs[-1, 0]
            lctx_y = his_traj_vcs[-1, 1]

            # If the historical trajectory is too short, treat it as static
            # obstacle.
            if len(his_x) < 2 or his_yaw is None:
                safe_area[track_id] = StaticSafeArea(
                    point=[lctx_x, lctx_y],
                    obs_cls=agent_cls,
                )
                continue

            # Get the expand trajectories in the global coordinates
            displace = np.sqrt(np.diff(his_x) ** 2 + np.diff(his_y) ** 2)
            mean_dis = np.mean(displace)

            if agent_cls == 1:
                # Pedestrain, generate circle safe area.
                safe_area[track_id] = PedestrainSafeArea(
                    point=[lctx_x, lctx_y],
                    radius=mean_dis * self.traj_len,
                    obs_cls=agent_cls,
                )
            else:
                # Get basic trajectoires for safe areas.
                if self.expand_traj_method == "CV":
                    yaw = his_yaw[-1]
                    start_idx = 1 if self.use_his_traj else 0
                    dis_array = (
                        np.arange(start_idx, self.traj_len + 1) * mean_dis
                    )
                    dis_x = np.cos(yaw) * dis_array + his_x[-1]
                    dis_y = np.sin(yaw) * dis_array + his_y[-1]
                    exp_traj_glob = np.stack([dis_x, dis_y], axis=1)
                else:
                    raise ValueError(
                        "Unsupported trajectory expanding method "
                        f"{self.expand_traj_method}."
                    )

                # Trans "basic trajectories" to vcs coordinates
                x, y = exp_traj_glob[:, 0], exp_traj_glob[:, 1]
                x, y = Affine2D.coord_translate(
                    x, y, seq_center.pos_x, seq_center.pos_y
                )
                x, y = Affine2D.coord_rotate(x, y, seq_center.yaw)
                exp_traj_vcs = np.stack([x, y], axis=1)

                if self.use_his_traj:
                    exp_traj_vcs = np.concatenate(
                        [his_traj_vcs, exp_traj_vcs], axis=0
                    )
                tmp_safe_area = VehicleSafeArea(
                    traj=exp_traj_vcs,
                    obs_cls=agent_cls,
                    expand_base=self.expand_base,
                    expand_ratio=self.expand_ratio,
                )
                if tmp_safe_area.static:
                    tmp_safe_area = StaticSafeArea(
                        point=[lctx_x, lctx_y],
                        obs_cls=agent_cls,
                    )
                else:
                    # Trans historical traj from vcs to image
                    traj_x_vcs = tmp_safe_area.traj[:, 0]
                    traj_y_vcs = tmp_safe_area.traj[:, 1]
                    traj_x_img, traj_y_img = Affine2D.coord_scale(
                        traj_x_vcs,
                        traj_y_vcs,
                        self.resolution,
                        self.resolution,
                    )
                    if self.reverse:
                        img_coords = np.array([-traj_x_img, -traj_y_img]).T
                    else:
                        img_coords = np.array([traj_x_img, traj_y_img]).T
                    img_coords = img_coords + self.seq_center_img_offset
                    tmp_safe_area.set_img_coord_traj(img_coords)

                safe_area[track_id] = tmp_safe_area

        sample["safe_area"] = safe_area
        return sample
