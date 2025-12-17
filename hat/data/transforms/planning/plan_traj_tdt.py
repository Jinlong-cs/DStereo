# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image
from skimage.draw import polygon

from hat.core.traj_pred_typing import PathLike
from hat.core.traj_pred_utils import TdtCoordHelper as TCH
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = [
    "TDTEgoCentricGt",
    "TDTFilterObstacles",
    "TDTGenStatesAndMask",
    "TDTOccupancyMapRender",
    "TDTGetBEVLocalMapByTimestamp",
    "TDTGetTrajPredObjectsInfo",
]


@OBJECT_REGISTRY.register
class TDTEgoCentricGt:  # noqa: D205,D400
    """Transform the obstacle coordinates to its ego-centric VCS (Vehicle
    Coordinate System). This transfrom is necessary if the user requires
    ego-centric ground truth trajectories. \

    We assume that each sequence of trajectory only has context frames
    and a short range of future frames (like 6s), so our ego-centric
    ground truth trajectories are just future positions transformed to
    the last context frame's coordinate system. This transformation
    assumes that the new system's x axis is the direction of the obs_yaw. \

    To use, the user should construct a `EgoCentricGt` instance. After
    instantiation, the callable function will return a new dictionary that
    contains the following changes: \
        1. the `seq_df` was added two columns 'object_centric_x' and \
            'object_centric_y'.
    """

    def __init__(self):
        """Initialize method."""
        pass

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'last_context_frame_id'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'frame_id', 'track_id', 'x', 'y', 'obs_yaw'.

        Args:
            sample (Dict): An input sample dictionary.

        Returns:
            sample (Dict): The sample dictionary whose 'seq_df' has two new
                columns 'object_centric_x' and 'object_centric_y.
        """
        seq_df = sample["seq_df"].copy()
        last_context_frame_id = sample["last_context_frame_id"]
        seq_df_vals = seq_df.values
        seq_df_cols = seq_df.columns
        col_x = seq_df_cols.get_loc("x")
        col_y = seq_df_cols.get_loc("y")
        col_obs_yaw = seq_df_cols.get_loc("obs_yaw")
        col_track_id = seq_df_cols.get_loc("track_id")
        last_frame_mask = seq_df["frame_id"] == last_context_frame_id
        lcf_track_ids = seq_df_vals[last_frame_mask, col_track_id]

        seq_df["object_centric_x"] = np.nan
        seq_df["object_centric_y"] = np.nan

        for track_id in lcf_track_ids:
            # For each object in the last context frame, we get its coordinate
            # and obs_yaw to conduct coordinate transformation to car-centric
            # coordinate system for all future frames of this object
            track_id_mask = seq_df["track_id"] == track_id
            obj_center = seq_df_vals[track_id_mask & last_frame_mask, :]
            obj_center = (
                obj_center[:, [col_x, col_y, col_obs_yaw]]
                .squeeze()
                .astype("float64")
            )

            # Extract all future x, y coordinates of the particular object
            object_coords = seq_df_vals[track_id_mask, :]
            object_coords = object_coords[:, [col_x, col_y]].astype("float64")
            obj_centric_coords = TCH.global_phy_to_agent_centric_phy(
                object_coords, obj_center[:2], obj_center[-1]
            ).astype("float64")
            seq_df.loc[
                track_id_mask, ["object_centric_x", "object_centric_y"]
            ] = obj_centric_coords

        sample["seq_df"] = seq_df
        return sample


@OBJECT_REGISTRY.register
class TDTFilterObstacles:  # noqa: D205,D400
    """Filter out obstacles based on the distance with the ego car and the
    speed, is used for multi agent pipeline to select valid track id to
    predict. \

    To use, the user should construct a `FilterObstacles`. After instantiation,
    the callable function will return a new dict that contains the following
    changes: \
        1. the `track_ids`, `valid_track_ids` and `drift_ids` was added.
    """

    def __init__(
        self,
        is_training,
        is_multiagent,
        ego_track_id,
        ped_cyc_type_id,
        leaving_mode="all",
        valid_distance=None,
        static_thr=None,
        num_frame_thr=None,
        bounce_thr=None,
        ped_static_thr=None,
        ped_drift_thr=None,
        use_given_lcf_yaw=True,
        use_given_obs_yaw=False,
        classify_by_shape=False,
        ped_shape_thr=1,
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
            valid_distance (float): the valid distance threshold of ego car and
                obstacles [m].
            static_thr (float): if the distance between two frames is smaller
                than the threshold, then filter out this obstacle. [m]
            num_frame_thr (int): the minimum (historical and future) frame
                number of valid tracks.
            bounce_thr (float): the maximum yaw diff between two adjacent
                frames. [Rad]
            ped_static_thr (float): if the distance between two frames is
                smaller than this threshold, filter out this pedestrain. [m]
            ped_drift_thr (float): the threshold to check whether the
                pedestrain is drifting. If the pedestrain moves too fast along
                the current direction of the ego vehicle, maybe it is an
                drifting perception. [m]
            use_given_lcf_yaw (bool, optional): whether to use the last
                context frame perception yaw in yaw bouncing detection. Default
                to True.
            use_given_obs_yaw (bool, optional): whether to use the entire
                perception yaw array in yaw bouncing detection. Default to
                False.
            classify_by_shape (bool, optional): the mode to classify
                pedestrains and vehicles. If true, means to classify by the
                length and width of obstacles. else, directly use the
                classification in the df.
            ped_shape_thr (float, optional): the maximum length and width of
                pedestrains. Defaults to 1.
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
        self.ped_static_thr = (
            ped_static_thr if (ped_static_thr is not None) else static_thr
        )
        self.ped_drift_thr = ped_drift_thr
        self.classify_by_shape = classify_by_shape
        self.ped_shape_thr = ped_shape_thr
        self.use_given_lcf_yaw = use_given_lcf_yaw or use_given_obs_yaw
        self.use_given_obs_yaw = use_given_obs_yaw

        self.if_filter_distance = (
            True if (valid_distance is not None) else False
        )
        self.if_filter_num_ctx = True if (num_frame_thr is not None) else False
        self.if_filter_stationary = True if (static_thr is not None) else False
        self.if_filter_bounce = True if (bounce_thr is not None) else False
        self.if_detect_drift_ped = (
            True if (ped_drift_thr is not None) else False
        )

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'last_context_frame_id'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'track_id', 'x', 'y', 'obs_yaw', 'pos_x', 'pos_y', 'yaw'
                'classification', 'obs_length', 'obs_width', 'timestamp'.

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample with added 'track_ids', 'valid_track_ids'
                and 'drift_ids'.
        """
        # Generate valid track ids.
        df = sample["seq_df"]
        last_context_frame_id = sample["last_context_frame_id"]
        vals = df.values
        cols = df.columns
        col_track_id = cols.get_loc("track_id")
        last_frame_mask = df.frame_id == last_context_frame_id
        ctx_frame_mask = df.frame_id <= last_context_frame_id
        future_frame_mask = df.frame_id > last_context_frame_id
        track_ids = vals[last_frame_mask, col_track_id]
        sample["track_ids"] = track_ids.tolist()
        sample["track_ids"].sort()

        if not self.is_multiagent:
            sample["valid_track_ids"] = [self.ego_track_id]
            sample["drift_ids"] = []
            return sample

        phy_x_col = cols.get_loc("x")
        phy_y_col = cols.get_loc("y")
        phy_yaw_col = cols.get_loc("obs_yaw")
        phy_ego_x_col = cols.get_loc("pos_x")
        phy_ego_y_col = cols.get_loc("pos_y")
        ego_yaw_col = cols.get_loc("yaw")
        class_col = cols.get_loc("classification")
        obs_length_col = cols.get_loc("obs_length")
        obs_width_col = cols.get_loc("obs_width")

        # Filter the valid track ids.
        valid_track_ids = []
        drift_ids = []
        for track_id in track_ids:
            agent_mask = df["track_id"] == track_id
            lcf_agent_mask = last_frame_mask & agent_mask
            ctx_agent_mask = ctx_frame_mask & agent_mask
            future_agent_mask = future_frame_mask & agent_mask

            # 0.0. Extract the trajectory information from the dataframe.
            # -- position
            his_x = vals[ctx_agent_mask, phy_x_col].astype("float64")
            his_y = vals[ctx_agent_mask, phy_y_col].astype("float64")
            lcf_x = vals[lcf_agent_mask, phy_x_col].astype("float64")
            lcf_y = vals[lcf_agent_mask, phy_y_col].astype("float64")
            ego_lcf_x = vals[lcf_agent_mask, phy_ego_x_col].astype("float64")
            ego_lcf_y = vals[lcf_agent_mask, phy_ego_y_col].astype("float64")
            future_x = vals[future_agent_mask, phy_x_col].astype("float64")
            future_y = vals[future_agent_mask, phy_y_col].astype("float64")
            all_x = vals[agent_mask, phy_x_col].astype("float64")
            all_y = vals[agent_mask, phy_y_col].astype("float64")
            # -- yaw
            ego_yaw = vals[lcf_agent_mask, ego_yaw_col].astype("float64")
            lcf_yaw = vals[lcf_agent_mask, phy_yaw_col].astype("float64")
            all_yaw = vals[agent_mask, phy_yaw_col].astype("float64")
            # -- shape and class
            his_width = vals[ctx_agent_mask, obs_width_col].astype("float")
            his_length = vals[ctx_agent_mask, obs_length_col].astype("float")
            his_class = vals[ctx_agent_mask, class_col].astype("int")

            # 0.1. Choose leaving agents.
            if self.leaving_mode == "ego":
                if track_id != self.ego_track_id:
                    continue
            elif self.leaving_mode == "vehicles":
                filter_flag = False
                for type_id in self.ped_cyc_type_id:
                    if type_id in his_class:
                        filter_flag = True
                if filter_flag:
                    continue

            # 0.2. If needed, skip the filter of the ego vehicle.
            if track_id == self.ego_track_id and not self.is_training:
                valid_track_ids.append(track_id)
                continue

            # 1. Filter by frame numbers in context and future frames.
            # -- The obstacle that appears in very few historical frames cannot
            #    provide enough information to describe the trajectory.
            # -- The obstacle that appears in very few future frames cannot
            #    provide a long-enough ground-truth for training.
            if self.if_filter_num_ctx or self.if_filter_bounce:
                if his_x.shape[0] < self.num_frame_thr:
                    continue
            if self.if_filter_num_ctx and self.is_training:
                if future_x.shape[0] < self.num_frame_thr:
                    continue

            # 2. Filter by classification.
            # -- Once an obstacle is detected as pedestrain or cyclist, it must
            #    be always pedestrain or cyclist in the context frames.
            #    Otherwise, the perception or tracking may be wrong, skip.
            unstable_class, track_is_ped = self._detect_unstable_obs_class(
                his_width, his_length, his_class
            )
            if unstable_class:
                continue

            # 3. Filter by distance.
            # -- Obstacles that are too far away usually have large perception
            #    errors. This distance threshold should be set according to
            #    the perception equipment.
            if self.if_filter_distance:
                ego_agent_distance = np.sqrt(
                    (lcf_x - ego_lcf_x) ** 2 + (lcf_y - ego_lcf_y) ** 2
                )
                if ego_agent_distance > self.valid_distance:
                    continue

            # 4. Filter the stationary obstacles by displacement.
            # -- Filter out the stationary obstacles. If the mode is training,
            #    we use the entire trajectory, otherwise, we just use the
            #    historical trajectory.
            if self.if_filter_stationary:
                sta_x = all_x if self.is_training else his_x
                sta_y = all_y if self.is_training else his_y
                sta_thr = (
                    self.ped_static_thr if track_is_ped else self.static_thr
                )
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
                    continue

            # 5. Filter or project the drifting pedestrains.
            # -- The pedestrains at the edge of the perception area may have
            #    position bounce. It seems that the pedestrain is moving at
            #    a very fast velocity (i.e., drifting), which is unreasonable.
            #    If the obstacle is drifting along the ego velocity direction,
            #    it will not influence the ego vehicle, skip.
            if self.if_detect_drift_ped and track_is_ped:
                is_drifting, need_filt = self._detect_drift_peds(
                    his_x, his_y, ego_yaw, track_is_ped
                )
                if is_drifting:
                    drift_ids.append(track_id)
                if need_filt:
                    continue

            # 6. Filter the yaw bounce trajectories.
            # -- Here, we filter the trajectories that have too large yaw
            #    difference. According to the perception, the used `yaw`
            #    can be derived directly from perception or by manually
            #    calculation.
            if self.if_filter_bounce and track_id != self.ego_track_id:
                _, need_filt = self._detect_yaw_bounce_obs(
                    his_x, his_y, future_x, future_y, lcf_yaw, all_yaw
                )
                if need_filt:
                    continue

            valid_track_ids.append(track_id)

        if len(valid_track_ids):
            valid_track_ids.sort()
        if len(drift_ids):
            drift_ids.sort()
        sample["valid_track_ids"] = valid_track_ids
        sample["drift_ids"] = drift_ids
        return sample

    def _detect_unstable_obs_class(self, his_width, his_length, his_class):
        """Detect whether the obstacle has unstable classification results.

        Once an obstacle is detected as pedestrain or cyclist, it must be
        always pedestrain or cyclist in the context frames. Otherwise, the
        perception or tracking may be wrong.

        Args:
            his_width (np.array): the historical obstacle width.
            his_length (np.array): the historical obstacle length.
            his_class (np.array): the historical obstacle classification.

        Returns:
            unstable_class (bool): whether the obstacle has unstable
                classifications.
            track_is_ped (bool): whether this obstacle is pedestrain or
                cyclist.
        """
        track_is_ped = False
        unstable_class = False
        if self.classify_by_shape:
            is_ped = (his_width <= self.ped_shape_thr) & (
                his_length <= self.ped_shape_thr
            )
            if True in is_ped:
                if np.sum(is_ped) != len(his_width):
                    unstable_class = True
                track_is_ped = True
        else:
            # Exclusive type_ids: the first element is pedestrain id.
            exclusive_type = self.ped_cyc_type_id
            for type_id in exclusive_type:
                if type_id in his_class:
                    track_is_ped = type_id == exclusive_type[0]
                    if np.sum(his_class == type_id) != len(his_class):
                        unstable_class = True
                        break
        return unstable_class, track_is_ped

    def _detect_drift_peds(self, his_x, his_y, ego_yaw, track_is_ped=False):
        """Detect whether the pedestrain is drifting.

        The pedestrains at the edge of the perception area may have position
        bounce. It seems that the pedestrain is moving at a very fast velocity
        (i.e., drifting), which is unreasonable. If the obstacle is drifting
        along the ego velocity direction, it will not influence the ego
        vehicle, thus we can skip this pedestrain during trajectory prediction.

        Args:
            his_x (np.array): the historical x coordinates.
            his_y (np.array): the historical y coordinates.
            ego_yaw (float): the ego vehicle yaw in the last context frame.
            track_is_ped (bool, optional): whether the obstacle is pedestrain.
                Default to False. If this parameter is False, this function
                will not perform any detection.

        Returns:
            is_drifting (bool): whether the obstacle is drifting pedestrain.
            need_filt (bool): whether this obstacle should be filtered out.
        """
        assert len(his_x) == len(his_y), (
            "The length of the given x and y coordinate lists should"
            f"be equal, but {len(his_x)} vs {len(his_y)}."
        )
        is_drifting, need_filt = False, False
        if len(his_x) <= 1:
            return is_drifting, True
        his_yaw = np.arctan2(np.diff(his_y), np.diff(his_x))
        # The `velo` variables here are not strict velocities, because they
        # are not divided by delta t.
        his_velo = np.sqrt(np.diff(his_x) ** 2 + np.diff(his_y) ** 2)
        max_velo = np.max(his_velo)
        if track_is_ped and max_velo >= self.ped_drift_thr:
            # The pedestrain moves too fast, maybe an drifting
            # perception. Save the id for postprocessing.
            his_vcs_yaw = his_yaw - ego_yaw
            his_vcs_x_velo = his_velo * np.cos(his_vcs_yaw)
            is_drifting = True
            # Why 0.9 here? 0.9 ~= cos(30°), i.e., this drifting
            # obstacle is within +-30 degree compared to ego yaw.
            if np.max(his_vcs_x_velo) > 0.9 * self.ped_drift_thr:
                return is_drifting, True
        return is_drifting, need_filt

    def _detect_yaw_bounce_obs(
        self, his_x, his_y, future_x, future_y, lcf_yaw, all_yaw
    ):
        """Detect whether the obstacle yaw is bouncing.

        In this function, we detect the trajectories that have too large yaw
        difference. According to the perception, the used `yaw` can be derived
        directly from perception or by manually calculation.

        Args:
            his_x (np.array): the historical x coordinates.
            his_y (np.array): the historical y coordinates.
            future_x (np.array): the future x coordinates.
            future_y (np.array): the future y coordinates.
            lcf_yaw (float): the obstacle yaw in the last context frame.
            all_yaw (np.array): the obstacle yaw in the entire trajectory.

        Returns:
            is_bounce (bool): whether the obstacle yaw is bouncing.
            need_filt (bool): whether this obstacle should be filtered out.
        """
        assert len(his_x) == len(his_y), (
            "The length of the given historical x and y coordinate lists"
            f"should be equal, but {len(his_x)} vs {len(his_y)}."
        )
        assert len(future_x) == len(future_y), (
            "The length of the given future x and y coordinate lists should"
            f"be equal, but {len(future_x)} vs {len(future_y)}."
        )
        is_bounce, need_filt = False, False
        yaw_array = []
        # 6.1 Based on calculated yaw.
        # -- History yaw.
        his_yaw = np.arctan2(np.diff(his_y), np.diff(his_x))
        yaw_array.append(his_yaw.reshape([-1]))
        # -- Last context frame yaw, use perception result.
        if self.use_given_lcf_yaw:
            yaw_array.append(lcf_yaw.reshape([-1]))
        # -- Future context frame yaw.
        if self.is_training:
            future_yaw = np.arctan2(np.diff(future_y), np.diff(future_x))
            yaw_array.append(future_yaw.reshape([-1]))
        yaw = np.concatenate(yaw_array, axis=0)
        if yaw.shape[0] <= 1:
            return is_bounce, True
        yaw_diff = np.diff(np.unwrap(yaw))
        max_yaw_diff = np.max(np.abs(yaw_diff))
        # 6.2 Based on given yaw.
        given_yaw_diff = np.diff(np.unwrap(all_yaw))
        max_given_yaw_diff = np.max(np.abs(given_yaw_diff))

        if self.use_given_obs_yaw:
            # The `max_yaw_diff` larger than pi/3 means the given yaw
            # is wrong, skip. Otherwise, we use the given yaw to detect
            # bounce.
            if (
                max_yaw_diff > np.pi / 3
                or max_given_yaw_diff > self.bounce_thr
            ):
                return True, True
        else:
            # Otherwise, we use the calucated delta_yaw to detect
            # bounce.
            if max_yaw_diff > self.bounce_thr:
                return True, True
        return is_bounce, need_filt


@OBJECT_REGISTRY.register
class TDTGenStatesAndMask:
    """(Sample and) Generate structured states and masks for the obstacles.

    This transform method requires the sample to have the key `track_ids`.
    Therefore, it must be defined after `GenFutureTrackids` or
    `FilterObstacles`. \

    To use, the user should construct a `GenStatesAndMask` instance with the
    required initialization parameters. After instantiation, the callable
    function will return a updated dict that contains the five ew elements as
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

    If you want to use the `states` in `MultiPathTransform`, please put the
    required multipath x&y in the first two columns in `state_col_name`.
    """

    def __init__(
        self,
        seq_length: int,
        context_frames: int,
        seq_period: int,
        state_col_name: tuple = ("object_centric_x", "object_centric_y"),
        enable_incomplete_gts: bool = False,
    ):
        """Initialize method.

        Args:
            seq_length (int): length of the entire sequence.
            context_frames (int): length of the context (historical) frames.
            seq_period (int): period of the sequence.
            state_col_name (list): the column names related to the structured
                states. The user must ensure that these columns are exist in
                the `seq_df` of the input sample.
            enable_incomplete_gts (bool, optional): whether to predict cases
                that do not have complete ground-truth future trajectories
                (i.e., the number of frames is smaller than `traj_len`).
                Defaults to False.
                If the `data_source` is 'nuscenes' and the current stage is
                    validation, this parameter must be False. Because the
                    metric calcuator in NuScenes api does not support the
                    comparison between a complete prediction result and an
                    incomplete ground-truth trajectory.
                Otherwise, it is okay to set this parameter as True.
        """
        self.seq_length = seq_length
        self.context_frames = context_frames
        self.future_frames = seq_length // seq_period - context_frames
        self.seq_period = seq_period
        self.state_col_name = state_col_name
        self.enable_incomplete_gts = enable_incomplete_gts

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'seq_index'.
            3. 'last_context_frame_id'.
            4. 'track_ids'

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                `track_id`, `frame_id` and all the keys in
                `self.state_col_name`.

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the input sample dictionary with five additional
                items: `states`, `masks`, `context_states`, `context_masks`,
                `classification`, `state_complete_track_ids`.
        """
        df = sample["seq_df"]
        frame_slice = sample["seq_index"].frame_slice
        track_ids = sample["track_ids"]
        lcf = sample["last_context_frame_id"]

        df_cols = df.columns
        df_values = df.values
        col_frame_id = df_cols.get_loc("frame_id")
        col_track_id = df_cols.get_loc("track_id")
        col_class = df_cols.get_loc("classification")
        col_states = list(
            map(lambda x: df_cols.get_loc(x), self.state_col_name)
        )

        # 0. Get the masks for context and future frames. If seq_period != 0,
        # need to sample.
        _start, _stop = frame_slice.start, frame_slice.stop
        seq_period = self.seq_period
        ctx_frames_arr = np.arange(
            start=_start, stop=lcf + 1, step=seq_period, dtype=np.int64
        )
        future_frames_arr = np.arange(
            start=lcf + seq_period, stop=_stop, step=seq_period, dtype=np.int64
        )
        ctx_mask = np.in1d(df_values[:, col_frame_id], ctx_frames_arr)
        future_mask = np.in1d(df_values[:, col_frame_id], future_frames_arr)

        # 1. Initialize result value arrays.
        # -- future states: [num_obj, traj_len, num_states]
        # -- future masks: [num_obj, traj_len]
        # -- context states: [num_obj, num_ctx, num_states]
        # -- context masks: [num_obj, num_ctx]
        # -- classification: [num_obj]
        len_track_ids = len(track_ids)
        len_state_cols = len(self.state_col_name)
        states = np.zeros(
            shape=[len_track_ids, self.future_frames, len_state_cols],
            dtype=np.float64,
        )
        masks = np.zeros(
            shape=[len_track_ids, self.future_frames], dtype=np.int64
        )
        ctx_states = np.zeros(
            shape=[len_track_ids, self.context_frames, len_state_cols],
            dtype=np.float64,
        )
        ctx_masks = np.zeros(
            shape=[len_track_ids, self.context_frames], dtype=np.int64
        )
        classification = np.zeros(shape=[len_track_ids], dtype=np.int64)

        # 2. Fill values for the result value arrays. If needed, save
        # the track_id of the valid obstacles that have nice state values.
        state_complete_track_ids = []
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
            # ---- lcf + seq_period is the 0th array element
            obj_future_arr_idx = (obj_future_frame_ids - lcf) // seq_period - 1
            obj_future_arr_idx = obj_future_arr_idx.astype(np.int64)
            # ---- use negative index, lcf is the -1st array element
            obj_ctx_arr_idx = (obj_ctx_frame_ids - lcf) // seq_period - 1
            obj_ctx_arr_idx = obj_ctx_arr_idx.astype(np.int64)
            # -- extract states and classification. Classification should be
            #    unique, so we take the first value of the unique value list
            obj_future_vals = df_values[future_mask & track_id_mask, :]
            obj_context_vals = df_values[ctx_mask & track_id_mask, :]
            obj_future_states = obj_future_vals[:, col_states]
            obj_ctx_states = obj_context_vals[:, col_states]
            obj_class = np.unique(obj_context_vals[:, col_class])[0]

            # Here, a valid track id means that the vehicle has a complete
            #   future trajectory (the number of frames is equal to
            #   `self.future_frames`).
            if (
                self.enable_incomplete_gts
                or len(obj_future_states) == self.future_frames
            ):
                state_complete_track_ids.append(track_id)

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
        sample["state_complete_track_ids"] = state_complete_track_ids
        return sample


@OBJECT_REGISTRY.register
class TDTOccupancyMapRender:
    """The occupancy map render.

    Here, we render all the obstacle bounding boxes in each context
    frames to a occupancy map. \

    To use, the user should construct a `OccupancyMapRender` instance with the
    required initialization parameters. The user should use this tranform after
    `GenBoundingBox` and `PhyToImage` (or `PhyToBEV`). \

    After instantiation, the callable function will return a new dict that
    contains the following changes: \
        1. a new key 'rendered_obs' was added.
    """

    def __init__(
        self,
        map_height: int,
        map_width: int,
        context_frames: int,
        render_ego: bool,
        render_fut: bool = False,
        normalize: bool = True,
    ):
        """Initialize method.

        Args:
            map_height (int): rendered map height.
            map_width (int): rendered map width.
            context_frames (int): number of context frames.
            render_ego (bool): render ego vehicle or obs.
            render_fut (bool, optional): render future occupancy
            normalize (bool, optional): whether to normalize the road map
                from [0, 255] -> [-1, 1]. Default to True.
        """
        self.map_height = map_height
        self.map_width = map_width
        self.context_frames = context_frames
        self.render_ego = render_ego
        self.render_fut = render_fut
        self.normalize = normalize

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        The following keys are required:
            1. 'seq_index'.
            2. 'seq_df'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'frame_id', 'img_x0', 'img_y0', 'img_x1', 'img_y1',
                'img_x2', 'img_y2', 'img_x3', 'img_y3', 'img_ego_x0',
                'img_ego_y0', 'img_ego_x1', 'img_ego_y1', 'img_ego_x2',
                'img_ego_y2', 'img_ego_x3', 'img_ego_y3'
            2. Therefore, this tranform method must be used after
                `GenBoundingBox` and `PhyToImage` (or `PhyToBEV`).

        Args:
            sample (Dict): input original sample. The `sample` should be
                obtained by the `__getitem__` of trajectory dataset (base
                class `hdlt.data.datasets.traj_pred_dataset.BaseTrajDataset`)

        Returns:
            sample (Dict): the updated sample as described above.
        """
        seq_df = sample["seq_df"]
        seq_index = sample["seq_index"]
        seq_frame_slice = seq_index.frame_slice
        frame_range = range(
            seq_frame_slice.start, seq_frame_slice.stop, seq_frame_slice.step
        )
        # Render historical frames as polygons.
        past_bboxes = []
        for frame_id in frame_range[: self.context_frames]:
            frame_df = seq_df.loc[seq_df["frame_id"] == frame_id]
            past_bboxes.append(
                self.render_one_frame_bboxes(
                    frame_df, self.map_height, self.map_width
                )
            )
        rendered_obs = np.concatenate([*past_bboxes], axis=2)
        # Norm the occumancy map. [0, 255] -> [-1, 1].
        rendered_obs = (rendered_obs - 128) / 128

        if self.render_ego:
            sample["plan_rendered_ego"] = rendered_obs
        else:
            sample["rendered_obs"] = rendered_obs
            if self.render_fut:
                # Render fut frames as polygons by planning.
                fut_bboxes = []
                for frame_id in frame_range[self.context_frames :]:
                    frame_df = seq_df.loc[seq_df["frame_id"] == frame_id]
                    fut_bboxes.append(
                        self.render_one_frame_bboxes(
                            frame_df, self.map_height, self.map_width
                        )
                    )

                rendered_obs_fut = np.concatenate([*fut_bboxes], axis=2)
                rendered_obs_fut = (rendered_obs_fut - 128) / 128
                sample["rendered_obs_fut"] = rendered_obs_fut

        return sample

    def render_one_frame_bboxes(
        self, frame_df, map_height, map_width
    ):  # noqa: D205,D400
        """Render obstacle bounding bboxes (in one frame) and
        generate the occupancy map. The values are 0~255.

        Args:
            frame_df: (pd.DataFrame): a DataFrame with the obstacle
                position information of a certain timestamp.
            map_height (int): the height of the rendered map.
            map_width (int): the width of the rendered map.

        Returns:
            ndarray: Rendered bboxes ndarray for the frame
        """
        if frame_df.empty:
            return np.zeros([map_height, map_width, 1])

        # Pad the base map to allow incomplete polygon rendering.
        base = np.zeros([3 * map_height, 3 * map_width])

        # by planning
        if self.render_ego:
            # Get ego vehicle bounding boxes.
            ego_bbox_cols = [
                "img_ego_x0",
                "img_ego_y0",
                "img_ego_x1",
                "img_ego_y1",
                "img_ego_x2",
                "img_ego_y2",
                "img_ego_x3",
                "img_ego_y3",
            ]
            ego_bbox = np.unique(frame_df[ego_bbox_cols], axis=0)
            img_bboxes = ego_bbox
        else:
            # Get the vehicle bounding boxes.
            obs_bbox_cols = [
                "img_x0",
                "img_y0",
                "img_x1",
                "img_y1",
                "img_x2",
                "img_y2",
                "img_x3",
                "img_y3",
            ]
            obs_frame = frame_df.loc[frame_df["track_id"] != -42]
            obs_bbox = obs_frame[obs_bbox_cols].values
            img_bboxes = obs_bbox

        for img_bbox in img_bboxes:
            v_x = img_bbox[::2]
            v_y = img_bbox[1::2]
            if v_x.size == 0 or v_y.size == 0:
                continue
            elif (
                np.min(v_x) < -map_height
                or np.max(v_x) >= 2 * map_height
                or np.min(v_y) < -map_width
                or np.max(v_y) >= 2 * map_width
            ):
                continue
            else:
                xx, yy = polygon(v_x, v_y)
                base[xx + map_height, yy + map_width] = 255
        base = base[
            map_height : 2 * map_height, map_width : 2 * map_width, None
        ]
        return base


@OBJECT_REGISTRY.register
class TDTGetTrajPredObjectsInfo:
    """Get the information of all the valid objects for trajectory prediction.

    This transform method requires the sample to have the following keys,
    which means that is must be used after some transform methods. \

    +-----------------------------+----------------------------------------+
    |    requires                 |  needed transforms                     |
    +=============================+========================================+
    | track_ids, valid_track_ids  |  FilterObstacles or GenFutureTrackids  |
    | states, masks               |  GenStatesAndMask                      |
    +-----------------------------+----------------------------------------+

    """

    def __init__(
        self,
        scene_img_height: int,
        scene_img_width: int,
        ego_track_id: int,
        use_given_obs_yaw: bool,
        peds_use_given_obs_yaw: bool,
        veh_type_id: int,
        ped_cyc_type_id: List,
        detect_peds_by_shape: bool = False,
        ped_shape_thr: float = 1,
        timestamp_unit: str = "ms",
        use_state_vectors: bool = True,
        use_instant_state_vectors: bool = False,
        num_state_vectors: int = 3,
        if_clip_state_vectors: bool = True,
        state_vectors_log: List = (True, False, False),
        state_vectors_clip_bound: List = (5, 5, 0.5),
        state_vectors_scales: List = (1, 1, 10),
    ):
        """Initialize method.

        Args:
            scene_img_height (int): rendered map height.
            scene_img_width (int): rendered map width.
            ego_track_id (int): the ego vehicle track id.
            use_given_obs_yaw (bool): whether to use the perception yaw in the
                vehicle image coordinates.
            peds_use_given_obs_yaw (bool): whether to use the perception yaw
                in the pedestrain (cyclist) image coordinates.
            veh_type_id (int): the classification id of the vehicle.
            ped_cyc_type_id (list): the classification id of the pedestrain
                and cyclist.
            detect_peds_by_shape (bool, optional): whether to classify
                pedestrains by the length and width of obstacles. Default to
                False, i.e., directly use the classification in the df.
            ped_shape_thr (float, optional): the maximum length and width of
                pedestrains. Defaults to 1.
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
        """
        self.scene_img_height = scene_img_height
        self.scene_img_width = scene_img_width
        self.ego_track_id = ego_track_id
        self.use_given_obs_yaw = use_given_obs_yaw
        self.peds_use_given_obs_yaw = peds_use_given_obs_yaw
        # Parameters about pedestrain detection.
        self.veh_type_id = veh_type_id
        self.ped_cyc_type_id = ped_cyc_type_id
        self.ped_type_id, self.cyc_type_id = ped_cyc_type_id
        self.detect_peds_by_shape = detect_peds_by_shape
        self.ped_shape_thr = ped_shape_thr
        self.label_remapping = {
            self.veh_type_id: 0,
            self.ped_type_id: 1,
            self.cyc_type_id: 2,
        }
        # Parameters about state vectors.
        self.use_state_vectors = use_state_vectors
        self.use_instant_state_vectors = use_instant_state_vectors
        self.num_state_vectors = num_state_vectors
        self.if_clip_state_vectors = if_clip_state_vectors
        self.state_vectors_log = state_vectors_log
        self.state_vectors_clip_bound = state_vectors_clip_bound
        self.state_vectors_scales = state_vectors_scales
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
            7. (optional) 'valid_track_ids', 'state_complete_track_ids'

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'img_x', 'img_y', 'yaw', 'img_obs_yaw', 'track_id', 'frame_id',
                'x', 'y', 'obs_yaw', 'timestamp', 'classification',
                'obs_length', 'obs_width'

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample includes new keys 'valid_class',
                'valid_img_coords', 'valid_masks', 'future_trajectories',
                'resize_ratio', 'state_vectors' and update keys 'seq_df',
                'rendered_obs', 'valid_track_ids'.
        """
        # Extract information from sample.
        seq_df = sample["seq_df"].copy()
        last_context_frame_id = sample["last_context_frame_id"]
        # -- from FilterObstacles or GenFutureTrackids
        full_track_ids = sample["track_ids"]
        if "valid_track_ids" in sample:
            center_ids = sample["valid_track_ids"]
        else:
            center_ids = sample["track_ids"]
        # -- from GenStatesAndMask
        full_future_trajs = sample["states"]
        full_future_masks = sample["masks"]
        if "state_complete_track_ids" in sample:
            state_complete_track_ids = sample["state_complete_track_ids"]
            center_ids = list(set(center_ids) & set(state_complete_track_ids))

        # Filter the out-of-scene rows according to img coords.
        seq_df["last_context_frame_id"] = last_context_frame_id
        scene_bounded_row_mask = (
            (seq_df["img_x"] > 0)
            & (seq_df["img_x"] < self.scene_img_height)
            & (seq_df["img_y"] > 0)
            & (seq_df["img_y"] < self.scene_img_width)
        )
        seq_df = seq_df.loc[scene_bounded_row_mask]

        # Get column indices.
        seq_df_vals = seq_df.values
        seq_df_cols = seq_df.columns
        track_id_col = seq_df_cols.get_loc("track_id")
        frame_id_col = seq_df_cols.get_loc("frame_id")
        phy_x_col = seq_df_cols.get_loc("x")
        phy_y_col = seq_df_cols.get_loc("y")
        phy_yaw_col = seq_df_cols.get_loc("obs_yaw")
        img_x_col = seq_df_cols.get_loc("img_x")
        img_y_col = seq_df_cols.get_loc("img_y")
        img_yaw_col = seq_df_cols.get_loc("img_obs_yaw")
        stamp_col = seq_df_cols.get_loc("timestamp")
        class_col = seq_df_cols.get_loc("classification")
        obs_length_col = seq_df_cols.get_loc("obs_length")
        obs_width_col = seq_df_cols.get_loc("obs_width")

        lcf_mask = seq_df_vals[:, frame_id_col] == last_context_frame_id
        ctx_mask = seq_df_vals[:, frame_id_col] <= last_context_frame_id

        # Extract the information for valid obstacles (objects for prediction).
        valid_cls = []
        valid_track_ids = []
        valid_fut_trajs = []
        valid_traj_masks = []
        valid_img_coords = []
        valid_his_phy_coords = []
        for track_id in center_ids:
            track_id_mask = seq_df_vals[:, track_id_col] == track_id
            track_id_index = full_track_ids.index(track_id)
            agent_fut_traj = full_future_trajs[track_id_index]
            agent_fut_mask = full_future_masks[track_id_index]
            cur_mask = lcf_mask & track_id_mask
            cur_ctx_mask = ctx_mask & track_id_mask
            if (not np.any(cur_mask)) or (not np.any(agent_fut_mask)):
                continue

            lcf_df_vals = seq_df_vals[cur_mask, :]
            img_coords_cols = [img_x_col, img_y_col, img_yaw_col]
            agent_img_coords = lcf_df_vals[:, img_coords_cols].reshape(-1)
            agent_class = lcf_df_vals[:, class_col][0]

            ctx_df_vals = seq_df_vals[cur_ctx_mask, :]
            his_img_x = ctx_df_vals[:, img_x_col].astype("float64")
            his_img_y = ctx_df_vals[:, img_y_col].astype("float64")
            used_phy_cols = [phy_x_col, phy_y_col, phy_yaw_col, stamp_col]
            his_phy_coords = ctx_df_vals[:, used_phy_cols].astype("float64")

            # Detect whether the obstacle is pedestrain (cyclist).
            track_is_ped = False
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
                    # says that it is pedestrain. We give it a vehicle type.
                    if agent_class == self.ped_type_id:
                        agent_class = self.veh_type_id
            else:
                track_is_ped = agent_class in self.ped_cyc_type_id

            # label remapping: -1: others, 0: vehicle, 1: pedestrain,
            #                   2: cyclist.
            remap_class = -1
            if agent_class in self.label_remapping:
                remap_class = self.label_remapping[agent_class]

            # Decide to use the perception yaw result or use the calculated yaw
            # in `agent_img_coords`. The given yaw of the ego vehicle is from
            # odometry and is stable enough, skip this selection.
            cal_yaw_flag = False
            if track_id != self.ego_track_id:
                if track_is_ped and (not self.peds_use_given_obs_yaw):
                    cal_yaw_flag = True
                elif (not track_is_ped) and (not self.use_given_obs_yaw):
                    cal_yaw_flag = True
                cal_yaw_flag &= len(his_img_x) >= 2
            if cal_yaw_flag:
                his_img_yaw = np.arctan2(
                    np.diff(his_img_y), np.diff(his_img_x)
                )
                agent_img_coords[2] = his_img_yaw[-1]

            valid_cls.append(remap_class)
            valid_track_ids.append(track_id)
            valid_fut_trajs.append(agent_fut_traj)
            valid_traj_masks.append(agent_fut_mask)
            valid_img_coords.append(agent_img_coords)
            valid_his_phy_coords.append(his_phy_coords)

        if self.use_state_vectors:
            state_vectors = self._extract_state_vectors(
                valid_track_ids, valid_his_phy_coords
            )
            sample["state_vectors"] = state_vectors

        # Update the sample dict.
        sample["seq_df"] = seq_df
        sample["valid_class"] = valid_cls
        sample["valid_track_ids"] = valid_track_ids
        sample["valid_img_coords"] = valid_img_coords
        sample["valid_masks"] = valid_traj_masks
        sample["future_trajectories"] = valid_fut_trajs
        num_agents = len(valid_track_ids)
        sample["resize_ratio"] = np.array([[1.0, 1.0]] * num_agents)
        return sample

    def _extract_state_vectors(self, valid_track_ids, valid_his_phy_coords):
        """Get agent state vectors (velocity, acceleration and yaw rate).

        Args:
            valid_track_ids (List): the valid track id list.
            valid_his_phy_coords (List): the historical coornates of the valid
                obstacles.

        Returns:
            state_vectors (np.array, [num_valid_obj, num_state_vectors]): the
                agent state vectors.
        """
        state_vectors = np.zeros(
            [len(valid_track_ids), self.num_state_vectors]
        )
        for idx, (track_id, his_phy) in enumerate(
            zip(valid_track_ids, valid_his_phy_coords)
        ):
            len_ctx = his_phy.shape[0]
            # The first-order variable requires a length of at least 2.
            if len_ctx < 2:
                continue
            his_x = his_phy[:, 0]
            his_y = his_phy[:, 1]
            his_yaw = his_phy[:, 2]
            his_stamp = his_phy[:, 3] / self.stamp_scale
            # -- Calculate velocity.
            his_time_diff = his_stamp[1:] - his_stamp[:-1]
            his_velo = (
                np.sqrt(
                    (his_x[1:] - his_x[:-1]) ** 2
                    + (his_y[1:] - his_y[:-1]) ** 2
                )
                / his_time_diff
            )
            # -- Calculate yaw rate.
            if track_id != self.ego_track_id:
                if not self.use_given_obs_yaw and (len_ctx > 2):
                    his_yaw = np.arctan2(np.diff(his_y), np.diff(his_x))
            his_yaw = np.unwrap(his_yaw)
            his_yaw_rate = (his_yaw[1:] - his_yaw[:-1]) / his_time_diff[-1]
            if self.use_instant_state_vectors:
                state_vectors[idx, 0] = his_velo[-1]
                state_vectors[idx, 2] = his_yaw_rate[-1]
            else:
                state_vectors[idx, 0] = np.mean(his_velo)
                state_vectors[idx, 2] = np.mean(his_yaw_rate)
            # The Second-order variable requires a length of at least 3.
            if len_ctx < 3:
                continue
            # -- Calculate acceleration.
            his_acc = (his_velo[1:] - his_velo[:-1]) / his_time_diff[-1]
            if self.use_instant_state_vectors:
                state_vectors[idx, 1] = his_acc[-1]
            else:
                state_vectors[idx, 1] = np.mean(his_acc)

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
        return state_vectors


@OBJECT_REGISTRY.register
class TDTGetBEVLocalMapByTimestamp:
    """Get BEV local map by timestamp.

    The local maps are pre-rendered and indexed by timestamps. Given a certain
    timestamp, we can set a vehicle (in most cases, it is the ego vehicle) as
    the center vehicle, and bind its postion with fixed BEV coordinates. Then,
    we can get the local map surrouding the center vehicle. We save the maps
    as figures with timestamps as file names. \

    Since origin BEV map is named by timestamps, the input dictionary must
    have the key `lcf_timestamp`. This means that this transform must be used
    after `GetLcfTimeStamp`. \

    To use, the user should construct a `GetBEVLocalMapByTimestamp`. After
    instantiation, the callable function will return a new dict that contains
    the following changes: \
        1. the `road_map` was added.
    """

    def __init__(
        self,
        image_dir: PathLike,
        image_suffix: str,
        map_height: int,
        map_width: int,
        data_token2path_mapping: dict,
        is_viz: bool = False,
        render_mapping: Optional[dict] = None,
        map_path_func: Optional[Callable] = None,
    ):
        """Initialize method.

        Args:
            image_dir (PathLike): the root path to the image directory.
            image_suffix (str): the suffix of the images.
            map_height (int): rendered map height.
            map_width (int): rendered map width.
            data_token2path_mapping (dict): the keys are the date tokens, and
                the values are the path prefix to the origin map file. Usually,
                the bev map files of different dates are stored in different
                paths, and `image_dir` is just their largest common path.
            is_vlz (bool, optional): whether the map is for visualization.
                Default to False.
            render_mapping (dict, optional): the color mapping of different
                road elements. Default to None. If the user wants to get a
                local map that renders different road elements (such as lane
                lines, curbs, sidewalks, etc.) as different colors, he should
                input the mapping between element classifications and color
                configurations here. If this parameter is None, this function
                will directly return the origin map from the file.
        """
        self.image_dir = image_dir
        self.image_suffix = image_suffix
        self.map_height = map_height
        self.map_width = map_width
        self.data_token2path_mapping = data_token2path_mapping
        self.is_viz = is_viz
        self.render_mapping = render_mapping
        if map_path_func is not None:
            self.map_path_func = map_path_func
        else:
            self.map_path_func = self.default_map_path_func

    def __call__(self, sample: dict):
        """Callable function.

        The following keys are required:
            1. `dataset_prefix`.
            2. `date_token`.
            3. `lcf_timestamp`.

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample with added 'road_map'.
        """
        plate = sample["dataset_prefix"]
        date_token = sample["date_token"]
        timestamp = str(int(sample["lcf_timestamp"]))
        date = date_token.split("-")[0]
        date_key = plate + date + "_viz" if self.is_viz else plate + date

        bev_bucket_path = self.data_token2path_mapping[date_key]
        bev_map_path = self.map_path_func(
            self.image_dir,
            bev_bucket_path,
            plate,
            date_token,
            timestamp,
            self.image_suffix,
        )
        need_render = True
        if not os.path.exists(bev_map_path):
            logger.warning(f"Image file {bev_map_path} does not exist!")
            road_map = np.zeros((self.map_height, self.map_width))
        else:
            with Image.open(bev_map_path) as im:
                im = np.array(im)
                road_map = im
                if len(im.shape) == 3:
                    if np.max(im) < 20:
                        road_map = im[:, :, 0]
                        need_render = True
                    else:
                        need_render = False
        if need_render and self.render_mapping is not None:
            for key, value in self.render_mapping.items():
                road_map[road_map == key] = value
        sample["road_map"] = road_map[:, :, None]

        # TODO use driveble area (bikun)
        # bev_mask_path = bev_map_path.split('.png')[0] + '_drivable.png'
        # if not os.path.exists(bev_mask_path):
        #     logger.warning(f"Image file {bev_mask_path} does not exist!")
        #     drivable_mask = np.zeros((self.map_height, self.map_width))
        # else:
        #     drivable_mask = cv2.imread(bev_mask_path)[:, :, 0]
        #     # make mask 0~1
        #     drivable_mask = drivable_mask.max() - drivable_mask
        #     drivable_mask[drivable_mask > 0] = 1

        # sample["road_mask"] = drivable_mask[:,:,None]

        return sample

    @staticmethod
    def default_map_path_func(
        image_dir, bev_bucket_path, plate, date_token, timestamp, image_suffix
    ):
        date = date_token.split("-")[0]
        bev_map_path = os.path.join(
            image_dir,
            bev_bucket_path,
            plate + date + "_D",
            date_token,
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


def gt_path_func_for_data_pipeline_p3c(
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
                date_token_dir = f"{date_token_suffix}/{bev_bucket_path[1][0]}"
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
