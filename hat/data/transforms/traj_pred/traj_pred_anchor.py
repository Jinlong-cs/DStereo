# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import yaml

from hat.core.traj_pred_typing import ANCHOR_TYPE
from hat.core.traj_pred_utils import (
    Affine2D,
    interpolate_trajectory_from_path,
    is_point_in_convex_polygon,
    normalize_yaw,
    trace_and_concat_driveline,
)
from hat.core.traj_pred_viz_utils import load_anchors
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "GetBestAnchors",
    "SampleNaviTrajAnchor",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class GetBestAnchors:  # noqa: D205,D400
    """Get the closest (most similar) anchor between ground truth
    and the anchor sets.

    In this class, the original anchor set is manually splitted into
    several classes with specific semantics (e.g. fast straight, etc).
    This class can get the closest anchor to the ground truth (based
    on a specific measure).

    This transform method should be used after `GetTrajPredObjectsInfo`.
    The callable function will return a updated dict that contains the
    following two new elements: \
        1. selected_anchors (np.array, [num_obj, num_anc_types, traj_len, \
            2]): the selected anchors in different anchor types. \
        2. selected_anchors_mask (np.array, [num_obj, num_anc_types]): \
            the masks of different anchor types. \
    """

    def __init__(
        self,
        anchor_cfg: Dict,
        anchor_type: Dict,
        anchor_type_dict: Dict,
        avg_l2_threshold: float = 5,
        final_l2_threshold: float = 10,
        mode: str = "l2",
    ):
        """Initialize method.

        Args:
            anchor_cfg (Dict): the anchor configuration dictionary with
                the following keys:
                1. 'anchor_file': the file path of the pickle anchors.
                2. 'anchor_method': the method to obtain the anchors, it
                    should be one of [kmeans, uniform].
                3. 'anchor_num': the number of anchors.
            anchor_type (Dict): the manually classified anchor types. The
                keys are the type id and the values are the corresponding
                description string.
            anchor_type_dict (Dict): the mapping between manually classified
                anchor types and the anchor indices of this type in the
                original anchor set.
            avg_l2_threshold (float): the threshold of average L2 distance.
                If the differences between one anchor and the ground truth
                is larger than this threshold, this anchor is invalid in
                this case. Default to 5.
            final_l2_threshold (float): the threshold of the L2 distance
                between the final points of anchors and the grouth truth.
                Defaults to 10.
            mode (str): the mode to choose the closest anchor. It should
                in ["l2", "weighted_l2"].
        """
        self.anchors = load_anchors(anchor_cfg, type="numpy")
        self.num_anchors = self.anchors.shape[0]
        self.traj_len = self.anchors.shape[1]
        self.avg_l2_threshold = avg_l2_threshold
        self.final_l2_threshold = final_l2_threshold
        self.mode = mode
        loaded_anchor_types = list(anchor_type_dict.keys())
        allowed_anchor_types = [v for k, v in anchor_type.items()]
        for key in loaded_anchor_types:
            assert (
                key in allowed_anchor_types
            ), f"The anchor type {key} is not allowed."
        anchor_split_dict = {}
        for idx, key in anchor_type.items():
            indices = anchor_type_dict[key]
            if len(indices):
                anchor_split_dict[idx] = self.anchors[indices, :, :]
            else:
                anchor_split_dict[idx] = []
        self.anchor_split_dict = anchor_split_dict
        self.num_anc_types = len(allowed_anchor_types)

    @staticmethod
    def get_min_l2_distance_index(trajectory, anchors, mask=None):
        """Find the nearest anchor index for trajectory.

        Args:
            trajectory (np.array, [traj_len, 2]): the ground-truth trajectory.
            anchors (np.array, [num_anchor, traj_len, 2]): the anchors.
            mask (np.array, [traj_len]): the mask.

        Returns:
            index (int) the index of the min l2 distance anchor.
            min_avg_l2 (float): the min value of the mean L2 distance.
        """
        # Vectorized implementation.
        trj_len = trajectory.shape[0]
        if mask is None:
            mask = np.ones([trj_len])
        dist = np.sqrt(
            np.sum((anchors - trajectory[None, :, :]) ** 2, axis=-1)
        )
        dist[np.where(np.isnan(dist))[0]] = 0
        loss = np.sum(dist * mask[None, :], axis=-1)
        loss /= np.sum(mask, axis=-1)[None] + 1e-6
        min_dis_idx = np.argmin(loss, axis=-1)
        min_avg_l2 = loss[min_dis_idx]
        return min_dis_idx, min_avg_l2

    @staticmethod
    def get_min_weighted_distance_index(trajectory, anchors, mask=None):
        """Find the nearest anchor index for trajectory.

        The measure is the weighted sum of mean L2 distance and final L2
        distance.

        Args:
            trajectory (np.array, [traj_len, 2]): the ground-truth trajectory.
            anchors (np.array, [num_anchor, traj_len, 2]): the anchors.
            mask (np.array, [traj_len]): the mask.

        Returns:
            index (int) the index of the min l2 distance anchor.
            min_avg_l2 (float): the min value of the mean L2 distance.
            min_final_l2 (float): the min value of the final L2 distance.
        """
        # Vectorized implementation.
        trj_len = trajectory.shape[0]
        if mask is None:
            mask = np.ones([trj_len])
        dist = np.sqrt(
            np.sum((anchors - trajectory[None, :, :]) ** 2, axis=-1)
        )
        dist[np.where(np.isnan(dist))[0]] = 0
        avg_l2_loss = np.sum(dist * mask[None, :], axis=-1)
        avg_l2_loss /= np.sum(mask, axis=-1)[None] + 1e-6

        last_valid_idx = np.where(np.cumsum(mask[::-1])[::-1] == 1)[0]
        if len(last_valid_idx):
            final_l2_loss = dist[:, last_valid_idx[0]]
            loss = avg_l2_loss + final_l2_loss
            min_dis_idx = np.argmin(loss, axis=-1)
            min_avg_l2 = avg_l2_loss[min_dis_idx]
            min_final_l2 = final_l2_loss[min_dis_idx]
        else:
            min_dis_idx = np.argmin(avg_l2_loss, axis=-1)
            min_avg_l2 = avg_l2_loss[min_dis_idx]
            min_final_l2 = 100
        return min_dis_idx, min_avg_l2, min_final_l2

    def __call__(self, sample):
        """Callable function.

        The following keys are required in `sample`:
            1. 'future_trajectories'.
            2. 'valid_masks'.

        Args:
            sample (Dict): input original sample. The `sample` should be
                obtained by the `__getitem__` of trajectory dataset (base
                class `hat.data.datasets.traj_pred_dataset.BaseTrajDataset`)

        Returns:
            sample (Dict): the updated sample as described above.
        """
        future_trajectories = sample["future_trajectories"]
        masks = sample["valid_masks"]
        num_obs = len(future_trajectories)
        best_anchors = np.zeros(
            [num_obs, self.num_anc_types, self.traj_len, 2]
        )
        best_anchors_mask = np.ones([num_obs, self.num_anc_types])
        for idx, (gt_traj, mask) in enumerate(zip(future_trajectories, masks)):
            for key, anchors in self.anchor_split_dict.items():
                if not len(anchors):
                    best_anchors[idx][key] = 0
                    continue
                if self.mode == "l2":
                    (
                        cur_idx,
                        avg_l2_loss,
                    ) = self.get_min_l2_distance_index(gt_traj, anchors, mask)
                    best_anchors[idx][key] = anchors[cur_idx]
                    if avg_l2_loss > self.avg_l2_threshold:
                        best_anchors_mask[idx][key] = 0
                elif self.mode == "weighted_l2":
                    (
                        cur_idx,
                        avg_l2_loss,
                        final_l2_loss,
                    ) = self.get_min_weighted_distance_index(
                        gt_traj, anchors, mask
                    )
                    best_anchors[idx][key] = anchors[cur_idx]
                    if (
                        avg_l2_loss > self.avg_l2_threshold
                        and final_l2_loss > self.final_l2_threshold
                    ):
                        best_anchors_mask[idx][key] = 0
                else:
                    raise ValueError(f"Unknown mode {self.mode}")

        sample["selected_anchors"] = best_anchors
        sample["selected_anchors_mask"] = best_anchors_mask
        return sample


@OBJECT_REGISTRY.register
class SampleNaviTrajAnchor:
    """Sample anchors from the navigation information.

    This transform method requires the sample to have the following keys,
    which means that is must be used after some transform methods. \

    +-----------------------------+----------------------------------------+
    |    requires                 |  needed transforms                     |
    +=============================+========================================+
    | seq_center                  |  GenSeqCenter                          |
    | valid_track_ids             |  FilterObstacles or GenFutureTrackids  |
    | lcf_timestamp               |  GetLcfTimeStamp                       |
    | seq_df with `img_x` col     |  PhyToBEV or PhyToImg                  |
    +-----------------------------+----------------------------------------+

    Moreover, this transform requires the sample from dataset to have the
    keys `navi_info`. Therefore, the dataset should be
    `AutoMultiAgentNaviDataset`.
    """

    def __init__(
        self,
        bev_origin_x: float,
        bev_origin_y: float,
        img_resolution: float,
        num_anchors: int,
        traj_len: int,
        navi_file_dir: str,
        navi_info_path_mapping: Dict,
        navi_file_path_func: Callable,
        anchor_classify_func: Callable,
        navi_file_suffix: str = "/lane2driveline.yaml",
        anchor_type_version: str = "classical",
        reverse: bool = False,
        velo_scales: Sequence[float] = (1.0,),
    ):
        """Initialize method.

        Args:
            bev_origin_x (float): the x coordinate of the bev orgin [m]
                in the bev map coordinate.
            bev_origin_y (float): the y coordinate of the bev orgin [m]
                in the bev map coordinate.
            img_resolution (float): query resolution of the map.
            num_anchors (float): the number of the anchors.
            traj_len (int): the length of the sampled trajectories.
            navi_file_dir (str): the root path of navigation files.
            navi_info_path_mapping (Dict): the keys are the date tokens, and
                the values are the path prefix to the navigation files.
                Usually, the navigation files of different plates and dates
                are stored in different aths, and `navi_file_dir` is just their
                largest common path.
            navi_file_path_func (Callable): given the plate and timestamp, this
                function returns the absolute path to the queried navi file.
            anchor_classify_func (Callable): a simple classification function
                for sampled trajectories. The classification types are defined
                in ANCHOR_TYPE.
            navi_file_suffix (str): the navigation file name.
            anchor_type_version (str): the used anchor type name. It should be
                one key in ANCHOR_TYPE.
            reverse (bool, optional): if the direction of VCS and image
                coordinate system are on the contrary (like in the BEV
                scenario), reverse should be True.
            velo_scales (List, optional): scales of the vehicle velocity to
                sample the navigation path to generate anchors..
        """
        assert (
            anchor_type_version in ANCHOR_TYPE
        ), f"Undefined anchor type set called {anchor_type_version}"
        self.img_resolution = img_resolution
        self.num_anchors = num_anchors
        self.traj_len = traj_len
        self.navi_file_dir = navi_file_dir
        self.navi_file_suffix = navi_file_suffix
        self.navi_info_path_mapping = navi_info_path_mapping
        self.navi_file_path_func = navi_file_path_func
        self.anchor_classify_func = anchor_classify_func
        self.reverse = reverse
        self.velo_scales = velo_scales

        self.seq_center_img_offset = np.array(
            [
                int(bev_origin_x / img_resolution),
                int(bev_origin_y / img_resolution),
            ]
        )
        self.anchor_type = ANCHOR_TYPE[anchor_type_version]
        self.anchor_str2idx = {v: k for k, v in self.anchor_type.items()}

    @staticmethod
    def trans_obstacle_bev_to_vcs(
        objects_xy, seq_center_img_offset, resolution, reverse
    ):
        """Trans the coordinates from BEV coordinates to VCS coordinates.

        Args:
            objects_xy (np.array, [num_obj, 2]): BEV coordinates
            seq_center_img_offset (ArrayLike): image coordinates of the map
                center of the dataframe.
            resolution (float): query resolution of the map.
            reverse (bool): if image is local BEV, and is drawn from the
                ego car upwards, the result need to reverse.

        Returns:
            vcs_coords (np.array, [num_obj, 2]): VCS coordinates
        """
        vcs_coords = objects_xy - seq_center_img_offset
        if reverse:
            vcs_coords = -vcs_coords
        inv_resolu = 1 / resolution
        x, y = Affine2D.coord_scale(
            vcs_coords[:, 0], vcs_coords[:, 1], inv_resolu, inv_resolu
        )
        vcs_coords = np.stack([x, y], axis=1)
        return vcs_coords

    def load_navi_information(self, plate, date_token, timestamp):
        """Load navigation information.

        Args:
            plate (str): the vehicle plate.
            date_token (str): the date token.
            timestamp (str): the time stamp of the navigation
                information.

        Returns:
            navi_info (Dict): the navigation information.
        """
        date = date_token.split("-")[0]
        date_key = plate + date
        timestamp = str(int(timestamp))
        navi_file = self.navi_file_path_func(
            self.navi_file_dir,
            self.navi_info_path_mapping[date_key],
            plate,
            date_token,
            timestamp,
            self.navi_file_suffix,
        )
        navi_info = None
        if os.path.exists(navi_file):
            with open(navi_file, "r") as f:
                navi_info = yaml.load(f, Loader=yaml.FullLoader)
            for _, data in navi_info.items():
                data["bounding_box"] = np.array(data["bounding_box"]).reshape(
                    [-1, 3]
                )
                data["drive_line"] = np.array(data["drive_line"]).reshape(
                    [-1, 3]
                )
        return navi_info

    @staticmethod
    def find_obstacle_located_logical_lane(
        all_track_ids: np.array,
        obs_vcs_coords: np.array,
        target_track_ids: Optional[Dict] = None,
        navi_information: Optional[Dict] = None,
    ):
        """Get the mapping between obstacles and their located lanes.

        Args:
            all_track_ids (np.array, [num_obj]): all track ids.
            obs_vcs_coords (np.array, [num_obj, 2]): the vcs coordinates
                of the obstacles.
            target_track_ids (np.array, [num_tar_obj]): the track id to
                search. Default to None, which means target_track_ids =
                all_track_ids.
            navi_information (Dict): the navigation information.
        """
        if navi_information is None:
            return {}
        if target_track_ids is None:
            target_track_ids = all_track_ids

        track_id_2_lane_id = {}
        for track_id in target_track_ids:
            obs_x, obs_y = obs_vcs_coords[
                np.where(all_track_ids == track_id)[0], :
            ][0]
            cur_lane_id = None
            for lane_id, lane in navi_information.items():
                if lane.get("bounding_box") is None:
                    continue
                lane_bbox = lane["bounding_box"]
                if len(lane_bbox) != 0:
                    max_x, max_y, _ = np.max(lane_bbox, axis=0)
                    min_x, min_y, _ = np.min(lane_bbox, axis=0)
                    lane_bbox = np.array(
                        [
                            [min_x, max_y],
                            [max_x, max_y],
                            [max_x, min_y],
                            [min_x, min_y],
                        ]
                    )
                    if is_point_in_convex_polygon(
                        obs_x, obs_y, lane_bbox[:, 0], lane_bbox[:, 1]
                    ):
                        if lane["is_logical"]:
                            cur_lane_id = lane_id
                            break
            track_id_2_lane_id[track_id] = cur_lane_id
        return track_id_2_lane_id

    def fill_location_column_in_seq_df(
        self,
        df,
        plate: str,
        date_token: str,
        lctx_frame_id: str,
        mode: str = "lctx",
        valid_track_ids: Optional[List] = None,
        navi_info: Optional[Dict] = None,
    ):
        """Fill the location columns in the data frame.

        Args: \
            df (DataFrame): the data frame. \
            plate (str): the vehicle plate. \
            date_token (str): the date token of the current dataset. \
            lctx_frame_id (str): the time stamp of the last context \
                frame.\
            mode (str, optional): how many timestamps should be filled. \
                Default to "lctx". \
            valid_track_ids (List): the valid track id list. \
            navi_info (Dict, optional): the navigation info. Default to \
                None. If the mode is "lctx" and the navi_info is not None, \
                the function will directly use the given navi_info. \
                Otherwise, the function will load navi_info from files. We \
                recommend the user manually input this parameter to avoid the \
                big IO time delay. \

        Returns: \
            df (DataFrame): the updated data frame. \
            stamp_to_navi_info (Dict): the mapping between time stamps to \
                the corresponding navigation information. \
        """
        vals = df.values
        cols = df.columns
        track_id_col = cols.get_loc("track_id")
        timestamp_col = cols.get_loc("timestamp")
        img_x_col = cols.get_loc("img_x")
        img_y_col = cols.get_loc("img_y")

        if mode == "ctx":
            # Fill all the context frames.
            ctx_frame_mask = df.frame_id <= lctx_frame_id
            all_timestamps = np.unique(
                vals[ctx_frame_mask, timestamp_col].astype("int")
            )
        elif mode == "lctx":
            # Just fill the last context frames.
            lctx_frame_mask = df.frame_id == lctx_frame_id
            all_timestamps = np.unique(
                vals[lctx_frame_mask, timestamp_col].astype("int")
            )
        else:
            all_timestamps = np.unique(vals[:, timestamp_col].astype("int"))

        stamp_to_navi_info = {}
        last_logical_lane_dict = {}
        for stamp in all_timestamps:
            stamp_mask = df["timestamp"] == stamp
            track_ids = vals[stamp_mask, track_id_col].astype("int")
            img_coords_x = vals[stamp_mask, img_x_col].astype("float")
            img_coords_y = vals[stamp_mask, img_y_col].astype("float")
            img_coords = np.stack([img_coords_x, img_coords_y], axis=1)
            if mode == "lctx" and valid_track_ids is not None:
                fill_track_ids = valid_track_ids
            else:
                fill_track_ids = track_ids
            # 1. Trans the obstacles in the last context frame to the VCS
            # coordinates.
            vcs_coords = self.trans_obstacle_bev_to_vcs(
                img_coords,
                self.seq_center_img_offset,
                self.img_resolution,
                self.reverse,
            )
            # 2. Load lane and drive line information file.
            if mode == "lctx" and navi_info is not None:
                navi_information = navi_info
            else:
                navi_information = self.load_navi_information(
                    plate, date_token, stamp
                )
            # 3. Find the obstacle located lane.
            idx_2_lane = self.find_obstacle_located_logical_lane(
                track_ids, vcs_coords, fill_track_ids, navi_information
            )
            for t_id in fill_track_ids:
                t_id = int(t_id)
                track_id_mask = df["track_id"] == t_id
                track_id_mask = track_id_mask & stamp_mask
                # 2.1 save lane id
                if idx_2_lane[t_id] is not None:
                    lane_id = idx_2_lane[t_id]
                    last_logical_lane_dict[t_id] = idx_2_lane[t_id]
                elif t_id in last_logical_lane_dict:
                    lane_id = last_logical_lane_dict[t_id]
                else:
                    lane_id = None
                if lane_id is not None:
                    df.loc[track_id_mask, "lane_id"] = lane_id
                    df.loc[track_id_mask, "in_logical_lane"] = 1
                else:
                    df.loc[track_id_mask, "lane_id"] = np.nan
                    df.loc[track_id_mask, "in_logical_lane"] = 0
                # 2.2 save coordinates in vcs coordinates.
                obs_x, obs_y = vcs_coords[np.where(track_ids == t_id)[0], :][0]
                df.loc[track_id_mask, "vcs_obs_x"] = obs_x
                df.loc[track_id_mask, "vcs_obs_y"] = obs_y
            if len(fill_track_ids) == 0:
                df["lane_id"] = np.nan
                df["in_logical_lane"] = 0
                df["vcs_obs_x"] = 0
                df["vcs_obs_y"] = 0
            stamp_to_navi_info[stamp] = navi_information
        return df, stamp_to_navi_info

    def __call__(self, sample):
        """Callable function.

        The following keys are required in `sample`:
            1. "seq_df"
            2. "valid_track_ids"
            3. "valid_class"
            4. "last_context_frame_id"
            5. "lcf_timestamp"
            7. "seq_center"
            8. "dataset_prefix"
            9. "date_token"
            10. "navi_info"

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample includes new keys 'sampled_anchors',
                'sampled_anchors_mask', 'valid_lane_chain_dict',
                'valid_drive_lines_dict'.
        """
        assert "navi_info" in sample, "Navigation information not found."
        df = sample["seq_df"]
        valid_track_ids = sample["valid_track_ids"]
        valid_cls = sample["valid_class"]
        last_context_frame_id = sample["last_context_frame_id"]
        timestamp = sample["lcf_timestamp"]
        seq_center = sample["seq_center"]
        plate = sample["dataset_prefix"]
        date_token = sample["date_token"]
        navi_info = sample["navi_info"]
        df, stamp_to_navi_info = self.fill_location_column_in_seq_df(
            df,
            plate,
            date_token,
            last_context_frame_id,
            mode="lctx",
            valid_track_ids=valid_track_ids,
            navi_info=navi_info,
        )

        num_obj = len(valid_track_ids)
        if num_obj == 0:
            sample["sampled_anchors"] = None
            sample["sampled_anchors_mask"] = None
            sample["valid_lane_chain_dict"] = None
            sample["valid_drive_lines_dict"] = None
            return sample

        vals = df.values
        cols = df.columns
        phy_x_col = cols.get_loc("x")
        phy_y_col = cols.get_loc("y")
        lane_logical_col = cols.get_loc("in_logical_lane")
        lane_id_col = cols.get_loc("lane_id")
        vcs_obs_x_col = cols.get_loc("vcs_obs_x")
        vcs_obs_y_col = cols.get_loc("vcs_obs_y")
        last_frame_mask = df.frame_id == last_context_frame_id
        ctx_frame_mask = df.frame_id <= last_context_frame_id

        all_anchors = np.zeros([num_obj, self.num_anchors, self.traj_len, 2])
        all_anchor_mask = np.zeros([num_obj, self.num_anchors])
        lane_chains_dict = {}
        drive_lines_dict = {}

        for t_idx, (track_id, track_cls) in enumerate(
            zip(valid_track_ids, valid_cls)
        ):
            agent_mask = df["track_id"] == track_id
            ctx_agent_mask = ctx_frame_mask & agent_mask
            lctx_agent_mask = last_frame_mask & agent_mask

            his_x = vals[ctx_agent_mask, phy_x_col].astype("float64")
            his_y = vals[ctx_agent_mask, phy_y_col].astype("float64")
            if len(his_x) < 2:
                continue
            in_logical_lane = vals[lctx_agent_mask, lane_logical_col].astype(
                "int"
            )[0]
            obs_x = vals[lctx_agent_mask, vcs_obs_x_col].astype("float")[0]
            obs_y = vals[lctx_agent_mask, vcs_obs_y_col].astype("float")[0]
            lane_id = vals[lctx_agent_mask, lane_id_col][0]
            his_yaw = np.arctan2(np.diff(his_y), np.diff(his_x))
            his_velo = np.sqrt(np.diff(his_x) ** 2 + np.diff(his_y) ** 2)
            lctx_velo = his_velo[-1]
            lctx_yaw_vcs = his_yaw[-1] - seq_center.yaw

            # Trace and concat the driveline.
            start_lane_id = lane_id
            if in_logical_lane:
                from_this_lane = True
            else:
                from_this_lane = False
            drive_lines, lane_chains = trace_and_concat_driveline(
                obs_x,
                obs_y,
                start_lane_id,
                stamp_to_navi_info[timestamp],
                from_this_lane,
            )
            lane_chains_dict[track_id] = lane_chains
            drive_lines_dict[track_id] = drive_lines

            # Sample from the driveline to get anchor trajectories.
            # -- Suppose the last context frame velocity is v, we
            #   sample by v, 1.5v and 0.5v here to cover the uncertainty
            #   of velocity.
            sampled_lines = []
            sampled_lines_mask = []
            sampled_s = np.arange(1, self.traj_len + 1, 1) * lctx_velo
            scaled_sampled_s = [
                sampled_s * scale for scale in self.velo_scales
            ]

            if drive_lines is None:
                continue
            for drive_line in drive_lines:
                if drive_line is None:
                    continue
                drive_line = np.concatenate(
                    [np.array([[obs_x, obs_y]]), drive_line], axis=0
                )
                for s in scaled_sampled_s:
                    (
                        sampled_driveline,
                        sampled_mask,
                    ) = interpolate_trajectory_from_path(drive_line, s)
                    sampled_lines.append(sampled_driveline)
                    sampled_lines_mask.append(sampled_mask)

            # Trans the sampled drive lines to obstacle centric coordinates.
            sampled_lines_centric = []
            for drive_line in sampled_lines:
                if len(drive_line) < 2:
                    continue
                x, y = drive_line[:, 0], drive_line[:, 1]
                x, y = Affine2D.coord_translate(x, y, obs_x, obs_y)
                x, y = Affine2D.coord_rotate(x, y, lctx_yaw_vcs)
                fut_yaw = np.arctan2(np.diff(y), np.diff(x))
                yaw_diff = abs(
                    normalize_yaw(fut_yaw[0]) - normalize_yaw(lctx_yaw_vcs)
                )
                if yaw_diff > np.pi / 18:
                    continue
                sampled_lines_centric.append(np.stack([x, y], axis=1))

            # Classify the sampled anchor trajectories.
            traj_types = []
            for traj in sampled_lines_centric:
                cur_type = self.anchor_classify_func(traj, track_cls)
                traj_types.append(cur_type)

            # Generate anchor array.
            for traj, traj_type in zip(sampled_lines_centric, traj_types):
                idx = int(self.anchor_str2idx[traj_type])
                all_anchors[t_idx, idx, :, :] = traj
                all_anchor_mask[t_idx, idx] = 1

        sample["sampled_anchors"] = all_anchors
        sample["sampled_anchors_mask"] = all_anchor_mask
        sample["valid_lane_chain_dict"] = lane_chains_dict
        sample["valid_drive_lines_dict"] = drive_lines_dict
        return sample
