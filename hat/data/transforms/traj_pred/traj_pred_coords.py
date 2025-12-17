# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import pickle
import random
from typing import Dict, List, Optional

import numpy as np

from hat.core.traj_pred_typing import SeqCenter, SeqIndex, VehicleSafeArea
from hat.core.traj_pred_utils import Affine2D
from hat.core.traj_pred_utils import TdtCoordHelper as TCH
from hat.data.datasets.traj_pred_dataset import BaseTrajDataset
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "GenSeqCenter",
    "GenBoundingBox",
    "GenObsShape",
    "PhyToBEV",
    "PhyToImage",
    "EgoCentricGt",
    "LateralSmoothing",
    "DetectCurve",
    "GenVisLanes",
    "GenObstaclesGoalsV2",
    "GenObstaclesGoalsV3",
]

logger = logging.getLogger(__name__)


class RandomCenterAffine:  # noqa: D205,D400
    """Randomly apply affine(translation + rotation) to the center of a
    sequence dataframe.

    We sample noise from two pre-defined uniform distribution and add them to
    the SeqCenter.
    """

    def __init__(
        self,
        max_x_perturb: float,
        max_y_perturb: float,
        max_yaw_perturb: float,
    ) -> None:
        """Initialize method.

        Args:
            max_x_perturb: how much to disturb the object position in
                the middle frame along x axis [m]. Setting this number too
                high will possibly result in empty seq_df.
            max_y_perturb: how much to disturb along y axis [m].
            max_yaw_perturb: how much to disturb the object yaw in the
                middle frame [Deg]. There is no restriction of yaw in this
                transform.
        """

        self.max_x_perturb = max_x_perturb
        self.max_y_perturb = max_y_perturb
        max_yaw_perturb = np.array(max_yaw_perturb) * np.pi / 180
        self.max_yaw_perturb = max_yaw_perturb

    def __call__(self, sample: Dict) -> Dict:
        """Call to randomly apply center perturbation to a sequence dataframe.

        The following keys are required in `sample`:
        1. 'seq_center'.

        Args:
            sample (Dict): input raw sample. The `sample` should be
                obtained by the `__getitem__` of trajectory dataset (base
                class `hat.data.datasets.traj_pred_dataset.BaseTrajDataset`)

        Returns:
            sample (Dict): output perturbed sample.
        """
        seq_center = sample["seq_center"]

        # Random affine transformation sampling.
        # Note: Don't use numpy's random.random() here, doing this will somehow
        # erase the randomness which is pretty weird.
        dist_perturb_x = random.uniform(
            -self.max_x_perturb, self.max_x_perturb
        )
        dist_perturb_y = random.uniform(
            -self.max_y_perturb, self.max_y_perturb
        )
        dist_perturb = np.array([dist_perturb_x, dist_perturb_y])
        yaw_perturb = random.uniform(
            -self.max_yaw_perturb, self.max_yaw_perturb
        )

        # Change seq_center according to sampled perturbation.
        new_pos_x, new_pos_y = np.array(seq_center[:2]) + dist_perturb
        new_yaw = (seq_center[-1] + yaw_perturb) % (2 * np.pi)
        new_seq_center = SeqCenter(new_pos_x, new_pos_y, new_yaw)

        sample["seq_center"] = new_seq_center

        return sample


@OBJECT_REGISTRY.register
class GenSeqCenter:
    """Generate SeqCenter for a sample."""

    def __init__(
        self,
        anchor_frame: str = "middle_seq",
        context_frames: int = None,
        force_zero_yaw: bool = False,
        augmentation: bool = False,
        max_x_perturb: float = 2.0,
        max_y_perturb: float = 2.0,
        max_yaw_perturb: float = 10.0,
        ego_track_id: float = -BaseTrajDataset.ANSWER,
    ) -> None:
        """Initialize method.

        Note(jingchu.liu): the `force_zero_yaw` argument is added to meet the
        need of legacy Tensorflow Dataset pipeline and visualization modules
        in which yaw is assumed to be zero.

        Args:
            anchor_frame: which frame to extract the center coordinates of
                the sequence. Choose from 'middle_seq', 'last_context',
                'first_seq', 'agent'. Defaults to 'middle_seq'.
            context_frames: number of context frames.
            force_zero_yaw: force the yaw of the generate SeqCenter to be
                zero.
            augmentation: whether to use center random affine augmentation
                (translation + rotation).
            max_x_perturb: the maximum perturbation value in x coordinates.
            max_y_perturb: the maximum perturbation value in y coordinates.
            max_yaw_perturb: the maximum perturbation value of yaw.
            ego_track_id: the track id of the ego vehicle.
        """
        self.anchor_frame = anchor_frame
        self.context_frames = context_frames
        self.force_zero_yaw = force_zero_yaw
        self.augmentation = augmentation
        self.ego_track_id = ego_track_id
        if self.augmentation:
            self.affine = RandomCenterAffine(
                max_x_perturb, max_y_perturb, max_yaw_perturb
            )

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        The following keys are required in `sample`:
            1. "seq_df".
            2. "seq_index".
            3. "last_context_frame_id"

        Args:
            sample (Dict): input original sample. The `sample` should be
                obtained by the `__getitem__` of trajectory dataset (base
                class `hat.data.datasets.traj_pred_dataset.BaseTrajDataset`)

        Returns:
            sample (Dict): output sample with a new 'seq_center' field.
        """
        seq_df = sample["seq_df"]
        last_context_frame_id = sample["last_context_frame_id"]

        if isinstance(sample["seq_index"], SeqIndex):
            seq_index = sample["seq_index"]
            agent_index = None
            if self.anchor_frame == "agent":
                agent_index = sample["agent_index"]
            seq_center = TCH.get_seq_center(
                df=seq_df,
                seq_index=seq_index,
                anchor_frame=self.anchor_frame,
                context_frames=self.context_frames,
                last_context_frame_id=last_context_frame_id,
                agent_index=agent_index,
            )
        elif isinstance(sample["seq_index"], List):
            self_traj = seq_df[seq_df["track_id"] == self.ego_track_id]
            pos_x = float(
                self_traj[self_traj["frame_id"] == last_context_frame_id][
                    "pos_x"
                ]
            )
            pos_y = float(
                self_traj[self_traj["frame_id"] == last_context_frame_id][
                    "pos_y"
                ]
            )
            yaw = float(
                self_traj[self_traj["frame_id"] == last_context_frame_id][
                    "yaw"
                ]
            )
            seq_center = SeqCenter(pos_x, pos_y, yaw)
        else:
            raise ValueError("Unsupported type of sample['seq_index'].")

        if self.force_zero_yaw:
            seq_center = SeqCenter(seq_center.pos_x, seq_center.pos_y, 0.0)

        sample["seq_center"] = seq_center
        if self.augmentation:
            self.affine(sample)

        return sample


@OBJECT_REGISTRY.register
class GenBoundingBox:  # noqa: D205,D400
    """Calculate the bounding boxes corner coordinates given physical
    center, shape and yaw. \

    To use, the user should construct a `GenBoundingBox` instance with the
    required initialization parameters. After instantiation, the callable
    function will return a new dict that contains the following changes: \
        1. the obstacle bounding box columns are added to (or just updated) \
            `seq_df`. The corresponding columns are 'x0', 'x1', 'x2', 'x3', \
            'y0', 'y1', 'y2', 'y3', 'z0', 'z1', 'z2', 'z3'.
    """

    # fmt: off
    EGO_COLS = ["pos_x", "pos_y", "yaw", "width", "length"]
    EGO_UPDATE_COLS = [
        "ego_x0", "ego_x1", "ego_x2", "ego_x3",
        "ego_y0", "ego_y1", "ego_y2", "ego_y3",
        "ego_z0", "ego_z1", "ego_z2", "ego_z3",
    ]
    OBS_COLS = ["x", "y", "obs_yaw", "obs_width", "obs_length"]
    OBS_UPDATE_COLS = [
        "x0", "x1", "x2", "x3", "y0", "y1", "y2", "y3",
        "z0", "z1", "z2", "z3",
    ]
    # fmt: on

    def __init__(
        self,
        gen_ego_bbox: bool = False,
        clockwise: bool = True,
        min_shape: float = None,
        const_z_value: float = None,
    ):
        """Initialize method.

        Args:
            gen_ego_bbox: if this parameter is True, the transform generates
                ego bounding box. Otherwise, it generates obstacle bounding
                box.
            clockwise: whether or not the corner points are sorted clockwise
                (in bird-eye view).
            min_shape: the min shape (length and width) of the obstacle [m].
            const_z_value: the constant z-coordinates of bounding box. If
                this value is None, this class will not update the columns
                "z0", "z1", "z2", "z3".
        """
        self.gen_ego_bbox = gen_ego_bbox
        self.clockwise = clockwise
        self.min_shape = min_shape
        self.const_z_value = const_z_value
        self.used_cols = self.EGO_COLS if gen_ego_bbox else self.OBS_COLS
        self.update_cols = (
            self.EGO_UPDATE_COLS if gen_ego_bbox else self.OBS_UPDATE_COLS
        )

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'x', 'y', 'obs_width', 'obs_length', 'obs_yaw'.

        Args:
            sample (Dict): input original sample. The `sample` should be
                obtained by the `__getitem__` of trajectory dataset (base
                class `hat.data.datasets.traj_pred_dataset.BaseTrajDataset`)

        Returns:
            sample (Dict): the updated sample as described above.
        """
        # Get the key of columns (to use or to update).
        x_key, y_key, yaw_key, w_key, l_key = self.used_cols
        bbox_x_keys = self.update_cols[0:4]
        bbox_y_keys = self.update_cols[4:8]
        bbox_z_keys = self.update_cols[8:12]

        seq_df = sample["seq_df"].copy()
        df_vals = seq_df.values
        df_cols = seq_df.columns
        phy_x_col = df_cols.get_loc(x_key)
        phy_y_col = df_cols.get_loc(y_key)
        phy_yaw_col = df_cols.get_loc(yaw_key)
        width_col = df_cols.get_loc(w_key)
        length_col = df_cols.get_loc(l_key)
        pos_cols = [phy_x_col, phy_y_col]
        shape_cols = [length_col, width_col]

        # Calculate bounding box shapes.
        obs_shapes = df_vals[:, shape_cols].astype("float64").reshape(-1, 2)
        if self.min_shape is not None:
            obs_shapes = np.maximum(obs_shapes, self.min_shape)
            obs_shape_vals = seq_df[[w_key, l_key]].values
            obs_shape_vals = np.maximum(obs_shape_vals, self.min_shape)
            seq_df[[w_key, l_key]] = obs_shape_vals

        # Calculate bounding boxes.
        # -- caluclate corner offsets: [df_length, 2, 4]
        if self.clockwise:
            offsets = np.array(
                [
                    [0.5, 0.5, -0.5, -0.5],  # x0, x1, x2, x3
                    [0.5, -0.5, -0.5, 0.5],  # y0, y1, y2, y3
                ]
            )
        else:
            offsets = np.array(
                [
                    [0.5, -0.5, -0.5, 0.5],  # x0, x1, x2, x3
                    [0.5, 0.5, -0.5, -0.5],  # y0, y1, y2, y3
                ]
            )
        offsets = obs_shapes[:, :, None] * offsets[None, :, :]
        # -- caluclate rotate matrix: [df_length, 4] -> [df_length, 2, 2]
        yaw = df_vals[:, [phy_yaw_col]].astype("float64")
        yaw = yaw.reshape(-1)[:, None]
        rotate = np.concatenate(
            (np.cos(yaw), -np.sin(yaw), np.sin(yaw), np.cos(yaw)), axis=-1
        )
        rotate = rotate.reshape([-1, 2, 2])
        # -- caluclate corners: [df_length, 2, 4]
        coors = df_vals[:, pos_cols].astype("float64")
        coors = coors.reshape(-1, 2)[:, :, None]
        corners = np.matmul(rotate, offsets) + coors
        # -- update the columns.
        seq_df[bbox_x_keys] = corners[:, 0, :]
        seq_df[bbox_y_keys] = corners[:, 1, :]
        if self.const_z_value is not None:
            seq_df[bbox_z_keys] = self.const_z_value

        sample["seq_df"] = seq_df
        return sample


@OBJECT_REGISTRY.register
class GenObsShape:  # noqa: D205,D400
    """Get the width and length of obstacles. \

    This transform should be used after `GenFutureTrackids`. \

    To use, the user should construct a `GenObsShape` instance with the
    required initialization parameters. After instantiation, the callable
    function will return a new dict that contains the following
    changes: \
    1. the `obs_length` was added.
    2. the `obs_width` was added.
    """

    def __init__(self):
        """Initialize method.

        Args:
            None
        """
        pass

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'
            2. 'track_ids'
            3. 'last_context_frame_id'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'obs_width', 'obs_length'.

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample with added 'obs_width' (list) and \
                           'obs_length' (list)
        """
        df = sample["seq_df"]
        track_ids = sample["track_ids"]
        last_context_frame_id = sample["last_context_frame_id"]
        vals = df.values
        cols = df.columns
        col_obs_width = cols.get_loc("obs_width")
        col_obs_length = cols.get_loc("obs_length")
        last_frame_mask = df.frame_id == last_context_frame_id

        obs_widths = []
        obs_lengths = []

        for track_id in track_ids:
            track_id_mask = df.track_id == track_id
            mask = track_id_mask & last_frame_mask

            obs_width = vals[mask, col_obs_width][0]
            obs_length = vals[mask, col_obs_length][0]

            obs_widths.append(obs_width)
            obs_lengths.append(obs_length)

        sample["obs_width"] = obs_widths
        sample["obs_length"] = obs_lengths

        return sample


@OBJECT_REGISTRY.register
class PhyToImage:
    """Transform trajectories from physical to image coordinate system.

    This class receive a raw sample from the dataset, and then apply
    'global_phy_df_to_local_img_df' to obtain the image coordinates of the
    object. \

    To use, the user should construct a `PhyToImage` instance with the
    required initialization parameters. After instantiation, the callable
    function will return a new dict that contains the following changes: \
        1. the image coordinate columns are added to `seq_df`; \
    """

    def __init__(
        self,
        map_height: int,
        map_width: int,
        resolution: float,
        reverse: bool = False,
    ):
        """Initialize method.

        Args:
            map_height: height of the map.
            map_width: width of the map.
            resolution: query resolution of the map.
            reverse: if the direction of VCS and image coordinate system
                is on the contrary (like in the BEV scenario), reverse
                should be True.
        """
        self.h = map_height
        self.w = map_width
        self.resolution = resolution
        self.reverse = reverse

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        The following keys are required:
            1. 'seq_center'.
            2. 'seq_df'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'pos_x', 'pos_y', 'ego_x0', 'ego_x1', 'ego_x2', 'ego_x3',
                'ego_y0', 'ego_y1', 'ego_y2', 'ego_y3', 'x', 'y',
                'x0', 'x1', 'x2', 'x3', 'y0', 'y1', 'y2', 'y3',
                'width', 'length', 'obs_width', 'obs_length', 'yaw'.

        Args:
            sample (Dict): input original sample. The `sample` should be
                obtained by the `__getitem__` of trajectory dataset (base
                class `hat.data.datasets.traj_pred_dataset.BaseTrajDataset`)

        Returns:
            sample (Dict): the updated sample as described above.
        """
        seq_center = sample["seq_center"]
        phy_seq_df = sample["seq_df"].copy()

        img_seq_df = TCH.global_phy_df_to_local_img_df(
            phy_df=phy_seq_df,
            seq_center=seq_center,
            seq_center_img_offset=np.array([self.h // 2, self.w // 2]),
            resolution=self.resolution,
            reverse=self.reverse,
        )

        sample["seq_df"] = img_seq_df

        return sample


@OBJECT_REGISTRY.register
class PhyToBEV(PhyToImage):
    """Transform trajectories from physical to BEV coordinate system.

    The differences between PhyToBEV and PhyToImage is whether the image
    offset is image center. In the BEV coordinates, the direction of VCS
    and BEV coordinate system are on the contrary (Unlike the usual
    image coordinate systems)
    """

    def __init__(
        self,
        bev_origin_x: float,
        bev_origin_y: float,
        resolution: float,
        reverse: bool = True,
    ):
        """Initialize method.

        Args:
            bev_origin_x: the x coordinate of the bev orgin [m] in the
                bev map coordinate.
            bev_origin_y: the y coordinate of the bev orgin [m] in the
                bev map coordinate.
            resolution: query resolution of the map.
            reverse: if the direction of VCS and image coordinate system
                are on the contrary (like in the BEV scenario), reverse
                should be True.
        """
        bev_image_origin_x = int(bev_origin_x / resolution)
        bev_image_origin_y = int(bev_origin_y / resolution)
        fake_map_height = bev_image_origin_x * 2
        fake_map_width = bev_image_origin_y * 2
        kwargs = {
            "map_height": fake_map_height,
            "map_width": fake_map_width,
            "resolution": resolution,
            "reverse": reverse,
        }
        super(PhyToBEV, self).__init__(**kwargs)


@OBJECT_REGISTRY.register
class EgoCentricGt:  # noqa: D205,D400
    """Transform the obstacle coordinates to its ego-centric VCS (Vehicle
    Coordinate System). This transfrom is necessary if the user requires
    ego-centric ground truth trajectories. \

    We assume that each sequence of trajectory only has context frames
    and a short range of future frames (like 6s), so our ego-centric
    ground truth trajectories are just future positions transformed to
    the last context frame's coordinate system. This transformation
    assumes that the new system's x axis is the direction of the obs_yaw. \

    Since the perception yaw is unstable sometimes, especially for pedestrian
    and cyclist. Therefore, some manually designed yaw selection method
    should be used before this class. Thus this transform should be used
    after "SelectYawArray".

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
            3. 'track_yaw_dict'

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
        last_frame_mask = seq_df["frame_id"] == last_context_frame_id
        lcf_track_ids = sample["track_ids"]
        track_yaw_dict = sample["track_yaw_dict"]

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
            agent_yaw = track_yaw_dict[track_id]
            if agent_yaw is None:
                continue
            # Extract all future x, y coordinates of the particular object
            object_coords = seq_df_vals[track_id_mask, :]
            object_coords = object_coords[:, [col_x, col_y]].astype("float64")
            obj_centric_coords = TCH.global_phy_to_agent_centric_phy(
                object_coords, obj_center[:2], agent_yaw[-1]
            ).astype("float64")
            seq_df.loc[
                track_id_mask, ["object_centric_x", "object_centric_y"]
            ] = obj_centric_coords

        sample["seq_df"] = seq_df
        return sample


@OBJECT_REGISTRY.register
class LateralSmoothing:
    """Perform Lateral Smoothing on obstacle trajectories.

    Sometimes, the obstacle perception position may bounce, which
    cause the predicted trajectories have unstable direction within
    a time sequence. Here, we smooth the lateral bounce to make
    prediction directions more stable.

    To use, the user should construct a `LateralSmoothing` instance with the
    required initialization parameters. After instantiation, the callable
    function will update the 'x' and 'y' columns of the input dataframe.

    Since this class change the vehicle position, the user should use
    `GenBoundingBox` and `PhyToImage` (or `PhyToBEV`) after this transform
    method.

    We recommend not to use this transform during training, because it may
    cause more time consuming and weaken the noise of data.
    """

    def __init__(self, squashing_scale: float = 10.0):
        """Initialize method.

        Args:
            squashing_scale (float, optional): the scale of the squashing
                function `sqrt(x * scale) / scale`. Defaults to 10.0.
        """
        self.squashing_scale = squashing_scale

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'last_context_frame_id'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'frame_id', 'x', 'y', 'obs_yaw'.

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample with updated 'x', 'y'.
        """
        seq_df = sample["seq_df"].copy()
        lcf_id = sample["last_context_frame_id"]

        for track_id in seq_df.track_id.unique():
            df = seq_df[seq_df.track_id == track_id]
            xy = df[["x", "y"]].values.astype("float64")
            yaw = df["obs_yaw"].values.astype("float64")
            frame_ids = list(df["frame_id"].values)
            if lcf_id not in frame_ids:
                continue
            lcf_offset = frame_ids.index(lcf_id)
            xy_diff = np.diff(xy, axis=0)
            rot = np.array(
                [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
            )[:, :, :-1]
            sl_diff = np.einsum("ij, jki -> ik", xy_diff, rot)
            sl_diff_mod = sl_diff[:]
            sl_diff_mod[:, 1] = (
                np.sign(sl_diff[:, 1])
                * np.sqrt(np.abs(sl_diff[:, 1]) * self.squashing_scale)
                / self.squashing_scale
            )
            xy_diff_mod = np.einsum(
                "ij, jki -> ik", sl_diff_mod, rot.transpose([1, 0, 2])
            )
            xy_diff_mod = np.concatenate(
                [np.ones([1, 2]), xy_diff_mod], axis=0
            )
            xy_mod = np.cumsum(xy_diff_mod, axis=0)
            # Align the coordinates in the last context frame to the original
            # values.
            xy_mod = (
                xy[lcf_offset, :][None, :]
                - xy_mod[lcf_offset, :][None, :]
                + xy_mod
            )
            seq_df.loc[seq_df.track_id == track_id, "x"] = xy_mod[:, 0]
            seq_df.loc[seq_df.track_id == track_id, "y"] = xy_mod[:, 1]

        sample["seq_df"] = seq_df
        return sample


@OBJECT_REGISTRY.register
class DetectCurve:
    """Detect whether the ego trajectory or road is curve.

    Sometimes, we need to know whether a road segment is curve, but we do not
    always have structured road information. In these cases, we need to
    estimate that the road segment is curve if there are enough curve
    trajectories. \

    In this function, we perform the linear fit based on the vehicle
    trajectory. If the residual is larger than a threshold, the trajectory is
    curve. \

    To use, the user should construct a `DetectCurve`. After instantiation,
    the callable function will return a new dict that contains the following
    changes: \
        1. the `is_curve` was added.
    """

    def __init__(
        self,
        seq_length: int,
        threshold: float = 1e-2,
        only_detect_ego: bool = True,
    ):
        """Initialize method.

        Args:
            seq_length (int): length of the sequence.
            threshold (float): the threshold of straight trajectories.
            only_detect_ego (bool, optional): whether to only detect
                ego vehicle. Default to True.
        """
        self.seq_length = seq_length
        self.threshold = threshold
        self.only_detect_ego = only_detect_ego

    @staticmethod
    def detect_curve_by_residual(x, y, threshold):
        """Detect whether the trajectory is curve.

        Args:
            x (np.array): the x coordinate list.
            y (np.array): the y coordinate list.
            threshold (float): the threshold of straight trajectories.
        """
        y = y[:, None]
        coefficient = np.vstack([x, np.ones_like(x)]).T
        residual = np.linalg.lstsq(coefficient, y, rcond=None)[1]
        if len(residual):
            return residual[0] > threshold
        else:
            return False

    @staticmethod
    def detect_curve(x, y, threshold):
        """Detect whether the trajectory is curve by length.

        Args:
            x (np.array): the x coordinate list.
            y (np.array): the y coordinate list.
            threshold (float): the threshold of straight trajectories.
        """
        dis_end_points = np.sqrt((y[-1] - y[0]) ** 2 + (x[-1] - x[0]) ** 2)
        dis_all_points = 0
        for i in range(1, len(x)):
            dis_all_points += np.sqrt(
                (y[i] - y[i - 1]) ** 2 + (x[i] - x[i - 1]) ** 2
            )
        if dis_end_points < 0.01:
            return True
        ratio = dis_all_points / dis_end_points
        return ratio > threshold

    def __call__(self, sample: dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'seq_df'.
            2. 'last_context_frame_id'.

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'frame_id', 'pos_x', 'pos_y', 'x', 'y', 'track_id'

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample includes a new value `is_curve`.
        """
        seq_df = sample["seq_df"]
        vals = seq_df.values
        cols = seq_df.columns
        frame_id_col = cols.get_loc("frame_id")
        phy_ego_x_col = cols.get_loc("pos_x")
        phy_ego_y_col = cols.get_loc("pos_y")
        phy_obs_x_col = cols.get_loc("x")
        phy_obs_y_col = cols.get_loc("y")
        track_id_col = cols.get_loc("track_id")
        # Detect whether ego trajectory is curve.
        _, unique_idx = np.unique(vals[:, frame_id_col], return_index=True)
        ego_x = vals[unique_idx, phy_ego_x_col].astype("float64")
        ego_y = vals[unique_idx, phy_ego_y_col].astype("float64")
        ego_curve = self.detect_curve(ego_x, ego_y, self.threshold)
        sample["is_curve"] = ego_curve
        if self.only_detect_ego:
            return sample
        # Detect whether has curve trajectories.
        lctx_frame_id = sample["last_context_frame_id"]
        lctx_mask = vals[:, frame_id_col] == lctx_frame_id
        track_ids = np.unique(vals[lctx_mask, track_id_col])
        has_curve_obs = False
        for track_id in track_ids:
            track_id_mask = vals[:, track_id_col] == track_id
            obs_x = vals[track_id_mask, phy_obs_x_col].astype("float64")
            obs_y = vals[track_id_mask, phy_obs_y_col].astype("float64")
            cur_obs_curve = self.detect_curve(obs_x, obs_y, self.threshold)
            has_curve_obs |= cur_obs_curve
        sample["is_curve"] = ego_curve or has_curve_obs
        return sample


@OBJECT_REGISTRY.register
class GenObstaclesGoalsV2:
    """Generate goal coords, and the goal (index) closest to the trajs' end_point.

    The difference between GenObstaclesGoalsV2 and V1 is that, in V1 we collect
    points in the center coordinate system of each obstacle, while in V2 we
    first collect points from VCS system and then convert points to each
    obstacle's center coordinate system. \

    This transform method requires the sample to have the following keys,
    which means that is must be used after some transform methods. \

    +------------------------------+----------------------------------------+
    |    requires                  |  needed transforms                     |
    +==============================+========================================+
    | track_ids,                   |   GenFutureTrackids,                   |
    | valid_track_ids,             |   FilterObstacles,                     |
    | context_states               |   GenStatesAndMask,                    |
    | future_trajectories          |   GetTrajPredObjectsInfo,              |
    | struct_road_feats,           |   VectorNetStructuredMapServer,        |
    | struct_num_road_elements,    |   VectorNetStructuredMapServer,        |
    | track_yaw_dict               |   SelectYawArray                       |
    | agent_classes                |   RemapObsCls                          |
    +------------------------------+----------------------------------------+

    To use, the user should construct a `GenObstaclesGoals`.
    The callable function will return a new dict that contains the following
    changes: \
    1. the `goal_coords` was added.
    2. the `nearest_goal_idxs` was added.
    3. the `nearest_road_idxs` was added.
    4. the `end_points` was added.
    5. the `vis_safe_area` was added.(for visualize temperally)
    6. the `safe_area_coords` was added.(for visualize temperally)
    7. the `vis_lanes` was added.(for visualize temperally)
    8. the `file_names` was added.(for visualize temperally)
    """

    def __init__(
        self,
        num_goals: int,
        goal_coords_scale: Optional[List],
        pad_coords: np.ndarray,
        map_origin_params: List,
        element_keys: List,
        image_coordinates: str = "bev",
        valid_img_coords_key: str = "valid_img_coords",
        include_road_ele: bool = True,
        road_ele_divide_num: int = 2,
        dense_goals_dis: int = 1,
        use_safe_area: bool = True,
        expand_traj_method: str = "CV",
        use_his_traj: bool = False,
        expand_base: int = 4,
        expand_ratio: float = 0.2,
        edge_divide_num: int = 5,
        vertical_divide_num: int = 10,
        hashv: float = 1.0,
        goal_interval: int = 1,
        input_num_ele_seg: int = 128,
        use_diff_trajs: bool = True,
    ):
        """Initialize method.

        Args:
            num_goals: 为每个障碍物生成采样点的数量.
            goal_coords_scale:采样点坐标数值的缩放比例
            pad_coords:补充采样点坐标
            map_origin_params: 图像坐标系下中心点坐标的偏移量.
            element_keys: 道路元素种类.
            image_coordinates: 图像坐标系的名字.
            valid_img_coords_key: valid_img_coords key的名称.
            include_road_ele: 是否在线段(道路元素或安全区)上采点.
            road_ele_divide_num: 每个道路元素的线段采集 road_ele_divide_num 个点.
            dense_goals_dis:使用该参数决定两侧采的点与线段的距离
            use_safe_area: 是否增加安全区采点
            expand_traj_method: use_safe_area=True时,使用该参数决定安全区的生成方式，默认使用"CV",
                即以当前时刻速度,"匀速"生成安全区
            use_his_traj:use_safe_area=True时,使用该参数决定是否使用历史轨迹生成安全区
            expand_base: use_safe_area=True时,该参数决定了安全区的宽度
            expand_ratio: use_safe_area=True时,使用该参数表示安全区宽度的扩展速率:
                [1 + (i-1) * expand_ratio] * expand_base.
            edge_divide_num: use_safe_area=True时,使用该参数表示每个安全区的边沿线段被分为几段.
            vertical_divide_num: use_safe_area=True时,
                使用该参数表示每个安全区的边沿线段做垂线(edge_divide_num决定每个线段做几条垂线),
                每条垂线采vertical_divide_num个点
            hashv:可以控制去重的效果,值越大,去重后采样点越稀疏
            num_points_persides:路口道路垂线的每侧采多少点
            goal_interval:polyline上每隔多少点取1个点
            input_num_ele_seg:densetnt需要输入backbone的num_ele_seg
            use_diff_trajs:是否使用差分轨迹
        """
        self.num_goals = num_goals
        self.road_ele_divide_num = road_ele_divide_num
        self.pad_coords = pad_coords
        self.dense_goals_dis = dense_goals_dis
        self.include_road_ele = include_road_ele
        self.use_safe_area = use_safe_area
        self.expand_traj_method = expand_traj_method
        self.use_his_traj = use_his_traj
        self.expand_base = expand_base
        self.expand_ratio = expand_ratio
        self.edge_divide_num = edge_divide_num
        self.vertical_divide_num = vertical_divide_num
        self.hashv = hashv
        self.goal_interval = goal_interval
        self.input_num_ele_seg = input_num_ele_seg
        self.goal_coords_scale = goal_coords_scale
        self.use_diff_trajs = use_diff_trajs
        self.map_origin_params = map_origin_params
        self.element_keys = element_keys
        self.image_coordinates = image_coordinates
        self.valid_img_coords_key = valid_img_coords_key

        # The basic feats is ["start_x", "start_y", "end_x", "end_y"]
        self.coor_feat_col_idx = [[0, 1], [2, 3]]

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

    def __call__(self, sample: dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'context_states'.
            2. 'future_trajectories'.
            3. 'track_ids'.
            4. 'valid_track_ids'.
            5. 'struct_road_feats'.
            6. 'struct_num_road_elements'.
            7. 'concat_dataset_index'.
            8. 'dataset_index'.
            9. 'track_yaw_dict'.
            10. 'agent_classes'.

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the updated sample as described above.
        """
        # List[array],len=num_obs
        all_obs_nearest_road_idxs = []
        all_obs_nearest_goal_idxs = []
        all_obs_goal_coords = []
        all_obs_goal_masks = []
        # List[List[array]],all_obs_vis_safe_area.len=num_obs
        all_obs_safe_area_polylines = []
        # List[array],all_obs_safe_area_coords.len=num_obs
        all_obs_safe_area_coords = []
        # List[array],all_obs_safe_area_coords.len=num_obs
        all_obs_end_points = []

        if self.use_diff_trajs:
            # List[array],all_obs_diff_trajs.len=num_obs
            all_obs_diff_trajs = []

        # important info
        valid_img_coords = sample[self.valid_img_coords_key]
        valid_track_ids = sample["valid_track_ids"]

        # set ctx_trajectories
        sample["ctx_trajectories"] = [
            sample["context_states"][sample["track_ids"].index(i)]
            for i in valid_track_ids
        ]

        all_feats = self.get_static_element_goal_feats(sample)

        for idx, track_id in enumerate(valid_track_ids):

            # 一个障碍物的所有采点集合，array.shape=(num_points,2)
            one_obs_goal_coords = []

            # future_traj(np.array):shape=(12,2),某valid障碍车未来轨迹的 center 坐标
            future_traj = sample["future_trajectories"][idx]

            agent_cls = sample["agent_classes"][
                sample["track_ids"].index(track_id)
            ]
            before_hash_points = self.trans_static_goal_to_centric(
                all_feats, agent_cls, valid_img_coords[idx]
            )

            # 计算差分轨迹的gt
            if self.use_diff_trajs:
                zero_arr = np.array([0, 0])
                # 为future_traj增加一行:(0,0),tmp_future_traj.shape=(13,2)
                tmp_future_traj = np.r_[[zero_arr], future_traj]
                # 差分轨迹，shape=(12,2)
                diff_fut_traj = np.diff(tmp_future_traj, axis=0)
                all_obs_diff_trajs.append(diff_fut_traj)

            # history_traj_tmp(np.array):shape=(4,2),某valid障碍车历史轨迹的 center 坐标
            history_traj = sample["ctx_trajectories"][idx]
            # ===保存每个障碍物的数据
            # 安全区采点集合,List[tuple],len=num_safe_area_points,tuple=(x,y)
            one_obs_safe_area_coords = []
            # List[array]
            one_obs_safe_area_polylines = np.zeros([1, 2])

            # ===得到安全区
            traj_len = future_traj.shape[0]
            tmp_safe_area = None
            his_yaw = sample["track_yaw_dict"][track_id]  # shape=(4,)
            ctx_frame_mask = [
                xy[0] ** 2 + xy[1] ** 2 != 0 for xy in history_traj[:-1]
            ]
            ctx_frame_mask.append(True)  # ctx_frame_mask:List[bool],len=4
            valid_his_traj = history_traj[ctx_frame_mask]
            his_x = valid_his_traj[:, 0]
            his_y = valid_his_traj[:, 1]
            if self.use_safe_area:
                # TODO(zifan.li):行人安全区的处理
                if (
                    np.sum(ctx_frame_mask) >= 2
                    and his_yaw is not None
                    and agent_cls != 1
                ):  # 历史帧+当前帧小于2，无法推断安全区，默认给坐标全为 0

                    # Get the expand trajectories in the global coordinates
                    displace = np.sqrt(
                        np.diff(his_x) ** 2 + np.diff(his_y) ** 2
                    )  # array.shape=(num_valid_his_frames-1,)
                    mean_dis = np.mean(displace)  # float,平均速度 m/0.5s
                    mean_dis += 2  # 提高上限可以预防未来时间内加速，导致轨迹终点超出安全区
                    # mean_dis=max(mean_dis,3)# 设置下限可以决定安全区的最短长度
                    # 以当前时刻速度和前进方向，匀速扩展安全区(前进方向)
                    if self.expand_traj_method == "CV":
                        start_idx = 1 if self.use_his_traj else 0
                        dis_x = (
                            np.arange(start_idx, traj_len + 1) * mean_dis
                        )  # array.shape=(traj_len or traj_len+1,)
                        dis_y = np.zeros(
                            traj_len + 1 - start_idx
                        )  # array.shape=(traj_len or traj_len+1,)
                        exp_traj = np.stack(
                            [dis_x, dis_y], axis=1
                        )  # array.shape=(traj_len or traj_len+1,2)
                    else:  # 目前只支持匀速
                        raise ValueError(
                            "Unsupported trajectory expanding method "
                            f"{self.expand_traj_method}."
                        )
                    # 若历史轨迹也要加安全区 use_his_traj
                    if self.use_his_traj:
                        exp_traj = np.concatenate(
                            [valid_his_traj, exp_traj], axis=0
                        )
                    # 获取机动车的安全区
                    tmp_safe_area = VehicleSafeArea(
                        traj=exp_traj,
                        obs_cls=agent_cls,
                        expand_base=self.expand_base,
                        expand_ratio=self.expand_ratio,
                    )
                    if tmp_safe_area.static:
                        tmp_safe_area = None
                # ===得到安全区内的线段
                if tmp_safe_area is not None:

                    one_obs_safe_area_polylines = tmp_safe_area.exp_bound_pts

                    # one_obs_safe_area_polylines 中的点的排布：
                    # 先上边后下边
                    # 上边从左到右，下边从左到右
                    top_edge = one_obs_safe_area_polylines[
                        0::2, :
                    ]  # array,shape=(13,2)
                    bottom_edge = one_obs_safe_area_polylines[
                        1::2, :
                    ]  # array,shape=(13,2)
                    one_obs_safe_area_polylines = np.vstack(
                        (top_edge, bottom_edge)
                    )

                    # ===安全区内的线段上采点
                    # 上侧边沿的线段
                    points_on_top_edge = self.collect_points_from_lines(
                        top_edge,
                        divide_num=self.edge_divide_num,
                        include_road_ele=True,
                    )
                    # 下侧边沿的线段
                    points_on_bottom_edge = self.collect_points_from_lines(
                        bottom_edge,
                        divide_num=self.edge_divide_num,
                        include_road_ele=True,
                    )

                    # 以前进方向为对称轴，两侧各有一个边沿上的点，这两个点关于前进方向是对称的，在两点的线段上采点，
                    for i in range(len(points_on_bottom_edge)):
                        # point:tuple.len=2

                        vertical_line = np.vstack(
                            (points_on_top_edge[i], points_on_bottom_edge[i])
                        )  # array.shape=(2,2)
                        # 安全区的对称线段上采点
                        vertical_points = self.collect_points_from_lines(
                            vertical_line,
                            divide_num=self.vertical_divide_num,
                            include_road_ele=True,
                        )
                        one_obs_safe_area_coords.extend(vertical_points)

                all_obs_safe_area_polylines.append(
                    [one_obs_safe_area_polylines]
                )
                if len(one_obs_safe_area_coords) > 0:
                    one_obs_safe_area_coords = np.vstack(
                        one_obs_safe_area_coords
                    )
                    before_hash_points = np.vstack(
                        [before_hash_points, one_obs_safe_area_coords]
                    )
                    all_obs_safe_area_coords.append(one_obs_safe_area_coords)
                else:
                    all_obs_safe_area_coords.append([])

            # ===去重的算法，当道路元素和安全区的点重合时有限去除道路元素的点
            after_hash_points = self.remove_dup_points(before_hash_points)

            # 初始采样点中去除重合度高的点
            # visit = {}
            # p_i = math.pi
            # for point in before_hash_points:
            #     hash = self.get_hash(point)
            #     if hash not in visit:
            #         visit[hash] = True
            #         if point[0] < 0:  # 第二第三象限的点，认为在障碍物后方
            #             angle = math.atan2(point[1], point[0])
            #             if angle > 3 * p_i / 4 or angle < (-3 * p_i / 4):
            #                 # if angle>5*p_i/6 or angle<(-5*p_i/6):
            #                 continue
            #         points_front_obs.append(point)
            front_points = []
            for point in after_hash_points:
                angle = np.arctan2(point[1], point[0])
                if angle > 0.75 * np.pi or angle < (-0.75 * np.pi):
                    continue
                front_points.append(point)
            front_points = np.vstack(front_points)

            displace = np.sqrt(
                np.diff(his_x) ** 2 + np.diff(his_y) ** 2
            )  # array.shape=(num_valid_his_frames-1,)
            if len(displace):
                mean_dis = np.mean(displace)
            else:
                mean_dis = 0
            abs_dis = np.sqrt(np.sum(front_points ** 2, axis=-1))
            abs_dis = np.abs(abs_dis - mean_dis * traj_len)
            sort_dis_idx = np.argsort(abs_dis)
            front_points = front_points[sort_dis_idx, :]

            if len(front_points) >= self.num_goals:
                one_obs_goal_coords = front_points[: self.num_goals, :]
                track_goal_mask = np.ones(self.num_goals)
            else:
                # ===从不足 num_goals 的采样点个数补零
                pad_point = [
                    self.pad_coords
                    for _ in range(self.num_goals - len(front_points))
                ]
                pad_point = np.array(pad_point)
                one_obs_goal_coords = np.concatenate(
                    [front_points, pad_point], axis=0
                )
                track_goal_mask = np.zeros(self.num_goals)
                track_goal_mask[: len(front_points)] = 1

            # array.shape=(num_goals,2)
            all_obs_goal_coords.append(one_obs_goal_coords)
            all_obs_goal_masks.append(track_goal_mask)

            # ===得到距离gt预测终点最近的采样点的索引
            # TODO(dukai.dong):gt终点无效的障碍物应该被过滤
            end_point = future_traj[-1]  # array.shape=(2,)
            one_obs_nearest_goal_idx = np.argmin(
                np.sum((one_obs_goal_coords - end_point) ** 2, axis=-1),
                axis=-1,
            )
            all_obs_end_points.append(end_point)
            all_obs_nearest_goal_idxs.append(one_obs_nearest_goal_idx)

            # 获取道路元素
            # [num_ele,polyline_len,2]
            road_feats = sample["struct_road_feats"][idx, :, :, :2]
            one_obs_nearest_road_idx = np.argmin(
                np.min(np.sum((road_feats - end_point) ** 2, axis=-1), axis=-1)
            )
            all_obs_nearest_road_idxs.append(one_obs_nearest_road_idx)

        # 差分轨迹
        if self.use_diff_trajs:
            sample["diff_fut_trajs"] = all_obs_diff_trajs

        # goal_coords:采样点坐标
        # nearest_goal_idxs:距离gt轨迹终点最近的道路元素
        # nearest_road_idxs:距离gt轨迹终点最近的采样点索引

        # shape=(num_valid_obs, num_goals, 2)
        sample["goal_coords"] = np.stack(all_obs_goal_coords)
        # shape=(num_valid_obs, num_goals)
        sample["goal_masks"] = np.stack(all_obs_goal_masks)

        sample["nearest_goal_idxs"] = np.array(
            all_obs_nearest_goal_idxs
        )  # shape=(num_valid_obs,)
        sample["nearest_road_idxs"] = np.array(
            all_obs_nearest_road_idxs
        )  # shape=(num_valid_obs,)
        sample["end_points"] = np.vstack(
            all_obs_end_points
        )  # shape=(num_valid_obs, 2)

        # vis_safe_area:用于可视化的安全区
        # safe_area_coords:用于可视化的安全区采样点
        # List[[array]],array.shape=(num_safe_area_polylines,2),len=num_obs
        sample["vis_safe_area"] = all_obs_safe_area_polylines
        # array.shape=(num_obs,num_safe_area_goals,2),len=num_obs
        sample["safe_area_coords"] = all_obs_safe_area_coords

        # 缩放比例
        if len(self.goal_coords_scale) == 0:
            self.goal_coords_scale = [1]
        sample["goal_coords_scale"] = self.goal_coords_scale
        return sample

    def get_static_element_goal_feats(self, sample):
        if "struct_road" not in sample:
            struct_feats = {}
        else:
            struct_feats = sample["struct_road"]
        all_feats = {}
        for key in self.element_keys:
            if key not in struct_feats or len(struct_feats[key]) == 0:
                continue
            if key == "stopline":
                continue
            key_vcs_struct_coords = struct_feats[key]
            ori_goal_coords = []
            vcs_goal_coords = []
            for polyline_list in key_vcs_struct_coords:

                # (num_polyline,3,2)
                tmp_polyline = np.array(polyline_list)

                # array.shape=(num_points,2)
                polyline = np.concatenate(
                    [tmp_polyline[:, :2, 0], tmp_polyline[-1:, :2, 1]]
                )

                if key == "crosswalk":
                    around_polyline_points = self.collect_points_from_lines(
                        polyline,
                        divide_num=7,
                        include_road_ele=False,
                        num_points_leftside=3,
                        num_points_rightside=3,
                        dense_goals_dis=self.dense_goals_dis * 2,
                    )
                elif key == "solid_lane":
                    around_polyline_points = self.collect_points_from_lines(
                        polyline,
                        divide_num=self.road_ele_divide_num,
                        include_road_ele=self.include_road_ele,
                        num_points_leftside=1,
                        num_points_rightside=1,
                        dense_goals_dis=self.dense_goals_dis,
                    )  # 每个 polyline 都采集 n 个点
                elif key == "roadedge":
                    around_polyline_points = self.collect_points_from_lines(
                        polyline,
                        divide_num=self.road_ele_divide_num,
                        include_road_ele=True,
                        num_points_leftside=0,
                        num_points_rightside=0,
                        dense_goals_dis=self.dense_goals_dis,
                    )  # 每个 polyline 都采集 n 个点
                vcs_goal_coords.extend(around_polyline_points)
                ori_goal_coords.extend(polyline)

            if len(vcs_goal_coords):
                vcs_goal_coords = np.vstack(vcs_goal_coords)
                ori_goal_coords = np.vstack(ori_goal_coords)
            all_feats[key] = vcs_goal_coords
            all_feats[f"ori_{key}"] = ori_goal_coords

        if "solid_lane" in all_feats:
            vcs_goal_coords = all_feats["solid_lane"]
            vcs_goal_coords = self.remove_dup_points(vcs_goal_coords)
            all_feats["solid_lane"] = vcs_goal_coords

        return all_feats

    def trans_static_goal_to_centric(self, all_feats, agent_cls, img_coords):
        zero_point = np.zeros([1, 2])
        offset_x, offset_y = self.img_center_offset
        img_x, img_y, img_yaw = img_coords
        img_resolu = self.img_resolu
        if self.reverse:
            vcs_x = -(img_x - offset_x / img_resolu) * img_resolu
            vcs_y = -(img_y - offset_y / img_resolu) * img_resolu
            vcs_yaw = img_yaw - np.pi
        else:
            vcs_x = (img_x - offset_x / img_resolu) * img_resolu
            vcs_y = (img_y - offset_y / img_resolu) * img_resolu
            vcs_yaw = img_yaw

        tmp_feats = [zero_point]
        for key, vcs_goal_coords in all_feats.items():
            if key == "cross_walk" and agent_cls != 1:
                continue
            trans_x, trans_y = Affine2D.coord_translate(
                vcs_goal_coords[:, 0], vcs_goal_coords[:, 1], vcs_x, vcs_y
            )
            trans_x, trans_y = Affine2D.coord_rotate(trans_x, trans_y, vcs_yaw)
            # 某障碍物centric坐标系下采样点坐标,shape=(num_goals,2)
            points = np.vstack([trans_x, trans_y]).transpose(1, 0)
            tmp_feats.append(points)
        tmp_feats = np.concatenate(tmp_feats, axis=0)
        return tmp_feats

    def remove_dup_points(self, points):
        ret_points = []
        visit = {}
        for point in points:
            hash_val = self.get_hash(point)
            if hash_val not in visit:
                visit[hash_val] = True
                ret_points.append(point)
        return np.vstack(ret_points)

    def get_hash(self, point: np.ndarray):
        """计算二维坐标的哈希值.

        Args:
            point: the x,y coordinate tuple (x,y).
        """
        return round((point[0] + 500) * self.hashv) * 1000000 + round(
            (point[1] + 500) * self.hashv
        )

    def get_interpolation_point(
        self, point_a: np.array, point_b: np.array, ratio: float
    ):
        """得到a,b两点间的一点c.

        Args:
            point_a (np.array): the x,y coordinate for point a.
            point_b (np.array): the x,y coordinate for point b.
            ratio (float): ac/ab=ratio.
        """
        return np.array(
            (
                point_a[0] * (1 - ratio) + point_b[0] * ratio,
                point_a[1] * (1 - ratio) + point_b[1] * ratio,
            )
        )

    def collect_points_from_lines(
        self,
        polyline: np.array,
        divide_num: int = 2,
        include_road_ele: bool = True,
        num_points_leftside: int = 0,
        num_points_rightside: int = 0,
        dense_goals_dis: float = 1.0,
    ):
        """
        在线段上或线段两侧进行采点.

        Args:
            polyline: 线段由 num_points 个点组成,shape=(num_points,2).
            divide_num: 每个线段采集 divide_num 个点.
            include_road_ele: 是否在线段上采点.
            num_points_leftside: 左侧采集多少点.
            num_points_rightside: 右侧采集多少点.
            dense_goals_dis: 两侧采样点的间距

        Returns:
            all_points (List): 列表元素为array,shape=(2,).
        """

        all_points = []
        road_ele_points = []  # 道路元素上的点
        point_pre = None
        start_point = np.array((polyline[0, 0], polyline[0, 1]))  # 先增加线段起点
        road_ele_points.append(start_point)
        for i, point in enumerate(polyline):
            if i > 0:
                # for k in range(1, divide_num):
                for k in range(
                    1, divide_num + 1
                ):  # 在线段的 1/n,2/n,...,n/n 处分别采点，其中 n/n 是线段终点
                    inter_points = self.get_interpolation_point(
                        point_pre, point, k / divide_num
                    )
                    road_ele_points.append(inter_points)
            point_pre = point  # 前一个采样点的坐标(x1,y1)
        if include_road_ele:
            all_points.extend(road_ele_points)

        road_ele_points = np.vstack(road_ele_points)

        if num_points_leftside > 0 or num_points_rightside > 0:
            rot_mat = np.array([[0, 1], [-1, 0]])
            # shape=(num_road_ele_points-1,2)
            der_points = np.diff(road_ele_points, axis=0)
            if len(der_points) > 0:
                valid_point_mask = ~np.all(der_points == 0, axis=-1)
                der_points = der_points[valid_point_mask, :]
                valid_point_mask = np.concatenate(
                    [np.array([True]), valid_point_mask]
                )
                road_ele_points = road_ele_points[valid_point_mask, :]
                scale = 1 / np.sqrt(np.sum(der_points ** 2, axis=-1))
                scale = np.repeat(scale[:, None], 2, 1)
                der_points *= scale * dense_goals_dis
                rot_arr = np.matmul(der_points, rot_mat)
                rot_arr = np.concatenate([rot_arr[0:1, :], rot_arr])
                for k in range(-num_points_leftside, num_points_rightside + 1):
                    if k != 0:
                        all_points.append(road_ele_points + k * rot_arr)
        return all_points


@OBJECT_REGISTRY.register
class GenObstaclesGoalsV3:
    """Generate goal coords - version 3.

    这个版本直接使用此前VectorNetStructuredMapServer的静态元素采样点衍生终点
    采样点，同时修改了安全区计算的方式（改为查表）。这样的采样方式更简单，且更容易
    软件端复现。
    """

    def __init__(
        self,
        num_goals: int,
        goal_coords_scale: Optional[List],
        pad_coords: np.ndarray,
        endpts_file_dir: str,
        use_diff_trajs: bool = True,
        fut_time: float = 6,
        offset: float = 1.875,
    ):
        """Initialize method.

        Args:
            num_goals: 为每个障碍物生成采样点的数量.
            goal_coords_scale: 采样点坐标数值的缩放比例.
            pad_coords: 补充采样点坐标.
            use_diff_trajs: 是否使用差分轨迹.
            fut_time: 预测的未来时间长度.
            max_safe_area_velo: 生成安全区的最大感知速度.
            offset: 基于静态元素坐标进行扰动得到采样点时的扰动距离(m).
        """
        self.num_goals = num_goals
        self.goal_coords_scale = goal_coords_scale
        self.pad_coords = pad_coords
        self.use_diff_trajs = use_diff_trajs
        self.fut_time = fut_time
        self.offset = offset

        # 读取根据数据集统计得到的终点集合
        with open(endpts_file_dir, "rb") as f:
            self.endpoint_anchors = pickle.load(f)
        self.endpts_stat = {}
        for k_cls in self.endpoint_anchors.keys():
            self.endpts_stat[k_cls] = {}
            for k in self.endpoint_anchors[k_cls].keys():
                pts = self.endpoint_anchors[k_cls][k]
                pts_x_mean = np.mean(pts[:, 0])
                pts_x_var = np.std(pts[:, 0])
                pts_y_mean = np.mean(pts[:, 1])
                pts_y_var = np.std(pts[:, 1])
                self.endpts_stat[k_cls][k] = [
                    pts_x_mean,
                    pts_x_var,
                    pts_y_mean,
                    pts_y_var,
                ]

    @staticmethod
    def remove_dup_points(points):
        """基于hash对采样点去重."""

        def get_hash(point: np.ndarray, hashv: float = 0.75):
            """计算二维坐标的哈希值.

            Args:
                point: the x,y coordinate tuple (x,y).
            """
            return round((point[0] + 500) * hashv) * 1000000 + round(
                (point[1] + 500) * hashv
            )

        ret_points = []
        visit = {}
        for point in points:
            hash_val = get_hash(point)
            if hash_val not in visit:
                visit[hash_val] = True
                ret_points.append(point)
        if len(ret_points):
            return np.vstack(ret_points)
        else:
            return []

    def __call__(self, sample: dict):
        """Callable function.

        The following keys are required in `sample`:
            1. 'struct_road_feats'.
            2. 'struct_road_masks'.
            3. 'ori_state_vectors'.
            4. 'future_trajectories'.

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the updated sample as described above.
        """
        road_feats = sample["struct_road_feats"]
        road_masks = sample["struct_road_masks"]
        obs_velo = sample["ori_state_vectors"][:, 0]
        future_trajs = sample["future_trajectories"]
        valid_track_ids = sample["valid_track_ids"]
        ctx_trajectories = [
            sample["context_states"][sample["track_ids"].index(i)]
            for i in valid_track_ids
        ]  # 障碍物坐标系下的历史轨迹

        y_offset = np.array([0, self.offset])[None, :]

        all_obs_nearest_road_idxs = []
        all_obs_nearest_goal_idxs = []
        all_obs_goal_coords = []
        all_obs_goal_masks = []
        # List[List[array]],all_obs_vis_safe_area.len=num_obs
        all_obs_end_points = []
        if self.use_diff_trajs:
            # List[array],all_obs_diff_trajs.len=num_obs
            all_obs_diff_trajs = []

        for (
            tmp_feat,
            tmp_mask,
            tmp_velo,
            fut_traj,
            track_id,
            history_traj,
        ) in zip(
            road_feats,
            road_masks,
            obs_velo,
            future_trajs,
            valid_track_ids,
            ctx_trajectories,
        ):
            agent_cls = sample["agent_classes"][
                sample["track_ids"].index(track_id)
            ]

            # 一个障碍物的所有采点集合，array.shape=(num_points,2)
            one_obs_goal_coords = []

            valid_tmp_feat = tmp_feat[tmp_mask.astype("bool"), :, 2:4]
            valid_tmp_feat = valid_tmp_feat.reshape([-1, 2])
            valid_tmp_feat = self.remove_dup_points(valid_tmp_feat)
            # 避免空图无采样点，下方vstack报错，将匀速外推的预瞄点加入
            front_points = [np.array([tmp_velo * self.fut_time, 0])]
            for point in valid_tmp_feat:
                angle = np.arctan2(point[1], point[0])
                if angle > 0.55 * np.pi or angle < (-0.55 * np.pi):
                    continue
                front_points.append(point)
            front_points = np.vstack(front_points)

            expand_points = np.concatenate(
                [
                    front_points + y_offset,
                    front_points - y_offset,
                ],
                axis=0,
            )

            # 根据查表添加动态点
            ctx_frame_mask = [
                xy[0] ** 2 + xy[1] ** 2 != 0 for xy in history_traj[:-1]
            ]
            ctx_frame_mask.append(True)  # ctx_frame_mask:List[bool],len=4
            valid_his_traj = history_traj[ctx_frame_mask]
            his_x = valid_his_traj[:, 0]
            his_y = valid_his_traj[:, 1]

            if np.sum(ctx_frame_mask) >= 2:  # 历史帧+当前帧小于2，无法推断安全区，默认给坐标全为 0
                velo_x = int(2 * np.diff(his_x).mean())
                velo_y = int(2 * np.diff(his_y).mean())
            else:
                velo_x = 0
                velo_y = 0

            if agent_cls == 0:  # 车辆，根据x方向速度分区
                cls_key = "veh_endpts"
                velo_key = max(0, min(velo_x, 30))
            elif agent_cls == 1:  # 行人，根据y方向速度分区
                cls_key = "ped_endpts"
                if velo_y < -0.4:
                    velo_key = -1
                elif velo_y < 0.4:
                    velo_key = 0
                else:
                    velo_key = 1
            else:  # 骑车人，根据x方向速度分区
                cls_key = "cyc_endpts"
                velo_key = max(0, min(velo_x, 15))

            one_obs_safe_area_coords = self.endpoint_anchors[cls_key][velo_key]
            endpts_x_mean = self.endpts_stat[cls_key][velo_key][0]
            endpts_x_var = self.endpts_stat[cls_key][velo_key][1]
            endpts_y_mean = self.endpts_stat[cls_key][velo_key][2]
            endpts_y_var = self.endpts_stat[cls_key][velo_key][3]

            expand_points = np.vstack(
                [expand_points, one_obs_safe_area_coords]
            )

            # ===去重的算法，当道路元素和安全区的点重合时有限去除道路元素的点

            expand_points = self.remove_dup_points(expand_points)

            # 根据到菱形等高线距离排序
            abs_dis_x = (
                np.abs(expand_points[:, 0] - endpts_x_mean) / endpts_x_var
            )
            abs_dis_y = (
                np.abs(expand_points[:, 1] - endpts_y_mean) / endpts_y_var
            )
            abs_dis = abs_dis_x + abs_dis_y
            sort_dis_idx = np.argsort(abs_dis)
            sorted_expand_points = expand_points[sort_dis_idx, :]

            # 采样和补零
            if len(sorted_expand_points) >= self.num_goals:
                one_obs_goal_coords = sorted_expand_points[: self.num_goals, :]
                track_goal_mask = np.ones(self.num_goals)
            else:
                # ===从不足 num_goals 的采样点个数补零
                pad_point = [
                    self.pad_coords
                    for _ in range(self.num_goals - len(sorted_expand_points))
                ]
                pad_point = np.array(pad_point)
                one_obs_goal_coords = np.concatenate(
                    [sorted_expand_points, pad_point], axis=0
                )
                track_goal_mask = np.zeros(self.num_goals)
                track_goal_mask[: len(sorted_expand_points)] = 1

            # array.shape=(num_goals,2)
            all_obs_goal_coords.append(one_obs_goal_coords)
            all_obs_goal_masks.append(track_goal_mask)

            # ===得到距离gt预测终点最近的采样点的索引
            # TODO(dukai.dong):gt终点无效的障碍物应该被过滤
            end_point = fut_traj[-1]  # array.shape=(2,)
            one_obs_nearest_goal_idx = np.argmin(
                np.sum((one_obs_goal_coords - end_point) ** 2, axis=-1),
                axis=-1,
            )
            all_obs_end_points.append(end_point)
            all_obs_nearest_goal_idxs.append(one_obs_nearest_goal_idx)

            # 获取道路元素
            # [num_ele,polyline_len,2]
            road_feats = tmp_feat[:, :, :2]
            one_obs_nearest_road_idx = np.argmin(
                np.min(np.sum((road_feats - end_point) ** 2, axis=-1), axis=-1)
            )
            all_obs_nearest_road_idxs.append(one_obs_nearest_road_idx)

            # 差分轨迹
            if self.use_diff_trajs:
                zero_arr = np.array([0, 0])
                # 为future_traj增加一行:(0,0),tmp_future_traj.shape=(13,2)
                tmp_future_traj = np.r_[[zero_arr], fut_traj]
                # 差分轨迹，shape=(12,2)
                diff_fut_traj = np.diff(tmp_future_traj, axis=0)
                all_obs_diff_trajs.append(diff_fut_traj)

        if self.use_diff_trajs:
            sample["diff_fut_trajs"] = all_obs_diff_trajs

        # goal_coords:采样点坐标
        # nearest_goal_idxs:距离gt轨迹终点最近的道路元素
        # nearest_road_idxs:距离gt轨迹终点最近的采样点索引

        # shape=(num_valid_obs, num_goals, 2)
        sample["goal_coords"] = np.stack(all_obs_goal_coords)
        # shape=(num_valid_obs, num_goals)
        sample["goal_masks"] = np.stack(all_obs_goal_masks)

        sample["nearest_goal_idxs"] = np.array(
            all_obs_nearest_goal_idxs
        )  # shape=(num_valid_obs,)
        sample["nearest_road_idxs"] = np.array(
            all_obs_nearest_road_idxs
        )  # shape=(num_valid_obs,)
        sample["end_points"] = np.vstack(
            all_obs_end_points
        )  # shape=(num_valid_obs, 2)
        sample["ctx_trajectories"] = [
            sample["context_states"][sample["track_ids"].index(i)]
            for i in sample["valid_track_ids"]
        ]

        # 缩放比例
        if len(self.goal_coords_scale) == 0:
            self.goal_coords_scale = [1]
        sample["goal_coords_scale"] = self.goal_coords_scale
        return sample


@OBJECT_REGISTRY.register
class GenVisLanes:
    """Generate lanes from vector coordinates, just for visualization.

    This transform method requires the sample to have the following keys,
    which means that is must be used after some transform methods. \

    +------------------------------+----------------------------------------+
    |    requires                  |  needed transforms                     |
    +==============================+========================================+
    | track_ids,                   |   GenFutureTrackids,                   |
    | valid_track_ids,             |   FilterObstacles,                     |
    | valid_img_coords             |   GetTrajPredObjectsInfo,              |
    | vcs_vector_map_feats         |   VectorNetStructuredMapServer,        |
    +------------------------------+----------------------------------------+

    To use, the user should construct a `GenObstaclesGoals`.
    The callable function will return a new dict that contains the following
    changes: \
    1. the `vis_lanes` was added.(for visualize temperally)
    2. the `file_names` was added.(for visualize temperally)
    """

    def __init__(
        self,
        map_origin_params: List,
        element_keys: List,
        image_coordinates: str = "bev",
        valid_img_coords_key: str = "valid_img_coords",
    ):
        """Initialize method.

        Args:
            map_origin_params: paramters to calculate the offset of the
                map center in the map coordinates.
            element_keys: elements to consider during extracting map
                features. It should be a subset of `SUPPORTED_KEYS`.
            image_coordinates: the name of the image coordinates. Defaults
                to "bev". It should be one of ["bev", "img]. The corresponding
                coordinate trans method is ["PhyToBEV", "PhyToImg"].
            valid_img_coords_key: the key name of the img coords of valid
                track ids.
        """
        self.map_origin_params = map_origin_params
        self.element_keys = element_keys
        self.image_coordinates = image_coordinates
        self.valid_img_coords_key = valid_img_coords_key

        # The basic feats is ["start_x", "start_y", "end_x", "end_y"]
        self.coor_feat_col_idx = [[0, 1], [2, 3]]

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

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. "struct_road"
            2. "valid_img_coords"

        Args:
            sample (Dict): input original sample.

        """

        all_obs_vis_lanes = []
        all_obs_file_names = []
        valid_img_coords = sample[self.valid_img_coords_key]
        struct_feats = sample["struct_road"]
        valid_track_ids = sample["valid_track_ids"]
        offset_x, offset_y = self.img_center_offset
        img_resolu = self.img_resolu
        for idx, track_id in enumerate(valid_track_ids):

            img_x, img_y, img_yaw = valid_img_coords[idx]

            if self.reverse:
                vcs_x = -(img_x - offset_x / img_resolu) * img_resolu
                vcs_y = -(img_y - offset_y / img_resolu) * img_resolu
                vcs_yaw = img_yaw - np.pi
            else:
                vcs_x = (img_x - offset_x / img_resolu) * img_resolu
                vcs_y = (img_y - offset_y / img_resolu) * img_resolu
                vcs_yaw = img_yaw

            one_obs_vis_lanes = {}

            for key in self.element_keys:
                if key not in struct_feats:
                    continue

                # List[arr],len=num_polyline
                key_vcs_list = []
                # List[List],len=num_polyline
                key_vcs_struct_coords = struct_feats[key]

                for polyline_list in key_vcs_struct_coords:

                    # len=num_points
                    polyline = []

                    for i, polyline_seg in enumerate(polyline_list):
                        # polyline_seg:[[x1,x2],[y1,y2],[z1,z2]]

                        if i == 0:
                            # array([[x1,y1],[x2,y2]])
                            tmp_arr = np.array(
                                [
                                    [polyline_seg[0][0], polyline_seg[1][0]],
                                    [polyline_seg[0][1], polyline_seg[1][1]],
                                ]
                            )
                        else:
                            tmp_arr = np.array(
                                [[polyline_seg[0][1], polyline_seg[1][1]]]
                            )
                        polyline.append(tmp_arr)

                    # array.shape=(num_points,2)
                    polyline = np.vstack(polyline)
                    trans_x, trans_y = Affine2D.coord_translate(
                        polyline[:, 0], polyline[:, 1], vcs_x, vcs_y
                    )
                    trans_x, trans_y = Affine2D.coord_rotate(
                        trans_x, trans_y, vcs_yaw
                    )
                    polyline[:, 0] = trans_x
                    polyline[:, 1] = trans_y
                    key_vcs_list.append(polyline)

                one_obs_vis_lanes[key] = key_vcs_list

            all_obs_vis_lanes.append(one_obs_vis_lanes)
            # 用于标志每个sample的每个障碍物
            file_name = (
                str(sample["concat_dataset_index"])
                + "_"
                + str(sample["dataset_index"])
                + "_"
                + str(int(track_id))
            )
            all_obs_file_names.append(file_name)

        sample["vis_lanes"] = all_obs_vis_lanes
        sample["file_names"] = all_obs_file_names

        return sample
