# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import math
import random
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.draw import line, polygon

from hat.core.box_utils import get_bev_bbox
from hat.core.point_geometry import coor_transformation
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "StateNormalize",
    "RasterizedDynDataGenerator",
    "RotateAugmentation",
    "CropAndResize",
    "FilteroutUselessAgents",
    "TargetGenerator",
    "DropUselessItems",
    "HWC2CHW",
    "AsType",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class StateNormalize(object):
    """
    Normalize the history state and raster_map, it is benifit for training \
    stability and quantized training.

    Args:
        scale: The reduction ratio of the raster_map.
    """

    def __init__(self, scale: float = 255):
        self.scale = scale

    def __call__(self, sample):
        if "history_state" in sample:
            T = sample["history_state"].shape[1]
            scale = np.arange(1, T + 1) / T * 40
            sample["history_state"] /= scale[..., None]
        if "raster_map" in sample:
            sample["raster_map"] /= self.scale
        return sample


@OBJECT_REGISTRY.register
class RasterizedDynDataGenerator(object):
    """
    Generate the rasterized_dynamic_data by history_state, which is a tensor \
    shaped H x W x state_length. The history_state will be filled to the \
    current positions of the agents.

    Args:
        max_rotate_angle: The history_state will rotate X radians, X is
            a random number in [-max_rotate_angle, +max_rotate_angle]. When
            max_rotate_angle <= 0, the history_state will not be rotated.
            When max_rotate_angle > 0, the 'RotateAugmentation' transforms
            need to be used after 'RasterizedDynDataGenerator'.
        history_mask_prob: The probability of removing several frames
            of history_state, for data augmentation.
        topology: If true, the topological relationship of agents will
            be drawn and concatenate to rasterized_dynamic_data in channel.
            The topology contains two channels, representing graph and
            centerlines respectively. Nodes in the graph are agents.
        topology_dist_thred: If the distance between any two agents is
            greater than it, the relative edge will be ignored.
        topology_color: The color of center line of lanes.
    """

    def __init__(
        self,
        max_rotate_angle: float = -1,
        history_mask_prob: float = -1,
        topology: bool = False,
        topology_dist_thred: float = 150,
        topology_color: Tuple[float, ...] = (255, 127, 0),
    ):
        self.max_rotate_angle = max_rotate_angle
        self.history_mask_prob = history_mask_prob
        self.topology = topology
        self.topology_dist_thred = topology_dist_thred
        self.topology_color = topology_color

    def __call__(self, sample: dict):
        N, T = sample["history"].shape[:2]
        H, W = sample["raster_map"].shape[:2]

        if self.max_rotate_angle > 0:
            angle = random.uniform(
                -self.max_rotate_angle, self.max_rotate_angle
            )
            sample["history_state"] = coor_transformation(
                sample["history_state"].reshape([-1, 2]), theta=angle
            ).reshape([N, T, -1])
            sample["angle"] = angle

        if random.uniform(0, 1) < self.history_mask_prob:
            mask_step = random.uniform(2, T)
            mask = np.arange(1, T + 1) <= mask_step
            sample["history_mask"] *= mask[None]
            sample["history"] *= mask[None, :, None]
            sample["history_state"] *= mask[None, :, None]

        sample["rasterized_dynamic_data"] = self.get_rasterized_dyn_data(
            sample["history_state"],
            (H, W),
            sample["history"][:, 0],
            sample.get("size", None),
            sample.get("yaw", None),
        )

        if self.topology:
            topology_map = np.zeros((H, W))
            for i in range(N):
                for j in range(N):
                    if i >= j:
                        continue
                    x_0, y_0 = sample["history"][i, 0]
                    x_1, y_1 = sample["history"][j, 0]
                    if (
                        x_0 < 0
                        or x_1 < 0
                        or y_0 < 0
                        or y_1 < 0
                        or x_0 >= H
                        or x_1 >= H
                        or y_0 >= W
                        or y_1 >= W
                    ):
                        continue

                    distance = math.sqrt((x_0 - x_1) ** 2, (y_0 - y_1) ** 2)
                    if distance > self.topology_dist_thred:
                        continue

                    xx, yy = line(
                        *sample["history"][i, 0].astype("int32"),
                        *sample["history"][j, 0].astype("int32")
                    )
                    xx = np.clip(xx, a_min=0, a_max=H - 1)
                    yy = np.clip(yy, a_min=0, a_max=W - 1)
                    topology_map[xx, yy, 0] = 1
            topology_map = np.stack(
                [
                    topology_map,
                    np.where(
                        np.all(
                            sample["raster_map"] == self.topology_color,
                            axis=-1,
                        ),
                        1,
                        0,
                    ),
                ],
                axis=-1,
            )
            sample["rasterized_dynamic_data"] = np.concatenate(
                [topology_map, sample["rasterized_dynamic_data"]], axis=-1
            )
        return sample

    @staticmethod
    def get_rasterized_dyn_data(
        states, img_size, coordinate, size=None, yaw=None, radius=5
    ):
        N = states.shape[0]
        states = np.reshape(states, [N, -1])
        rasterized_dynamic_data = np.zeros([*img_size, states.shape[1]])

        if size is None:  # draw agent as circles
            r = radius
            kernel = np.zeros([2 * r + 1, 2 * r + 1, 1])
            cv2.circle(kernel, (r, r), r, (1), -1)
            for i, pixel in enumerate(coordinate):
                x, y = pixel.astype(np.int32)
                if (
                    x - r < 0
                    or x + r + 1 > img_size[0]
                    or y - r < 0
                    or y + r + 1 > img_size[1]
                ):
                    continue
                rasterized_dynamic_data[
                    x - r : x + r + 1, y - r : y + r + 1
                ] += np.where(kernel, states[i][None, None], 0)
        else:  # draw agent bev bounding box
            bbox = get_bev_bbox(coordinate, size, yaw)

            for i, b in enumerate(bbox):
                xx, yy = polygon(
                    b.astype("int32")[:, 0], b.astype("int32")[:, 1]
                )
                xx = np.clip(xx, a_min=0, a_max=img_size[0] - 1)
                yy = np.clip(yy, a_min=0, a_max=img_size[1] - 1)
                rasterized_dynamic_data[xx, yy] = states[i]
        return rasterized_dynamic_data


@OBJECT_REGISTRY.register
class RotateAugmentation(object):
    """
    Rotate the raster_map, rasterized_dynamic_data and history_state for \
    data augmentation.

    Args:
        max_rotate_angle: The data will rotate X radians, X is
            a random number in [-max_rotate_angle, +max_rotate_angle].
            When the 'angle' item is in the input dict, the X will be between
            -angle and +angle.
        rotate_dynamic_data: If true, the filled state in
            rasterized_dynamic_data will be rotated.
        border_value: The value to fill in the out-of-bounds area
            caused by the rotation.
    """

    def __init__(
        self,
        max_rotate_angle: float = 0,
        rotate_dynamic_data: bool = True,
        border_value: float = 1,
    ):
        self.max_rotate_angle = max_rotate_angle
        self.rotate_dynamic_data = rotate_dynamic_data
        self.border_value = border_value

    def __call__(self, sample):

        H, W = sample["raster_map"].shape[:2]

        if "angle" not in sample:
            if self.max_rotate_angle <= 0:
                return sample
            angle = random.uniform(
                -self.max_rotate_angle, self.max_rotate_angle
            )
        else:
            angle = sample["angle"]
        rotation_mat = cv2.getRotationMatrix2D(
            (W / 2, H / 2), angle / math.pi * 180, 1
        )

        sample["raster_map"] = cv2.warpAffine(
            sample["raster_map"],
            rotation_mat,
            (W, H),
            borderValue=self.border_value,
        )
        temp = cv2.warpAffine(
            sample["rasterized_dynamic_data"],
            rotation_mat,
            (W, H),
            flags=cv2.INTER_NEAREST,
        )

        if "angle" not in sample and self.rotate_dynamic_data:
            if "history_state" in sample:
                N, T = sample["history_state"].shape[:2]
                sample["history_state"] = coor_transformation(
                    sample["history_state"].reshape([-1, 2]), theta=angle
                ).reshape([N, T, -1])

            sample["rasterized_dynamic_data"] = coor_transformation(
                temp.reshape([-1, 2]), theta=angle
            ).reshape([H, W, -1])
        else:
            sample["rasterized_dynamic_data"] = temp

        if "angle" in sample:
            sample.pop("angle")

        for key in ["future", "history"]:
            sample[key] = (
                coor_transformation(
                    sample[key] - np.array([H, W]) / 2, theta=angle
                )
                + np.array([H, W]) / 2
            )
        if "yaw" in sample:
            sample["yaw"] -= angle
        return sample


@OBJECT_REGISTRY.register
class CropAndResize(object):
    """
    Crop and resize the data into specific output_size and resolution.

    Args:
        output_size: The output size of raster_map and
            rasterized_dynamic_data.
        downsample_rate: Ratio of input to output resolution.
        input_center_ratio: Ratio of center coordinate to
            raster_map size of input data. For example, (0.5, 0.5) means the
            center coordinate is (0.5xH, 0.5xW). Center usually be set as the
            position of ego or specific agent.
        output_center_ratio: Ratio of center coordinates to
            raster_map size of output data.
        translation_aug: Whether to add translation augmentation.
        resize_dynamic_data: Whether to rescale the filled states in
            rasterized_dynamic_data.
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]],
        downsample_rate: float = 1,
        input_center_ratio: Tuple[int, int] = (0.5, 0.5),
        output_center_ratio: Tuple[int, int] = (0.5, 0.5),
        translation_aug: bool = False,
        resize_dynamic_data: bool = True,
    ):
        assert downsample_rate >= 1

        self.output_size = np.array(output_size)
        self.downsample_rate = np.array(downsample_rate)
        self.crop_size = self.output_size * self.downsample_rate

        self.crop_shift = np.array(
            [
                -self.crop_size[0] * output_center_ratio[0],
                +self.crop_size[0] * (1 - output_center_ratio[0]),
                -self.crop_size[1] * output_center_ratio[1],
                +self.crop_size[1] * (1 - output_center_ratio[1]),
            ]
        )

        self.input_center_ratio = input_center_ratio
        self.translation_aug = translation_aug
        self.resize_dynamic_data = resize_dynamic_data

    def __call__(self, sample):

        raster_map = sample["raster_map"]
        rasterized_dynamic_data = sample["rasterized_dynamic_data"]
        H, W = raster_map.shape[:2]
        center_pixel = np.round(
            np.array(
                [
                    H * self.input_center_ratio[0],
                    W * self.input_center_ratio[1],
                ]
            )
        )

        crop_range_x = (center_pixel[0] + self.crop_shift[:2]).astype(np.int16)
        crop_range_y = (center_pixel[1] + self.crop_shift[2:]).astype(np.int16)

        if self.translation_aug:
            min_margin = min(
                crop_range_x[0],
                crop_range_y[0],
                H - crop_range_x[1],
                W - crop_range_y[1],
            )
            aug_shift = np.random.uniform(
                -min_margin, min_margin, size=(2,)
            ).astype(np.int16)
            crop_range_x += aug_shift[0]
            crop_range_y += aug_shift[1]

        raster_map = raster_map[
            crop_range_x[0] : crop_range_x[1],
            crop_range_y[0] : crop_range_y[1],
        ]
        rasterized_dynamic_data = rasterized_dynamic_data[
            crop_range_x[0] : crop_range_x[1],
            crop_range_y[0] : crop_range_y[1],
        ]

        if self.downsample_rate != 1:
            if self.resize_dynamic_data:
                rasterized_dynamic_data /= self.downsample_rate
                sample["history_state"] /= self.downsample_rate

            image_size = np.array(raster_map.shape[:2])
            target_size = np.int32(np.round(image_size / self.downsample_rate))
            raster_map = cv2.resize(raster_map, tuple(target_size[::-1]))
            rasterized_dynamic_data = cv2.resize(
                rasterized_dynamic_data, tuple(target_size[::-1])
            )

        sample["raster_map"] = raster_map
        sample["rasterized_dynamic_data"] = rasterized_dynamic_data

        for key in ["future", "history"]:
            sample[key] = np.where(
                sample[key + "_mask"][..., None] == 0,
                0,
                (sample[key] - [crop_range_x[0], crop_range_y[0]])
                / self.downsample_rate,
            )

        return sample


@OBJECT_REGISTRY.register
class FilteroutUselessAgents(object):
    """
    Filter out agents that are not worthy of attention or that are not \
    suitable for training, mainly including: \
    (1) agents of unconcerned categories, \
    (2) agents beyond or near the boundary of raster_map, \
    (3) agents with incomplete trajectories, \
    (4) stationary agents, \
    (5) the agents with invalid flag.

    Args:
        keys_to_filter: Keys of the items to filter out in the
            data dict.
        ignore_cls: IDs of the categories to ignore.
        min_pixel_to_edge: The lower threshold of filter
            rule (2).
        min_future_frame: The lower threshold of valid future
            frames of filter rule (3).
        min_history_frame: The lower threshold of valid history
            frames of filter rule (3).
        shift_threshold: The upper threshold of stationary
            agents' shift.
        max_stationary_num: Maximum number of stationary agents in a
            single training data.
        verbose: If true, the info of filter process will be logged.
    """

    def __init__(
        self,
        keys_to_filter: List[str],
        ignore_cls: Optional[List] = None,
        min_pixel_to_edge: Optional[float] = None,
        min_future_frame: Optional[int] = None,
        min_history_frame: Optional[int] = None,
        shift_threshold: Optional[float] = None,
        max_stationary_num: int = 2,
        verbose: bool = False,
    ):

        for k in ["future", "history", "future_mask", "history_mask"]:
            if k not in keys_to_filter:
                keys_to_filter.append(k)
        self.keys_to_filter = keys_to_filter
        self.ignore_cls = ignore_cls
        self.min_pixel_to_edge = min_pixel_to_edge
        self.min_future_frame = min_future_frame
        self.min_history_frame = min_history_frame
        self.shift_threshold = shift_threshold
        self.max_stationary_num = max_stationary_num
        self.verbose = verbose

    def __call__(self, sample):
        if "valid_flag" in sample:
            self._filter(sample, sample["valid_flag"])
            sample.pop("valid_flag")

        agents_num = sample["future"].shape[0]
        record = {"total_agents_num": agents_num}
        # 1. filter out cls in ignore_cls
        if "category" in sample and self.ignore_cls is not None:
            mask = np.ones(agents_num).astype(np.bool_)
            for cls in self.ignore_cls:
                mask = np.logical_and(mask, sample["category"] != cls)
            self._filter(sample, mask)
        record["num_of_ignore_cls"] = sample["history"].shape[0]

        # 2. filter out agents near or out the edge of raster map
        if self.min_pixel_to_edge is not None:
            raster_map_size = sample["raster_map"].shape[:2]
            pixel_to_edge = np.minimum(
                raster_map_size - sample["history"][:, 0],
                sample["history"][:, 0],
            )
            pixel_to_edge = np.min(pixel_to_edge, axis=-1)
            mask = pixel_to_edge >= self.min_pixel_to_edge
            self._filter(sample, mask)
            record["out_or_near_edge"] = sample["history"].shape[0]

        # 3. filter out agents with incomplete future trajectory
        if self.min_future_frame is not None:
            mask = np.logical_and(
                sample["future"] < raster_map_size, sample["future"] >= 0
            )
            mask = np.all(mask, axis=-1)
            mask = np.logical_and(mask, sample["future_mask"])
            mask = np.sum(mask, axis=1) >= self.min_future_frame
            self._filter(sample, mask)
            record["incomplete_future_trajectory"] = sample["history"].shape[0]

        # 4. filter out agents with short history trajectory
        if self.min_history_frame is not None:
            mask = sample["history_mask"]
            mask = np.sum(mask, axis=1) >= self.min_history_frame
            self._filter(sample, mask)
            record["short_history_trajectory"] = sample["history"].shape[0]

        # 5. Balance number of stationary and moving agents
        if self.shift_threshold is not None:
            self._balance_sta_mov_agents(sample)
            record["after_filter"] = sample["history"].shape[0]

        if self.verbose:
            filter_msg = ""
            for k, v in record.items():
                filter_msg += "{}: {} | ".format(k, v)
            logger.info(filter_msg)

        return sample

    def _balance_sta_mov_agents(self, sample):
        history = sample["history"]
        future = sample["future"]

        #   5.1 Judging by the history state
        shift = np.linalg.norm(history - history[:, :1], ord=2, axis=-1)
        stationary_history_mask = np.logical_or(
            np.logical_not(sample["history_mask"]),
            shift <= self.shift_threshold,
        )
        stationary_history_mask = np.all(stationary_history_mask, axis=1)

        #   5.2 Judging by the future state
        shift = np.linalg.norm(future - history[:, :1], ord=2, axis=-1)
        stationary_future_mask = np.logical_or(
            np.logical_not(sample["future_mask"]),
            shift <= self.shift_threshold,
        )
        stationary_future_mask = np.all(stationary_future_mask, axis=1)

        stationary_mask = np.logical_and(
            stationary_history_mask, stationary_future_mask
        )

        #   5.3 split stationary and moving agents,
        #       and move extra stationary agents
        moving_mask = np.logical_not(stationary_mask)
        for key in self.keys_to_filter:
            if key not in sample:
                continue
            sample[key] = np.concatenate(
                [sample[key][moving_mask], sample[key][stationary_mask]],
                axis=0,
            )

        stationary_agents_num = np.sum(stationary_mask)
        moving_agents_num = np.sum(moving_mask)

        if stationary_agents_num > min(
            moving_agents_num, self.max_stationary_num
        ):
            mask = np.arange(stationary_agents_num) < min(
                moving_agents_num, self.max_stationary_num
            )
            np.random.shuffle(mask)
            mask = np.concatenate(
                [np.ones(moving_agents_num, np.bool_), mask], axis=0
            )
            self._filter(sample, mask)

    def _filter(self, sample, mask):
        if mask.sum() == 0:  # keep one agent at least
            mask[0] = True
        for key in self.keys_to_filter:
            if key in sample:
                sample[key] = sample[key][mask]
        return sample


@OBJECT_REGISTRY.register
class TargetGenerator(object):
    """

    Genterte the training labels for heatmap-based trajectory prediction \
    models, including 'future_traj', 'heatmap', 'offset' and 'passable_mask'.

    Args:
        heatmap_size: The size of 'heatmap' label. The 'offset' and
            'passable_mask' have the same size as 'heatmap'.
        bev2heatmap: Convert the coordinates in BEV to heatmap.
        variance: The variance of the Gaussian distribution in the
            heatmap.
        with_offset_map: Whether to return 'offset'.
        heading_norm: Whether to rotate all labels to a specific
            direction.
        gen_passable_mask: Whether to generate 'passable_mask'.
        passable_fn: Funtion to convert raster_map to passable_mask
            and defaults to self._default_passable_fn.
        heatmap_heading: Whether to associate this Gaussian distribution
            with the endpoint speed.
    """

    def __init__(
        self,
        heatmap_size: Union[int, List],
        bev2heatmap: Callable,
        variance: float = 20,
        with_offset_map: bool = True,
        heading_norm: bool = False,
        gen_passable_mask: bool = True,
        passable_fn: Optional[Callable] = None,
        heatmap_heading: bool = False,
    ):
        if isinstance(heatmap_size, int):
            heatmap_size = [heatmap_size, heatmap_size]

        self.heatmap_size = np.array(heatmap_size)
        self.bev2heatmap = bev2heatmap
        self.variance = variance
        self.with_offset_map = with_offset_map
        self.heading_norm = heading_norm
        self.heatmap_heading = heatmap_heading
        self.gen_passable_mask = gen_passable_mask

        if passable_fn is None:
            self.passable_fn = self._default_passable_fn
        else:
            self.passable_fn = passable_fn

    def __call__(self, sample):
        sample["future_traj"] = sample["future"] - sample["history"][:, 0:1]
        if self.heading_norm:
            assert "yaw" in sample
            sample["future_traj"] = coor_transformation(
                sample["future_traj"], -sample["yaw"][:, 0:1]
            )
        self.get_target_heatmap(sample)
        if self.gen_passable_mask:
            self.get_passable_mask(sample)
        return sample

    def get_target_heatmap(self, sample):

        max_future_frame = sample["future"].shape[1]
        corners = [
            [0, 0],
            [0, self.heatmap_size[1] - 1],
            [self.heatmap_size[0] - 1, self.heatmap_size[1] - 1],
            [self.heatmap_size[0] - 1, 0],
            [0, 0],
        ]
        boundaries = np.array([[corners[i], corners[i + 1]] for i in range(4)])

        pixel = self.bev2heatmap(
            np.pad(
                sample["future_traj"],
                ((0, 0), (1, 0), (0, 0)),
                mode="constant",
            )
        )

        target_pixel = []
        for pixel_agent in pixel:
            frame = max_future_frame
            while (pixel_agent[frame] > self.heatmap_size - 1).any() or (
                pixel_agent[frame] < 0
            ).any():
                frame -= 1
            if frame == max_future_frame:
                target_pixel.append(pixel_agent[frame])
            else:
                for bdy in boundaries:
                    flag, cross_point = self._line_cross(
                        bdy, pixel_agent[frame : frame + 2]
                    )
                    if flag:
                        target_pixel.append(cross_point)
                        break
        target_pixel = np.stack(target_pixel, axis=0)

        target_pixel = np.maximum(
            np.minimum(target_pixel, self.heatmap_size - 1), 0.0
        )

        raw_pixel = np.indices(self.heatmap_size).transpose([1, 2, 0])

        offset = -raw_pixel[None] + target_pixel[:, None, None]
        if self.with_offset_map:
            sample["offset"] = offset

        if self.heatmap_heading:
            speed_vector = np.mean(
                sample["future"][:, -5:-1] - sample["future"][:, -6:-2],
                axis=-2,
            )
            speed = np.clip(
                np.linalg.norm(speed_vector, ord=2, axis=-1),
                a_min=1e-5,
                a_max=None,
            )
            theta = np.arctan(speed_vector[:, 0] / speed_vector[:, 1])
            theta = np.where(speed_vector[:, 1] > 0, theta, theta + math.pi)
            offset_rot = coor_transformation(offset, theta)
            scale = np.stack(
                [np.where(speed > 0.5, 2, 1), np.where(speed > 0.5, 0.5, 1)],
                axis=-1,
            )
            distance = np.sum(
                np.power(offset_rot * scale[:, None, None], 2), axis=-1
            )
        else:
            distance = np.sum(np.power(offset, 2), axis=-1)

        heatmap = np.exp(-0.5 * distance / self.variance)
        sample["heatmap"] = heatmap

    def get_passable_mask(self, sample):
        passable_mask = []
        raster_map = sample["raster_map"].copy()

        if sample["history"].shape[0] > 2:
            drivable_area = self.passable_fn(raster_map)
        else:
            drivable_area = None

        for pixel in sample["history"][:, 0]:
            if drivable_area is not None:
                croped = self._crop(pixel, drivable_area, padding=1)[..., 0]
            else:
                croped = self.passable_fn(
                    self._crop(pixel, raster_map, padding=255)
                )

            croped = cv2.resize(
                croped, (self.heatmap_size[0], self.heatmap_size[1])
            )
            passable_mask.append(croped)
        sample["passable_mask"] = np.stack(passable_mask, axis=0)

    def _line_cross(self, l1, l2):
        """Determine whether two line segments cross, and if so, return the \
        coordinates of the cross point."""
        p1, p2 = l1
        p3, p4 = l2
        x1 = (p1[1] - p3[1]) * (p4[0] - p3[0]) - (p1[0] - p3[0]) * (
            p4[1] - p3[1]
        )
        x2 = (p2[0] - p1[0]) * (p4[1] - p3[1]) - (p2[1] - p1[1]) * (
            p4[0] - p3[0]
        )
        x3 = (p1[1] - p3[1]) * (p2[0] - p1[0]) - (p1[0] - p3[0]) * (
            p2[1] - p1[1]
        )
        x4 = (p2[0] - p1[0]) * (p4[1] - p3[1]) - (p2[1] - p1[1]) * (
            p4[0] - p3[0]
        )
        if x2 == 0 or x4 == 0:
            return False, None
        r = x1 / x2
        s = x3 / x4
        if r >= 0 and r <= 1 and s >= 0 and r <= 1:
            cp = np.round(p1 + r * (p2 - p1)).astype("int32")
            cp2 = np.round(p3 + s * (p4 - p3)).astype("int32")
            assert np.all(cp == cp2)
            return True, cp
        else:
            return False, None

    def _crop(self, pixel, image, padding=0):
        """Take pixel as the center, extract a partial image of heatmap_size \
        in the image."""
        if self.heading_norm:
            raise NotImplementedError

        channel = image.shape[2]

        corners = np.array(
            [[0, 0], [self.heatmap_size[0], self.heatmap_size[1]]]
        )
        shift = self.bev2heatmap(corners, True)
        size = np.abs(shift[1] - shift[0]).astype(np.int32)

        pixel = np.round(pixel.copy())
        range_row = [
            max(0, pixel[0] + shift[0, 0]),
            min(image.shape[0] - 1, pixel[0] + shift[1, 0] - 1),
        ]
        range_col = [
            max(0, pixel[1] + shift[0, 1]),
            min(image.shape[1] - 1, pixel[1] + shift[1, 1] - 1),
        ]

        croped = image[
            int(range_row[0]) : int(range_row[1]),
            int(range_col[0]) : int(range_col[1]),
        ]

        x = int(max(-(shift[0, 0] + pixel[0]), 0))
        y = int(max(-(shift[0, 1] + pixel[1]), 0))
        img = np.ones([*size, channel]) * padding
        img[x : x + croped.shape[0], y : y + croped.shape[1]] = croped

        return img

    def _default_passable_fn(self, raster_map):
        return (raster_map[:, :, 0:1] >= 1e-4).astype("uint8")


@OBJECT_REGISTRY.register
class DropUselessItems(object):
    def __init__(self, keys: List):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample.pop(key)
        return sample


@OBJECT_REGISTRY.register
class HWC2CHW(object):
    def __init__(self, keys: List):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if key not in sample:
                continue
            ranks = list(range(len(sample[key].shape)))
            ranks[-1], ranks[-2], ranks[-3] = ranks[-2], ranks[-3], ranks[-1]
            sample[key] = np.transpose(sample[key], ranks)

        return sample


@OBJECT_REGISTRY.register
class AsType(object):
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, sample):
        for k in sample.keys():
            sample[k] = sample[k].astype(self.dtype)
        return sample
