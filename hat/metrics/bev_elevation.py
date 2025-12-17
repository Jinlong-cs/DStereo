# Copyright (c) Horizon Robotics. All rights reserved.
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

try:
    from aidisdk.experiment import Table
except ImportError:
    Table = None

try:
    from torchvision.transforms.functional import InterpolationMode, resize
except ImportError:
    InterpolationMode = None
    resize = None

from hat.core.bev_elevation_utils import (
    crop_roi_vcs_range,
    decimal_div,
    decimal_minus,
    get_area_threshod_mask,
    get_resolution,
    max_freespace_contain_ego,
)
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.utils.aidi import EvalResult
from hat.utils.apply_func import _as_list

__all__ = [
    "ANCBevElevationMetric",
    "ANCBEVFreespaceMetric",
    "FreespaceBoundaryExtractor",
]

logger = logging.getLogger(__name__)


class FreespaceBoundaryExtractor(object):
    r"""Extract boundary of freespace.

    Extract freepace boundary by transmit rays from ego. Specifically,
    for 0-1 binary freespace: 1) at ego front center, transmit 180
    rays, i.e.,  transmit one ray for each degree. Along each ray,
    the nearest freepace_no point, to ego front center, is defined as
    boundary point; 2) similar boundary point definition for ego rear
    center; 3) for ego left side, transmit evenly spaced 20 rays along
    ego left border, the nearest freepace_no point, to each side ray
    emission source, is defined as boundary point; 4) similar boundary
    point definition for ego right border. The overall sketch is as
    follows:

    * * * * * * * * * * * * 180 front rays * * * * * * * * * * * * *
    *                     \\\\\\\....//////                        *
    *                         \\\\..////                           *
    *                           __\/__                             *
    *     *      ---------------|     |---------------     *       *
    *  side rays ---------------| ego |--------------- side rays   *
    *     *      ---------------|     |---------------     *       *
    *     *      ---------------|_____|---------------     *       *
    *                              /\                              *
    *                          ////..\\\\                          *
    *                       //////....\\\\\\                       *
    * * * * * * * * * * * * 180  rear rays * * * * * * * * * * * * *

    Args:
        center: vcs origin coord, (u, v) order.
        spatial_resolution: spitial resolution, order (u, v),
            corresponding to (y, x) order in vcs.
        ego_size: ego size, (bottom, right, top, left) order, measured
            in meters, denotes the distance from ego bottom, right, top,
            left to vcs orgin.
        ignore_index: ignore cls id for freespace.
        polar_ray_num: num of rays transmit from ego front and rear
            center in polar coord system, for search front and rear
            boundary point of freespace.
        side_ray_num: num of rays transmit from ego left and right
            border in u-v coord system, for search left and right
            boundary point of freespace.
        save_dir: boundary points visualize save dir.
    """

    def __init__(
        self,
        center: Sequence[Union[int, float]],
        spatial_resolution: Sequence[float],  # (u, v)<--> (y, x) order
        ego_size: Sequence[float] = (1.1, 1.1, 4.0, 1.1),
        ignore_index: int = 255,
        polar_ray_num: int = 360,
        side_ray_num: int = 20,
        save_dir: Optional[str] = None,
    ):
        self.center_x = int(center[0])
        self.center_y = int(center[1])
        self.ego_size = ego_size
        self.ego_top = int(ego_size[2] / spatial_resolution[1])
        self.ego_right = int(ego_size[1] / spatial_resolution[0])
        self.ego_bottom = int(ego_size[0] / spatial_resolution[1])
        self.ego_left = int(ego_size[3] / spatial_resolution[0])
        self.ego_front_y = self.center_y - self.ego_top
        self.ego_bottom_y = self.center_y + self.ego_bottom
        self.ego_right_x = self.center_x + self.ego_right
        self.ego_left_x = self.center_x - self.ego_left
        self.spatial_resolution = spatial_resolution
        self.polar_ray_num = polar_ray_num
        self.side_ray_num = side_ray_num
        self.ignore_index = ignore_index
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def line_intersect_image_border(
        self, image: np.ndarray, pt1: Tuple, pt2: Tuple
    ):
        """Get intersect point between line and image border.

        Connect a line L between pt1 and pt2, then get the
        intersect point coord between line L and image border.

        Args:
            image: shape (h, w).
            pt1: tuple of u-v coord, (u, v) order.
            pt2: tuple of u-v coord, (u, v) order.

        Returns:
            tuple: intersect point coord, (u, v) order.
        """
        mask = np.zeros_like(image)
        cv2.line(mask, pt1, pt2, color=1, thickness=1)
        # only remain pixel masked with 1 on image border
        mask[1:-1, 1:-1] = 0
        h_index, w_index = np.where(mask == 1)
        intersect_point = (w_index[0], h_index[0])
        return intersect_point

    def front_rear_boundary_points(
        self,
        input: np.ndarray,
        polar_origin: Tuple,
        view: str,
        ray_num=360,
    ):
        """Get boundary points by ray from ego front or rear.

        Args:
            input: shape (h, w).
            polar_origin: polar coord origin in data, order (u, v).
            view: values in ["front", "rear"], means emission ray
                from ego front or rear.
            ray_num: the num of emission ray from ego front center
                or rear center.

        Returns:
            list: list of array, polar data, polar coord and
                cartesion coord.
        """
        data = input.copy()
        # handle ignore in freespace region
        data[data == self.ignore_index] = 0

        if view == "front":
            degree_range = (180, 360)
        elif view == "rear":
            degree_range = (0, 180)
        else:
            raise TypeError("only support front and back range.")
        x, y = polar_origin[0], polar_origin[1]
        max_radius = int(np.sqrt(x ** 2 + y ** 2))

        data_polar = cv2.warpPolar(
            data.copy().astype(np.uint8),
            None,
            (x, y),
            max_radius,
            cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR,
        )

        data_polar = cv2.resize(
            data_polar,
            (data_polar.shape[1], ray_num),
            interpolation=cv2.INTER_NEAREST,
        )

        delta_deg = 360 / ray_num
        degrees = np.arange(degree_range[0], degree_range[1], delta_deg)
        start_id = int(degree_range[0] / delta_deg)
        # get the nearest non freespace point along each ray
        points = []
        for id in range(degrees.shape[0]):
            for r in range(max_radius):
                if data_polar[start_id + id, r] != 0:  # freespace_yes=0
                    points.append([start_id + id, r])
                    break
                if r == max_radius - 1:
                    points.append([start_id + id, r])
                    break
        polar_points = np.array(points)

        # convert polar coord to cartesian coord
        radius = polar_points[:, 1]
        cart_x, cart_y = cv2.polarToCart(
            radius.astype(np.float32),
            degrees.astype(np.float32),
            angleInDegrees=1,
        )
        cart_points = np.hstack([cart_x + x, cart_y + y]).astype(int)

        # clip radius and cart points beyond data size
        # get ignore boundary point mask
        ignore_pts_mask = np.zeros((cart_points.shape[0],))
        for i in range(cart_points.shape[0]):
            u, v = cart_points[i, :]
            # clip to valid boundary
            if u < 0 or v < 0 or u >= data.shape[1] or v >= data.shape[0]:
                u, v = self.line_intersect_image_border(
                    data, (u, v), polar_origin
                )
            r = int(np.sqrt((x - u) ** 2 + (y - v) ** 2))
            cart_points[i, :] = (u, v)
            polar_points[i, 1] = r
            # 3x3 window match ignore
            if (
                input[v - 1 : v + 2, u - 1 : u + 2] == self.ignore_index
            ).any():
                ignore_pts_mask[i] = 1

        # real polar radius measued in meters
        # for different (u, v) spatial resolution
        real_polar_points = np.zeros_like(polar_points)
        if self.spatial_resolution[0] != self.spatial_resolution[1]:
            for i in range(polar_points.shape[0]):
                deg_id, r = polar_points[i, :]
                deg = delta_deg * deg_id
                rad = deg / 180.0 * np.pi
                real_u = r * np.cos(rad) * self.spatial_resolution[0]
                real_v = r * np.sin(rad) * self.spatial_resolution[1]
                real_r = np.sqrt(real_u ** 2 + real_v ** 2)
                real_polar_points[i, :] = (deg_id, real_r)

        return (
            data_polar,
            polar_points,
            real_polar_points,
            cart_points,
            ignore_pts_mask,
        )

    def left_right_boundary_points(
        self, input, crop_coord, view="right", ray_num=20
    ):
        """Get left and right boundary points by ray from ego.

        Args:
            data: shape (h, w)
            crop_coord: coord of roi in data.
            view: ray to left or right. Defaults
                to "right".
            ray_num: the num of emission ray along ego
                left or right border.

        Returns:
            list: left or right boundary points cartesian coord,
                and corresponding distance to ego.
        """
        data = input.copy()
        # handle ignore in freespace region
        data[data == self.ignore_index] = 0

        top, bottom, left, right = crop_coord
        data = data[top:bottom, left:right].copy()
        data = cv2.resize(
            data,
            (data.shape[1], ray_num + 1),
            interpolation=cv2.INTER_NEAREST,
        )
        # transmit ray from center to left or right
        if view == "right":
            mask = data != 0
        else:
            mask = data[:, ::-1] != 0

        # (n, ) with u coord
        side_points_u = np.where(
            mask.any(axis=1),
            mask.argmax(axis=1),
            mask.shape[1] - 1,
        )[:-1]
        side_points_v = np.arange(ray_num)
        if view == "left":
            side_points_u = side_points_u + 1
            side_points_v = side_points_v + 1

        side_points = np.stack([side_points_v, side_points_u], axis=-1)

        side_cart_points = side_points[:, ::-1].copy()
        side_cart_points[:, 1] = (
            side_cart_points[:, 1]
            * (self.ego_bottom_y - self.ego_front_y)
            / ray_num
        ).astype(int)
        side_cart_points[:, 1] = side_cart_points[:, 1] + self.ego_front_y
        if view == "right":
            side_cart_points[:, 0] = self.center_x + side_cart_points[:, 0]
        else:
            side_cart_points[:, 0] = self.center_x - side_cart_points[:, 0]

        # get ignore boundary point mask
        ignore_pts_mask = np.zeros((side_cart_points.shape[0],))
        for i in range(side_cart_points.shape[0]):
            u, v = side_cart_points[i, :]
            # 　3x3 window match ignore
            if (
                input[v - 1 : v + 2, u - 1 : u + 2] == self.ignore_index
            ).any():
                ignore_pts_mask[i] = 1

        # real distance measued in meters
        # for different (u, v) spatial resolution
        real_side_points = np.zeros_like(side_points)
        if self.spatial_resolution[0] != self.spatial_resolution[1]:
            real_side_points = side_points * np.array(
                self.spatial_resolution[::-1]
            ).reshape(-1, 2)

        return side_points, real_side_points, side_cart_points, ignore_pts_mask

    def get_boundary_points(self, data, vis_boundary=False):
        h, w = data.shape[:2]

        # get polar and cartesion coord for ego head and ego rear
        (
            front_polar_img,
            front_polar_points,
            real_front_polar_points,
            front_cart_points,
            front_pts_ignore_mask,
        ) = self.front_rear_boundary_points(
            data,
            (self.center_x, self.ego_front_y),
            "front",
            self.polar_ray_num,
        )  # x<-->u, y<-->v
        (
            rear_polar_img,
            rear_polar_points,
            real_rear_polar_points,
            rear_cart_points,
            rear_pts_ignore_mask,
        ) = self.front_rear_boundary_points(
            data,
            (self.center_x, self.ego_bottom_y),
            "rear",
            self.polar_ray_num,
        )

        # right boundary
        (
            right_side_points,
            real_right_side_points,
            right_cart_points,
            right_pts_ignore_mask,
        ) = self.left_right_boundary_points(
            data,
            (self.ego_front_y, self.ego_bottom_y + 1, self.center_x, w),
            "right",
            self.side_ray_num,
        )
        # left boundary
        (
            left_side_points,
            real_left_side_points,
            left_cart_points,
            left_pts_ignore_mask,
        ) = self.left_right_boundary_points(
            data,
            (self.ego_front_y, self.ego_bottom_y + 1, 0, self.center_x),
            "left",
            self.side_ray_num,
        )

        # stitch in counter-clockwise
        target_cart_points = np.concatenate(
            [
                front_cart_points,
                right_cart_points,
                rear_cart_points,
                left_cart_points[::-1, :],
            ],
            axis=0,
        )
        vcs_bdry_pts_ignore_mask = np.concatenate(
            [
                front_pts_ignore_mask,
                right_pts_ignore_mask,
                rear_pts_ignore_mask,
                left_pts_ignore_mask[::-1],
            ],
            axis=0,
        )

        center = np.array([self.center_y, self.center_x]).reshape(
            -1, 2
        )  # (v, u)

        # shape (n, 2), in (x, y) <--> (v, u),
        vcs_bdry_pts = (center - target_cart_points[:, ::-1]) * np.array(
            self.spatial_resolution[::-1]
        ).reshape(-1, 2)

        if self.spatial_resolution[0] != self.spatial_resolution[1]:
            vcs_bdry_pts_radius = np.concatenate(
                [
                    real_front_polar_points[:, 1],
                    real_right_side_points[:, 1],
                    real_rear_polar_points[:, 1],
                    real_left_side_points[::-1, 1],
                ],
                axis=0,
            )
        else:
            target_radius = np.concatenate(
                [
                    front_polar_points[:, 1],
                    right_side_points[:, 1],
                    rear_polar_points[:, 1],
                    left_side_points[::-1, 1],
                ],
                axis=0,
            )
            vcs_bdry_pts_radius = target_radius * self.spatial_resolution[0]

        if vis_boundary:
            self.draw_freespace(
                target_cart_points,
                data,
                front_polar_points,
                front_polar_img,
                rear_polar_points,
                rear_polar_img,
                front_cart_points,
                rear_cart_points,
                right_cart_points,
                left_cart_points,
                vcs_bdry_pts_ignore_mask,
            )

        return vcs_bdry_pts, vcs_bdry_pts_radius, vcs_bdry_pts_ignore_mask

    def draw_freespace(
        self,
        target_cart_points: np.ndarray,
        freespace_img: np.ndarray,
        front_polar_points: Optional[np.ndarray] = None,
        front_polar_img: Optional[np.ndarray] = None,
        rear_polar_points: Optional[np.ndarray] = None,
        rear_polar_img: Optional[np.ndarray] = None,
        front_cart_points: Optional[np.ndarray] = None,
        rear_cart_points: Optional[np.ndarray] = None,
        right_cart_points: Optional[np.ndarray] = None,
        left_cart_points: Optional[np.ndarray] = None,
        pts_ignore_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Draw bounary points of freespace data.

        Draw boundary points and emission rays during boundary points
        extraction. Mainly, for visualization check of boundary points
        extraction.

        Args:
            target_cart_points: shape (n, 2).
            freespace_img: freespace data with values in [0, 1,
                ignore_index], shape (h, w).
            front_polar_points: boundary points from emission rays for
                ego front center, in polar coord system, shape (n, 2).
            front_polar_img: freespace_img converted into polar
                coordinate system. shape (h1, w1).
            rear_polar_points: boundary points from emission rays for
                ego rear center, in polar coord system, shape (n, 2).
            rear_polar_img: freespace_img converted into polar
                coordinate system. shape (h2, w2).
            front_cart_points: boundary points from emission rays for
                ego front center, in u-v coordinates, shape (n, 2).
            rear_cart_points: boundary points from emission rays for
                ego rear center, in u-v coordinates, shape (n, 2).
            right_cart_points: boundary points extracting from ego
                right border, in u-v coordinates, shape (n, 2).
            left_cart_points: boundary points extracting from ego
                left border, in u-v coordinates, shape (n, 2).
            pts_ignore_mask: boundary points ignore mask, shape (n,),
                where 1 means ignore.
        """
        # obvious visualization
        freespace_img = freespace_img.copy()
        freespace_img[freespace_img == 1] = 128
        front_polar_img[front_polar_img == 1] = 128
        rear_polar_img[rear_polar_img == 1] = 128

        # color visualization
        freespace_img = cv2.cvtColor(freespace_img, cv2.COLOR_GRAY2RGB)
        front_polar_img = cv2.cvtColor(front_polar_img, cv2.COLOR_GRAY2RGB)
        rear_polar_img = cv2.cvtColor(rear_polar_img, cv2.COLOR_GRAY2RGB)

        # boundary point draw in cartesian
        for i in range(len(target_cart_points) - 1):
            if pts_ignore_mask[i] or pts_ignore_mask[i + 1]:
                color = [0, 255, 0]
            else:
                color = [255, 0, 255]
            cv2.line(
                freespace_img,
                (target_cart_points[i][0], target_cart_points[i][1]),
                (target_cart_points[i + 1][0], target_cart_points[i + 1][1]),
                color=color,
                thickness=1,
            )
        if pts_ignore_mask[-1] or pts_ignore_mask[0]:
            color = [0, 255, 0]
        else:
            color = [255, 0, 255]
        cv2.line(
            freespace_img,
            (target_cart_points[-1][0], target_cart_points[-1][1]),
            (target_cart_points[0][0], target_cart_points[0][1]),
            color=color,
            thickness=1,
        )

        cv2.imwrite(
            f"{self.save_dir}/freespace_boundary_cartesian.png",
            freespace_img,
        )

        # polar draw
        if front_polar_points is not None and front_polar_img is not None:
            for i in range(len(front_polar_points) - 1):
                cv2.line(
                    front_polar_img,
                    (front_polar_points[i][1], front_polar_points[i][0]),
                    (
                        front_polar_points[i + 1][1],
                        front_polar_points[i + 1][0],
                    ),
                    color=[255, 0, 255],
                    thickness=1,
                )
            cv2.imwrite(
                f"{self.save_dir}/front_polar_img.png", front_polar_img
            )

        if rear_polar_points is not None and rear_polar_img is not None:
            for i in range(len(rear_polar_points) - 1):
                cv2.line(
                    rear_polar_img,
                    (rear_polar_points[i][1], rear_polar_points[i][0]),
                    (
                        rear_polar_points[i + 1][1],
                        rear_polar_points[i + 1][0],
                    ),
                    color=[255, 0, 255],
                    thickness=1,
                )
            cv2.imwrite(f"{self.save_dir}/rear_polar_img.png", rear_polar_img)

        # cartesian draw in counter-clockwise
        start_id = 0
        end_id = 0
        if front_cart_points is not None:
            end_id = start_id + len(front_cart_points)
            front_pts_ignore_mask = pts_ignore_mask[start_id:end_id]
            for i in range(len(front_cart_points)):
                if front_pts_ignore_mask[i]:
                    color = [0, 255, 0]
                else:
                    color = [255, 0, 255]
                cv2.line(
                    freespace_img,
                    (front_cart_points[i][0], front_cart_points[i][1]),
                    (self.center_x, self.ego_front_y),
                    color=color,
                    thickness=1,
                )
            start_id = end_id

        if right_cart_points is not None:
            end_id = start_id + len(right_cart_points)
            right_pts_ignore_mask = pts_ignore_mask[start_id:end_id]
            for i in range(len(right_cart_points)):
                if right_pts_ignore_mask[i]:
                    color = [0, 255, 0]
                else:
                    color = [255, 0, 255]
                cv2.line(
                    freespace_img,
                    (right_cart_points[i][0], right_cart_points[i][1]),
                    (self.ego_right_x, right_cart_points[i][1]),
                    color=color,
                    thickness=1,
                )
            start_id = end_id

        if rear_cart_points is not None:
            end_id = start_id + len(rear_cart_points)
            rear_pts_ignore_mask = pts_ignore_mask[start_id:end_id]
            for i in range(len(rear_cart_points)):
                if rear_pts_ignore_mask[i]:
                    color = [0, 255, 0]
                else:
                    color = [255, 0, 255]
                cv2.line(
                    freespace_img,
                    (rear_cart_points[i][0], rear_cart_points[i][1]),
                    (self.center_x, self.ego_bottom_y),
                    color=color,
                    thickness=1,
                )
            start_id = end_id

        if left_cart_points is not None:
            end_id = start_id + len(left_cart_points)
            left_pts_ignore_mask = pts_ignore_mask[start_id:end_id]
            for i in range(len(left_cart_points)):
                if left_pts_ignore_mask[i]:
                    color = [0, 255, 0]
                else:
                    color = [255, 0, 255]
                cv2.line(
                    freespace_img,
                    (left_cart_points[i][0], left_cart_points[i][1]),
                    (self.ego_left_x, left_cart_points[i][1]),
                    color=color,
                    thickness=1,
                )
        cv2.imwrite(
            f"{self.save_dir}/freespace_plar_ray_cartesian.png", freespace_img
        )


@OBJECT_REGISTRY.register
class ANCBEVFreespaceMetric(EvalMetric):
    """ANCBEVFreespaceMetric segmentation results.

    Args:
        seg_class: A list of classes the segmentation dataset
            includes, the order should be the same as the label.
        bev_size: Bev size, in pixel.(order is (h,w)).
        vcs_range: Vcs range, in meter.(order is
            (bottom,right,top,left), e.g.(-30, -51.2, 72.4, 51.2)).
        eval_cfg: eval setting dict, for example, defined as::

            {
                "eval_vcs_range": None,
                "smallobj_area_thresh_ub": 12.0,
                "smallobj_area_thresh_lb": 0.04,
                "pr_cfg": {
                    "depth_intervals": (-32.4, 0, 20, 50, 80, 100),
                    "area_intervals": (0.5, 1.0, 4.0, 12.0),
                    "dist_interval_dxy_threshs": {
                        (0, 5): 0.5,
                        (5, 10): 1.0,
                    },
                },
                "boundary_cfg": {
                    "dist_intervals": (0, 3, 10, 20),  # in meters
                    "dist_interval_dxy_threshs": {
                        (0, 3): 5.0,
                        (3, 10): 10.0,
                    },
                },
                metric_modes: ["iou", "boundary_error", "pr"],
            }

            where "eval_vcs_range" is roi vcs range to eval, ordered in
            (bottom, right, top, left), same vcs origin with vcs_range;
            "smallobj_area_thresh_ub" is max area for small object, measured
            in m^2, for compute small obj iou; "depth_intervals" for
            metrics in each depth interval; "area_intervals" for different
            size object, measured in m^2. "dist_interval_dxy_threshs",
            object center distance thresh for pr matching, according
            to distance to ego. measured in meter. "metric_mode", list of
            metric names, value in ["iou", "boundary_error", "pr"],
            corresponding to iou, boundary error, small object pr metrics.
        name: Name of this metric instance for display, also used as
            monitor params for Checkpoint.
        ignore_index: The label index that will be ignored in evaluation.
        global_ignore_index: The label index that will be ignored in
            global evaluation,such as:mIoU,mAcc,aAcc.Supporting list of label
            index.
        save_metric_path: the path to save metric results.
        result_prefix: Prefix of aidi eval result.
    """

    def __init__(
        self,
        seg_class: List[str],
        bev_size: Sequence[int],
        vcs_range: Sequence[float],
        eval_cfg: Dict,
        task_name: Optional[bool] = "bev_freespace",
        gt_name: Optional[bool] = "gt_bev_freespace",
        pred_name: Optional[bool] = "pred_bev_freespace",
        name: str = "FreespaceMetric",
        ignore_index: int = 255,
        global_ignore_index: Union[Sequence, int] = 255,
        save_metric_path: str = None,
        result_prefix: str = "",
    ):
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.raw_vcs_range = vcs_range

        metric_modes = eval_cfg.get("metric_modes", ["iou"])
        assert len(metric_modes) > 0
        for _ in metric_modes:
            assert _ in ["iou", "boundary_error", "pr"]
        self.metric_modes = metric_modes

        eval_vcs_range = eval_cfg["eval_vcs_range"]
        depth_intervals = eval_cfg["depth_intervals"]
        obj_area_thresh_ub = eval_cfg.get("smallobj_area_thresh_ub", 12.0)
        obj_area_thresh_lb = eval_cfg.get("smallobj_area_thresh_lb", 12.0)
        self.smallobj_background_dilate_k = eval_cfg.get(
            "smallobj_background_dilate_k", 9
        )
        self.max_dist = (
            np.sqrt(
                (max(abs(vcs_range[0]), abs(vcs_range[2]))) ** 2
                + (max(abs(vcs_range[1]), abs(vcs_range[3]))) ** 2
            ).astype(int)
            + 1
        )
        pr_cfg = eval_cfg.get(
            "pr_cfg",
            {
                "area_intervals": (0, 12.0),
                "dist_interval_dxy_threshs": {(0, self.max_dist): 1.0},
            },
        )
        area_intervals = pr_cfg.get("area_intervals", (0, 12.0))
        self.dxy_threshs = pr_cfg.get(
            "dist_interval_dxy_threshs", {(0, self.max_dist): 1.0}
        )
        self.eval_cfg = eval_cfg
        self.num_classes = len(seg_class)
        self.eval_vcs_range = eval_vcs_range
        self.task_name = task_name
        self.gt_name = gt_name
        self.pred_name = pred_name
        self.seg_class = seg_class
        self.name = name
        self.ignore_index = ignore_index
        self.global_save_index_list = [
            index
            for index in range(len(seg_class))
            if index not in _as_list(global_ignore_index)
        ]

        if save_metric_path:
            save_dir, _ = os.path.split(save_metric_path)
            os.makedirs(save_dir, exist_ok=True)
        self.save_metric_path = save_metric_path

        self.iou_metric_suffix = ["intersect", "union", "pred_label", "label"]
        self.pr_metric_name_suffix = ["tp", "fp", "fn", "dxy"]
        self.bdry_metric_name_suffix = [
            "num",
            "mean",
            "std",
            "50_percentile",
            "90_percentile",
            "95_percentile",
            "max",
        ]

        # convert gt_centers in u-v to vcs coord
        self.spatial_resolution = get_resolution(self.vcs_range, bev_size)[
            ::-1
        ]  # (y, x) <--> (u, v)
        self.pixel_area = np.prod(self.spatial_resolution)
        if eval_vcs_range is not None:
            eval_w = decimal_div(
                abs(decimal_minus(eval_vcs_range[3], eval_vcs_range[1])),
                self.spatial_resolution[0],
            )
            eval_h = decimal_div(
                abs(decimal_minus(eval_vcs_range[2], eval_vcs_range[0])),
                self.spatial_resolution[1],
            )
            self.vcs_range = eval_vcs_range
            self.bev_size = (int(eval_h), int(eval_w))

        self.vcs_origin_coord = (
            decimal_div(self.vcs_range[3], self.spatial_resolution[0]),
            decimal_div(self.vcs_range[2], self.spatial_resolution[1]),
        )  # (u, v)

        # pr by depth and area thresh
        self.full_depth_intervals = [
            [start, end]
            for start, end in zip(depth_intervals[:-1], depth_intervals[1:])
        ]
        self.physic_area_intervals = [0] + list(area_intervals)
        area_intervals = [
            int(area / self.pixel_area) for area in area_intervals
        ]  # convert to pixel num
        full_area_intervals = [0] + list(area_intervals)
        self.full_area_intervals = [
            [start, end]
            for start, end in zip(
                full_area_intervals[:-1], full_area_intervals[1:]
            )
        ]
        self.full_area_intervals.append(
            [full_area_intervals[0], full_area_intervals[-1]]
        )
        self.get_depth_interval_mask()

        # boundar eval
        self.boundary_error_delta = min(self.spatial_resolution)
        self.boundary_error_arr_len = (
            int(self.max_dist / self.boundary_error_delta) + 1
        )
        self.boundary_extractor = FreespaceBoundaryExtractor(
            self.vcs_origin_coord,
            spatial_resolution=self.spatial_resolution,
            ignore_index=self.ignore_index,
        )
        dist_intervals = eval_cfg.get("dist_intervals", (0, 20))
        self.full_dist_intervals = [
            (start, end)
            for start, end in zip(dist_intervals[:-1], dist_intervals[1:])
        ]
        self.num_dist_intervals = len(self.full_dist_intervals)
        self.get_distance_interval_mask()

        self.obj_area_thresh_ub = int(obj_area_thresh_ub / self.pixel_area)
        self.obj_area_thresh_lb = int(obj_area_thresh_lb / self.pixel_area)
        if obj_area_thresh_ub > 0:
            self.small_obj_render_label = 1
            self.small_obj_global_save_index_list = [
                index
                for index in range(len(seg_class))
                if (
                    index not in _as_list(global_ignore_index)
                    and index == self.small_obj_render_label
                )
            ]
        self.result_prefix = result_prefix
        super(ANCBEVFreespaceMetric, self).__init__(name)

    def save_res_json(self, res):
        if self.save_metric_path:
            with open(self.save_metric_path, "w") as f:
                f.write(json.dumps(res, ensure_ascii=False, indent=1))

    def get_depth_interval_mask(self):
        depth_interval_mask = {}
        for depth_interval in self.full_depth_intervals:
            start, end = depth_interval
            mask = np.zeros(self.bev_size, np.bool)

            depth_roi_range = (
                max(start, self.vcs_range[0]),
                self.vcs_range[1],
                min(end, self.vcs_range[2]),
                self.vcs_range[3],
            )
            top, left, crop_h, crop_w = crop_roi_vcs_range(
                mask, depth_roi_range, self.vcs_range, return_crop_coord=True
            )
            mask[top : top + crop_h, left : left + crop_w] = 1

            depth_interval_mask[
                f"{depth_interval[0]}_{depth_interval[1]}"
            ] = mask
        self.depth_interval_mask = depth_interval_mask

    def get_distance_interval_mask(self):
        dist_interval_mask = {}
        ego_size = (
            self.boundary_extractor.ego_size
        )  # same order with vcs range
        for dist_interval in self.full_dist_intervals:
            start, end = dist_interval

            mask = np.zeros(self.bev_size, np.bool)
            outer_roi_range = (
                max(-ego_size[0] - end, self.vcs_range[0]),
                max(-ego_size[1] - end, self.vcs_range[1]),
                min(ego_size[2] + end, self.vcs_range[2]),
                min(ego_size[3] + end, self.vcs_range[3]),
            )
            top, left, crop_h, crop_w = crop_roi_vcs_range(
                mask, outer_roi_range, self.vcs_range, return_crop_coord=True
            )
            mask[top : top + crop_h, left : left + crop_w] = 1

            inner_roi_range = (
                max(-ego_size[0] - start, self.vcs_range[0]),
                max(-ego_size[1] - start, self.vcs_range[1]),
                min(ego_size[2] + start, self.vcs_range[2]),
                min(ego_size[3] + start, self.vcs_range[3]),
            )
            top, left, crop_h, crop_w = crop_roi_vcs_range(
                mask, inner_roi_range, self.vcs_range, return_crop_coord=True
            )
            mask[top : top + crop_h, left : left + crop_w] = 0

            dist_interval_mask[f"{dist_interval[0]}_{dist_interval[1]}"] = mask
        self.dist_interval_mask = dist_interval_mask

    def get_metric_names(self):
        metric_names = []
        for depth_interval in self.full_depth_intervals:
            # small obj pr metric
            for area_interval in self.full_area_intervals:
                prefix = "area_%d_%d_depth_%d_%d_" % (
                    area_interval[0],
                    area_interval[1],
                    depth_interval[0],
                    depth_interval[1],
                )
                for metric_suffix in self.pr_metric_name_suffix:
                    metric_names.append(prefix + metric_suffix)

        return metric_names

    def _init_states(self):
        for suffix in self.iou_metric_suffix:
            self.add_state(
                suffix,
                default=torch.zeros((self.num_classes,)),
                dist_reduce_fx="sum",
            )
            if self.obj_area_thresh_ub > 0:
                self.add_state(
                    f"small_obj_{suffix}",
                    default=torch.zeros((self.num_classes,)),
                    dist_reduce_fx="sum",
                )

        for suffix in self.pr_metric_name_suffix:
            self.add_state(
                name=f"smallobj_{suffix}",
                default=torch.zeros(1),
                dist_reduce_fx="sum",
            )

        for name in self.get_metric_names():
            self.add_state(
                name,
                default=torch.zeros(1),
                dist_reduce_fx="sum",
            )

        # boundary error array for percentile compute
        self.add_state(
            "boundary_error_arr",
            default=torch.zeros(
                self.num_dist_intervals,
                self.boundary_error_arr_len,
                dtype=torch.int32,
            ),
            dist_reduce_fx="sum",
        )

    def get_small_obj_mask(self, label: torch.Tensor):
        """Get small object mask from label.

        Note that freespace label with values in {0, 1, 255}, where
        0 is freespace_yes, 1 is freespace_no, 255 is ignore index.

        Args:
            label: (b, h, w)

        Returns:
            torch.Tensor: mask denotes small object.
        """
        img = label.cpu().numpy().astype(np.uint8)
        output = np.zeros(img.shape)
        for i in range(img.shape[0]):
            smallobj_mask = get_area_threshod_mask(
                img[i, :, :],
                upper_area_thresh=self.obj_area_thresh_ub,
                lower_area_thresh=self.obj_area_thresh_lb,
                ignore_index=self.ignore_index,
                mask_render_label=self.small_obj_render_label,
            )
            # dilate the mask to include surroundings of small objects
            k = self.smallobj_background_dilate_k
            smallobj_mask = cv2.dilate(
                smallobj_mask,
                np.ones((k, k), np.uint8),
            )
            output[i, :, :] = smallobj_mask

        return label.new_tensor(output) == self.small_obj_render_label

    def center_distance_match(
        self, gt_centers: np.ndarray, pred_centers: np.ndarray
    ):
        """Accord center distance to match pred and gt.

        Args:
            gt_centers: (N, 2), ordered in (u, v).
            pred_centers: (K, 2), ordered in (u, v).
        """
        gt_assign = np.zeros(shape=len(gt_centers), dtype=np.int) - 1
        matched_pred = np.zeros(shape=len(pred_centers), dtype=np.int)
        valid_gt = np.ones(shape=len(gt_centers), dtype=np.bool)
        matched_dxy = np.zeros(shape=len(gt_centers), dtype=np.float32)

        # convert (u, v) to vcs coord (y, x)
        pred_centers = (
            np.array(self.vcs_origin_coord).reshape(1, 2) - pred_centers
        ) * np.array(self.spatial_resolution).reshape(1, 2)
        gt_centers = (
            np.array(self.vcs_origin_coord).reshape(1, 2) - gt_centers
        ) * np.array(self.spatial_resolution).reshape(
            1, 2
        )  # (y, x)

        gt_dist = np.sqrt(np.sum(np.square(gt_centers), axis=-1))  # (N,)
        for idx, _ in enumerate(pred_centers):
            pred_center = pred_centers[idx : idx + 1, :]  # (1, 2)
            dxy = np.sqrt(
                np.sum(np.square(pred_center - gt_centers), axis=-1)
            )  # (N, )

            dxy[valid_gt == 0] = 1000
            min_idx = np.argmin(dxy)
            min_dxy = dxy[min_idx]
            for dist_interval, _dxy_thresh in self.dxy_threshs.items():
                if dist_interval[0] <= gt_dist[min_idx] < dist_interval[1]:
                    break

            if min_dxy < _dxy_thresh:
                gt_assign[min_idx] = idx
                matched_pred[idx] = 1
                valid_gt[min_idx] = 0
                matched_dxy[min_idx] = min_dxy

        return matched_pred.sum(), matched_dxy.sum()

    def calculate_pr_per_depth_interval(
        self,
        gt_centers: List[List[float]],
        gt_contours_area: List[float],
        pred_centers: List[List[float]],
        pred_contours_area: List[float],
    ):
        """Calculate tp, fn, fp for each depth interval.

        Args:
            gt_centers: list of gt object center coord (u, v).
            gt_contours_area: list of gt object area, correspond
                to gt_centers.
            pred_centers: list of pred object center coord (u, v).
            pred_contours_area: list of pred object area, correspond
                to pred_centers.
        """
        for area_interval in self.full_area_intervals:
            lower_area, upper_area = area_interval
            for depth_interval in self.full_depth_intervals:
                prefix = "area_%d_%d_depth_%d_%d_" % (
                    area_interval[0],
                    area_interval[1],
                    depth_interval[0],
                    depth_interval[1],
                )
                mask = self.depth_interval_mask[
                    f"{depth_interval[0]}_{depth_interval[1]}"
                ]
                valid_gt_centers = []
                valid_pred_centers = []
                for gt_center, gt_area in zip(gt_centers, gt_contours_area):
                    cx, cy = gt_center
                    if (
                        lower_area <= gt_area < upper_area
                        and mask[int(cy), int(cx)]
                    ):
                        valid_gt_centers.append(gt_center)
                for pred_center, pred_area in zip(
                    pred_centers, pred_contours_area
                ):
                    cx, cy = pred_center
                    if (
                        lower_area <= pred_area < upper_area
                        and mask[int(cy), int(cx)]
                    ):
                        valid_pred_centers.append(pred_center)
                valid_tp = 0
                valid_dxy = 0
                if len(valid_gt_centers) > 0 and len(valid_pred_centers) > 0:
                    valid_tp, valid_dxy = self.center_distance_match(
                        np.array(valid_gt_centers),
                        np.array(valid_pred_centers),
                    )
                    setattr(
                        self,
                        prefix + "tp",
                        getattr(self, prefix + "tp") + valid_tp,
                    )
                    setattr(
                        self,
                        prefix + "dxy",
                        getattr(self, prefix + "dxy") + valid_dxy,
                    )

                setattr(
                    self,
                    prefix + "fn",
                    getattr(self, prefix + "fn")
                    + len(valid_gt_centers)
                    - valid_tp,
                )
                setattr(
                    self,
                    prefix + "fp",
                    getattr(self, prefix + "fp")
                    + len(valid_pred_centers)
                    - valid_tp,
                )

    def calculate_small_obj_pr(self, label: torch.Tensor, pred: torch.tensor):
        label = label.cpu().numpy().astype(np.uint8)
        pred = pred.detach().cpu().numpy().astype(np.uint8)

        # TODO: support large objects use iou as matching
        for i in range(label.shape[0]):
            gt_centers = []
            pred_centers = []

            gt_centers_all = []
            pred_centers_all = []
            gt_contours_area_all = []
            pred_contours_area_all = []

            # get roi max freespace region contain ego
            gt_max_freespace_mask = max_freespace_contain_ego(
                label[i, :, :], self.vcs_range, use_erode=True
            )
            pred_max_freespace_mask = max_freespace_contain_ego(
                pred[i, :, :], self.vcs_range, use_erode=True
            )
            label[i, :, :][gt_max_freespace_mask < 1] = 0  # freespace_yes=0
            pred[i, :, :][pred_max_freespace_mask < 1] = 0

            # set ignore to freespace
            pred[i, :, :][label[i, :, :] == self.ignore_index] = 0
            label[i, :, :][label[i, :, :] == self.ignore_index] = 0
            # TODO: remove noise mainly for autolabel !!!
            label[i, :, :] = cv2.medianBlur(label[i, :, :], 3)
            pred[i, :, :] = cv2.medianBlur(pred[i, :, :], 3)

            contours, _ = cv2.findContours(
                label[i, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for contour in contours:
                cx, cy = contour.sum(axis=(0, 1)) / contour.shape[0]
                gt_centers_all.append([cx, cy])
                gt_contours_area_all.append(cv2.contourArea(contour))
                contour_area = cv2.contourArea(contour)
                if (
                    contour_area < self.obj_area_thresh_ub
                    and contour_area > self.obj_area_thresh_lb
                ):
                    gt_centers.append([cx, cy])

            contours, _ = cv2.findContours(
                pred[i, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for contour in contours:
                cx, cy = contour.sum(axis=(0, 1)) / contour.shape[0]
                pred_centers_all.append([cx, cy])
                pred_contours_area_all.append(cv2.contourArea(contour))
                contour_area = cv2.contourArea(contour)
                if contour_area < self.obj_area_thresh_ub:
                    pred_centers.append([cx, cy])

            # overall small object precision-recall
            tp = 0
            dxy = 0
            if len(gt_centers) > 0 and len(pred_centers) > 0:
                tp, dxy = self.center_distance_match(
                    np.array(gt_centers),
                    np.array(pred_centers),
                )
                self.smallobj_tp += tp
                self.smallobj_dxy += dxy
            self.smallobj_fn += len(gt_centers) - tp
            self.smallobj_fp += len(pred_centers) - tp

            # depth interval small object precision-recall
            self.calculate_pr_per_depth_interval(
                gt_centers_all,
                gt_contours_area_all,
                pred_centers_all,
                pred_contours_area_all,
            )

    def calculate_iou(
        self,
        num_classes: int,
        label: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        # only one pred and one gt used in MeanIOU calculation.
        pred_label = pred.detach()
        if mask is not None:
            mask = torch.logical_and(label != self.ignore_index, mask)
        else:
            mask = label != self.ignore_index
        pred_label = pred_label[mask].float()
        label = label[mask].float()

        intersect = pred_label[pred_label == label]

        area_intersect = torch.histc(
            intersect, bins=num_classes, max=num_classes - 1
        )
        area_pred_label = torch.histc(
            pred_label, bins=num_classes, max=num_classes - 1
        )
        area_label = torch.histc(label, bins=num_classes, max=num_classes - 1)
        area_union = area_pred_label + area_label - area_intersect

        return area_intersect, area_union, area_pred_label, area_label

    def crop_roi(self, img):
        if self.eval_vcs_range is None:
            return img
        output = crop_roi_vcs_range(
            torch.permute(img, (1, 2, 0)),  # (h, w, c)
            self.eval_vcs_range,
            self.raw_vcs_range,
        )
        return torch.permute(output, (2, 0, 1))

    def calculate_boundary_error(
        self, label: torch.Tensor, pred: torch.tensor
    ):
        """Calculate boundary error for freespace.

        Args:
            label: (b, h, w)
            pred: (b, h, w)
        """
        label = label.cpu().numpy().astype(np.uint8)
        pred = pred.detach().cpu().numpy().astype(np.uint8)  # binary
        for i in range(label.shape[0]):
            label[i, ...] = cv2.medianBlur(label[i, :, :], 3)
            pred[i, ...] = cv2.medianBlur(pred[i, :, :], 3)

            (
                gt_vcs_bdry_pts,
                gt_vcs_bdry_pts_radius,
                vcs_bdry_pts_ignore_mask,
            ) = self.boundary_extractor.get_boundary_points(label[i, ...])
            (
                pred_vcs_bdry_pts,
                pred_vcs_bdry_pts_radius,
                _,
            ) = self.boundary_extractor.get_boundary_points(pred[i, ...])
            bdry_pts_vu = np.stack(
                [
                    (self.vcs_range[2] - gt_vcs_bdry_pts[:, 0])
                    / self.spatial_resolution[1],  # vcs x
                    (self.vcs_range[3] - gt_vcs_bdry_pts[:, 1])
                    / self.spatial_resolution[0],  # vcs y
                ],
                axis=1,
            ).astype(np.int)
            for dist_interval in self.full_dist_intervals:
                dist_mask = (
                    self.dist_interval_mask[
                        f"{dist_interval[0]}_{dist_interval[1]}"
                    ][bdry_pts_vu[:, 0], bdry_pts_vu[:, 1]]
                    > 0
                )
                valid_mask = (1 - vcs_bdry_pts_ignore_mask) > 0
                valid_mask = np.logical_and(dist_mask, valid_mask)
                dist_error = np.abs(
                    gt_vcs_bdry_pts_radius[valid_mask]
                    - pred_vcs_bdry_pts_radius[valid_mask]
                )
                # count boundary error
                index = self.full_dist_intervals.index(dist_interval)
                for cur_error in dist_error:
                    error_index = int(cur_error / self.boundary_error_delta)
                    self.boundary_error_arr[index, error_index] += 1

    def update(self, label, pred):
        """
        Update internal buffer with latest predictions.

        Note that the statistics are not available until
        you call self.get() to return the metrics.

        Args:
            label: gt.
            preds: model output.

        """
        # TODO: ben.hu, to unify pred and label shape, before calling BEVFreespaceMetric.  # noqa
        label = label[self.task_name][self.gt_name][self.gt_name]
        pred = _as_list(pred[self.pred_name])[0]
        if label.shape[-2:] != pred.shape[-2:]:
            label = resize(
                label,
                pred.shape[-2:],
                interpolation=InterpolationMode.NEAREST,
            )
        pred = self.crop_roi(pred)
        label = self.crop_roi(label)

        # compute pr
        if "pr" in self.metric_modes:
            self.calculate_small_obj_pr(label, pred)

        # boundary error
        if "boundary_error" in self.metric_modes:
            self.calculate_boundary_error(label, pred)

        # iou
        (
            area_intersect,
            area_union,
            area_pred_label,
            area_label,
        ) = self.calculate_iou(self.num_classes, label, pred)
        self.intersect += area_intersect
        self.union += area_union
        self.pred_label += area_pred_label
        self.label += area_label

        if self.obj_area_thresh_ub > 0:
            area_thresh_mask = self.get_small_obj_mask(label)
            (
                area_intersect,
                area_union,
                area_pred_label,
                area_label,
            ) = self.calculate_iou(
                self.num_classes, label, pred, area_thresh_mask
            )
            self.small_obj_intersect += area_intersect
            self.small_obj_union += area_union
            self.small_obj_pred_label += area_pred_label
            self.small_obj_label += area_label

    def calculate_iou_metrics(
        self,
        intersect,
        union,
        label,
        global_save_index_list,
        name,
        class_ids,
    ):
        def _save_tensor_ele(x: torch.Tensor, save_index_list: list):
            _save_tensor = [x[index : index + 1] for index in save_index_list]
            return torch.cat(_save_tensor, dim=0)

        def _tensor_nan_mean(x: torch.Tensor):
            """Compute the mean value, ignoring NaNs."""

            tmp_value = x.cpu().numpy()
            _mean_iou = np.nanmean(tmp_value)
            mean_iou = x.new_tensor(_mean_iou)
            return mean_iou

        res_dict = {}
        all_acc = (
            _save_tensor_ele(intersect, global_save_index_list).sum()
            / _save_tensor_ele(label, global_save_index_list).sum()
        )
        acc = intersect / label
        iou = intersect / union

        summary_str = "\n~~~~ %s Summary metrics ~~~~\n" % (name)
        summary_str += "Summary:\n"
        line_format = "{:<15} {:>10} {:>10} {:>10}\n"
        summary_str += line_format.format("Scope", "mIoU", "mAcc", "aAcc")

        miou = _tensor_nan_mean(_save_tensor_ele(iou, global_save_index_list))
        macc = _tensor_nan_mean(_save_tensor_ele(acc, global_save_index_list))
        iou_str = "{:.2f}".format(miou.cpu().item() * 100)
        acc_str = "{:.2f}".format(macc.cpu().item() * 100)
        all_acc_str = "{:.2f}".format(all_acc * 100)
        summary_str += line_format.format(
            "global", iou_str, acc_str, all_acc_str
        )

        summary_str += "Per Class Results:\n"
        line_format = "{:<15} {:>10} {:>10}\n"
        summary_str += line_format.format("Class", "IoU", "Acc")

        res_dict["global"] = {
            f"{self.result_prefix} mIOU": float(iou_str),
            f"{self.result_prefix} mAcc": float(acc_str),
            f"{self.result_prefix} aAcc": float(all_acc_str),
        }
        table_data = []
        for i in class_ids:
            iou_str = "{:.2f}".format(iou[i].cpu().item() * 100)
            acc_str = "{:.2f}".format(acc[i].cpu().item() * 100)
            summary_str += line_format.format(
                self.seg_class[i], iou_str, acc_str
            )
            res_dict[f"{i}"] = {"IOU": float(iou_str), "Acc": float(acc_str)}
            table_data.append(
                {"Class": f"{i}", "IoU": float(iou_str), "Acc": float(acc_str)}
            )
        logger.info(summary_str)

        summary = res_dict["global"]
        tables = [
            Table(
                name=self.result_prefix + "_" + name,
                columns=["Class", "IoU", "Acc"],
                data=table_data,
            )
        ]

        eval_result = EvalResult(
            summary=summary,
            tables=tables,
        )

        return miou, res_dict, eval_result

    def calculate_small_obj_pr_metrics(self):
        # precesion-recall log
        summary_str = "\n~~~~ Precision-Recall metrics by depth ~~~~\n"
        area_show = [
            [start, end]
            for start, end in zip(
                self.physic_area_intervals[:-1], self.physic_area_intervals[1:]
            )
        ]
        area_show.append(
            [self.physic_area_intervals[0], self.physic_area_intervals[-1]]
        )
        for i, area_interval in enumerate(self.full_area_intervals):
            area_strs = "object area {:.1f} ~ {:.1f}m^2:\n".format(
                area_show[i][0], area_show[i][1]
            )
            summary_str += area_strs
            line_format = "{:<15}"
            recall_strs = ["Recall(%)"]
            precision_strs = ["Precision(%)"]
            dxy_strs = ["dxy(m)"]
            all_tp = 0
            all_fn = 0
            all_fp = 0
            depth_interval_strs = ["depth(m)"]
            for depth_interval in self.full_depth_intervals:
                line_format += " {:>10}"
                depth_interval_strs.append(
                    f"{depth_interval[0]}_{depth_interval[1]}"
                )
                prefix = "area_%d_%d_depth_%d_%d_" % (
                    area_interval[0],
                    area_interval[1],
                    depth_interval[0],
                    depth_interval[1],
                )
                tp = getattr(self, prefix + "tp").cpu().item()
                fn = getattr(self, prefix + "fn").cpu().item()
                fp = getattr(self, prefix + "fp").cpu().item()
                dxy = getattr(self, prefix + "dxy").cpu().item()
                recall = tp / max(tp + fn, 1)
                precision = tp / max(tp + fp, 1)
                dxy = dxy / max(tp, 1)
                recall_strs.append("{:.2f}".format(recall * 100))
                precision_strs.append("{:.2f}".format(precision * 100))
                dxy_strs.append("{:.2f}".format(dxy))
                all_tp += tp
                all_fn += fn
                all_fp += fp

            line_format += "\n"
            summary_str += line_format.format(*depth_interval_strs)
            summary_str += line_format.format(*recall_strs)
            summary_str += line_format.format(*precision_strs)
            summary_str += line_format.format(*dxy_strs)
            all_tp_strs = ["TP"] + [" "] * len(self.full_depth_intervals)
            all_tp_strs[1] = str(int(all_tp))
            summary_str += line_format.format(*all_tp_strs)
            all_fn_strs = ["FN"] + [" "] * len(self.full_depth_intervals)
            all_fn_strs[1] = str(int(all_fn))
            summary_str += line_format.format(*all_fn_strs)
            all_fp_strs = ["FP"] + [" "] * len(self.full_depth_intervals)
            all_fp_strs[1] = str(int(all_fp))
            summary_str += line_format.format(*all_fp_strs)
            summary_str += "\n"
        logger.info(summary_str)

    def calculate_boundary_metrics(self):
        # boundary-eval log
        summary_str = (
            "\n~~~~ Freespace boundary error by distance to ego ~~~~\n"  # noqa
        )
        line_format = "{:<18}"
        stats_num_strs = ["Num"]
        stats_tp_strs = ["TP"]
        stats_mean_strs = ["Mean(m)"]
        stats_std_strs = ["Std(m)"]
        stats_max_strs = ["Max(m)"]
        stats_50_percentile_strs = ["50_percentile(m)"]
        stats_90_percentile_strs = ["90_percentile(m)"]
        stats_95_percentile_strs = ["95_percentile(m)"]
        dist_interval_strs = ["dist(m)"]
        self.boundary_error_mean = []
        for dist_interval in self.full_dist_intervals:
            dxy_thresh = self.eval_cfg["boundary_cfg"][
                "dist_interval_dxy_threshs"
            ][dist_interval]
            tp_truncate_index = min(
                int(dxy_thresh / self.boundary_error_delta) + 1,
                self.boundary_error_arr_len,
            )
            line_format += " {:>10}"
            dist_interval_strs.append(f"{dist_interval[0]}_{dist_interval[1]}")
            index = self.full_dist_intervals.index(dist_interval)
            num_all = self.boundary_error_arr[index, :].sum().cpu().item()
            num_tp = (
                self.boundary_error_arr[index, :tp_truncate_index]
                .sum()
                .cpu()
                .item()
            )
            stats_num_strs.append(str(num_all))
            stats_tp_strs.append(str(num_tp))
            # mean
            mean = 0
            for i in range(tp_truncate_index):
                mean += (
                    i
                    * self.boundary_error_delta
                    * self.boundary_error_arr[index, i]
                )
            mean = mean.cpu().item() / max(num_tp, 1)
            self.boundary_error_mean.append(mean)
            stats_mean_strs.append("{:.2f}".format(mean))
            # std
            var = 0
            for i in range(tp_truncate_index):
                if self.boundary_error_arr[index, i] > 0:
                    boundary_error = i * self.boundary_error_delta
                    var += (
                        self.boundary_error_arr[index, i]
                        * (boundary_error - mean) ** 2
                    )
            std = (var / max(num_tp, 1)) ** 0.5
            stats_std_strs.append("{:.2f}".format(std))
            # 50 percentile
            perc_50_loc = int(num_tp * 0.5)
            count = 0
            for i in range(tp_truncate_index):
                if count >= perc_50_loc:
                    break
                count += self.boundary_error_arr[index, i]
            perc_50_error = i * self.boundary_error_delta
            stats_50_percentile_strs.append("{:.2f}".format(perc_50_error))
            # 90 percentile
            perc_90_loc = int(num_tp * 0.9)
            count = 0
            for i in range(tp_truncate_index):
                if count >= perc_90_loc:
                    break
                count += self.boundary_error_arr[index, i]
            perc_90_error = i * self.boundary_error_delta
            stats_90_percentile_strs.append("{:.2f}".format(perc_90_error))
            # 95 percentile
            perc_95_loc = int(num_tp * 0.95)
            count = 0
            for i in range(tp_truncate_index):
                if count >= perc_95_loc:
                    break
                count += self.boundary_error_arr[index, i]
            perc_95_error = i * self.boundary_error_delta
            stats_95_percentile_strs.append("{:.2f}".format(perc_95_error))
            # max
            max_v = 0
            for i in range(tp_truncate_index - 1, -1, -1):
                if self.boundary_error_arr[index, i] > 0:
                    break
            max_v = i * self.boundary_error_delta
            stats_max_strs.append("{:.2f}".format(max_v))
        line_format += "\n"
        summary_str += line_format.format(*dist_interval_strs)
        summary_str += line_format.format(*stats_num_strs)
        summary_str += line_format.format(*stats_tp_strs)
        summary_str += line_format.format(*stats_mean_strs)
        summary_str += line_format.format(*stats_std_strs)
        summary_str += line_format.format(*stats_50_percentile_strs)
        summary_str += line_format.format(*stats_90_percentile_strs)
        summary_str += line_format.format(*stats_95_percentile_strs)
        summary_str += line_format.format(*stats_max_strs)
        summary_str += "\n"
        logger.info(summary_str)

    def compute(self):
        """Get evaluation metrics."""
        eval_result = None
        output, res_dict, eval_result = self.calculate_iou_metrics(
            self.intersect,
            self.union,
            self.label,
            self.global_save_index_list,
            "MeanIoU",
            class_ids=range(self.num_classes),
        )

        if self.obj_area_thresh_ub > 0:
            (
                small_obj_output,
                small_obj_res_dict,
                small_obj_eval_result,
            ) = self.calculate_iou_metrics(
                self.small_obj_intersect,
                self.small_obj_union,
                self.small_obj_label,
                self.global_save_index_list,
                "Small Object MeanIoU ({:.3f}m^2 < area < {:.3f}m^2)".format(
                    self.eval_cfg["smallobj_area_thresh_lb"],
                    self.eval_cfg["smallobj_area_thresh_ub"],
                ),
                class_ids=range(self.num_classes),
            )
            res_dict.update(
                {
                    "small_object_surrounding": small_obj_res_dict["0"],
                    "small_object_body": small_obj_res_dict["1"],
                }
            )
            for k, v in small_obj_eval_result.summary.items():
                eval_result.summary.update({f"Small Object {k}": v})
            eval_result.tables.extend(small_obj_eval_result.tables)

        self.save_res_json(res_dict)

        # small obj recall
        if "pr" in self.metric_modes:
            summary_str = "\n~~~~ Small Object ({:.3f}m^2 ".format(
                self.eval_cfg["smallobj_area_thresh_lb"]
            ) + "< area < {:.3f}m^2) metrics ~~~~\n".format(
                self.eval_cfg["smallobj_area_thresh_ub"]
            )
            line_format = "{:<15} {:>10} {:>10} {:>10} {:>10}\n"
            summary_str += line_format.format(
                "Recall(%)", "Precision(%)", "TP", "FN", "FP"
            )
            tp = self.smallobj_tp.cpu().item()
            fn = self.smallobj_fn.cpu().item()
            fp = self.smallobj_fp.cpu().item()
            recall = tp / max(tp + fn, 1)
            recall_str = "{:.2f}".format(recall * 100)
            precision = tp / max(tp + fp, 1)
            precision_str = "{:.2f}".format(precision * 100)
            summary_str += line_format.format(
                recall_str,
                precision_str,
                str(int(tp)),
                str(int(fn)),
                str(int(fp)),  # noqa
            )
            logger.info(summary_str)

            # small object pr metrics by depth and area
            self.calculate_small_obj_pr_metrics()

        # boundary metrics
        if "boundary_error" in self.metric_modes:
            self.calculate_boundary_metrics()

        return eval_result


@OBJECT_REGISTRY.register
class ANCBevElevationMetric(ANCBEVFreespaceMetric):
    """Calculate error of bev&voxel elevation task.

    ANCBevElevationMetric support computing metrics, including
    regression metrics using in monodepth(a1, a2, a3, mae,
    mse, rmse, mse_log, rmse_log, abs_rel, sq_rel) and
    classification metric(percision, recall, accuracy).

    Args:
        bev_elevation_type: elevationb gt type, support
            "HDE_REG", "HDE_45_7", "HDE_7_7".
        metrics_type: support "vismask" and "elevation",
            must provide one for valid computing.
        bev_size: Bev size, in pixel.(order is (h,w)).
        vcs_range: Vcs range, in meter.(order is
            (bottom,right,top,left), e.g.(-30, -51.2, 72.4, 51.2)).
        eval_cfg: eval setting dict, for example, defined as::

            {
                "eval_vcs_range": None,
                "depth_intervals": (0, 5, 10, 25, 50, 100),
                "height_intervals": (0, 10, 25, 50),
            }

            where "eval_vcs_range" is roi vcs range to eval, ordered in
            (bottom, right, top, left), same vcs origin with vcs_range;
            "depth_intervals" for metrics in each depth interval,
            measured in meter; "height_intervals" for metrics in each
            height interval, measured in centimeter.
        height_names: show height names if bev_elevation_type is
            classification.
        save_metric_path: path to save result.
        use_mask_type: include "freespace", "vismask"
        ignore_index: class index to ignore.
    """

    def __init__(
        self,
        bev_elevation_type: str,
        metrics_type: Sequence[str],
        bev_size: Sequence[int],
        vcs_range: Sequence[float],
        eval_cfg: Dict,
        height_names: Sequence[str] = None,
        save_metric_path: str = None,
        use_mask_type: Sequence = None,
        ignore_index: int = 255,
        result_prefix: str = "wide",
    ):
        assert (
            metrics_type == ["vis_mask"]
            or metrics_type == ["elevation"]
            or metrics_type == ["vis_mask", "elevation"]
        )
        assert bev_elevation_type in ["HDE_REG", "HDE_45_7", "HDE_7_7"]
        self.bev_elevation_type = bev_elevation_type
        self.metrics_type = metrics_type

        self.use_mask_type = use_mask_type
        self.ignore_index = ignore_index
        self.result_prefix = result_prefix
        self.name = []
        if "vismask" in self.metrics_type:
            self.name.append("vismask_miou")
        if "elevation" in self.metrics_type:
            if self.bev_elevation_type != "HDE_REG":
                self.name.append("elevation_miou")
            else:
                self.name.append("elevation_reg_errs")

        if bev_elevation_type == "HDE_REG":
            self.cls_nums = 9  # remap to 9 class for visualisation
        else:
            self.cls_nums = int(bev_elevation_type.split("_")[-1])
        self.cls_ids = list(range(self.cls_nums))

        self.metric_intervals = {}
        depth_intervals = eval_cfg["depth_intervals"]
        height_intervals = eval_cfg["height_intervals"]
        h0 = height_intervals[0]
        for h in height_intervals[1:]:
            self.metric_intervals[(h0, h)] = []
            for d0, d1 in zip(depth_intervals[:-1], depth_intervals[1:]):
                self.metric_intervals[(h0, h)].append([d0, d1])

        self.save_metric_path = save_metric_path
        if height_names is None:
            height_names = list(map(str, self.cls_ids))
        super(ANCBevElevationMetric, self).__init__(
            seg_class=height_names,
            bev_size=bev_size,
            vcs_range=vcs_range,
            eval_cfg=eval_cfg,
            name=self.name,
            ignore_index=ignore_index,
            save_metric_path=save_metric_path,
            result_prefix=result_prefix,
        )

        dist_map = self.init_distance_mask(bev_size, vcs_range)
        self.dist_map = dist_map
        self.dist_masks = {}
        for d0, d1 in zip(depth_intervals[:-1], depth_intervals[1:]):
            mask = torch.from_numpy((dist_map >= d0) * (dist_map < d1))
            mask = self.crop_roi(torch.unsqueeze(mask, 0))
            self.dist_masks[f"{d0}_{d1}"] = mask

    def init_distance_mask(self, bev_size, vcs_range):
        h, w = bev_size
        # vcs: x is usual y
        y, x = np.meshgrid(range(w), range(h), indexing="xy")
        spatial_ratio = [
            decimal_div(
                abs(decimal_minus(self.vcs_range[2], self.vcs_range[0])), h
            ),
            decimal_div(
                abs(decimal_minus(self.vcs_range[3], self.vcs_range[1])), w
            ),
        ]
        x = vcs_range[2] - x * spatial_ratio[0]
        y = vcs_range[3] - y * spatial_ratio[1]

        xy = (x ** 2 + y ** 2) ** 0.5

        return xy

    def _init_states(self):
        state_name_suffixes = ["intersect", "pred_label", "label", "union"]
        if "vismask" in self.metrics_type:
            for suffix in state_name_suffixes:
                self.add_state(
                    name=f"vismask_area_{suffix}",
                    default=torch.zeros(2),
                    dist_reduce_fx="sum",
                )

        if "elevation" in self.metrics_type:
            # regression metrics
            # a1, a2, a3, mae, mse, rmse, mse_log, rmse_log, abs_rel, sq_rel # noqa
            self.add_state(
                name="elevation_reg_errs",
                default=torch.zeros(1, 10),
                dist_reduce_fx="sum",
            )
            for (h0, h1), dep_intervals in self.metric_intervals.items():
                for d0, d1 in dep_intervals:
                    name = f"{h0}_{h1}_{d0}_{d1}"
                    self.add_state(
                        name=name,
                        default=torch.zeros(2),
                        dist_reduce_fx="sum",
                    )
            if self.bev_elevation_type != "HDE_REG":
                for suffix in state_name_suffixes:
                    self.add_state(
                        name=f"elevation_area_{suffix}",
                        default=torch.zeros(self.cls_nums),
                        dist_reduce_fx="sum",
                    )

        self.add_state(
            name="num_inst",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )

    def calculate_regression_error(
        self,
        gt: torch.tensor,
        pred: torch.tensor,
    ):
        # see detail in https://github.com/nianticlabs/monodepth2.
        gt = gt.float()
        pred = pred.float()
        thresh = torch.max(gt / pred, pred / gt)
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        diff = gt - pred
        mae = diff.abs().mean()
        mse = diff.pow(2).mean()
        rmse = mse.sqrt()

        mse_log = (gt.log() - pred.log()).pow(2).mean()
        rmse_log = mse_log.sqrt()

        abs_rel = (diff.abs() / gt).mean()
        sq_rel = (diff.pow(2) / gt).mean()
        return [
            a1,
            a2,
            a3,
            mae,
            mse,
            rmse,
            mse_log,
            rmse_log,
            abs_rel,
            sq_rel,
        ]

    def statistic_regression_error(
        self, gt: torch.tensor, pred: torch.tensor, valid_mask: torch.tensor
    ):
        for (h0, h1), dep_intervals in self.metric_intervals.items():
            valid_mask_h = valid_mask * (gt >= h0 / 100) * (gt < h1 / 100)
            for d0, d1 in dep_intervals:
                valid_mask_d = self.dist_masks[f"{d0}_{d1}"]
                valid_mask_d = valid_mask_d.type_as(valid_mask_h)
                valid_mask_hd = valid_mask_h * valid_mask_d
                h_error = torch.abs(gt[valid_mask_hd] - pred[valid_mask_hd])
                h_error_num = valid_mask_hd.sum().float()
                h_error = h_error.sum()
                name = f"{h0}_{h1}_{d0}_{d1}"
                val = getattr(self, name)
                val[0] += h_error
                val[1] += h_error_num
                setattr(self, name, val)

    def update_metrics(
        self,
        elevation_gt: torch.tensor,
        elevation_pred: torch.tensor,
        vis_mask_gt: dict,
        vis_mask_pred: torch.tensor,
        confidence_pred: torch.tensor,
        bev_freespace_gt: torch.tensor,
    ):
        elevation_gt = self.crop_roi(elevation_gt)
        elevation_pred = self.crop_roi(elevation_pred)
        valid_mask = elevation_gt != self.ignore_index
        if bev_freespace_gt is not None and "freespace" in self.use_mask_type:
            bev_freespace_gt = self.crop_roi(bev_freespace_gt)
            valid_mask = valid_mask * (bev_freespace_gt != 0)
        if vis_mask_gt is not None and "vismask" in self.use_mask_type:
            vis_mask_gt = vis_mask_gt["vismask"]
            vis_mask_gt = self.crop_roi(vis_mask_gt)
            valid_mask = valid_mask * (vis_mask_gt != 0)
        elevation_gt_valid = elevation_gt[valid_mask]
        if self.bev_elevation_type == "HDE_REG":
            elevation_pred = torch.squeeze(elevation_pred, 1)
        elevation_pred_valid = elevation_pred[valid_mask]

        # vis mask
        elevation_reg_errs = 0.0
        vismask_area_intersect = 0.0
        vismask_area_pred_label = 0.0
        vismask_area_label = 0.0
        vismask_area_union = 0.0
        elevation_area_intersect = 0.0
        elevation_area_pred_label = 0.0
        elevation_area_label = 0.0
        elevation_area_union = 0.0

        if "vismask" in self.metrics_type:
            (
                vismask_area_intersect,
                vismask_area_union,
                vismask_area_pred_label,
                vismask_area_label,
            ) = self.calculate_iou(2, vis_mask_gt, vis_mask_pred)
        if self.bev_elevation_type == "HDE_REG":
            elevation_reg_errs = self.calculate_regression_error(
                elevation_gt_valid,
                elevation_pred_valid,
            )
            self.statistic_regression_error(
                elevation_gt, elevation_pred, valid_mask
            )
        else:
            (
                elevation_area_intersect,
                elevation_area_union,
                elevation_area_pred_label,
                elevation_area_label,
            ) = self.calculate_iou(
                self.cls_nums, elevation_gt_valid, elevation_pred_valid
            )
            elevation_reg_errs = self.calculate_regression_error(
                elevation_gt_valid + 1,
                elevation_pred_valid + 1,
            )

        return (
            elevation_reg_errs,
            vismask_area_intersect,
            vismask_area_pred_label,
            vismask_area_label,
            vismask_area_union,
            elevation_area_intersect,
            elevation_area_pred_label,
            elevation_area_label,
            elevation_area_union,
        )

    def update(
        self,
        elevation_gts: torch.tensor,
        timestamps: torch.tensor,
        vis_mask_gts: dict = None,
        bev_freespace_gts: torch.tensor = None,
        elevation_preds: torch.tensor = None,
        vis_mask_preds: torch.tensor = None,
        confidence_preds: torch.tensor = None,
    ):
        """Update batch statistics for final metrics.

        Args:
            elevation_gts: elevation gt, (b, 1, h, w).
            timestamps: timestamps, (b, 1).
            vis_mask_gts: vismask (b, h, w), agent (b, h, w).
            bev_freespace_gts: freespace gt to constraint elevation
                eval region,  (b, h, w).
            elevation_preds: elevation pred, (b, h, w).
            vis_mask_preds: vis mask pred, (b, h, w).
            confidence_preds: confidence pred, (b, h, w).
        """

        (
            elevation_reg_errs,
            vismask_area_intersect,
            vismask_area_pred_label,
            vismask_area_label,
            vismask_area_union,
            elevation_area_intersect,
            elevation_area_pred_label,
            elevation_area_label,
            elevation_area_union,
        ) = self.update_metrics(
            elevation_gts,
            elevation_preds,
            vis_mask_gts,
            vis_mask_preds,
            confidence_preds,
            bev_freespace_gts,
        )
        if "vismask" in self.metrics_type:
            self.vismask_area_intersect += vismask_area_intersect
            self.vismask_area_pred_label += vismask_area_pred_label
            self.vismask_area_label += vismask_area_label
            self.vismask_area_union += vismask_area_union
        if "elevation" in self.metrics_type:
            if self.bev_elevation_type != "HDE_REG":
                self.elevation_area_intersect += elevation_area_intersect
                self.elevation_area_pred_label += elevation_area_pred_label
                self.elevation_area_label += elevation_area_label
                self.elevation_area_union += elevation_area_union
            self.elevation_reg_errs += torch.cat(
                [i.unsqueeze(0) for i in elevation_reg_errs]
            )
        self.num_inst += 1

    def calculate_regression_metrics(
        self,
    ):
        reg_errors = self.elevation_reg_errs / self.num_inst

        summary_str = "Regreesion Results:\n"
        line_format = (
            "{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}\n"
        )
        res_dict = {}
        metric_name = [
            "a1",
            "a2",
            "a3",
            "mae",
            "mse",
            "rmse",
            "mse_log",
            "rmse_log",
            "abs_rel",
            "sq_rel",
        ]
        summary_str += line_format.format(*metric_name)
        reg_errors_str = ""
        for i in range(reg_errors.shape[-1]):
            reg_error_i = reg_errors[0, i].cpu().item()
            res_dict[metric_name[i]] = round(reg_error_i, 2)
            reg_errors_str += "{:.2f}     ".format(reg_error_i)
        summary_str += reg_errors_str
        logger.info(summary_str)

        tables = []
        tables.append(
            Table(
                name=f"{self.result_prefix} Regreesion Results",
                columns=list(res_dict.keys()),
                data=[res_dict],
            )
        )
        res_dict_intervals = self.print_metric_intervals()
        _columns, _rows = [], []
        data = []
        for k, _ in res_dict_intervals.items():
            h0, h1, d0, d1 = k.split("_")
            _columns.append(f"{d0}_{d1}m")
            _rows.append(f"{h0}_{h1}cm")
        _columns = list(set(_columns))
        _rows = list(set(_rows))

        for row in _rows:
            h0, h1 = row[:-2].split("_")
            _data = {"height/depth": row}
            for col in _columns:
                d0, d1 = col[:-1].split("_")
                _data.update({col: res_dict_intervals[f"{h0}_{h1}_{d0}_{d1}"]})
            data.append(_data)

        tables.append(
            Table(
                name=f"{self.result_prefix} Average Error in Regions",
                columns=[
                    "height/depth",
                    "0_5m",
                    "5_10m",
                    "10_25m",
                    "25_50m",
                    "50_100m",
                ],
                data=data,
            )
        )
        eval_result = EvalResult(tables=tables)

        return reg_errors, res_dict, res_dict_intervals, eval_result

    def print_metric_intervals(self):
        res_dict = {}
        log_line = "\nAverage Error in Regions:\n"
        log_line += "height/depth".ljust(15)
        for dep_interval in self.dist_masks.keys():
            log_line += f"{dep_interval}m".ljust(10)
        log_line += "\n"
        for (h0, h1), dep_intervals in self.metric_intervals.items():
            log_line += f"{h0}_{h1}cm".ljust(15)
            for d0, d1 in dep_intervals:
                key = f"{h0}_{h1}_{d0}_{d1}"
                val = getattr(self, key).cpu().numpy()
                val_mean = round(float(val[0] / val[1] * 100), 3)
                log_line += f"{val_mean}".ljust(10)
                res_dict[key] = val_mean
            log_line += "\n"
        logger.info(log_line)
        return res_dict

    def compute(self):
        val = []
        res_dict = {}
        if "vismask" in self.metrics_type:
            (
                vismask_miou,
                iou_res_dict_vismask,
                eval_result,
            ) = self.calculate_iou_metrics(
                self.vismask_area_intersect,
                self.vismask_area_union,
                self.vismask_area_label,
                global_save_index_list=[0, 1],
                name="vismask",
                class_ids=[0, 1],
            )
            val.append(vismask_miou)
            res_dict.update({"iou_vismask": iou_res_dict_vismask})
        if "elevation" in self.metrics_type:
            if self.bev_elevation_type == "HDE_REG":
                (
                    reg_errors,
                    res_dict_ele,
                    res_dict_ele_intervals,
                    eval_result,
                ) = self.calculate_regression_metrics()
                val.append(reg_errors)
                res_dict.update({"reg_errors": res_dict_ele})
                res_dict.update(
                    {"reg_errors_intervals": res_dict_ele_intervals}
                )
            else:
                (
                    elevation_miou,
                    iou_res_dict_ele,
                    eval_result,
                ) = self.calculate_iou_metrics(
                    self.elevation_area_intersect,
                    self.elevation_area_union,
                    self.elevation_area_label,
                    global_save_index_list=self.cls_ids,
                    name="elevation",
                    class_ids=self.cls_ids,
                )
                val.append(elevation_miou)
                res_dict.update({"iou_ele": iou_res_dict_ele})
        self.save_res_json(res_dict)
        return eval_result
