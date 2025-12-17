# Copyright (c) Horizon Robotics. All rights reserved.
import json
import logging
import os
from cmath import pi
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import LineString, Polygon

try:
    from aidisdk.experiment import Table
except ImportError:
    Table = None

from hat.metrics.bev.utils.bev_discrete_obj import get_topk_min_value
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.utils.aidi import EvalResult

__all__ = ["ANCBEVParkingrodMetric"]

logger = logging.getLogger(__name__)

ignore_dict = {"Y": 0, "N": 1}
rod_type_dict = {
    "limited_strip": 0,
    "limited_rod": 1,
    "other": 2,
}


@OBJECT_REGISTRY.register
class ANCBEVParkingrodMetric(EvalMetric):
    """Compute bev parkingrod point detection metric.

    .. code-block:: none

        *************************************************
        *                      6-9m                     *
        *     *************************************     *
        *     *                2-6m               *     *
        *     *     *************************     *     *
        *     *     *          0-2m         *     *     *
        *     *     *     *************     *     *     *
        *     *     *     *     *     *     *     *     *
        *     *     *     *     *     *     *     *     *
        *     *     *     *     *     *0-2m *2-6m *6-9m *
        *     *     *     ****center***     *     *     *
        *     *     *     *     *     *     *     *     *
        *     *     *     *************     *     *     *
        *     *     *                       *     *     *
        *     *     *************************     *     *
        *     *                                   *     *
        *     *************************************     *
        *                                               *
        *************************************************

    Args:
        rod_distance_threshold: distance threshold to judge whether
            endpoint prediction is right for long parkingrod.
        strip_distance_threshold: distance threshold to judge whether
            endpoint prediction is right for short parking strip.
        angle_threshold: angle threshold to judge whether
            endpoint prediction is right.
        resolution_m_per_grid: convert bev pixel to vcs meter.
        eval_vcs_range: max vcs range (right, top, left, bottom)
            for validation.
        dep_intervals: (tuple of int): Depth range to validation.
        vcs_range: perception vcs range.
        eval_result_path: eval result json path.
        ego_vcs_range: supposed car vcs range.
        ignore_distance_threshold: distance threshold to judge
            detection matching with ignore label.
        threshold_2sigma: topk threshold to compute 2sigma value.
        eps: a value to fill none-instance for concat.
        name: Name of this metric instance for display.
        result_prefix (str): Prefix of aidi eval result.
        eval_parking_slot_rod (bool): Flag to evaluate the parkingrod
        of the current parking slot.
        ego_vcs_range_rod: vcs range of the ego vehicle, used to
        determine the intersection of the ego vehicle and the parking spot.

    """

    def __init__(
        self,
        rod_distance_threshold: float,
        strip_dictance_threshold: float,
        angle_threshold: float,
        resolution_m_per_grid: float,
        eval_vcs_range: tuple,
        vcs_range: tuple,
        ego_vcs_range: tuple,
        name: str = "bevparkingrod",
        eval_result_path: str = None,
        dep_intervals: tuple = None,
        ignore_distance_threshold: float = 4.0,
        threshold_2sigma: float = 0.05,
        eps: float = -1e-9,
        result_prefix: str = "",
        eval_parking_slot_rod: bool = True,
        ego_vcs_range_rod: tuple = (-0.5, -0.5, 3, 0.5),
    ):
        self.name = name
        self.rod_distance_threshold = rod_distance_threshold
        self.strip_dictance_threshold = strip_dictance_threshold
        self.angle_threshold = angle_threshold
        self.resolution_m_per_grid = resolution_m_per_grid
        self.eval_vcs_range = eval_vcs_range
        self.ego_vcs_range = ego_vcs_range
        self.ego_vcs_range_rod = ego_vcs_range_rod
        self.vcs_range = vcs_range
        self.eval_result_path = eval_result_path
        self.ignore_distance_threshold = ignore_distance_threshold
        self.dep_intervals = dep_intervals
        self.threshold_2sigma = threshold_2sigma
        self._device = None
        self.full_dep_intervals = None
        self.eval_parking_slot_rod = eval_parking_slot_rod

        self.eps = eps
        self.result_prefix = result_prefix

        if dep_intervals is not None:
            self.dep_intervals = [0] + list(dep_intervals)
            assert (
                max(abs(self.eval_vcs_range[2]), abs(self.eval_vcs_range[0]))
                > self.dep_intervals[-1]
            )
            self.full_dep_intervals = [
                f"({start}, {end})"
                for start, end in zip(
                    self.dep_intervals[:-1], self.dep_intervals[1:]
                )
            ]
        super(ANCBEVParkingrodMetric, self).__init__(name)

    def _init_states(self):
        for name in self.get_names_sum():
            self.add_state(
                name,
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
        for name in self.get_names_cat():
            self.add_state(
                name,
                default=[],
                dist_reduce_fx="cat",
            )

    def get_names_sum(self):
        names = ["all_right", "all_label", "all_fp"]
        if self.full_dep_intervals is not None:
            for depth in self.full_dep_intervals:
                names += [depth + "_right"]
                names += [depth + "_label"]
                names += [depth + "_fp"]
        if (
            self.eval_parking_slot_rod
        ):  # enable eval parkingrod of the parking_slot flag
            names += ["parking_slot_rod" + "_right"]
            names += ["parking_slot_rod" + "_label"]
            names += ["parking_slot_rod" + "_fp"]
        return names

    def get_names_cat(self):
        names = [
            "all_horizon_diff",
            "all_vertical_diff",
            "all_angle_diff",
            "all_system_diff",
        ]
        if self.full_dep_intervals is not None:
            for depth in self.full_dep_intervals:
                names += [depth + "_horizon_diff"]
                names += [depth + "_vertical_diff"]
                names += [depth + "_angle_diff"]
                names += [depth + "_system_diff"]
        if self.eval_parking_slot_rod:
            names += ["parking_slot_rod" + "_horizon_diff"]
            names += ["parking_slot_rod" + "_vertical_diff"]
            names += ["parking_slot_rod" + "_angle_diff"]
            names += ["parking_slot_rod" + "_system_diff"]
        return names

    def reset(self) -> None:
        super().reset()

    def get_vcs_range(self, depth):
        """Compute eval vcs range appending on depth.

        depth: eval depth value from self vcs range
        """
        eval_vcs_range = (
            self.ego_vcs_range[0] - depth,
            self.ego_vcs_range[1] - depth,
            self.ego_vcs_range[2] + depth,
            self.ego_vcs_range[3] + depth,
        )
        return eval_vcs_range

    def is_parking_slot_rod(self, rod_label_point, slot_label_point):
        ego_bev_range = (
            (self.vcs_range[3] - self.ego_vcs_range_rod[3])  # left
            / self.resolution_m_per_grid,
            (self.vcs_range[2] - self.ego_vcs_range_rod[2])  # top
            / self.resolution_m_per_grid,
            (self.vcs_range[3] - self.ego_vcs_range_rod[1])  # right
            / self.resolution_m_per_grid,
            (self.vcs_range[2] - self.ego_vcs_range_rod[0])  # bottom
            / self.resolution_m_per_grid,
        )
        ego_bev_points = [  # TODO
            [ego_bev_range[0], ego_bev_range[1]],  # top left
            [ego_bev_range[2], ego_bev_range[1]],  # top right
            [ego_bev_range[2], ego_bev_range[3]],  # bottom right
            [ego_bev_range[0], ego_bev_range[3]],  # bottom left
        ]
        slot_bev_points = [
            slot_label_point[0],
            slot_label_point[1],
            slot_label_point[2],
            slot_label_point[3],
        ]

        assert len(ego_bev_points) == len(
            slot_bev_points
        ), "The two poly dimensions of the contrast should be equal"
        ego_poly = Polygon(ego_bev_points)
        slot_poly = Polygon(slot_bev_points)
        if ego_poly.intersects(slot_poly):  # is_intersect_slot == True
            assert len(rod_label_point) == 2, "Line should have 2 points"

            line_seg = LineString(rod_label_point)

            return slot_poly.intersects(line_seg)
        else:
            return False

    def is_in_vcs_range(self, eval_vcs_range, ptx, pty):
        """Determine whether the point is in eval_vcs_range.

        eval_vcs_range: eval vcs range. (bottom, right, top, left)
        ptx: x of point in bev Coordinate.
        pty: y of point in bev Coordinate.
        """

        eval_bev_range = (
            (self.vcs_range[3] - eval_vcs_range[3])
            / self.resolution_m_per_grid,
            (self.vcs_range[2] - eval_vcs_range[2])
            / self.resolution_m_per_grid,
            (self.vcs_range[3] - eval_vcs_range[1])
            / self.resolution_m_per_grid,
            (self.vcs_range[2] - eval_vcs_range[0])
            / self.resolution_m_per_grid,
        )
        in_range_xmin = ptx > eval_bev_range[0]
        in_range_xmax = ptx < eval_bev_range[2]
        in_range_ymin = pty > eval_bev_range[1]
        in_range_ymax = pty < eval_bev_range[3]
        in_range_x = np.logical_and(in_range_xmin, in_range_xmax)
        in_range_y = np.logical_and(in_range_ymin, in_range_ymax)
        in_range = np.logical_and(in_range_x, in_range_y)
        return in_range  # True or False

    def update(self, annos_bev_parkingrod_obj, preds_label):
        """
        Update fuse msg.

        rod_labels_bs: batch * n * 17
            ignore:
            [[x0, y0, x1, y1, slot_x0, slot_y0, slot_x1, slot_y1, slot_x2, slot_y2, slot_x3, slot_y3, 0.0, 0.0, len_rod, rod_type, ignore_type], [], ...]  # noqa
            normal:
            [[x_center, y_center, rod_type, x0, y0, x1, y1, slot_x0, slot_y0, slot_x1, slot_y1, slot_x2, slot_y2, slot_x3, slot_y3, rod_type, ignore_type], [], ...]  # noqa
        rod_dets_bs: batch * n * 6
            [[x0, y0, x1, y1, score], [], ...]
        """
        rod_labels_bs = np.array(annos_bev_parkingrod_obj["parking_rod"].cpu())
        rod_dets_bs = np.array(preds_label)
        assert len(rod_labels_bs.shape) == 3
        batch_size = rod_labels_bs.shape[0]

        for bs in range(batch_size):
            rod_labels = rod_labels_bs[bs]
            rod_dets = np.array(rod_dets_bs[bs])
            # delete empty data
            idx_valid = np.sum(rod_labels, axis=1) > np.finfo(np.float32).eps
            rod_labels = rod_labels[idx_valid]

            # filter ignore labels
            idx_rod_ignore = rod_labels[:, -1] == ignore_dict["Y"]
            rod_ignores = rod_labels[idx_rod_ignore]
            idx_not_ignore = np.logical_not(idx_rod_ignore)
            # delete labels out of valid range
            idx_in_range = self.is_in_vcs_range(
                self.eval_vcs_range,
                rod_labels[:, 0],
                rod_labels[:, 1],
            )
            idx_in_ego_range = self.is_in_vcs_range(
                self.ego_vcs_range,
                rod_labels[:, 0],
                rod_labels[:, 1],
            )
            idx_not_in_ego_range = np.logical_not(idx_in_ego_range)
            idx_rod_normal = np.logical_and(idx_not_in_ego_range, idx_in_range)
            idx_rod_normal = np.logical_and(idx_not_ignore, idx_rod_normal)
            normal_rod_label = rod_labels[idx_rod_normal]
            if self._device is None and torch.cuda.is_available():
                self._device = torch.device("cuda")

            # collect label msg
            # classify labels to different range
            for rod_label in normal_rod_label:
                name = "all_label"
                value_ddp = getattr(self, name, None) + torch.tensor(1)
                setattr(self, name, value_ddp)
                ptx = rod_label[0]
                pty = rod_label[1]
                if self.full_dep_intervals is not None:
                    for idx, depth in enumerate(self.full_dep_intervals):
                        start_depth = self.dep_intervals[idx]
                        if start_depth == 0:
                            start_vcs_range = (0, 0, 0, 0)
                        else:
                            start_vcs_range = self.get_vcs_range(start_depth)
                        end_depth = self.dep_intervals[idx + 1]
                        end_vcs_range = self.get_vcs_range(end_depth)
                        is_in_start_range = self.is_in_vcs_range(
                            start_vcs_range, ptx, pty
                        )
                        is_in_end_range = self.is_in_vcs_range(
                            end_vcs_range, ptx, pty
                        )
                        if not is_in_start_range and is_in_end_range:
                            name = depth + "_label"
                            value_ddp = getattr(
                                self, name, None
                            ) + torch.tensor(1)
                            setattr(self, name, value_ddp)
                            break
                if self.eval_parking_slot_rod:
                    rod_label_point = rod_label[3:7]
                    rod_label_point = rod_label_point.reshape(2, 2)
                    slot_label_point = rod_label[7:15]
                    slot_label_point = slot_label_point.reshape(4, 2)

                    if self.is_parking_slot_rod(
                        rod_label_point, slot_label_point
                    ):
                        name = "parking_slot_rod" + "_label"
                        value_ddp = getattr(self, name, None) + torch.tensor(1)
                        setattr(self, name, value_ddp)
            # add a eps-tensor to keep dim consistent for concat
            names = [
                "all_horizon_diff",
                "all_vertical_diff",
                "all_angle_diff",
                "all_system_diff",
            ]
            for name in names:
                value_ddp = getattr(self, name, None) + [
                    torch.tensor(self.eps, device=self._device)
                ]
                setattr(self, name, value_ddp)
            if self.full_dep_intervals is not None:
                for depth in self.full_dep_intervals:
                    names = [
                        depth + "_angle_diff",
                        depth + "_horizon_diff",
                        depth + "_vertical_diff",
                        depth + "_system_diff",
                    ]
                    for name in names:
                        value_ddp = getattr(self, name, None) + [
                            torch.tensor(self.eps, device=self._device)
                        ]
                        setattr(self, name, value_ddp)

            if len(rod_dets.shape) != 2:
                continue
            # delete dets out of valid_range
            rod_det_center_x = (rod_dets[:, 0] + rod_dets[:, 2]) / 2
            rod_det_center_y = (rod_dets[:, 1] + rod_dets[:, 3]) / 2
            idx_in_vcs_range = self.is_in_vcs_range(
                self.eval_vcs_range,
                rod_det_center_x,
                rod_det_center_y,
            )
            idx_in_ego_range = self.is_in_vcs_range(
                self.ego_vcs_range,
                rod_det_center_x,
                rod_det_center_y,
            )
            idx_not_in_ego_range = np.logical_not(idx_in_ego_range)
            idx_in_range = np.logical_and(
                idx_not_in_ego_range, idx_in_vcs_range
            )
            rod_dets = rod_dets[idx_in_range]

            # filter detection matching with ignore label
            rod_dets_valid_status = np.zeros(len(rod_dets), dtype=np.bool)
            rod_dets_valid_status.fill(True)
            for rod_ignore in rod_ignores:
                rod_ignore_point_len = int(rod_ignore[-3])
                rod_ignore_point = rod_ignore[0:rod_ignore_point_len]
                rod_ignore_point = rod_ignore_point.reshape(
                    rod_ignore_point_len // 2, 2
                )
                rod_ignore_label_center = np.sum(rod_ignore_point, axis=0) / 2
                for idx, det in enumerate(rod_dets):
                    rod_det_point = det[:4].reshape(2, 2)
                    rod_det_point_center = np.sum(rod_det_point, axis=0) / 2
                    center_distance = np.linalg.norm(
                        rod_det_point_center - rod_ignore_label_center
                    )
                    if center_distance < self.ignore_distance_threshold:
                        rod_dets_valid_status[idx] = False
            rod_dets = rod_dets[rod_dets_valid_status]
            gt_recall_list = []
            pred_fp_list = []
            for rod_det in rod_dets:
                bool_FP = True
                rod_det_point = rod_det[:4].reshape(2, 2)
                vec_det = [
                    (rod_det_point[1][0] - rod_det_point[0][0]),
                    (rod_det_point[1][1] - rod_det_point[0][1]),
                ]
                center_det = np.sum(rod_det_point, axis=0) / 2
                center_det_reshape = center_det.reshape(1, 2)

                rod_label_points = normal_rod_label[:, 3:7].reshape(-1, 2, 2)
                center_labels = np.sum(rod_label_points, axis=1) / 2
                distance_diffs = np.linalg.norm(
                    center_det_reshape - center_labels, axis=1
                )
                sorted_labels = np.argsort(distance_diffs)
                for idx_label in sorted_labels:
                    # rod_det_point: [[x0, y0], [x1, y1]]
                    # rod_label_point: [[x0, y0], [x1,y1]]
                    rod_label_point = normal_rod_label[idx_label, 3:7]
                    rod_label_point = rod_label_point.reshape(2, 2)
                    slot01_label_point = normal_rod_label[idx_label, 7:11]
                    slot01_label_point = slot01_label_point.reshape(2, 2)
                    slot_label_point = normal_rod_label[idx_label, 7:15]
                    slot_label_point = slot_label_point.reshape(4, 2)
                    vec_label = [
                        (rod_label_point[1][0] - rod_label_point[0][0]),
                        (rod_label_point[1][1] - rod_label_point[0][1]),
                    ]
                    angle_diff = self.compute_angle(vec_det, vec_label)
                    angle_diff = angle_diff * 180 / pi
                    if angle_diff > 90:
                        angle_diff = 180 - angle_diff

                    distance_diff = distance_diffs[idx_label]
                    if (
                        distance_diff < self.rod_distance_threshold
                        and angle_diff < self.angle_threshold
                    ):
                        if idx_label in gt_recall_list:  # deduplication
                            continue
                        gt_recall_list.append(idx_label)
                        bool_FP = False
                        name = "all_right"
                        value_ddp = getattr(self, name, None) + torch.tensor(1)
                        setattr(self, name, value_ddp)
                        name = "all_angle_diff"
                        angle_ddp = getattr(self, name, None) + [
                            torch.as_tensor(angle_diff, device=self._device)
                        ]
                        setattr(self, name, angle_ddp)
                        self.update_position_diff(
                            location_det=rod_det_point,
                            location_label=rod_label_point,
                            slot01_label=slot01_label_point,
                            name="all",
                        )
                        if self.full_dep_intervals is not None:
                            for idx, depth in enumerate(
                                self.full_dep_intervals
                            ):
                                start_depth = self.dep_intervals[idx]
                                if start_depth == 0:
                                    start_vcs_range = (0, 0, 0, 0)
                                else:
                                    start_vcs_range = self.get_vcs_range(
                                        start_depth
                                    )
                                end_depth = self.dep_intervals[idx + 1]
                                end_vcs_range = self.get_vcs_range(end_depth)
                                # gt result in the start and end range or not
                                is_in_start_range = self.is_in_vcs_range(
                                    start_vcs_range,
                                    normal_rod_label[idx_label][0],
                                    normal_rod_label[idx_label][1],
                                )
                                is_in_end_range = self.is_in_vcs_range(
                                    end_vcs_range,
                                    normal_rod_label[idx_label][0],
                                    normal_rod_label[idx_label][1],
                                )
                                if not is_in_start_range and is_in_end_range:
                                    name = depth + "_right"
                                    value_ddp = getattr(
                                        self, name, None
                                    ) + torch.tensor(1)
                                    setattr(self, name, value_ddp)
                                    name = depth + "_angle_diff"
                                    angle_ddp = getattr(self, name, None) + [
                                        torch.as_tensor(
                                            angle_diff, device=self._device
                                        )
                                    ]
                                    setattr(self, name, angle_ddp)
                                    self.update_position_diff(
                                        location_det=rod_det_point,
                                        location_label=rod_label_point,
                                        slot01_label=slot01_label_point,
                                        name=depth,
                                    )
                                    break
                        if self.eval_parking_slot_rod:
                            if self.is_parking_slot_rod(
                                rod_det_point, slot_label_point
                            ):
                                parking_slot_rod_name = "parking_slot_rod"
                                name = parking_slot_rod_name + "_right"
                                value_ddp = getattr(
                                    self, name, None
                                ) + torch.tensor(1)
                                setattr(self, name, value_ddp)
                                name = parking_slot_rod_name + "_angle_diff"
                                angle_ddp = getattr(self, name, None) + [
                                    torch.as_tensor(
                                        angle_diff, device=self._device
                                    )
                                ]
                                setattr(self, name, angle_ddp)
                                self.update_position_diff(
                                    location_det=rod_det_point,
                                    location_label=rod_label_point,
                                    slot01_label=slot01_label_point,
                                    name=parking_slot_rod_name,
                                )
                        break
                if bool_FP:
                    name = "all_fp"
                    value_ddp = getattr(self, name, None) + torch.tensor(1)
                    setattr(self, name, value_ddp)
                    pred_fp_list.append(rod_det)

            # collect rod_dets msg
            # classify fp to different range
            for rod_det in pred_fp_list:
                name = "all_fp"
                ptx = (rod_det[0] + rod_det[2]) / 2
                pty = (rod_det[1] + rod_det[3]) / 2
                if self.full_dep_intervals is not None:
                    for idx, depth in enumerate(self.full_dep_intervals):
                        start_depth = self.dep_intervals[idx]
                        if start_depth == 0:
                            start_vcs_range = (0, 0, 0, 0)
                        else:
                            start_vcs_range = self.get_vcs_range(start_depth)
                        end_depth = self.dep_intervals[idx + 1]
                        end_vcs_range = self.get_vcs_range(end_depth)
                        is_in_start_range = self.is_in_vcs_range(
                            start_vcs_range, ptx, pty
                        )
                        is_in_end_range = self.is_in_vcs_range(
                            end_vcs_range, ptx, pty
                        )
                        if not is_in_start_range and is_in_end_range:
                            name = depth + "_fp"
                            value_ddp = getattr(
                                self, name, None
                            ) + torch.tensor(1)
                            setattr(self, name, value_ddp)
                            break
                if self.eval_parking_slot_rod:
                    for rod_label in normal_rod_label:
                        slot_label_point = rod_label[7:15]
                        slot_label_point = slot_label_point.reshape(4, 2)

                        if self.is_parking_slot_rod(
                            rod_det[:4].reshape(2, 2), slot_label_point
                        ):  # rod_det: rod input predict
                            name = "parking_slot_rod" + "_fp"
                            value_ddp = getattr(
                                self, name, None
                            ) + torch.tensor(1)
                            setattr(self, name, value_ddp)
                            break  # avoid calculate pred twice

    def compute_angle(self, vec_det, vec_label):
        """Get parkingrod angle diff of radius.

        Args:
            vec_det: np.array, [x0, y0]
            vec_label: np.array, [x1, y1]
        """
        vec_det = np.array(vec_det)
        vec_label = np.array(vec_label)
        angle_diff = F.cosine_similarity(
            torch.tensor(vec_det), torch.tensor(vec_label), dim=0
        )
        if abs(1 - angle_diff) < 1e-5:
            angle_diff = torch.tensor(1 - 1e-5)
        elif abs(-1 - angle_diff) < 1e-5:
            angle_diff = torch.tensor(-1 + 1e-5)
        angle_diff = torch.acos(angle_diff)
        angle_diff = torch.as_tensor(angle_diff, dtype=torch.float32)
        return angle_diff

    def update_position_diff(
        self,
        location_det,
        location_label,
        slot01_label,
        name,
    ):
        """Update horizon/vertical center position diff.

        Args:
            location_det: np.array, [[x0, y0], [x1, y1]]
            location_label: np.array, [[x0, y0], [x1, y1]]
            slot01_label: np.array, [[slot_x0, slot_y0], [slot_x1, slot_y1]]
            rod_name: parking_rod type
        """
        center_det = np.sum(location_det, axis=0) / 2
        center_label = np.sum(location_label, axis=0) / 2
        center_slot01 = np.sum(slot01_label, axis=0) / 2
        position_diff = np.linalg.norm(center_det - center_label)
        vec_label2det = [
            center_det[0] - center_label[0],
            center_det[1] - center_label[1],
        ]

        vec_label = [
            location_label[1][0] - location_label[0][0],
            location_label[1][1] - location_label[0][1],
        ]
        # get two vertical lines of parkingrod in two directions
        vec_label_vertical1 = [-vec_label[1], vec_label[0]]
        vec_label_vertical1 = vec_label_vertical1 / (
            np.linalg.norm(vec_label_vertical1) + 1e-9
        )
        point_label_vertical1 = center_label + vec_label_vertical1
        vectical1_to_slot = np.linalg.norm(
            point_label_vertical1 - center_slot01
        )
        vec_label_vertical2 = [vec_label[1], -vec_label[0]]
        vec_label_vertical2 = vec_label_vertical2 / (
            np.linalg.norm(vec_label_vertical2) + 1e-9
        )
        point_label_vertical2 = center_label + vec_label_vertical2
        vertical2_to_slot = np.linalg.norm(
            point_label_vertical2 - center_slot01
        )
        # select the vertical line towards the entrance of slot
        if vectical1_to_slot < vertical2_to_slot:
            vec_label_vertical = vec_label_vertical1
        else:
            vec_label_vertical = vec_label_vertical2

        angle = self.compute_angle(vec_label2det, vec_label_vertical)
        system_diff = position_diff * torch.cos(angle)
        vertical_diff = abs(system_diff)
        horizon_diff = abs(position_diff * torch.sin(angle))
        name_system = name + "_system_diff"
        setattr(
            self,
            name_system,
            getattr(self, name_system)
            + [torch.as_tensor(system_diff, device=self._device)],
        )
        name_horizon = name + "_horizon_diff"
        setattr(
            self,
            name_horizon,
            getattr(self, name_horizon)
            + [torch.as_tensor(horizon_diff, device=self._device)],
        )
        name_vertical = name + "_vertical_diff"
        setattr(
            self,
            name_vertical,
            getattr(self, name_vertical)
            + [torch.as_tensor(vertical_diff, device=self._device)],
        )

    def get(self):
        format_metrics = self.compute()
        total_names = []
        total_values = []

        tables = []
        columns = ["name", "value"]
        data = []
        output_stats = {"all": {}}
        all_format_metric = format_metrics["all"]
        summary_str = "\n************** %s Metric **************\n" % (
            self.name
        )
        summary_str += "\n***eval vcs range:{}***\n".format(
            self.eval_vcs_range
        )
        for k, v in all_format_metric.items():
            total_names += [k]
            total_values += [v]
            line_format = "{:<35}\t{:>10.3f}\n"
            summary_str += line_format.format(k, v)
            output_stats["all"][k] = round(float(v), 3)
            data.append({"name": k, "value": round(float(v), 3)})

        tables.append(
            Table(
                name="{} eval vcs range: {}".format(
                    self.result_prefix, self.eval_vcs_range
                ),
                columns=columns,
                data=data,
            )
        )
        summary_str += "\n"

        if self.full_dep_intervals is not None:
            for depth in self.full_dep_intervals:
                columns = ["name", "value"]
                data = []
                summary_str += "\n*******eval depth: {}*******\n".format(depth)
                output_stats.update({depth: {}})
                format_metric = format_metrics[depth]
                for k, v in format_metric.items():
                    total_names += [k]
                    total_values += [v]
                    line_format = "{:<35}\t{:>10.3f}\n"
                    summary_str += line_format.format(k, v)
                    output_stats[depth][k] = round(float(v), 3)
                    data.append({"name": k, "value": round(float(v), 3)})
                tables.append(
                    Table(
                        name="{} eval depth: {}".format(
                            self.result_prefix, depth
                        ),
                        columns=columns,
                        data=data,
                    )
                )
                summary_str += "\n"
        if self.eval_parking_slot_rod:
            columns = ["name", "value"]
            data = []
            output_stats.update({"parking_slot_rod": {}})
            parking_format_metric = format_metrics["parking_slot_rod"]  # TODO
            summary_str += (
                "\n************** %s Eval parking slot rod **************\n"
            )

            for k, v in parking_format_metric.items():
                total_names += [k]
                total_values += [v]
                line_format = "{:<35}\t{:>10.3f}\n"
                summary_str += line_format.format(k, v)
                output_stats["all"][k] = round(float(v), 3)
                data.append({"name": k, "value": round(float(v), 3)})

            tables.append(
                Table(
                    name="{} eval parking slot rod".format(self.result_prefix),
                    columns=columns,
                    data=data,
                )
            )
            summary_str += "\n"

        logger.info(summary_str)
        if self.eval_result_path is not None:
            save_dir, _ = os.path.split(self.eval_result_path)
            os.makedirs(save_dir, exist_ok=True)
            with open(self.eval_result_path, "w", encoding="utf-8") as f:
                json.dump(output_stats, f, ensure_ascii=False, indent=1)

        eval_result = EvalResult(tables=tables)

        return self.name, eval_result

    def compute(self):
        format_metrics = {}
        all_format_metric = {}
        format_metrics["all"] = all_format_metric
        num_all_right = int(getattr(self, "all_right", None).cpu().numpy())
        num_all_label = int(getattr(self, "all_label", None).cpu().numpy())
        num_all_fp = int(getattr(self, "all_fp", None).cpu().numpy())
        precision_all = (
            0
            if num_all_right == 0
            else num_all_right / (num_all_right + num_all_fp)
        )
        recall_all = 0 if num_all_label == 0 else num_all_right / num_all_label
        all_format_metric["Precision(%)"] = precision_all * 100
        all_format_metric["Recall(%)"] = recall_all * 100
        all_format_metric["TP_cnt"] = num_all_right
        all_format_metric["Label_cnt"] = num_all_label
        all_format_metric["Pred_cnt"] = num_all_right + num_all_fp

        all_horizon_diff = getattr(self, "all_horizon_diff", None)
        if isinstance(all_horizon_diff, List):
            all_horizon_diff = [diff.tolist() for diff in all_horizon_diff]
            all_horizon_diff = torch.tensor(all_horizon_diff)
        all_horizon_diff = all_horizon_diff.cpu()
        all_horizon_diff_mask = all_horizon_diff >= 0
        all_horizon_diff = all_horizon_diff[all_horizon_diff_mask]
        all_format_metric["Position_horizon_diff_mean(m)"] = (
            torch.mean(all_horizon_diff) * self.resolution_m_per_grid
        )
        all_format_metric["Position_horizon_diff_std(m)"] = (
            torch.std(all_horizon_diff) * self.resolution_m_per_grid
        )
        all_format_metric["Position_horizon_diff_2sigma(m)"] = (
            get_topk_min_value(all_horizon_diff, self.threshold_2sigma)
            * self.resolution_m_per_grid
        )

        all_vertical_diff = getattr(self, "all_vertical_diff", None)
        if isinstance(all_vertical_diff, List):
            all_vertical_diff = [diff.tolist() for diff in all_vertical_diff]
            all_vertical_diff = torch.tensor(all_vertical_diff)
        all_vertical_diff = all_vertical_diff.cpu()
        all_vertical_diff_mask = all_vertical_diff >= 0
        all_vertical_diff = all_vertical_diff[all_vertical_diff_mask]
        all_format_metric["Position_vertical_diff_mean(m)"] = (
            torch.mean(all_vertical_diff) * self.resolution_m_per_grid
        )
        all_format_metric["Position_vertical_diff_std(m)"] = (
            torch.std(all_vertical_diff) * self.resolution_m_per_grid
        )
        all_format_metric["Position_vertical_diff_2sigma(m)"] = (
            get_topk_min_value(all_vertical_diff, self.threshold_2sigma)
            * self.resolution_m_per_grid
        )

        all_angle_diff = getattr(self, "all_angle_diff", None)
        if isinstance(all_angle_diff, List):
            all_angle_diff = [diff.tolist() for diff in all_angle_diff]
            all_angle_diff = torch.tensor(all_angle_diff)
        all_angle_diff = all_angle_diff.cpu()
        all_angle_diff_mask = all_angle_diff >= 0
        all_angle_diff = all_angle_diff[all_angle_diff_mask]
        all_format_metric["angle_diff_mean"] = torch.mean(all_angle_diff)
        all_format_metric["angle_diff_std"] = torch.std(all_angle_diff)
        all_format_metric["angle_diff_2sigma"] = get_topk_min_value(
            all_angle_diff, self.threshold_2sigma
        )

        all_system_diff = getattr(self, "all_system_diff", None)
        if isinstance(all_system_diff, List):
            all_system_diff = [diff.tolist() for diff in all_system_diff]
            all_system_diff = torch.tensor(all_system_diff)
        all_system_diff = all_system_diff.cpu()
        all_system_diff_mask = all_system_diff != self.eps
        all_system_diff = all_system_diff[all_system_diff_mask]
        all_format_metric["Position_system_diff_mean(m)"] = (
            torch.mean(all_system_diff) * self.resolution_m_per_grid
        )
        all_system_diff_mask_add = all_system_diff >= 0
        all_system_diff_add = all_system_diff[all_system_diff_mask_add]
        all_format_metric["Position_system_diff_std_正向(m)"] = (
            torch.std(all_system_diff_add) * self.resolution_m_per_grid
        )
        all_format_metric["Position_system_diff_2sigma_正向(m)"] = (
            get_topk_min_value(all_system_diff_add, self.threshold_2sigma)
            * self.resolution_m_per_grid
        )
        all_system_diff_mask_sub = all_system_diff < 0
        all_system_diff_sub = all_system_diff[all_system_diff_mask_sub]
        all_system_diff_sub = -all_system_diff_sub
        all_format_metric["Position_system_diff_std_负向(m)"] = (
            torch.std(all_system_diff_sub) * self.resolution_m_per_grid
        )
        all_format_metric["Position_system_diff_2sigma_负向(m)"] = (
            get_topk_min_value(all_system_diff_sub, self.threshold_2sigma)
            * self.resolution_m_per_grid
        )

        if self.full_dep_intervals is not None:
            if self.eval_parking_slot_rod:
                traverse_list = self.full_dep_intervals + ["parking_slot_rod"]
            else:
                traverse_list = self.full_dep_intervals
            for depth in traverse_list:
                format_metric = {}
                format_metrics[depth] = format_metric
                num_right = int(
                    getattr(self, depth + "_right", None).cpu().numpy()
                )
                num_label = int(
                    getattr(self, depth + "_label", None).cpu().numpy()
                )
                num_fp = int(getattr(self, depth + "_fp", None).cpu().numpy())
                precision = (
                    0 if num_right == 0 else num_right / (num_right + num_fp)
                )
                recall = 0 if num_label == 0 else num_right / num_label
                format_metric["Precision(%)"] = precision * 100
                format_metric["Recall(%)"] = recall * 100
                format_metric["TP_cnt"] = num_right
                format_metric["Lable_cnt"] = num_label
                format_metric["Pred_cnt"] = num_right + num_fp

                horizon_diff = getattr(self, depth + "_horizon_diff", None)
                if isinstance(horizon_diff, List):
                    horizon_diff = [diff.tolist() for diff in horizon_diff]
                    horizon_diff = torch.tensor(horizon_diff)
                horizon_diff = horizon_diff.cpu()
                horizon_diff_mask = horizon_diff >= 0
                horizon_diff = horizon_diff[horizon_diff_mask]
                format_metric["Position_horizon_diff_mean(m)"] = (
                    torch.mean(horizon_diff) * self.resolution_m_per_grid
                )
                format_metric["Position_horizon_diff_std(m)"] = (
                    torch.std(horizon_diff) * self.resolution_m_per_grid
                )
                format_metric["Position_horizon_diff_2sigma(m)"] = (
                    get_topk_min_value(horizon_diff, self.threshold_2sigma)
                    * self.resolution_m_per_grid
                )

                vertical_diff = getattr(self, depth + "_vertical_diff", None)
                if isinstance(vertical_diff, List):
                    vertical_diff = [diff.tolist() for diff in vertical_diff]
                    vertical_diff = torch.tensor(vertical_diff)
                vertical_diff = vertical_diff.cpu()
                vertical_diff_mask = vertical_diff >= 0
                vertical_diff = vertical_diff[vertical_diff_mask]
                format_metric["Position_vertical_diff_mean(m)"] = (
                    torch.mean(vertical_diff) * self.resolution_m_per_grid
                )
                format_metric["Position_vertical_diff_std(m)"] = (
                    torch.std(vertical_diff) * self.resolution_m_per_grid
                )
                format_metric["Position_vertical_diff_2sigma(m)"] = (
                    get_topk_min_value(vertical_diff, self.threshold_2sigma)
                    * self.resolution_m_per_grid
                )

                angle_diff = getattr(self, depth + "_angle_diff", None)
                if isinstance(angle_diff, List):
                    angle_diff = [diff.tolist() for diff in angle_diff]
                    angle_diff = torch.tensor(angle_diff)
                angle_diff = angle_diff.cpu()
                angle_diff_mask = angle_diff >= 0
                angle_diff = angle_diff[angle_diff_mask]
                format_metric["angle_diff_mean"] = torch.mean(angle_diff)
                format_metric["angle_diff_std"] = torch.std(angle_diff)
                format_metric["angle_diff_2sigma"] = get_topk_min_value(
                    angle_diff, self.threshold_2sigma
                )

                system_diff = getattr(self, depth + "_system_diff", None)
                if isinstance(system_diff, List):
                    system_diff = [diff.tolist() for diff in system_diff]
                    system_diff = torch.tensor(system_diff)
                system_diff = system_diff.cpu()
                system_diff_mask = system_diff != self.eps
                system_diff = system_diff[system_diff_mask]
                format_metric["Position_system_diff_mean(m)"] = (
                    torch.mean(system_diff) * self.resolution_m_per_grid
                )
                system_diff_mask_add = system_diff >= 0
                system_diff_add = system_diff[system_diff_mask_add]
                format_metric["Position_system_diff_std_正向(m)"] = (
                    torch.std(system_diff_add) * self.resolution_m_per_grid
                )
                format_metric["Position_system_diff_2sigma_正向(m)"] = (
                    get_topk_min_value(system_diff_add, self.threshold_2sigma)
                    * self.resolution_m_per_grid
                )
                system_diff_mask_sub = system_diff < 0
                system_diff_sub = system_diff[system_diff_mask_sub]
                system_diff_sub = -system_diff_sub
                format_metric["Position_system_diff_std_负向(m)"] = (
                    torch.std(system_diff_sub) * self.resolution_m_per_grid
                )
                format_metric["Position_system_diff_2sigma_负向(m)"] = (
                    get_topk_min_value(system_diff_sub, self.threshold_2sigma)
                    * self.resolution_m_per_grid
                )

        return format_metrics
