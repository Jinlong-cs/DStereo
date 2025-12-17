# Copyright (c) Horizon Robotics. All rights reserved.
import glob
import json
import logging
import os
from math import pi
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from aidisdk.experiment import Image, Table
except ImportError:
    Image = None
    Table = None

from shapely.geometry import Polygon

from hat.metrics.bev.metric_save import reverse_table
from hat.metrics.bev.utils.bev_discrete_obj import (
    cal_iou_by_polygon,
    get_topk_min_value,
)
from hat.metrics.metric import EvalMetric
from hat.metrics.metric_3dv_utils import NpEncoder
from hat.registry import OBJECT_REGISTRY
from hat.utils.aidi import EvalResult

__all__ = ["ANCBEVPSDMetric"]

logger = logging.getLogger(__name__)

slot_type_dict = {
    "Not_normal_slot": -1,
    "Vertical": 0,
    "Stereo": 0,
    "Parallel": 1,
    "Oblique": 2,
    "Handicapped": 3,
    "Not_slot": 4,
}
ignore_dict = {"Y": 0, "N": 1}
occupancy_dict = {"Y": 0, "N": 1}


def gen_fuse_result_map(det_results, img):
    # [[junction_locations*8, junction_types*4, junction_visibilities*4,
    # slot_occupancies*1, slot_types*1, junction_orientations*8,
    # slot_orientations*2, slot_score, junction_score*4],...]
    slot_occupancy_color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
    slot_type_mark_dict = {0: "V", 1: "P", 2: "S"}
    img1, img2 = img.copy(), img.copy()
    det_results = np.array(det_results)
    if len(det_results) == 0:
        return img
    # point_directions = det_results[:, 18:26].reshape(-1, 4, 2)
    junction_locations = det_results[:, 0:8].reshape(-1, 4, 2)
    slot_centers = junction_locations.mean(axis=-2)
    slot_directions = slot_centers + 30 * det_results[:, 26:28]
    # junction_directions = junction_locations + 30 * point_directions
    center_points = junction_locations.mean(1).astype(np.int)
    for i, junction_location in enumerate(junction_locations):
        cv2.line(
            img1,
            tuple(junction_location[0].astype(np.int)),
            tuple(junction_location[1].astype(np.int)),
            (255, 0, 0),
            1,
        )
        cv2.line(
            img1,
            tuple(junction_location[1].astype(np.int)),
            tuple(junction_location[2].astype(np.int)),
            (255, 0, 0),
            1,
        )
        cv2.line(
            img1,
            tuple(junction_location[2].astype(np.int)),
            tuple(junction_location[3].astype(np.int)),
            (255, 0, 0),
            1,
        )
        cv2.line(
            img1,
            tuple(junction_location[3].astype(np.int)),
            tuple(junction_location[0].astype(np.int)),
            (255, 0, 0),
            1,
        )
        cv2.line(
            img1,
            tuple(slot_centers[i].astype(np.int)),
            tuple(slot_directions[i].astype(np.int)),
            (255, 0, 100),
            2,
        )

        cv2.fillPoly(
            img2,
            [junction_location.astype(np.int)],
            slot_occupancy_color_dict[int(det_results[i][16])],
        )
        pass
    for i, center_point in enumerate(center_points):
        cv2.putText(
            img1,
            slot_type_mark_dict[int(det_results[i][17])],
            tuple(center_point),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
    img1 = cv2.addWeighted(img1, 0.8, img2, 0.2, 0)
    return img1


@OBJECT_REGISTRY.register
class ANCBEVPSDMetric(EvalMetric):
    """Compute bev psd point detection metric.

    .. code-block:: none

        *************************************************
        *               10-16m(only front)              *
        *************************************************
        *                      6-10m                    *
        *     *************************************     *
        *     *                2-6m               *     *
        *     *     *************************     *     *
        *     *     *    0-2m(include ego)  *     *     *
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
        iou_threshold: iou threshold to judge whether
            detect slot or not.
        global_max_distance: coordinate distance threshold to judge
            whether global corner point prediction is right.
        local_max_distance: coordinate distance threshold to judge
            whether local corner point prediction is right.
        validation_bev_range_list: store the range boundary of the above chart.
        resolution_m_per_grid: spatial resolution
            between vcs range and bev range.
        slot_type: The slot category of the model output.
        ignore_det_iou_thresh: For large ignore area, if
            intersection > min(gt_ignore, pred) * ignore_det_iou_thresh,
            ignore the pred result.
        eval_result_path: PSD metric are stored in json format.
        device: The architecture used in the calculation. eg:cpu or gpu.
        loc_2sigma: A 2sigma difference location value. unit: meters.
        save_badcase: ensure whether compute and save badcase info to json.
        name: Name of this metric instance for display.
        result_prefix: Prefix of aidi eval result.
        vis_image_dir: Save dir of visual images.
        upload_image2aidi: Whether upload visunal images to aidi experiment
            manager or not.
    """

    def __init__(
        self,
        iou_threshold: float,
        global_max_distance: float,
        local_max_distance: float,
        validation_bev_range_list: list,
        resolution_m_per_grid: float,
        loc_2sigma: float = 0.22,
        slot_type: tuple = ("Vertical", "Parallel", "Oblique"),
        ignore_det_iou_thresh: float = 0.7,
        device: str = None,
        eval_result_path: str = None,
        save_badcase: bool = False,
        name: str = "bevpsd",
        result_prefix: str = "",
        vis_image_dir: Optional[str] = None,
        upload_image2aidi: bool = False,
    ):
        self.slot_type = slot_type
        self.iou_threshold = iou_threshold
        self.global_max_distance = global_max_distance
        self.local_max_distance = local_max_distance
        self.validation_bev_range_list = validation_bev_range_list
        self.resolution_m_per_grid = resolution_m_per_grid
        self.ignore_det_iou_thresh = ignore_det_iou_thresh
        self._device = device
        self.eval_result_path = eval_result_path
        self.name = name
        self.loc_2sigma = loc_2sigma
        self.save_badcase = save_badcase
        self.result_prefix = result_prefix
        self.vis_image_dir = vis_image_dir
        self.upload_image2aidi = upload_image2aidi
        super(ANCBEVPSDMetric, self).__init__(name)

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
        names = [
            "num_test_set",
        ]
        return names

    def get_names_cat(self):
        """
        Update msg.

        instances_list: such as below.

            .. code-block:: none

                [
                    [
                        dets_junction01_mean_x,
                        dets_junction01_mean_y,
                        TP/FP flag,
                        position_diff_0,
                        position_diff_1,
                        angle_diff,
                        slot heading angle diff,
                        horizon_diff,
                        vertical_diff,
                        occpuacy flag,
                        slot_det_type,

                    ],

                    [],[],[],
                ]

        gt_list: such as below.

            .. code-block:: none

            [
                [
                    slots_junction01_mean_x,
                    slots_junction01_mean_y,
                    slots_type_label,

                ],

                [], [], [],

            ]

        """
        names = [
            "instances_list",
            "gt_list",
        ]
        return names

    def reset(self) -> None:
        super().reset()

    def cal_poly_area(self, gt: List, pred: List):
        """Calculate polygon area.

        Args:
            gt: [[x1, y1], [x2, y2], [x3, y3]...]
                groundtruth polygon result.
            pred: [[x1, y1], [x2, y2], [x3, y3]...]
                predict polygon result.
        """
        gt = tuple(tuple(x) for x in gt)
        pred = tuple(tuple(x) for x in pred)

        if np.isnan(gt).any() or np.isnan(pred).any():
            return False, 0, 0, 0
        # Note:The corners of the polygon must be greater than 3
        if len(gt) < 3 or len(pred) < 3:
            return False, 0, 0, 0
        gt = Polygon(gt)
        pred = Polygon(pred)
        if not gt.is_valid or not pred.is_valid:
            return False, 0, 0, 0
        gt_area = Polygon(gt).area
        pred_area = Polygon(pred).area
        try:
            inter_area = Polygon(gt).intersection(Polygon(pred)).area
            return True, gt_area, pred_area, inter_area
        except ValueError:
            logger.warning(f"Convex polygons fail, gt:{gt}, pred:{pred}")
            return False, 0, 0, 0

    def is_in_bev_range(self, bev_range, pts_x, pts_y):
        idx_in_range_xmin = pts_x[:] >= bev_range[0]
        idx_in_range_xmax = pts_x[:] <= bev_range[2]
        idx_in_range_ymin = pts_y[:] >= bev_range[1]
        idx_in_range_ymax = pts_y[:] <= bev_range[3]
        idx_in_range_x = np.logical_and(idx_in_range_xmin, idx_in_range_xmax)
        idx_in_range_y = np.logical_and(idx_in_range_ymin, idx_in_range_ymax)
        idx_in_range = np.logical_and(idx_in_range_x, idx_in_range_y)
        return idx_in_range

    def update(self, annos_bev_psd_obj, decode_label):
        """
        Update fuse msg.

        .. code-block:: none

            global_label: batch * n * 16.
                ignore or Not_normal_slot: [[x1, y1, x2, y2, x3, y3, x4, y4,
                        x5, y5, x6, y6, occupancy, len_global, slot_type,
                        ignore_type],
                        [], [], ...
                    ]

                normal: [[centerx, centery, slot_type, slot_vec_x, slot_vec_y,
                        occupancy, x1, y1, x2, y2, x3, y3, x4, y4, slot_type,
                        ignore_type],
                        [], [], ...
                    ]

            local_label: batch * n * 22

                ignore or Not_normal_slot: [[x1, y1, x2, y2, x3, y3, x4,
                        y4, x5, y5, x6, y6, 0, 0, 0, 0, 0, 0, 0, len_local,
                        slot_type, ignore_type],
                        [], [],...
                    ]

                normal: [[x1, y1, x2, y2, x3, y3, x4, y4, p1_vec_x,
                        p1_vec_y, p2_vec_x, p2_vec_y, p3_vec_x, p3_vec_y,
                        p4_vec_x, p4_vec_y, p1_type, p2_type, p3_type,
                        p4_type, slot_type, ignore_type],
                        [], [],...
                    ]
            det_results: batch* (N * 33).

                [
                    [x0,y0,x1,y1,x2,y2,x3,y3, type_j0, type_j1, type_j2,
                    type_j3,visibility_j0, visibility_j1, visibility_j2,
                    visibility_j3, occupancy of slot, type of slot,
                    sin_j0, cos_j0, sin_j1, cos_j1, sin_j2,
                    cos_j2, sin_j3, cos_j3, sin_slot, cos_slot],...

                ]

        """

        b_global_labels = np.array(annos_bev_psd_obj["global"].cpu())
        b_local_labels = np.array(annos_bev_psd_obj["local"].cpu())
        b_slots_det = np.array(decode_label["decode_slots"])
        assert len(b_global_labels.shape) == 3
        assert len(b_local_labels.shape) == 3
        batch_size = b_local_labels.shape[0]

        for bs in range(batch_size):
            name = "num_test_set"
            value_ddp = getattr(self, name, None) + torch.tensor(1)
            setattr(self, name, value_ddp)
            if self._device is None:
                self._device = value_ddp.device

            global_labels = b_global_labels[bs]
            local_labels = b_local_labels[bs]
            slots_det = np.array(b_slots_det[bs])
            # delete empty data
            valid_idx = (
                np.sum(global_labels, axis=1) > np.finfo(np.float32).eps
            )
            global_labels = global_labels[valid_idx]
            local_labels = local_labels[valid_idx]

            # Handicapped and Not_normal_slot as ignore
            idx_handicapped = (
                global_labels[:, -2] == slot_type_dict["Handicapped"]
            )
            idx_not_normal = (
                global_labels[:, -2] == slot_type_dict["Not_normal_slot"]
            )
            idx_type_ignore = global_labels[:, -1] == ignore_dict["Y"]
            idx_slot_ignore = np.logical_or(idx_not_normal, idx_type_ignore)
            idx_slot_ignore = np.logical_or(idx_slot_ignore, idx_handicapped)
            global_ignore = global_labels[idx_slot_ignore]
            idx_not_ignore = np.logical_not(idx_slot_ignore)

            # Not_slot as negtive, delete Not_slot label
            idx_not_slot = global_labels[:, -2] == slot_type_dict["Not_slot"]
            idx_filter_not_slot = np.logical_not(idx_not_slot)
            idx_slot_normal = np.logical_and(
                idx_not_ignore, idx_filter_not_slot
            )
            slots_global_label = global_labels[idx_slot_normal]
            slots_local_label = local_labels[idx_slot_normal]

            if len(slots_det.shape) != 2:
                continue
            slots_det_valid_status = np.ones(len(slots_det), dtype=np.bool)

            for slot_ignore in global_ignore:
                slot_ignore_point_len = int(slot_ignore[-3])
                slot_ignore_point = slot_ignore[0:slot_ignore_point_len]
                slot_ignore_point = slot_ignore_point.reshape(
                    slot_ignore_point_len // 2, 2
                )
                slot_ignore_poly = [tuple(i) for i in slot_ignore_point]
                for idx, d in enumerate(slots_det):
                    slot_det_point = d[:8].reshape(4, 2)
                    slot_det_poly = [tuple(i) for i in slot_det_point]
                    (
                        valid,
                        poly1_area,
                        poly2_area,
                        poly_insec,
                    ) = self.cal_poly_area(slot_ignore_poly, slot_det_poly)
                    if (
                        poly_insec
                        > min(poly1_area, poly2_area)
                        * self.ignore_det_iou_thresh
                        or not valid
                    ):
                        slots_det_valid_status[idx] = False
            slots_det = slots_det[slots_det_valid_status]
            dets_junction01_mean_x = (slots_det[:, 0] + slots_det[:, 2]) / 2
            dets_junction01_mean_y = (slots_det[:, 1] + slots_det[:, 3]) / 2
            slots_junction01_mean_x = (
                slots_global_label[:, 6] + slots_global_label[:, 8]
            ) / 2
            slots_junction01_mean_y = (
                slots_global_label[:, 7] + slots_global_label[:, 9]
            ) / 2
            # collect match msg between label and det
            gt_recall_list = []
            loc_badcase_list = []
            for slot_det_idx in range(len(slots_det)):
                slot_det_occupancy = slots_det[slot_det_idx][16]
                slot_det_type = slots_det[slot_det_idx][17]
                slot_det_angle = slots_det[slot_det_idx][26:28]
                slot_det_point = slots_det[slot_det_idx][:8].reshape(4, 2)
                slot_det_poly = [tuple(i) for i in slot_det_point]
                # TODO: accord dict optimize
                ins_list = [
                    dets_junction01_mean_x[slot_det_idx],
                    dets_junction01_mean_y[slot_det_idx],
                    0,  # TP:1, FP:0, IoUed but cls error:-1
                    0,  # position_diff_0,
                    0,  # position_diff_1,
                    0,  # angle_diff,
                    0,  # heading angle diff
                    0,  # horizon_diff
                    0,  # vertical_diff
                    0,  # occpuacy TP:1, FP:-1
                    slot_det_type,
                ]
                for idx_label in range(slots_global_label.shape[0]):
                    slot_label_occupancy = slots_global_label[idx_label, 5]
                    slot_label_type = slots_global_label[idx_label, -2]
                    slot_label_angle = slots_global_label[idx_label, 3:5]
                    slot_label_point = slots_local_label[idx_label, :8]
                    slot_label_point = slot_label_point.reshape(4, 2)
                    slot_label_poly = [tuple(i) for i in slot_label_point]
                    try:
                        iou = cal_iou_by_polygon(
                            slot_det_poly, slot_label_poly
                        )
                    except Exception:
                        iou = 0
                    if iou > self.iou_threshold:
                        if int(slot_det_type) == int(slot_label_type):
                            if idx_label in gt_recall_list:
                                continue
                            gt_recall_list.append(idx_label)

                            for slot_name in self.slot_type:
                                if slot_type_dict[slot_name] == int(
                                    slot_det_type
                                ):
                                    position_diff = (
                                        self.compute_point_position_diff(
                                            position_result=slot_det_point,
                                            position_label=slot_label_point,
                                        )
                                    )
                                    angle_diff = self.compute_slot_angle_diff(
                                        angle_result=slot_det_angle,
                                        angle_label=slot_label_angle,
                                    )
                                    heading_angle_diff = (
                                        self.compute_heading_angle_diff(
                                            slot_det_point,
                                            slot_label_point,
                                        )
                                    )
                                    junction_23_mean_diff = self.compute_horizon_vertical_position_diff(  # noqa
                                        slot_det_point[2:],
                                        slot_label_point[2:],
                                        slot_label_angle,
                                    )
                                    # If it is tp, replace the predict midpoint
                                    # with the midpoint position of groundtruth
                                    ins_list[0] = slots_junction01_mean_x[
                                        idx_label
                                    ]
                                    ins_list[1] = slots_junction01_mean_y[
                                        idx_label
                                    ]
                                    ins_list[2] = 1  # TP
                                    ins_list[3] = position_diff[0]
                                    ins_list[4] = position_diff[1]
                                    ins_list[5] = angle_diff
                                    ins_list[6] = heading_angle_diff
                                    ins_list[7] = junction_23_mean_diff[0]
                                    ins_list[8] = junction_23_mean_diff[1]
                                    if self.save_badcase:
                                        is_need_range = self.is_in_bev_range(
                                            self.validation_bev_range_list[1],
                                            slot_det_point[:2, 0],
                                            slot_det_point[:2, 1],
                                        )
                                        if is_need_range.any():
                                            pos_vcs_diff0 = (
                                                position_diff[0]
                                                * self.resolution_m_per_grid
                                            )
                                            pos_vcs_diff1 = (
                                                position_diff[1]
                                                * self.resolution_m_per_grid
                                            )
                                            if (
                                                min(
                                                    pos_vcs_diff0,
                                                    pos_vcs_diff1,
                                                )
                                                >= self.loc_2sigma
                                            ):
                                                loc_badcase = (
                                                    slot_det_point.tolist()
                                                )
                                                loc_badcase.append(
                                                    [
                                                        pos_vcs_diff0,
                                                        pos_vcs_diff1,
                                                    ]
                                                )
                                                loc_badcase_list.append(
                                                    loc_badcase
                                                )
                        else:
                            ins_list[2] = -1

                        if (
                            slot_det_occupancy == slot_label_occupancy
                            and slot_label_occupancy == occupancy_dict["Y"]
                        ):
                            ins_list[9] = 1
                        if (
                            slot_det_occupancy != slot_label_occupancy
                            and slot_label_occupancy == occupancy_dict["Y"]
                        ):
                            ins_list[9] = -1
                name = "instances_list"
                setattr(
                    self,
                    name,
                    getattr(self, name, [])
                    + [torch.tensor(ins_list, device=self._device)],
                )
            gt_recall_list.sort()
            for idx in range(slots_global_label.shape[0]):
                gt_lst = [
                    slots_junction01_mean_x[idx],
                    slots_junction01_mean_y[idx],
                    slots_global_label[idx][-2],
                ]
                name = "gt_list"
                setattr(
                    self,
                    name,
                    getattr(self, name, [])
                    + [torch.tensor(gt_lst, device=self._device)],
                )

    def compute_point_position_diff(
        self, position_result: np.array, position_label: np.array
    ):
        """
        Update point position diff.

        position_result:  [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        position_label:  [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        position_diff = position_result[:2] - position_label[:2]
        position_diff = np.sqrt(np.sum(position_diff ** 2, axis=-1))
        return position_diff

    def compute_slot_angle_diff(
        self, angle_result: np.array, angle_label: np.array
    ):
        """Compute point angle diff.

        angle_result: [x1, y1]
        angle_label: [x1, y1]
        """
        angle_diff = F.cosine_similarity(
            torch.tensor(angle_result), torch.tensor(angle_label), dim=0
        )
        if abs(1 - angle_diff) < 1e-5:
            angle_diff = torch.tensor(1 - 1e-5)
        elif abs(-1 - angle_diff) < 1e-5:
            angle_diff = torch.tensor(-1 + 1e-5)
        angle_diff = torch.acos(angle_diff) * 180 / pi
        return angle_diff

    def get_sline_unit_vec(self, point_data_list: np.array) -> List:
        """
        Get slot two sline angle unit vec.

        Args:
            point_data_list: np.array, [[x1, y1],[x2, y2],[x3, y3],[x4, y4]]

        Returns:
            vec0: [v1_x, v1_y]
            vec1: [v2_x, v2_y]
        """
        diff1 = point_data_list[3] - point_data_list[0]
        diff2 = point_data_list[2] - point_data_list[1]
        vec0 = (
            diff1 / (np.linalg.norm(diff1, axis=0) + np.finfo(np.float32).eps)
        ).tolist()
        vec1 = (
            diff2 / (np.linalg.norm(diff2, axis=0) + np.finfo(np.float32).eps)
        ).tolist()

        return vec0, vec1

    def compute_direct_angle_diff(
        self, pred_vec_list: np.array, label_vec_list: np.array
    ) -> List:
        """
        Compute two vector angle diff with direction.

        Args:
            pred_vec_list: [[x1, y1],[x2, y2], ...].
            label_vec_list: [[u1, v1],[u2, v2], ...].

        Returns:
            The storage Angle is positive
                clockwise(+°) and negative anticlockwise(-°)
        """
        assert len(pred_vec_list) == len(
            label_vec_list
        ), "The contrast vectors have different lengths"
        angle_diff = []
        for angle_idx in range(len(pred_vec_list)):
            v1, v2 = pred_vec_list[angle_idx], label_vec_list[angle_idx]
            norm = (
                np.linalg.norm(v1) * np.linalg.norm(v2)
                + np.finfo(np.float32).eps
            )
            is_clock_flag = np.rad2deg(np.arcsin(np.cross(v1, v2) / norm))
            theta = np.rad2deg(np.arccos(np.dot(v1, v2) / norm))
            if is_clock_flag < 0:
                angle_diff.append(theta)
            else:
                angle_diff.append(-theta)
        return angle_diff

    def compute_heading_angle_diff(
        self, pred_junctions: np.array, gt_junctions: np.array
    ):
        """Compute slot heading point angle diff.

        pred_junctions: [[x1, y1],[x2, y2],[x3, y3],[x4, y4]]
        gt_junctions: [[x1, y1],[x2, y2],[x3, y3],[x4, y4]]
        """
        angle_result = self.get_sline_unit_vec(pred_junctions)
        angle_label = self.get_sline_unit_vec(gt_junctions)
        angle_diff = self.compute_direct_angle_diff(angle_label, angle_result)
        heading_angle_diff = np.sum(angle_diff)
        return heading_angle_diff

    def compute_horizon_vertical_position_diff(
        self,
        end_points_pred: np.array,
        end_points_label: np.array,
        slot_label_angle: np.array,
    ):
        """Compute horizon and vertical position diff based label line direction.

        Args:
            end_points_pred: [[x0, y0], [x1, y1]]
                The two endpoint coordinate of a pred line.
            end_points_label: [[x0, y0], [x1, y1]]
                The two endpoint coordinate of a label line.
            slot_label_angle: slot center orientation vector label.
        """
        assert len(end_points_pred) == len(
            end_points_label
        ), "The contrast points have different lengths"
        center_pred = np.mean(end_points_pred, axis=0)
        center_label = np.mean(end_points_label, axis=0)
        center_diff = np.linalg.norm(center_pred - center_label)

        diff_vec = center_label - center_pred
        angle = self.compute_slot_angle_diff(diff_vec, slot_label_angle)
        rad = torch.deg2rad(angle)
        horizon_diff = abs(center_diff * torch.sin(rad))
        vertical_diff = abs(center_diff * torch.cos(rad))
        return [horizon_diff, vertical_diff]

    def generate_images(self, ins_results: List, image_dir: str):
        """Generate aidi images.

        Args:
            ins_results: Instances evaluation scores.
            image_dir: Visual images saved dir.
        """
        images = []
        for ins_result in ins_results:
            ins_result = ins_result.tolist()

            ts = int(ins_result[-1])
            fp_score = ins_result[2]
            junction_diff = ins_result[3]
            heading_angle = ins_result[6]
            image_path = glob.glob(os.path.join(image_dir, f"{ts}*"))
            if len(image_path) == 0:
                continue
            image_path = image_path[0]
            if not os.path.exists(image_path):
                continue
            image = Image(
                name="/".join(image_path.split("/")[-2:]),
                attrs={
                    "fp_score": fp_score,
                    "junction_diff": junction_diff,
                    "heading_angle": heading_angle,
                },
            )
            image.add_slice(data_or_path=image_path)
            images.append(image)

        return images

    def get(self):
        format_metrics, instances_list = self.compute()
        if self.eval_result_path is not None:
            save_dir, _ = os.path.split(self.eval_result_path)
            os.makedirs(save_dir, exist_ok=True)
            with open(self.eval_result_path, "w", encoding="utf-8") as f:
                json.dump(
                    format_metrics,
                    f,
                    ensure_ascii=False,
                    indent=1,
                    cls=NpEncoder,
                )
        names_range = []
        names_cnt, value_cnt = [], []
        names_err, value_err = [], []
        names_precison, value_precision = [], []
        names_recall, value_recall = [], []
        names_position, value_position = [], []
        names_angle, value_angle = [], []
        for k_, d in format_metrics.items():
            names_range += [k_]
            for k, v in d.items():
                if "TP" in k:
                    names_cnt += [k]
                    value_cnt += [v]
                if "GT" in k:
                    names_cnt += [k]
                    value_cnt += [v]
                if "FP" in k:
                    names_cnt += [k]
                    value_cnt += [v]
                if "error_rate" in k:
                    names_err += [k]
                    value_err += [v]
                if "precision" in k:
                    names_precison += [k]
                    value_precision += [v]
                if "recall" in k:
                    names_recall += [k]
                    value_recall += [v]
                if "position" in k:
                    names_position += [k]
                    value_position += [v]
                if "angle" in k:
                    names_angle += [k]
                    value_angle += [v]
        summary_str = "~~~~ BEV PSD Summary metrics ~~~~\n"
        start_str = "{:<50}\t"
        summary_str += start_str.format("Summary:")
        columns = ["Summary"]
        data = []
        summary = {}
        # table_data = []
        for name_r in names_range:
            line_format = "{:<10}\t"
            summary_str += line_format.format(name_r)
            columns.append(name_r)
        summary_str += "\n"
        name_lst = []
        for (name_metric, _) in zip(names_err, value_err):
            if name_metric in name_lst:
                continue
            line_format = "{:<50}\t"
            summary_str += line_format.format(name_metric)
            _data = {"Summary": name_metric}
            idx = 1
            for (name, value) in zip(names_err, value_err):
                if name == name_metric:
                    metric_format = "{:<10.3f}\t"
                    summary_str += metric_format.format(value)
                    _data.update({f"{columns[idx]}": float(value)})
                    idx += 1
            summary_str += "\n"
            name_lst.append(name_metric)
            data.append(_data)
        summary_str += "\n"
        name_lst = []
        for (name_metric, _) in zip(names_cnt, value_cnt):
            if name_metric in name_lst:
                continue
            line_format = "{:<50}\t"
            summary_str += line_format.format(name_metric)
            _data = {"Summary": name_metric}
            idx = 1
            for (name, value) in zip(names_cnt, value_cnt):
                if name == name_metric:
                    metric_format = "{:<10.3f}\t"
                    summary_str += metric_format.format(value)
                    _data.update({f"{columns[idx]}": float(value)})
                    idx += 1
            summary_str += "\n"
            name_lst.append(name_metric)
            data.append(_data)
        summary_str += "\n"
        name_lst = []
        for (name_metric, _) in zip(names_precison, value_precision):
            if name_metric in name_lst:
                continue
            line_format = "{:<50}\t"
            summary_str += line_format.format(name_metric)
            _data = {"Summary": name_metric}
            idx = 1
            for (name, value) in zip(names_precison, value_precision):
                if name == name_metric:
                    if (
                        name
                        in [
                            "precision_Vertical(%)",
                            "precision_Parallel(%)",
                            "precision_occupancy(%)",
                        ]
                        and idx == 1
                    ):
                        summary.update({f"psd_{name}": float(value)})
                    metric_format = "{:<10.3f}\t"
                    summary_str += metric_format.format(value)
                    _data.update({f"{columns[idx]}": float(value)})
                    idx += 1
            summary_str += "\n"
            name_lst.append(name_metric)
            data.append(_data)
        summary_str += "\n"
        name_lst = []
        for (name_metric, _) in zip(names_recall, value_recall):
            if name_metric in name_lst:
                continue
            line_format = "{:<50}\t"
            summary_str += line_format.format(name_metric)
            _data = {"Summary": name_metric}
            idx = 1
            for (name, value) in zip(names_recall, value_recall):
                if name == name_metric:
                    metric_format = "{:<10.3f}\t"
                    summary_str += metric_format.format(value)
                    _data.update({f"{columns[idx]}": float(value)})
                    if idx == 1:
                        summary.update({f"psd_{name}": float(value)})
                    idx += 1
            summary_str += "\n"
            name_lst.append(name_metric)
            data.append(_data)
        summary_str += "\n"
        name_lst = []
        for (name_metric, _) in zip(names_position, value_position):
            if name_metric in name_lst:
                continue
            line_format = "{:<50}\t"
            summary_str += line_format.format(name_metric)
            _data = {"Summary": name_metric}
            idx = 1
            for (name, value) in zip(names_position, value_position):
                if name == name_metric:
                    if name in ["position_mean_all(m)"] and idx == 1:
                        summary.update({f"psd_{name}": float(value)})
                    metric_format = "{:<10.3f}\t"
                    summary_str += metric_format.format(value)
                    _data.update({f"{columns[idx]}": float(value)})
                    idx += 1
            summary_str += "\n"
            name_lst.append(name_metric)
            data.append(_data)
        summary_str += "\n"
        name_lst = []
        for (name_metric, _) in zip(names_angle, value_angle):
            if name_metric in name_lst:
                continue
            line_format = "{:<50}\t"
            summary_str += line_format.format(name_metric)
            _data = {"Summary": name_metric}
            idx = 1
            for (name, value) in zip(names_angle, value_angle):
                if name == name_metric:
                    if name in ["angle_mean_all(°)"] and idx == 1:
                        summary.update({f"psd_{name}": float(value)})
                    metric_format = "{:<10.3f}\t"
                    summary_str += metric_format.format(value)
                    _data.update({f"{columns[idx]}": float(value)})
                    idx += 1
            summary_str += "\n"
            name_lst.append(name_metric)
            data.append(_data)
        logger.info(summary_str)
        total_names = []
        total_values = []
        for k, v in format_metrics.items():
            total_names += k
            total_values += v
        if data:
            columns, data = reverse_table(columns, data)
        tables = [
            Table(
                name=self.result_prefix + "_" + self.name,
                columns=columns,
                data=data,
            )
        ]

        images = None
        if self.upload_image2aidi and instances_list is not None:
            image_dir = self.vis_image_dir
            images = self.generate_images(instances_list, image_dir)
        eval_result = EvalResult(
            summary=summary,
            tables=tables,
            images=images,
        )
        return self.name, eval_result

    def get_single(self):
        format_metrics = self.compute()
        name = format_metrics.keys()
        value = format_metrics.values()

        names_cnt, value_cnt = [], []
        names_err, value_err = [], []
        names_precison, value_precision = [], []
        names_recall, value_recall = [], []
        names_position, value_position = [], []
        names_angle, value_angle = [], []
        for k, v in format_metrics.items():
            if "TP" in k:
                names_cnt += [k]
                value_cnt += [v]
            if "GT" in k:
                names_cnt += [k]
                value_cnt += [v]
            if "FP" in k:
                names_cnt += [k]
                value_cnt += [v]
            if "error_rate" in k:
                names_err += [k]
                value_err += [v]
            if "precision" in k:
                names_precison += [k]
                value_precision += [v]
            if "recall" in k:
                names_recall += [k]
                value_recall += [v]
            if "position" in k:
                names_position += [k]
                value_position += [v]
            if "angle" in k:
                names_angle += [k]
                value_angle += [v]

        summary_str = "~~~~ %s Summary metrics ~~~~\n" % (self.name)

        summary_str += "Summary:\n"
        for (name, value) in zip(names_err, value_err):
            line_format = "{:<50}\t{:>10.3f}\n"
            summary_str += line_format.format(name, value)
        for (name, value) in zip(names_cnt, value_cnt):
            line_format = "{:<50}\t{:>10.3f}\n"
            summary_str += line_format.format(name, value)
        summary_str += "\n"
        for (name, value) in zip(names_precison, value_precision):
            line_format = "{:<50}\t{:>10.3f}\n"
            summary_str += line_format.format(name, value)
        summary_str += "\n"
        for (name, value) in zip(names_recall, value_recall):
            line_format = "{:<50}\t{:>10.3f}\n"
            summary_str += line_format.format(name, value)
        summary_str += "\n"
        for (name, value) in zip(names_position, value_position):
            line_format = "{:<50}\t{:>10.3f}\n"
            summary_str += line_format.format(name, value)
        summary_str += "\n"
        for (name, value) in zip(names_angle, value_angle):
            line_format = "{:<50}\t{:>10.3f}\n"
            summary_str += line_format.format(name, value)
        logger.info(summary_str)

        total_names = (
            names_precison + names_recall + names_position + names_angle
        )
        total_values = (
            value_precision + value_recall + value_position + value_angle
        )
        return total_names, total_values

    def get_pr(
        self,
        metric: dict,
        num_pred: int,
        num_tp: int,
        num_gt: int,
        name: str = "all",
        mode: str = "all",
    ):
        """
        Update precision and recall across mode.

        Args:
            metric: record number of predict,label
                    and precision and recall
            mode: precision: only store precision
                  recall: only store recall
                  all: store precision and recall
            name: constitute the key of dict.
        """
        num_fp = num_pred - num_tp
        metric[f"TP_cnt_{name}"] = num_tp
        metric[f"FP_cnt_{name}"] = num_fp
        metric[f"GT_cnt_{name}"] = num_gt
        if mode == "precision" or mode == "all":
            metric[f"precision_{name}(%)"] = (
                0 if num_pred == 0 else (num_tp / num_pred) * 100
            )
        if mode == "recall" or mode == "all":
            metric[f"recall_{name}(%)"] = (
                0 if num_gt == 0 else (num_tp / num_gt) * 100
            )
        return metric

    def get_position(
        self,
        metric: dict,
        data: list,
        name: str = "all",
    ):
        """
        Update metric relate to position deviation.

        Args:
            metric: record mean deviation, standard deviation
                    and 2sigma result
            data: [pos_diff_1, pos_diff_2, ...]
            name: constitute the key of dict.
        """
        data = torch.tensor(data) * self.resolution_m_per_grid

        metric[f"position_mean_{name}(m)"] = torch.mean(data).item()
        metric[f"position_std_{name}(m)"] = torch.std(data).item()
        metric[f"position_2sigma_{name}(m)"] = get_topk_min_value(data, 0.05)
        return metric

    def get_angle(
        self,
        metric: dict,
        data: list,
        name: str = "all",
    ):
        """
        Update metric relate to angle deviation.

        Args:
            metric: record angle mean deviation, standard deviation
                    and 2sigma result
            data: [angle_diff_1, angle_diff_2, ...]
            name: constitute the key of dict.
        """
        data = torch.tensor(data)

        metric[f"angle_mean_{name}(°)"] = torch.mean(data).item()
        metric[f"angle_std_{name}(°)"] = (
            0 if data.sum() == 0 else torch.std(data).item()
        )
        metric[f"angle_2sigma_{name}(°)"] = get_topk_min_value(data, 0.05)
        return metric

    def gen_psd_metric(self, res, gt_list):
        metric = {}
        gt_all, pred_all, tp_all, ava_tp_all, iou_pred_all = 0, 0, 0, 0, 0
        (
            positions_01p_diff,
            positions_bottom_horizion_diff,
            positions_bottom_vertical_diff,
            angles_diff,
            angle_heading_diff,
            angle_heading_abs_diff,
        ) = ([], [], [], [], [], [])
        for idx in range(len(self.slot_type)):
            slot_type_inds = slot_type_dict[self.slot_type[idx]]
            idx_slot = res[:, -1] == slot_type_inds
            idx_gt_slot = (
                np.array([False])
                if not np.any(gt_list)
                else (gt_list[:, -1] == slot_type_inds)
            )
            num_gt = (
                0 if not np.any(idx_gt_slot) else len(gt_list[idx_gt_slot])
            )
            if not np.any(idx_slot):
                num_pred, num_tp, num_iou_pred, num_ava_tp = 0.0, 0.0, 0.0, 0.0
                (
                    pos0_diff,
                    pos1_diff,
                    positions_bottom_horizion_diff_slot,
                    positions_bottom_vertical_diff_slot,
                ) = (
                    np.array([0.0]),
                    np.array([0.0]),
                    np.array([0.0]),
                    np.array([0.0]),
                )
                (
                    slot_angle_diff,
                    angle_heading_diff_slot,
                    angle_heading_abs_diff_slot,
                ) = (
                    np.array([0.0]),
                    np.array([0.0]),
                    np.array([0.0]),
                )
            else:
                res_slot = res[idx_slot]
                res_tp = res_slot[res_slot[:, 2] == 1]
                res_iou = res_slot[res_slot[:, 2] != 0]
                num_iou_pred = len(res_iou)
                num_pred = len(res_slot)
                num_tp = len(res_tp)
                # The error of slot corner tp will not exceed 2.5m,
                # but the error of GT marking will be more than 2.5m,
                # so ignore it.
                res_tp = res_tp[
                    np.logical_and(res_tp[:, 3] < 2.5, res_tp[:, 4] < 2.5)
                ]
                num_ava_res_tp = len(res_tp)
                pos0_diff = res_tp[:, 3]
                pos1_diff = res_tp[:, 4]
                slot_angle_diff = res_tp[:, 5]
                angle_heading_diff_slot = res_tp[:, 6]
                angle_heading_abs_diff_slot = (
                    np.absolute(angle_heading_diff_slot) / 2
                )
                positions_bottom_horizion_diff_slot = res_tp[:, 7]
                positions_bottom_vertical_diff_slot = res_tp[:, 8]
                pos_flag = np.logical_and(pos0_diff < 0.5, pos1_diff < 0.5)
                angle_flag = angle_heading_abs_diff_slot < 1.5
                ava_flag = np.logical_and(pos_flag, angle_flag)
                num_ava_tp = len(res_tp[ava_flag])
            metric = self.get_pr(
                metric, num_pred, num_tp, num_gt, name=self.slot_type[idx]
            )
            metric = self.get_pr(
                metric,
                num_ava_res_tp,
                num_ava_tp,
                num_ava_res_tp,
                name=f"ava_{self.slot_type[idx]}",
                mode="precision",
            )
            metric = self.get_pr(
                metric,
                num_iou_pred,
                num_tp,
                num_gt,
                name=f"cls_acc_{self.slot_type[idx]}",
                mode="precision",
            )
            position_01p_diff_slot = pos0_diff.tolist() + pos1_diff.tolist()
            metric = self.get_position(
                metric,
                position_01p_diff_slot,
                name=self.slot_type[idx],
            )
            positions_bottom_horizion_diff_slot = (
                positions_bottom_horizion_diff_slot.tolist()
            )
            metric = self.get_position(
                metric,
                positions_bottom_horizion_diff_slot,
                name=f"bottom_midpoint_horizion_{self.slot_type[idx]}",
            )
            positions_bottom_vertical_diff_slot = (
                positions_bottom_vertical_diff_slot.tolist()
            )
            metric = self.get_position(
                metric,
                positions_bottom_vertical_diff_slot,
                name=f"bottom_midpoint_vertical_{self.slot_type[idx]}",
            )
            angle_diff_slot = slot_angle_diff.tolist()
            metric = self.get_angle(
                metric,
                angle_diff_slot,
                name=self.slot_type[idx],
            )
            angle_heading_abs_diff_slot = angle_heading_abs_diff_slot.tolist()
            metric = self.get_angle(
                metric,
                angle_heading_abs_diff_slot,
                name=f"heading_{self.slot_type[idx]}",
            )

            metric[f"angle_heading_system_mean_{self.slot_type[idx]}(°)"] = (
                0 if num_pred == 0 else np.mean(angle_heading_diff_slot)
            )
            tp_all += num_tp
            ava_tp_all += num_ava_tp
            gt_all += num_gt
            pred_all += num_pred
            iou_pred_all += num_iou_pred
            positions_01p_diff += position_01p_diff_slot
            angles_diff += angle_diff_slot
            angle_heading_abs_diff += angle_heading_abs_diff_slot
            angle_heading_diff += angle_heading_diff_slot.tolist()
            positions_bottom_horizion_diff += (
                positions_bottom_horizion_diff_slot
            )
            positions_bottom_vertical_diff += (
                positions_bottom_vertical_diff_slot
            )

        num_test_set = int(getattr(self, "num_test_set", None).cpu().numpy())
        err_rate = (
            0 if num_test_set == 0 else (pred_all - tp_all) / num_test_set
        )
        metric["error_rate(%)"] = err_rate * 100
        precision_occupancy = (
            0
            if np.sum(res[:, 9] != 0) == 0
            else np.sum(res[:, 9] == 1) / np.sum(res[:, 9] != 0)
        )
        metric["precision_occupancy(%)"] = precision_occupancy * 100
        metric = self.get_pr(
            metric,
            pred_all,
            tp_all,
            gt_all,
            name="all",
        )
        metric = self.get_pr(
            metric,
            tp_all,
            ava_tp_all,
            gt_all,
            name="ava_all",
            mode="precision",
        )
        metric = self.get_pr(
            metric,
            iou_pred_all,
            tp_all,
            gt_all,
            name="cls_acc_all",
            mode="precision",
        )
        metric = self.get_position(
            metric,
            positions_01p_diff,
            name="all",
        )
        metric = self.get_position(
            metric,
            positions_bottom_horizion_diff,
            name="bottom_midpoint_hor_all",
        )
        metric = self.get_position(
            metric,
            positions_bottom_vertical_diff,
            name="bottom_midpoint_ver_all",
        )
        metric = self.get_angle(
            metric,
            angles_diff,
            name="all",
        )
        metric = self.get_angle(
            metric,
            angle_heading_abs_diff,
            name="heading_all",
        )
        metric["angle_heading_system_mean(°)"] = (
            0 if num_pred == 0 else np.mean(angle_heading_diff)
        )
        return metric

    def collect_different_range_metric(
        self,
        results,
        gt_list,
        bev_range_list,
    ):
        results_range = {}
        if len(bev_range_list) < 2 or len(results) == 0:
            return {}
        ins_start_valid = self.is_in_bev_range(
            bev_range_list[0], results[:, 0], results[:, 1]
        )
        ins_end_valid = self.is_in_bev_range(
            bev_range_list[-1], results[:, 0], results[:, 1]
        )
        fn_start_valid = self.is_in_bev_range(
            bev_range_list[0], gt_list[:, 0], gt_list[:, 1]
        )
        fn_end_valid = self.is_in_bev_range(
            bev_range_list[-1], gt_list[:, 0], gt_list[:, 1]
        )
        ins_start_unvalid = np.logical_not(ins_start_valid)
        idx_ins_valid = np.logical_and(ins_start_unvalid, ins_end_valid)
        fn_start_unvalid = np.logical_not(fn_start_valid)
        idx_gt_valid = np.logical_and(fn_start_unvalid, fn_end_valid)
        if not np.any(idx_ins_valid):
            print("No result for valid range exists")
            return {}
        gt_valid_list = (
            np.array([[]])
            if not np.any(idx_gt_valid)
            else gt_list[idx_gt_valid]
        )
        results_range["all"] = self.gen_psd_metric(
            results[idx_ins_valid], gt_valid_list
        )
        for idx in range(len(bev_range_list) - 1):
            ins_start = self.is_in_bev_range(
                bev_range_list[idx], results[:, 0], results[:, 1]
            )
            ins_end = self.is_in_bev_range(
                bev_range_list[idx + 1], results[:, 0], results[:, 1]
            )
            fn_start = self.is_in_bev_range(
                bev_range_list[idx], gt_list[:, 0], gt_list[:, 1]
            )
            fn_end = self.is_in_bev_range(
                bev_range_list[idx + 1], gt_list[:, 0], gt_list[:, 1]
            )
            ins_not_start = np.logical_not(ins_start)
            idx_ins = np.logical_and(ins_not_start, ins_end)
            fn_not_start = np.logical_not(fn_start)
            idx_gt = np.logical_and(fn_not_start, fn_end)
            start_distance = round(
                (128 - bev_range_list[idx][1]) * self.resolution_m_per_grid
            )
            end_distance = round(
                (128 - bev_range_list[idx + 1][1]) * self.resolution_m_per_grid
            )
            start_name = (
                "0" if start_distance == 0 else str(start_distance - 4)
            )
            end_name = "0" if end_distance == 0 else (end_distance - 4)
            range_name = f"{start_name}_{end_name}(m)"
            if not np.any(idx_ins):
                print(f"No result for range {range_name} exists")
                continue
            fn_range_list = (
                np.array([[]]) if not np.any(idx_gt) else gt_list[idx_gt]
            )

            results_range[range_name] = self.gen_psd_metric(
                results[idx_ins], fn_range_list
            )
        return results_range

    def compute(self):
        instances_list = getattr(self, "instances_list", None)
        gt_list = getattr(self, "gt_list", None)
        if isinstance(instances_list, List):
            instances_list = [ins.tolist() for ins in instances_list]
            instances_list = torch.tensor(instances_list).cpu().numpy()
            gt_list = [fn.tolist() for fn in gt_list]
            gt_list = torch.tensor(gt_list).cpu().numpy()
        if isinstance(instances_list, torch.Tensor):
            instances_list = (
                getattr(self, "instances_list").reshape(-1, 11).cpu().numpy()
            )
            gt_list = getattr(self, "gt_list").reshape(-1, 3).cpu().numpy()
        format_metrics = self.collect_different_range_metric(
            instances_list,
            gt_list,
            self.validation_bev_range_list,
        )
        return format_metrics, instances_list
