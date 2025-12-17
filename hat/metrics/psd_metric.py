# Copyright (c) Horizon Robotics. All rights reserved.
import json
import logging
from math import pi, sqrt

import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon

from hat.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = ["PSDMetric"]

logger = logging.getLogger(__name__)


class ShapeTool(object):
    """Tool function used to calculate polygon property."""

    def __init__(self):
        pass

    def get_iou(self, poly1, poly2):

        poly1 = poly1 if isinstance(poly1, Polygon) else Polygon(poly1)
        poly2 = poly2 if isinstance(poly2, Polygon) else Polygon(poly2)

        inter = self.get_intersection(poly1, poly2)
        union = self.get_union(poly1, poly2)

        iou = 0 if union == 0 else inter / float(union)
        return iou

    def get_union(self, poly1, poly2):

        assert isinstance(poly1, Polygon) and isinstance(poly2, Polygon)
        return poly1.union(poly2).area

    def get_intersection(self, poly1, poly2):

        poly1 = poly1 if isinstance(poly1, Polygon) else Polygon(poly1)
        poly2 = poly2 if isinstance(poly2, Polygon) else Polygon(poly2)
        return poly1.intersection(poly2).area

    def get_area(self, poly):

        poly = poly if isinstance(poly, Polygon) else Polygon(poly)
        return poly.area


@OBJECT_REGISTRY.register
class PSDMetric(EvalMetric):
    """Computes super psd point detection metric.

    Args:
        iou_threshold: iou threshold to judge whether
            detect slot.
        global_max_distance: coordinate distance threshold to judge
            whether global corner point prediction is right.
        local_max_distance: coordinate distance threshold to judge
            whether local corner point prediction is right.
        is_training: whether model is in training step.
    """

    def __init__(
        self,
        iou_threshold: float,
        global_max_distance: float,
        local_max_distance: float,
        is_training=False,
        get_log=True,
        ignore_threshold=0.7,
        name="PSDMetric",
    ):
        super().__init__(name=name)
        self.global_max_distance = global_max_distance
        self.local_max_distance = local_max_distance
        self.shape_tool = ShapeTool()
        self.iou_threshold = iou_threshold
        self.is_training = is_training
        self.get_log = get_log
        self.ignore_threshold = ignore_threshold
        self.reset()
        self.names = {
            "global": [
                "Point_Acc_01",
                "Point_Recall_01",
                "Point_Mean_Dist_01",
                "Point_Mean_Dist_Square_01",
            ],
            "local": [
                "Point_Acc_01",
                "Point_Recall_01p_>=1",
                "Point_Recall_01p_>1",
                "Point_Mean_Dist_01",
                "Point_Mean_Dist_Square_01",
            ],
            "2sigma": [
                "All_Point_2sigma",
                "All_Angle_2sigma",
                "Vertical_Point_2sigma",
                "Paraller_Point_2sigma",
                "Oblique_Point_2sigma",
                "Vertical_Angle_2sigma",
                "Paraller_Angle_2sigma",
                "Oblique_Angle_2sigma",
            ],
            "acc": [
                "Fuse_Slot_Acc",
                "Indoors_Acc",
                "Outoors_Acc",
                "Vertical_Acc",
                "Parallel_Acc",
                "Oblique_Acc",
                "Occupancy_Acc",
                "Classification_Acc",
            ],
            "recall": [
                "Fuse_Slot_Recall",
                "Indoors_Recall",
                "Outdoors_Recall",
                "Vertical_Recall",
                "Parallel_Recall",
                "Oblique_Recall",
            ],
            "position": [
                "Slot_Position_Diff_Mean",
                "Vertical_Position_Diff_Mean",
                "Parallel_Position_Diff_Mean",
                "Oblique_Position_Diff_Mean",
                "Slot_Position_Diff_Variance",
                "Vertical_Position_Diff_Variance",
                "Parallel_Position_Diff_Variance",
                "Oblique_Position_Diff_Variance",
                "All_Point_2sigma",
                "Vertical_Point_2sigma",
                "Paraller_Point_2sigma",
                "Oblique_Point_2sigma",
            ],
            "angle": [
                "Slot_Angle_Diff_Mean",
                "Vertical_Angle_Diff_Mean",
                "Parallel_Angle_Diff_Mean",
                "Oblique_Angle_Diff_Mean",
                "Slot_Angle_Diff_Variance",
                "Vertical_Angle_Diff_Variance",
                "Parallel_Angle_Diff_Variance",
                "Oblique_Angle_Diff_Variance",
                "All_Angle_2sigma",
                "Vertical_Angle_2sigma",
                "Paraller_Angle_2sigma",
                "Oblique_Angle_2sigma",
            ],
        }

    def reset(self):

        self.reset_local()
        self.reset_global()
        self.reset_fuse()
        self.reset_2sigma()

    def reset_local(self):

        self.local_total_point_right_GreaterEqual1Point = [0, 0, 0, 0]
        self.local_total_point_right_Greater1Point = [0, 0, 0, 0]
        self.local_total_point_label = [0, 0, 0, 0]
        self.local_total_point_pred = [0, 0, 0, 0]

        self.local_point_position_diff = [0, 0, 0, 0]
        self.local_point_position_diff_square = [0, 0, 0, 0]
        self.local_point_angle_diff = [0, 0, 0, 0]
        self.local_point_angle_diff_square = [0, 0, 0, 0]
        self.oneLabelPoint_nearby_MaxPoints = 0

    def reset_global(self):

        self.global_total_point_right = [0, 0, 0, 0]
        self.global_total_point_label = [0, 0, 0, 0]
        self.global_total_point_pred = [0, 0, 0, 0]
        self.global_total_point_right_distance = [0, 0, 0, 0]
        self.global_total_point_right_distance_square = [0, 0, 0, 0]

    def reset_2sigma(self):

        self.all_point_position_diff = []
        self.all_kinds_point_position_diff = [[], [], []]
        self.all_slot_angle_diff = []
        self.all_kinds_slot_angle_diff = [[], [], []]

    def reset_fuse(self):

        self.fuse_total_slot_right = 0
        self.fuse_total_slot_label = 0
        self.fuse_total_slot_pred = 0

        self.fuse_point_angle_diff = 0
        self.fuse_point_angle_diff_square = 0

        self.fuse_all_kinds_angle_diff = [0, 0, 0]
        self.fuse_all_kinds_angle_diff_square = [0, 0, 0]

        self.fuse_all_kinds_right = [0, 0, 0]
        self.fuse_all_kinds_label = [0, 0, 0]
        self.fuse_all_kinds_pred = [0, 0, 0]

        self.fuse_total_occupancy_right = 0
        self.fuse_total_occupancy_label = 0

        self.fuse_total_classification_right = 0
        self.fuse_total_classification_label = 0

        self.fuse_point_position_diff = 0
        self.fuse_point_position_diff_square = 0
        self.fuse_visiable_point_num = 0

        self.fuse_all_kinds_position_diff = [0, 0, 0]
        self.fuse_all_kinds_position_diff_square = [0, 0, 0]
        self.fuse_all_kinds_point_num = [0, 0, 0]

        self.fuse_all_scene_right = [0, 0]
        self.fuse_all_scene_label = [0, 0]
        self.fuse_all_scene_pred = [0, 0]

    def update(self, result_dict):
        labels = result_dict["labels"]
        preds_label = json.loads(result_dict["preds_label"])
        scene_list = result_dict["img_scene"]
        global_labels = [i[0] for i in labels]
        local_labels = [i[1] for i in labels]
        self.update_global(
            labels=global_labels, preds=preds_label["preds_global"]
        )
        self.update_local(
            labels=local_labels, preds=preds_label["preds_local"]
        )
        self.update_fuse(
            global_labels,
            local_labels,
            preds_label["preds_slots"],
            scene_list,
        )

    def update_point_position_diff(
        self, location_result, location_label, slot_type_label, point_label
    ):
        """Update point position diff.

        Args:
            location_result: np.array, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            location_label: np.array, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            point_label: np.array, [p1_type, p2_type, p3_type, p4_type]
        """
        location_diff = location_result[:2] - location_label[:2]
        location_diff = np.sqrt(np.sum(location_diff ** 2, axis=-1))
        self.all_point_position_diff.extend(location_diff.tolist())
        self.all_kinds_point_position_diff[
            int(slot_type_label.tolist())
        ].extend(location_diff.tolist())
        self.fuse_point_position_diff += np.sum(
            location_diff * point_label[:2]
        )
        self.fuse_point_position_diff_square += np.sum(
            (location_diff ** 2) * point_label[:2]
        )
        self.fuse_all_kinds_position_diff[
            int(slot_type_label.tolist())
        ] += np.sum(location_diff * point_label[:2])
        self.fuse_all_kinds_position_diff_square[
            int(slot_type_label.tolist())
        ] += np.sum((location_diff ** 2) * point_label[:2])
        self.fuse_visiable_point_num += point_label[:2].sum()
        self.fuse_all_kinds_point_num[
            int(slot_type_label.tolist())
        ] += point_label[:2].sum()

    def update_fuse(self, global_labels, local_labels, preds, scene_list):
        """Update fuse msg.

        Args:
            global_label: batch * n * 14.
                [[centerx, centery, slot_type, slot_vec_x, slot_vec_y, occupancy,  # noqa
                x1, y1, x2, y2, x3, y3, x4, y4,],
                [], [], ...
                ]
            local_label: batch * n * 21
                [[x1, y1, x2, y2, x3, y3, x4, y4,
                p1_vec_x, p1_vec_y, p2_vec_x, p2_vec_y, p3_vec_x, p3_vec_y, p4_vec_x, p4_vec_y,  # noqa
                p1_type, p2_type, p3_type, p4_type, slot_type],
                [], [],...
                ]
            preds: batch* (N * 28)
                [[x0,y0,x1,y1,x2,y2,x3,y3, type_j0, type_j1, type_j2,
                type_j3,visibility_j0, visibility_j1, visibility_j2,
                visibility_j3, occupancy of slot, type of slot,
                sin_j0, cos_j0, sin_j1, cos_j1, sin_j2,
                cos_j2, sin_j3, cos_j3, sin_slot, cos_slot],
                [],[]...
                ]
        """
        result_dict = {}
        result_dict.setdefault("visualized_result", [])
        result_dict.setdefault("pred_slots", [])
        for batch_idx in range(len(preds)):
            label = global_labels[batch_idx]
            label = np.array(label)
            local_label = local_labels[batch_idx]
            local_label = np.array(local_label)
            det_results = preds[batch_idx]
            if len(label) == 0:
                continue
            localtions_label = label[:, 6:]
            point_label = local_label[:, -5:-1]
            slot_angle_label = torch.from_numpy(label[:, 3:5])

            occupancy_labels = label[:, 5]
            slot_type_labels = label[:, 2]

            label_len, pred_len = 0, len(det_results)
            for slot_label in label:
                if int(slot_label[2]) != -1:
                    label_len += 1
                    self.fuse_all_kinds_label[int(slot_label[2])] += 1
            self.fuse_total_slot_label += label_len
            self.fuse_total_slot_pred += pred_len

            self.fuse_all_scene_label[scene_list[batch_idx]] += label_len
            self.fuse_all_scene_pred[scene_list[batch_idx]] += pred_len

            self.fuse_total_classification_label += label_len
            self.fuse_total_occupancy_label += label_len
            if not pred_len:
                continue
            localtions_label = localtions_label.reshape(-1, 4, 2)
            location_results = np.array(det_results)
            location_results = location_results[:, 0:8]
            location_results = location_results.reshape(-1, 4, 2)

            occupancy_results = np.array(det_results)[:, 16]
            slot_type_results = np.array(det_results)[:, 17]

            angle_results = torch.tensor(det_results)[:, 26:28]
            gt_recall_list = []
            for index, location_result in enumerate(location_results):
                poly1 = [tuple(i) for i in location_result]
                self.fuse_all_kinds_pred[int(det_results[index][17])] += 1
                occupancy_result = occupancy_results[index]
                slot_type_result = slot_type_results[index]
                for idx, localtion_label in enumerate(localtions_label):
                    poly2 = [tuple(i) for i in localtion_label]
                    try:
                        iou = self.shape_tool.get_iou(poly1, poly2)
                    except Exception:
                        iou = 0
                    occupancy_label = occupancy_labels[idx]
                    slot_type_label = slot_type_labels[idx]

                    if iou > self.iou_threshold:
                        if slot_type_label == -1:
                            self.fuse_total_slot_pred -= 1
                            self.fuse_all_scene_pred[
                                scene_list[batch_idx]
                            ] -= 1
                            self.fuse_all_kinds_pred[
                                int(det_results[index][17])
                            ] -= 1
                        elif slot_type_result == slot_type_label:
                            if idx in gt_recall_list:
                                continue
                            gt_recall_list.append(idx)
                            self.update_point_position_diff(
                                location_result,
                                localtion_label,
                                slot_type_label,
                                point_label[idx],
                            )
                            angle_diff = F.cosine_similarity(
                                angle_results[index],
                                slot_angle_label[idx],
                                dim=0,
                            )
                            angle_diff_value = angle_diff.tolist()
                            if abs(1 - angle_diff_value) < 1e-5:
                                angle_diff = torch.tensor(1 - 1e-5)
                            elif abs(-1 - angle_diff_value) < 1e-5:
                                angle_diff = torch.tensor(-1 + 1e-5)
                            angle_diff = torch.acos(angle_diff) * 180 / pi
                            self.all_slot_angle_diff.append(
                                angle_diff.tolist()
                            )
                            self.all_kinds_slot_angle_diff[
                                int(slot_type_label.tolist())
                            ].append(angle_diff.tolist())
                            self.fuse_point_angle_diff += angle_diff.tolist()
                            self.fuse_point_angle_diff_square += (
                                angle_diff.tolist() ** 2
                            )
                            self.fuse_all_kinds_angle_diff[
                                int(slot_type_label.tolist())
                            ] += angle_diff.tolist()
                            self.fuse_all_kinds_angle_diff_square[
                                int(slot_type_label.tolist())
                            ] += (angle_diff.tolist() ** 2)
                            self.fuse_total_slot_right += 1
                            self.fuse_all_scene_right[
                                scene_list[batch_idx]
                            ] += 1
                            self.fuse_all_kinds_right[
                                int(slot_type_label.tolist())
                            ] += 1

                            if occupancy_result == occupancy_label:
                                self.fuse_total_occupancy_right += 1
                            self.fuse_total_classification_right += 1
                            break
                    elif slot_type_label == -1:
                        try:
                            poly1_area = self.shape_tool.get_area(poly1)
                            poly2_area = self.shape_tool.get_area(poly2)
                            poly_insec = self.shape_tool.get_intersection(
                                poly1, poly2
                            )
                        except Exception:
                            continue
                        if (
                            poly_insec
                            > min(poly1_area, poly2_area)
                            * self.ignore_threshold
                        ):
                            self.fuse_total_slot_pred -= 1
                            self.fuse_all_scene_pred[
                                scene_list[batch_idx]
                            ] -= 1
                            self.fuse_all_kinds_pred[
                                int(det_results[index][17])
                            ] -= 1
                            break

    def update_global(self, labels, preds):
        for batch_idx in range(len(preds)):
            label = labels[batch_idx]
            if len(label) == 0:
                continue
            label = torch.tensor(label)
            localtions_label = label[:, 6:]
            slot_type_label = label[:, 2]
            slot_type_label = torch.where(slot_type_label != -1, 1, 0)
            len_junctions = 4
            for idx in range(len_junctions):
                self.global_total_point_label[idx] += torch.sum(
                    slot_type_label
                )
            slot_info_dict = preds[batch_idx]
            localtions_list = slot_info_dict[
                "slot_junctions_locations_list"
            ]  # [[[y1,x1],[y2,x2],[y3,x3],[y4,x4]],...]
            localtions_pred = torch.tensor(localtions_list)
            pred_len, label_len = len(localtions_pred), len(localtions_label)
            for idx in range(len_junctions):
                self.global_total_point_pred[idx] += pred_len
            if pred_len:
                slot_type_label = slot_type_label.unsqueeze(dim=0).unsqueeze(
                    dim=-1
                )
                slot_type_label = slot_type_label.repeat(pred_len, 1, 4)
                slot_type_label = slot_type_label.reshape(-1, 4)
                # slot_type_label: pred_len * label_len * 4
                localtions_pred = localtions_pred.view(
                    localtions_pred.shape[0], -1
                )
                localtions_pred = localtions_pred.unsqueeze(dim=1)
                localtions_pred = localtions_pred.repeat(1, label_len, 1)
                localtions_label = localtions_label.unsqueeze(dim=0)
                localtions_label = localtions_label.repeat(pred_len, 1, 1)
                diff_value = localtions_pred - localtions_label
                diff_value = diff_value.reshape(-1, 4, 2)
                diff_value = torch.sqrt(torch.sum(diff_value ** 2, dim=-1))
                point_right = torch.where(
                    diff_value < self.global_max_distance, 1, 0
                )
                for idx in range(len_junctions):
                    point_right_p = torch.zeros_like(point_right)
                    point_right_p[:, idx] = point_right[:, idx]
                    all_point_right = torch.sum(point_right_p)
                    point_right_p *= slot_type_label
                    real_point_right = torch.sum(point_right_p)
                    self.global_total_point_right[
                        idx
                    ] += real_point_right.tolist()
                    self.global_total_point_right_distance[idx] += torch.sum(
                        diff_value * point_right_p
                    ).tolist()
                    self.global_total_point_right_distance_square[
                        idx
                    ] += torch.sum(
                        diff_value * diff_value * point_right_p
                    ).tolist()
                    self.global_total_point_pred[idx] -= (
                        all_point_right.tolist() - real_point_right.tolist()
                    )

    def update_local(self, labels, preds):

        for batch_idx in range(len(preds)):
            label = labels[batch_idx]
            label = torch.tensor(labels[batch_idx])
            if len(label) == 0:
                continue
            (
                j0_dict,
                j1_dict,
                j2_dict,
                j3_dict,
            ) = preds[batch_idx]
            j0_locations, j1_locations, j2_locations, j3_locations = (
                j0_dict["j0_location_list"],
                j1_dict["j1_location_list"],
                j2_dict["j2_location_list"],
                j3_dict["j3_location_list"],
            )
            (
                j0_sline_angles,
                j1_sline_angles,
                j2_sline_angles,
                j3_sline_angles,
            ) = (
                j0_dict["j0_sline_angle_list"],
                j1_dict["j1_sline_angle_list"],
                j2_dict["j2_sline_angle_list"],
                j3_dict["j3_sline_angle_list"],
            )

            for idx, (locations, sline_angles) in enumerate(
                zip(
                    [j0_locations, j1_locations, j2_locations, j3_locations],
                    [
                        j0_sline_angles,
                        j1_sline_angles,
                        j2_sline_angles,
                        j3_sline_angles,
                    ],
                )
            ):
                locations_label = label[:, 2 * idx : 2 * idx + 2]
                locations_label = locations_label[label[:, 16 + idx] != 0]
                angles_label = label[:, 2 * idx + 8 : 2 * idx + 10]
                angles_label = angles_label[label[:, 16 + idx] != 0]
                pred_len, label_len = len(locations), len(locations_label)
                if pred_len and label_len:
                    locations = torch.tensor(
                        locations
                    )  # size:n*2 [[x, y],[],[]...]
                    locations = torch.unsqueeze(locations, dim=1)
                    locations = locations.repeat(1, label_len, 1)
                    locations_label = torch.unsqueeze(
                        locations_label, dim=0
                    )  # size:n*2 [[x, y],[],[]...]
                    locations_label = locations_label.repeat(pred_len, 1, 1)
                    diff_value = locations - locations_label
                    diff_value = torch.sqrt(torch.sum(diff_value ** 2, dim=-1))
                    point_right = torch.where(
                        diff_value < self.local_max_distance, 1, 0
                    )

                    sline_angles = torch.tensor(sline_angles)
                    sline_angles = torch.unsqueeze(sline_angles, dim=1)
                    sline_angles = sline_angles.repeat(1, label_len, 1)
                    angles_label = torch.unsqueeze(angles_label, dim=0)
                    angles_label = angles_label.repeat(pred_len, 1, 1)
                    similarity = F.cosine_similarity(
                        sline_angles, angles_label, dim=-1
                    )
                    inter_angle = torch.acos(similarity)
                    inter_angle = inter_angle * point_right * 180 / pi
                    self.local_point_angle_diff[idx] += torch.sum(
                        inter_angle
                    ).tolist()
                    self.local_point_angle_diff_square[idx] += torch.sum(
                        inter_angle * inter_angle
                    ).tolist()
                    self.local_point_position_diff[idx] += torch.sum(
                        torch.abs(diff_value) * point_right
                    ).tolist()
                    self.local_point_position_diff_square[idx] += torch.sum(
                        torch.abs(diff_value)
                        * torch.abs(diff_value)
                        * point_right
                    ).tolist()
                    for i in range(label_len):
                        label_i_matchPoints = torch.sum(
                            point_right[:, i]
                        ).tolist()
                        if label_i_matchPoints >= 1:
                            self.local_total_point_right_GreaterEqual1Point[
                                idx
                            ] += 1
                        if label_i_matchPoints > 1:
                            self.local_total_point_right_Greater1Point[
                                idx
                            ] += 1
                            if (
                                label_i_matchPoints
                                > self.oneLabelPoint_nearby_MaxPoints
                            ):
                                self.oneLabelPoint_nearby_MaxPoints = (
                                    label_i_matchPoints
                                )
                self.local_total_point_label[idx] += label_len
                self.local_total_point_pred[idx] += pred_len

    def get(self):
        if self.get_log:
            global_names, global_values = self.get_global()
            summary_str = "~~~~ PSD global Summary metrics ~~~~\n"
            summary_str += "Summary:\n"
            for i in range(len(global_names)):
                line_format = "{:<30}\t{:>10}\n"
                summary_str += line_format.format(
                    global_names[i], global_values[i]
                )
            logger.warning(summary_str)

            local_names, local_values = self.get_local()
            summary_str = "~~~~ PSD local Summary metrics ~~~~\n"
            summary_str += "Summary:\n"
            for i in range(len(local_names)):
                line_format = "{:<30}\t{:>10}\n"
                summary_str += line_format.format(
                    local_names[i], local_values[i]
                )
            logger.warning(summary_str)

        total_names, total_values = [], []
        gap_list = []
        idx = 0
        names, values = self.get_acc()
        total_names.extend(names)
        total_values.extend(values)
        idx += len(names) - 1
        gap_list.append(idx)

        names, values = self.get_recall()
        total_names.extend(names)
        total_values.extend(values)
        idx += len(names)
        gap_list.append(idx)

        names, values = self.get_position()
        total_names.extend(names)
        total_values.extend(values)
        idx += len(names)
        gap_list.append(idx)

        names, values = self.get_angle()
        total_names.extend(names)
        total_values.extend(values)
        idx += len(names)
        gap_list.append(idx)

        if self.get_log:
            summary_str = "~~~~ PSD Summary metrics ~~~~\n"
            summary_str += "Summary:\n"
            for i in range(len(total_names)):
                line_format = "{:<30}\t{:>10.3f}\n"
                summary_str += line_format.format(
                    total_names[i], total_values[i]
                )
                if i in gap_list:
                    summary_str += "\n"
            logger.warning(summary_str)
            total_values = [torch.tensor(i).cuda() for i in total_values]
        return total_names, total_values

    def get_global(self):

        point_acc = (
            0
            if not sum(self.global_total_point_pred[:2])
            else sum(self.global_total_point_right[:2])
            / float(sum(self.global_total_point_pred[:2]))
        )
        point_recall = (
            0
            if not sum(self.global_total_point_label[:2])
            else sum(self.global_total_point_right[:2])
            / float(sum(self.global_total_point_label[:2]))
        )
        point_mean_dist = (
            0
            if not sum(self.global_total_point_right[:2])
            else sum(self.global_total_point_right_distance[:2])
            / sum(self.global_total_point_right[:2])
        )
        point_mean_dist_square = (
            0
            if not sum(self.global_total_point_right[:2])
            else sum(self.global_total_point_right_distance_square[:2])
            / sum(self.global_total_point_right[:2])
        )
        values = [
            point_acc,
            point_recall,
            point_mean_dist,
            point_mean_dist_square,
        ]
        return self.names["global"], values

    def get_local(self):

        point_acc = (
            0
            if not sum(self.local_total_point_pred[:2])
            else sum(self.local_total_point_right_GreaterEqual1Point[:2])
            / sum(self.local_total_point_pred[:2])
        )
        point_recall_GreatEqual1Pnt = (
            0
            if not sum(self.local_total_point_label[:2])
            else sum(self.local_total_point_right_GreaterEqual1Point[:2])
            / sum(self.local_total_point_label[:2])
        )
        point_recall_Great1Pnt = (
            0
            if not sum(self.local_total_point_label[:2])
            else sum(self.local_total_point_right_Greater1Point[:2])
            / sum(self.local_total_point_label[:2])
        )
        position_diff = (
            0
            if not sum(self.local_total_point_right_GreaterEqual1Point[:2])
            else sum(self.local_point_position_diff[:2])
            / sum(self.local_total_point_right_GreaterEqual1Point[:2])
        )
        position_diff_square = (
            0
            if not sum(self.local_total_point_right_GreaterEqual1Point[:2])
            else sum(self.local_point_position_diff_square[:2])
            / sum(self.local_total_point_right_GreaterEqual1Point[:2])
        )
        values = [
            point_acc,
            point_recall_GreatEqual1Pnt,
            point_recall_Great1Pnt,
            position_diff,
            position_diff_square,
        ]
        return self.names["local"], values

    def get_2sigma(self):

        all_point_position_diff = torch.tensor(self.all_point_position_diff)
        all_point_2sigma, _ = torch.topk(
            all_point_position_diff,
            int(0.05 * len(self.all_point_position_diff)),
        )
        all_point_2sigma = (
            torch.tensor(0)
            if not len(all_point_2sigma)
            else all_point_2sigma[-1]
        )

        all_slot_angle_diff = torch.tensor(self.all_slot_angle_diff)
        all_angle_2sigma, _ = torch.topk(
            all_slot_angle_diff, int(0.05 * len(self.all_slot_angle_diff))
        )
        all_angle_2sigma = (
            torch.tensor(0)
            if not len(all_angle_2sigma)
            else all_angle_2sigma[-1]
        )

        all_kinds_point_position_diff = [
            torch.tensor(i) for i in self.all_kinds_point_position_diff
        ]
        all_kinds_slot_angle_diff = [
            torch.tensor(i) for i in self.all_kinds_slot_angle_diff
        ]

        all_kinds_point_position_2sigma = [
            torch.topk(i, int(0.05 * len(i)))[0]
            for i in all_kinds_point_position_diff
        ]
        all_kinds_point_position_2sigma = [
            torch.tensor(0) if not len(i) else i[-1]
            for i in all_kinds_point_position_2sigma
        ]

        all_kinds_slot_angle_2sigma = [
            torch.topk(i, int(0.05 * len(i)))[0]
            for i in all_kinds_slot_angle_diff
        ]
        all_kinds_slot_angle_2sigma = [
            torch.tensor(0) if not len(i) else i[-1]
            for i in all_kinds_slot_angle_2sigma
        ]

        values = []
        values.append(all_point_2sigma)
        values.append(all_angle_2sigma)
        values.extend(all_kinds_point_position_2sigma)
        values.extend(all_kinds_slot_angle_2sigma)
        return self.names["2sigma"], values

    def get_acc(self):

        slot_acc = (
            0
            if not self.fuse_total_slot_pred
            else self.fuse_total_slot_right / float(self.fuse_total_slot_pred)
        )
        scene_acc = []
        for i in range(2):
            scene_acc.append(
                0
                if not self.fuse_all_scene_pred[i]
                else self.fuse_all_scene_right[i] / self.fuse_all_scene_pred[i]
            )
        kind_acc = []
        for i in range(3):
            kind_acc.append(
                0
                if not self.fuse_all_kinds_pred[i]
                else self.fuse_all_kinds_right[i] / self.fuse_all_kinds_pred[i]
            )
        slot_occupancy_acc = (
            0
            if not self.fuse_total_occupancy_label
            else self.fuse_total_occupancy_right
            / float(self.fuse_total_occupancy_label)
        )
        slot_classification_acc = (
            0
            if not self.fuse_total_classification_label
            else self.fuse_total_classification_right
            / float(self.fuse_total_classification_label)
        )
        values = [slot_acc]
        values.extend(scene_acc)
        values.extend(kind_acc)
        values.append(slot_occupancy_acc)
        values.append(slot_classification_acc)
        return self.names["acc"], values

    def get_recall(self):

        slot_recall = (
            0
            if not self.fuse_total_slot_label
            else self.fuse_total_slot_right / float(self.fuse_total_slot_label)
        )
        scene_recall = []
        for i in range(2):
            scene_recall.append(
                0
                if not self.fuse_all_scene_label[i]
                else self.fuse_all_scene_right[i]
                / self.fuse_all_scene_label[i]
            )
        kind_recall = []
        for i in range(3):
            kind_recall.append(
                0
                if not self.fuse_all_kinds_label[i]
                else self.fuse_all_kinds_right[i]
                / self.fuse_all_kinds_label[i]
            )
        values = [slot_recall]
        values.extend(scene_recall)
        values.extend(kind_recall)
        return self.names["recall"], values

    def get_position(self):

        slot_position_diff_mean = (
            0
            if not self.fuse_visiable_point_num
            else self.fuse_point_position_diff
            / float(self.fuse_visiable_point_num)  # noqa
        )
        kind_position_diff_mean = [
            0 if not j else i / float(j)
            for i, j in zip(
                self.fuse_all_kinds_position_diff,
                self.fuse_all_kinds_point_num,
            )
        ]
        slot_position_diff_variance = (
            0
            if not self.fuse_visiable_point_num
            else sqrt(
                self.fuse_point_position_diff_square
                / float(self.fuse_visiable_point_num)
                - slot_position_diff_mean ** 2
            )
        )
        kind_position_diff_variance = [
            0 if not j else sqrt(i / float(j) - k ** 2)
            for i, j, k in zip(
                self.fuse_all_kinds_position_diff_square,
                self.fuse_all_kinds_point_num,
                kind_position_diff_mean,
            )
        ]

        all_point_position_diff = torch.tensor(self.all_point_position_diff)
        all_point_2sigma, _ = torch.topk(
            all_point_position_diff,
            int(0.05 * len(self.all_point_position_diff)),
        )
        all_point_2sigma = (
            torch.tensor(0)
            if not len(all_point_2sigma)
            else all_point_2sigma[-1]
        )

        all_kinds_point_position_diff = [
            torch.tensor(i) for i in self.all_kinds_point_position_diff
        ]

        all_kinds_point_position_2sigma = [
            torch.topk(i, int(0.05 * len(i)))[0]
            for i in all_kinds_point_position_diff
        ]
        all_kinds_point_position_2sigma = [
            torch.tensor(0) if not len(i) else i[-1]
            for i in all_kinds_point_position_2sigma
        ]

        values = []
        values.append(slot_position_diff_mean)
        values.extend(kind_position_diff_mean)
        values.append(slot_position_diff_variance)
        values.extend(kind_position_diff_variance)
        values.append(all_point_2sigma)
        values.extend(all_kinds_point_position_2sigma)
        return self.names["position"], values

    def get_angle(self):

        slot_angle_diff_mean = (
            0
            if not self.fuse_total_slot_right
            else self.fuse_point_angle_diff / float(self.fuse_total_slot_right)
        )
        kind_angle_diff_mean = [
            0 if not j else i / float(j)
            for i, j in zip(
                self.fuse_all_kinds_angle_diff, self.fuse_all_kinds_right
            )
        ]
        slot_angle_diff_variance = (
            0
            if not self.fuse_total_slot_right
            else sqrt(
                self.fuse_point_angle_diff_square
                / float(self.fuse_total_slot_right)
                - slot_angle_diff_mean ** 2
            )
        )
        kind_angle_diff_variance = [
            0 if not j else sqrt(i / float(j) - k ** 2)
            for i, j, k in zip(
                self.fuse_all_kinds_angle_diff_square,
                self.fuse_all_kinds_right,
                kind_angle_diff_mean,
            )
        ]
        all_slot_angle_diff = torch.tensor(self.all_slot_angle_diff)
        all_angle_2sigma, _ = torch.topk(
            all_slot_angle_diff, int(0.05 * len(self.all_slot_angle_diff))
        )
        all_angle_2sigma = (
            torch.tensor(0)
            if not len(all_angle_2sigma)
            else all_angle_2sigma[-1]
        )

        all_kinds_slot_angle_diff = [
            torch.tensor(i) for i in self.all_kinds_slot_angle_diff
        ]

        all_kinds_slot_angle_2sigma = [
            torch.topk(i, int(0.05 * len(i)))[0]
            for i in all_kinds_slot_angle_diff
        ]
        all_kinds_slot_angle_2sigma = [
            torch.tensor(0) if not len(i) else i[-1]
            for i in all_kinds_slot_angle_2sigma
        ]

        values = []
        values.append(slot_angle_diff_mean)
        values.extend(kind_angle_diff_mean)
        values.append(slot_angle_diff_variance)
        values.extend(kind_angle_diff_variance)
        values.append(all_angle_2sigma)
        values.extend(all_kinds_slot_angle_2sigma)

        return self.names["angle"], values
