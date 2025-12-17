from collections import defaultdict
from typing import List, Optional, Sequence

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

from hat.core.bev_elevation_utils import (
    calculate_points_on_line,
    compute_vismask,
    get_roi_resize_data,
)
from hat.data.transforms.bev_psd_target import ANCBEVPSDTargetGenerator
from hat.registry import OBJECT_REGISTRY

__all__ = ["ANCBEVParkingRodTargetGenerator"]


@OBJECT_REGISTRY.register
class ANCBEVParkingRodTargetGenerator(object):
    """Target module for bev parkingrod detection.

    Args:
        max_objs: size of parkingrod align to max_objs.
        len_gt_info: length of labels to get parkingrod gt info,
            such as type, coordinates, ignore.
        target_size: The size of parkingrod target, hxw.
        rod_weight_dict: weight of different type of parkingrod.
        num_parkingrod_class: the number of parkingrod type.
        roi_vcs_range: roi reception vcs range.
        dilate_rate: expansion factor for centersampling.
        lidar_blind_spot_range: Represents the vcs range
            for the panda blind spot.
        visable_condition: Store some threshold labels are visible.

            .. code-block:: none

                "visable_threshold": All the points(grids) in parkingrod,
                    if more than 1 points are in the visible area,
                    the label is retained.

        vismask_vcsrange_cfg: Raw offline vismask image size
            as key, and corresponding vcs range as value.
        use_vis_mask: whether to use vismask.
        coord_duplicate_thresh: The center point threshold of duplicate.
            default 0.2m.
        lidar_distance_threshold: Threshold of distance between
            corner point and lidar point.If the slot point is smaller
            than this threshold, the slot point is visible.
        with_psd: whether to train with psd task.
        reweight: Re-weight parameters and near range for regresssion, e.g.,
            [all_scale, near_scale, x0, y0, x1, y1].
    """

    def __init__(
        self,
        max_objs: int,
        len_gt_info: int,
        target_size: tuple,
        roi_vcs_range: tuple,
        rod_weight_dict: dict,
        num_parkingrod_class: int,
        resolution_m_per_grid: int,
        dilate_rate: int,
        use_vis_mask: bool,
        visable_condition: dict,
        with_psd: bool,
        vismask_vcsrange_cfg: Optional[Sequence[float]] = None,
        lidar_distance_threshold: float = 1.4,
        res_key: str = "bev_parkingrod_obj",
        reweight: List[float] = None,
    ):
        super().__init__()
        self.max_objs = max_objs
        self.len_gt_info = len_gt_info
        self.target_size = target_size
        self.roi_vcs_range = roi_vcs_range
        self.rod_weight_dict = rod_weight_dict
        self.num_parkingrod_class = num_parkingrod_class
        self.resolution_m_per_grid = resolution_m_per_grid
        self.dilate_rate = dilate_rate
        self.lidar_distance_threshold = lidar_distance_threshold
        self.use_vis_mask = use_vis_mask
        self.visable_condition = visable_condition
        self.vismask_vcsrange_cfg = vismask_vcsrange_cfg
        self.with_psd = with_psd
        self.lidar_blind_spot_range = {
            "Normal": (-4.0, -4.0, 7.0, 4.0),
            "Parallel": (-5.0, -5.0, 8.0, 5.0),
        }
        self.res_key = res_key
        self.rod_type_dict = {
            "limited_strip": 0,
            "limited_rod": 1,
            "other": 2,
        }
        self.slot_type_dict = {
            "Vertical": 0,
            "Parallel": 1,
            "Oblique": 2,
            "Stereo": 0,
            "Not_slot": -1,
            "Not_normal_slot": -1,
        }
        self.slot_occupancy_dict = {"Y": 0, "N": 1}
        self.slot_enable_dict = {"N": 0, "Y": 1}
        self.ignore_dict = {"Y": 0, "N": 1}
        self.scene_dict = {"Indoors": 0, "Outdoors": 1}
        self.coord_duplicate_thresh = 0.2
        assert reweight is None or len(reweight) == 6
        self.reweight = reweight

    def gen_rod_center(self, point_data_list):
        point_data_list = np.array(point_data_list)
        center = point_data_list.mean(0)
        return list(center)

    def vcs_to_bev(
        self,
        data_vcs: np.array,
        roi_vcs_range: tuple,
        resolution_m_per_grid: np.array,
    ) -> np.array:
        """Transfer vcs coordinate to bev coordinate.

        Args:
            data_vcs: vcs data of point. n*2
                [[x0, y0], [x1, y1], [x2, y2], ...]
            roi_vcs_range: roi_vcs_range. (bottom, right, top, left)
            resolution_m_per_grid: pixel per grid.

        Returns:
            data_bev: bev data of point. n*2
                [[h0, w0], [h1, w1], [h2, w2], ...]
        """
        data_bev = np.zeros_like(data_vcs)
        data_bev[:, 1] = (
            roi_vcs_range[2] - data_vcs[:, 0]
        ) / resolution_m_per_grid
        data_bev[:, 0] = (
            roi_vcs_range[3] - data_vcs[:, 1]
        ) / resolution_m_per_grid
        return data_bev

    def gt_filter_cover(
        self, rod_info_list: dict, slot_info_list: dict, has_vismask: bool
    ) -> dict:
        """Filter parkingrod depending on slot.

        Args:
            rod_info_list: label msg for parkingrod.
            slot_info_list: label msg for slot.
            rod_info_list: label msg for parkingrod.

                .. code-block:: none

                    [{
                        "attrs":{
                            "Parking_rod_type": "Y",
                            "ignore": "N",
                        },
                        "data_bev":[[],[]],
                    }, {}, {}]

            slot_info_list: label msg for slot.

                .. code-block:: none

                    [{
                        "attrs":{
                            "Enable":"Y",
                            "Occupancy": "N",
                            "Slot_type": "Vertical",
                            "ignore":"N",
                        },
                        "data_bev":[[],[]],

                    }, {}, {}]

        Returns:
            rod_info_list: update the ignore attrs depending on slot.
        """
        for rod_info in rod_info_list:
            rod_data_bev = np.array(rod_info["data_bev"])
            rod_point_data_list = rod_data_bev[:, :2].tolist()
            rod_point_0 = rod_point_data_list[0]
            rod_point_1 = rod_point_data_list[1]
            is_parkingrod_valid = False

            for slot_info in slot_info_list:
                slot_data_bev = np.array(slot_info["data_bev"])
                slot_occupancy = slot_info["attrs"]["Occupancy"]
                slot_ignore = slot_info["attrs"]["ignore"]
                slot_point_data_list = slot_data_bev[:, :2].tolist()
                slot_data_vcs_list = np.array(slot_info["data_vcs"]).tolist()
                slot_point_0 = slot_data_vcs_list[0]
                slot_point_1 = slot_data_vcs_list[1]
                slot_point_2 = slot_data_vcs_list[2]
                slot_point_3 = slot_data_vcs_list[3]
                slot_01center_int = [
                    int((slot_point_0[0] + slot_point_1[0]) / 2),
                    int((slot_point_0[1] + slot_point_1[1]) / 2),
                ]
                rod_line = LineString([rod_point_0, rod_point_1])
                slot_polygon = Polygon(slot_point_data_list)
                intersection = rod_line.intersection(slot_polygon)
                intersection_length = intersection.length
                inner_ratio = intersection_length / rod_line.length
                if inner_ratio > 0.7:
                    is_parkingrod_valid = True
                    rod_info["slot_luid"] = slot_info["luid"]
                    rod_info["slot_point"] = [
                        slot_point_0,
                        slot_point_1,
                        slot_point_2,
                        slot_point_3,
                    ]  # 01角点的xy
                    rod_info["slot_01center"] = slot_01center_int
                    if not has_vismask:
                        if (
                            self.ignore_dict[slot_ignore]
                            == self.ignore_dict["Y"]
                        ):
                            rod_info["attrs"]["ignore"] = "Y"
                    if (
                        self.slot_occupancy_dict[slot_occupancy]
                        == self.slot_occupancy_dict["Y"]
                    ):
                        rod_info["attrs"]["ignore"] = "Y"
                    break
                else:
                    continue
            if is_parkingrod_valid is False:
                rod_info["attrs"]["ignore"] = "Y"
                rod_info["slot_point"] = [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
                rod_info["slot_01center"] = [0.0, 0.0]
                rod_info["slot_luid"] = "Not_luid"
        return rod_info_list

    def merge_parking_strip(self, rod_info_list: dict) -> dict:
        """Merge strips in the same slot.

        Args:
            rod_info_list: label msg for parkingrod.

                .. code-block:: json

                    [{
                        "attrs":{
                            "Parking_rod_type": "Y",
                            "ignore": "N",

                        },
                        "data_bev":[[],[]],
                        "data_vcs":[[],[]],
                        "slot_point": [[], []],
                        "slot_01center": [],

                    }, {}, {}]

        Returns:
            merged_rod_info_list: only limited-rod type.
        """
        merged_rod_info_list = []
        merge_rod_dict = defaultdict(list)
        # The other two properties except the limited-strip are processed
        for rod_info in rod_info_list:
            if rod_info["attrs"]["Parking_rod_type"] == "limited_rod":
                merged_rod_info_list.append(rod_info)
            if rod_info["attrs"]["Parking_rod_type"] == "Other":
                rod_info["attrs"]["ignore"] = "Y"
                merged_rod_info_list.append(rod_info)
        # Based on whether to merge in the same parking-slot.
        for rod_info in rod_info_list:
            if rod_info["attrs"]["Parking_rod_type"] != "limited_rod":
                merge_rod_dict[rod_info["slot_luid"]].append(rod_info)
        for k in merge_rod_dict.keys():
            # The processing method of less than two limited-strip in one slot.
            if len(merge_rod_dict[k]) < 2:
                merge_rod_dict[k][0]["attrs"][
                    "Parking_rod_type"
                ] = "limited_rod"
                merged_rod_info_list.append(merge_rod_dict[k][0])
                continue
            # The processing method of more than two limited-strip in one slot,
            # and if it is not inside the slot.
            if len(merge_rod_dict[k]) > 2 or k == "Not_luid":
                for rod_info_nimiety in merge_rod_dict[k]:
                    rod_info_nimiety["attrs"][
                        "Parking_rod_type"
                    ] = "limited_rod"
                    rod_info_nimiety["attrs"]["ignore"] = "Y"
                    merged_rod_info_list.append(rod_info_nimiety)
                continue
            rod_info_idx = merge_rod_dict[k][0]
            rod_info_jdx = merge_rod_dict[k][1]
            rod_point_idx_0 = rod_info_idx["data_vcs"][0]
            rod_point_idx_1 = rod_info_idx["data_vcs"][1]
            rod_point_jdx_0 = rod_info_jdx["data_vcs"][0]
            rod_point_jdx_1 = rod_info_jdx["data_vcs"][1]
            distance_1 = np.linalg.norm(
                np.array(rod_point_idx_0) - np.array(rod_point_jdx_1)
            )
            distance_2 = np.linalg.norm(
                np.array(rod_point_idx_1) - np.array(rod_point_jdx_0)
            )
            if distance_1 >= distance_2:
                rod_info_idx["data_vcs"][0] = rod_point_idx_0
                rod_info_idx["data_vcs"][1] = rod_point_jdx_1
            else:
                rod_info_idx["data_vcs"][0] = rod_point_jdx_0
                rod_info_idx["data_vcs"][1] = rod_point_idx_1
            rod_info_idx["attrs"]["Parking_rod_type"] = "limited_rod"
            if (
                rod_info_idx["attrs"]["ignore"] == "Y"
                or rod_info_jdx["attrs"]["ignore"] == "Y"
            ):
                rod_info_idx["attrs"]["ignore"] = "Y"
            else:
                rod_info_idx["attrs"]["ignore"] = "N"
            merged_rod_info_list.append(rod_info_idx)
        return merged_rod_info_list

    def label_format(self, gt_info: dict):
        """
        Convert label to annotation.

        Args:
            gt_info: label msg.

        Returns:
            rod_label: n x self.len_gt_info.

            ignore: [[x0, y0, x1, y1, slot_x0, slot_y0, slot_x1, slot_y1,
                slot_x2, slot_y2, slot_x3, slot_y3, 0.0, 0.0, 4,
                type, ignore], [], [], ...].

            non_ignore: [[centerx, centery, rod_type,
                x0, y0, x1, y1,
                slot_x0, slot_y0, slot_x1, slot_y1,
                slot_x2, slot_y2, slot_x3, slot_y3,
                type, ignore], [], [], ...]

        """
        total_rod_label = []
        img_scene = self.scene_dict["Indoors"]
        has_vismask = False
        if self.use_vis_mask and "bev_occlusion_mask" in gt_info.keys():
            if gt_info["bev_occlusion_mask"]:
                has_vismask = True
        if (
            "Parking_rod" not in gt_info["gt_bev_parking_obj"].keys()
            or "Slot" not in gt_info["gt_bev_parking_obj"].keys()
        ):
            total_rod_label = np.zeros(
                (self.max_objs, self.len_gt_info), dtype=np.float32
            )
            return [total_rod_label, img_scene]
        if len(gt_info["gt_bev_parking_obj"]["Parking_rod"]) == 0:
            total_rod_label = np.zeros(
                (self.max_objs, self.len_gt_info), dtype=np.float32
            )
            return [total_rod_label, img_scene]

        slot_info_list = gt_info["gt_bev_parking_obj"]["Slot"]
        rod_info_list = gt_info["gt_bev_parking_obj"]["Parking_rod"]
        rod_info_list = self.gt_filter_cover(
            rod_info_list, slot_info_list, has_vismask
        )
        rod_info_list = self.merge_parking_strip(rod_info_list)

        if self.use_vis_mask:
            gt_info["gt_bev_parking_obj"]["Parking_rod"] = rod_info_list
            gt_info = self.set_ignore_from_vismask(gt_info)
            rod_info_list = gt_info["gt_bev_parking_obj"]["Parking_rod"]

        for rod_info in rod_info_list:
            rod_label = []
            rod_type = rod_info["attrs"]["Parking_rod_type"]
            ignore_type = rod_info["attrs"]["ignore"]
            rod_data_vcs = np.array(rod_info["data_vcs"])
            rod_data_vcs_list = rod_data_vcs[:, :2].tolist()
            slot01_data_vcs = np.array(rod_info["slot_point"][:2])
            slot23_data_vcs = np.array(rod_info["slot_point"][2:])
            if len(rod_data_vcs_list) != 2:
                continue
            # get roi_vcs_range GT
            gt_in_range = True
            for data in rod_data_vcs_list:
                x_min_valid = data[0] > self.roi_vcs_range[0]
                x_max_valid = data[0] < self.roi_vcs_range[2]
                y_min_valid = data[1] > self.roi_vcs_range[1]
                y_max_valid = data[1] < self.roi_vcs_range[3]
                in_range_x = np.logical_and(x_min_valid, x_max_valid)
                in_range_y = np.logical_and(y_min_valid, y_max_valid)
                point_in_range = np.logical_and(in_range_x, in_range_y)
                gt_in_range = np.logical_and(gt_in_range, point_in_range)
            if gt_in_range:
                rod_data_bev = self.vcs_to_bev(
                    rod_data_vcs,
                    self.roi_vcs_range,
                    self.resolution_m_per_grid,
                )
                slot01_data_bev = self.vcs_to_bev(
                    slot01_data_vcs,
                    self.roi_vcs_range,
                    self.resolution_m_per_grid,
                )
                slot23_data_bev = self.vcs_to_bev(
                    slot23_data_vcs,
                    self.roi_vcs_range,
                    self.resolution_m_per_grid,
                )
                point_data_list = rod_data_bev.tolist()
                slot01_data_list = slot01_data_bev.tolist()
                slot23_data_list = slot23_data_bev.tolist()

                if self.ignore_dict[ignore_type] == self.ignore_dict["Y"]:
                    for point in point_data_list:
                        rod_label.extend([float(i) for i in point[:2]])
                    len_label = len(rod_label)
                    for point in slot01_data_list:
                        rod_label.extend([float(i) for i in point[:2]])
                    for point in slot23_data_list:
                        rod_label.extend([float(i) for i in point[:2]])
                    rod_label.extend(
                        [
                            0.0
                            for _ in range(
                                self.len_gt_info - len(rod_label) - 2
                            )
                        ]
                    )
                    rod_label[-1] = len_label
                else:
                    center = self.gen_rod_center(point_data_list)
                    center = [float(i) for i in center[:2]]
                    rod_label.extend(center)
                    rod_label.append(self.rod_type_dict[rod_type])
                    for point in point_data_list:
                        rod_label.extend([float(i) for i in point[:2]])
                    for point in slot01_data_list:
                        rod_label.extend([float(i) for i in point[:2]])
                    for point in slot23_data_list:
                        rod_label.extend([float(i) for i in point[:2]])
                rod_label.append(self.rod_type_dict[rod_type])
                rod_label.append(self.ignore_dict[ignore_type])

                assert (
                    len(rod_label) == self.len_gt_info
                ), f"rod_label: {len(rod_label)} :{rod_label}"
                total_rod_label.append(rod_label)

        if len(total_rod_label) == 0:
            total_rod_label = np.zeros(
                (self.max_objs, self.len_gt_info), dtype=np.float32
            )
            return [total_rod_label, img_scene]

        assert (
            len(total_rod_label) < self.max_objs
        ), f"len(total_rod_label): {len(total_rod_label)}, max_objs: {self.max_objs}"  # noqa
        total_rod_label = np.array(total_rod_label, np.float32)
        return [total_rod_label, img_scene]

    def ignore_gt_with_line(
        self,
        data: dict,
        vis_mask: np.ndarray,
    ):
        """Ignore the groundtruth with parkingrod line.

        The methods is by determining the number of points
        in the visible area of the parkingrod line.

        Args:
            data: Store the dictionary of groundtruth.
            vis_mask: Provide a map with visible&invisible regions.

        Returns: label reserved only in the visible area.
        """
        rod_info_list = data["gt_bev_parking_obj"]["Parking_rod"]
        for rod_info in rod_info_list:
            if rod_info["attrs"]["ignore"] == "Y":
                continue
            data_vcs = np.array(rod_info["data_vcs"])
            data_bev = np.zeros_like(data_vcs)
            if len(data_bev) != 2:
                continue
            data_bev[:, 1] = (
                self.roi_vcs_range[2] - data_vcs[:, 0]
            ) / self.resolution_m_per_grid
            data_bev[:, 0] = (
                self.roi_vcs_range[3] - data_vcs[:, 1]
            ) / self.resolution_m_per_grid
            point_data_list = data_bev[:, :2].tolist()
            points_line = calculate_points_on_line(
                point_data_list[0], point_data_list[1]
            )
            visable_point_cnt = 0
            for point in points_line:
                col = point[0]
                row = point[1]
                col = (
                    col
                    if col < self.target_size[1]
                    else self.target_size[1] - 1
                )
                row = (
                    row
                    if row < self.target_size[0]
                    else self.target_size[0] - 1
                )
                col, row = max(0, col), max(0, row)
                if vis_mask[row, col] == 1:
                    visable_point_cnt += 1
                else:
                    continue
            if visable_point_cnt < self.visable_condition["visable_threshold"]:
                rod_info["attrs"]["ignore"] = "Y"
        return data

    def set_ignore_from_vismask(self, data):
        if "bev_occlusion_mask" not in data:
            return data
        ori_vismask = data.pop("bev_occlusion_mask")
        if "Parking_rod" not in data["gt_bev_parking_obj"].keys():
            return data
        if ori_vismask:
            ori_vismask = compute_vismask(
                ori_vismask,
                visible_flag=[1, 2],
                occlusion_flag=[0, 1, 2, 3],
            )
            vismask_size = ori_vismask.shape[:2]
            assert vismask_size in self.vismask_vcsrange_cfg
            vis_mask_vcs_range = self.vismask_vcsrange_cfg[vismask_size]
            vis_mask = get_roi_resize_data(
                ori_vismask,
                vis_mask_vcs_range,
                self.roi_vcs_range,
                self.target_size,
                pad_index=0,
            )
        else:
            return data
        data = self.ignore_gt_with_line(
            data,
            vis_mask,
        )
        return data

    def remove_rod_duplicates(self, data):
        """Remove duplicate parking rods from the input data.

        Args:
            data: The input data containing the parking rod information.

        Returns:
            The modified data with duplicate parking rods removed.
        """
        if "Parking_rod" not in data["gt_bev_parking_obj"].keys():
            return data
        new_rod_list = []
        rod_list = data["gt_bev_parking_obj"].pop("Parking_rod")
        for idx in range(len(rod_list)):
            is_duplicate = False
            rod_center_vcs_idx = np.array(rod_list[idx]["data_vcs"]).mean(0)
            for jdx in range(len(new_rod_list)):
                rod_center_vcs_jdx = np.array(
                    new_rod_list[jdx]["data_vcs"]
                ).mean(0)
                distance = np.linalg.norm(
                    rod_center_vcs_idx - rod_center_vcs_jdx
                )
                if distance <= self.coord_duplicate_thresh:
                    is_duplicate = True
                    break
            if not is_duplicate:
                new_rod_list.append(rod_list[idx])
        data["gt_bev_parking_obj"]["Parking_rod"] = new_rod_list
        return data

    def _collect_label(self, data):
        data = ANCBEVPSDTargetGenerator.set_ignore_type_from_lidar(
            data, self.lidar_blind_spot_range, self.lidar_distance_threshold
        )
        data = self.remove_rod_duplicates(data)
        rod_label, img_scene = self.label_format(data)
        if "bev_occlusion_mask" in data.keys():
            data.pop("bev_occlusion_mask")
        data[f"annos_{self.res_key}"] = {
            "parking_rod": rod_label,
        }
        data.setdefault(f"gt_{self.res_key}", {})
        return data

    def label_size_align(self, data, max_objs):
        # align label size to max_objs
        rod_label = data[f"annos_{self.res_key}"]["parking_rod"]
        if len(rod_label) > max_objs:
            data[f"annos_{self.res_key}"]["parking_rod"] = rod_label[
                :max_objs, :
            ]
        else:
            n_con = max_objs - len(rod_label)
            rod_con = np.zeros((n_con, self.len_gt_info), dtype=np.float32)
            data[f"annos_{self.res_key}"]["parking_rod"] = np.concatenate(
                (rod_label, rod_con), axis=0
            )
        return data

    def get_ignore_mask(self, rod_info):
        """Get mask region for ignore instance.

        Args:
            rod_info: msg for ignore label
                [x0, x1, y0, y1, slot_x0, slot_y0, slot_x1, slot_y1,
                slot_x2, slot_y2, slot_x3, slot_y3, 0.0, 0.0,
                len(point_coordinates),type, ignore].
        """
        mask_size = self.target_size
        mask = np.ones(mask_size, dtype=np.float)
        len_point = rod_info[-3]
        points = (np.array(rod_info[: int(len_point)]).reshape(-1, 2)).astype(
            np.int
        )
        if len(points) == 2:
            if points[0][0] == points[1][0] or points[0][1] == points[1][1]:
                cv2.line(mask, points[0], points[1], color=0)
            else:
                points_poly = np.append(points, [points[0][0], points[1][1]])
                points_poly = points_poly.reshape(-1, 2)
                cv2.fillPoly(mask, [points_poly], color=0)
        slot_points = (
            np.array(rod_info[int(len_point) : int(len_point) + 8]).reshape(
                -1, 2
            )
        ).astype(np.int)
        cv2.fillPoly(mask, [slot_points], color=0)
        return mask

    def get_rod_target(self, data):
        """Get target and grad for Parkingrod.

        example: if target_size=(h, w), data[f"gt_{self.res_key}"] will add
        {
        classification_obj: (h, w, 1)
        endpoint_offset_obj: (h, w, 4)
        classification_weight_mask: (h, w, 1)
        endpoint_offset_weight_mask: (h, w, 4)
        }
        """
        # rod_label:# [centerx, centery, rod_type, x0, y0, x1, y1,
        #                   slot_x0, slot_y0, slot_x1, slot_y1, type, ignore]
        rod_label = data[f"annos_{self.res_key}"]["parking_rod"]
        classfication_obj = np.zeros(
            (1, self.target_size[0], self.target_size[1]),
            dtype=np.float32,
        )
        endpoint_offset_obj = np.zeros(
            (4, self.target_size[0], self.target_size[1]),
            dtype=np.float32,
        )
        slot_01offset_obj = np.zeros(
            (4, self.target_size[0], self.target_size[1]),
            dtype=np.float32,
        )
        rod2slot_offset_obj = np.zeros(
            (2, self.target_size[0], self.target_size[1]),
            dtype=np.float32,
        )

        classification_weight_mask = np.ones_like(classfication_obj)
        endpoint_offset_weight_mask = np.zeros_like(endpoint_offset_obj)
        slot_01offset_weight_mask = np.zeros_like(slot_01offset_obj)
        rod2slot_offset_weight_mask = np.zeros_like(rod2slot_offset_obj)

        for rod_info in rod_label:
            # process ignore parkingrod
            if rod_info[-1] == self.ignore_dict["Y"]:
                ignore_mask = self.get_ignore_mask(rod_info)
                classification_weight_mask = (
                    ignore_mask * classification_weight_mask
                )
                endpoint_offset_weight_mask = (
                    ignore_mask * endpoint_offset_weight_mask
                )
                continue
            origin_H = rod_info[1]
            origin_W = rod_info[0]
            center_H = origin_H
            center_W = origin_W
            row, col = int(center_H), int(center_W)
            row = row if row < self.target_size[0] else self.target_size[0] - 1
            col = col if col < self.target_size[1] else self.target_size[1] - 1
            start_row, end_row = max(0, row - self.dilate_rate), min(
                self.target_size[0] - 1, row + self.dilate_rate
            )
            start_col, end_col = max(0, col - self.dilate_rate), min(
                self.target_size[1] - 1, col + self.dilate_rate
            )
            # centersampling
            # 1. classification confidence regression
            classfication_obj[
                0, start_row : end_row + 1, start_col : end_col + 1
            ] = 1.0
            # offset regression
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    # 2. endpoint_offset regression
                    for i in range(2):
                        endpoint_offset_obj[2 * i, row, col] = (
                            rod_info[2 * i + 3] - col
                        )
                        endpoint_offset_obj[2 * i + 1, row, col] = (
                            rod_info[2 * i + 4] - row
                        )
                    # 3. slot 01junction_point offset regression
                    for i in range(2):
                        slot_01offset_obj[2 * i, row, col] = (
                            rod_info[2 * i + 7] - col
                        )
                        slot_01offset_obj[2 * i + 1, row, col] = (
                            rod_info[2 * i + 8] - row
                        )
            # 4. rod_endpoint to slot_01junction_point offset regression
            rod_center_x = (rod_info[3] + rod_info[5]) / 2
            rod_center_y = (rod_info[4] + rod_info[6]) / 2
            slot_01center_x = (rod_info[7] + rod_info[9]) / 2
            slot_01center_y = (rod_info[8] + rod_info[10]) / 2
            rod2slot_offset_obj[
                0, start_row : end_row + 1, start_col : end_col + 1
            ] = (rod_center_x - slot_01center_x)
            rod2slot_offset_obj[
                1, start_row : end_row + 1, start_col : end_col + 1
            ] = (rod_center_y - slot_01center_y)

            classification_weight_mask[classfication_obj != 0] = 1.0
            endpoint_offset_weight_mask[endpoint_offset_obj != 0] = 1.0
            slot_01offset_weight_mask[slot_01offset_obj != 0] = 1.0
            rod2slot_offset_weight_mask[rod2slot_offset_obj != 0] = 1.0

            if self.reweight is not None:
                endpoint_offset_weight_mask[:4] *= self.reweight[0]
                endpoint_offset_weight_mask[
                    :4,
                    self.reweight[2] : self.reweight[4],
                    self.reweight[3] : self.reweight[5],
                ] *= self.reweight[1]

        # get parkingrod target for classify/offset/type/slot_offset
        rod_target_dict = {
            "classification_obj": classfication_obj.transpose(1, 2, 0),
            "endpoint_offset_obj": endpoint_offset_obj.transpose(1, 2, 0),
            "slot_01offset_obj": slot_01offset_obj.transpose(1, 2, 0),
            "rod2slot_offset_obj": rod2slot_offset_obj.transpose(1, 2, 0),
        }
        # get grad or mask to parkingrod target
        rod_grad_dict = {
            "classification_weight_mask": classification_weight_mask.transpose(
                1, 2, 0
            ),
            "endpoint_offset_weight_mask": endpoint_offset_weight_mask.transpose(  # noqa
                1, 2, 0
            ),
            "slot_01offset_weight_mask": slot_01offset_weight_mask.transpose(
                1, 2, 0
            ),
            "rod2slot_offset_weight_mask": rod2slot_offset_weight_mask.transpose(  # noqa
                1, 2, 0
            ),
        }
        for k in rod_target_dict.keys():
            data[f"gt_{self.res_key}"][k] = rod_target_dict[k]
        for k in rod_grad_dict.keys():
            data[f"gt_{self.res_key}"][k] = rod_grad_dict[k]
        return data

    def __call__(self, data):
        # rod_preds: self.len_gt_info*h*w
        # rod_labels: N x 17
        # [centerx, centery,rod_type,x1, y1, x2, y2,
        # slot_x0,slot_y0,slot_x1,slot_y1,slot_x2,slot_y2,slot_x3,slot_y3,
        # rod_type, ignore]
        assert isinstance(data, dict), "data is not dict type"
        data = self._collect_label(data)
        data = self.get_rod_target(data)
        data = self.label_size_align(data, self.max_objs)
        return data
