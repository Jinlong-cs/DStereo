# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Optional, Sequence

import cv2
import numpy as np

from hat.core.bev_elevation_utils import compute_vismask, get_roi_resize_data
from hat.core.center_utils import draw_umich_gaussian
from hat.data.transforms.auto_3dv import get_ctoff_map
from hat.data.transforms.real3d import draw_reg_map
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "ANCBEVPSDTargetGenerator",
    "ANCBEVPSDGlobalTargetGenerator",
    "ANCBEVPSDLocalTargetGenerator",
]


@OBJECT_REGISTRY.register
class ANCBEVPSDTargetGenerator(object):
    """Target module for bev psd task.

    Args:
        global_target: Global target module for bev psd task.
        local_target: Local target module for bev psd task.
        max_objs: size of global and local align to max_objs
        input_size: The size of input feature size.
        visable_condition: Store some threshold labels are visible.

            .. code-block:: none

                "extend_line_length": A line extending from the center of
                    parking-slot to the entrance line, default: 5(m).

                "visable_threshold": All the points(grids) along this line,
                    if more than 2 points are in the visible area,
                    the label is retained.

        vcs_range: The groundtruth vcs range, (bottom, right, top, left)
        resolution_m_per_grid:
            [spatial resolution width, spatial resolution height]
        vismask_vcsrange_cfg: Raw offline vismask image size
            as key, and corresponding vcs range as value.
        lidar_blind_spot_range: sequence of float:
            Represents the vcs range for the panda blind spot.
        lidar_distance_threshold: Threshold of distance between
            corner point and lidar point.If the slot point is smaller
            than this threshold, the slot point is visible.
        min_rate_visible: Threshold of whether the object is visible.
        res_key: result key of output.
        local_near_target: Local target module only for near range.

    Return:
        data:generate annos_{self.res_key}(global/local msg for metric)
            from gt_bev_parking_obj, and generate gt_{self.res_key}(target)
            for computing loss.

        e.g.

        .. code-block:: none

            data[f"annos_{self.res_key}"] = {"global":..., "local":...}
            data[f"gt_{self.res_key}"] = {
                "local_classification_obj":...,
                ...,
                "global_classification_obj":...
            }

    """

    def __init__(
        self,
        global_target,
        local_target,
        max_objs: int,
        input_size: tuple,
        visable_condition: dict,
        vcs_range: Sequence[float],
        resolution_m_per_grid: float,
        with_rod: bool,
        vismask_vcsrange_cfg: Optional[Sequence[float]] = None,
        use_vis_mask: bool = False,
        lidar_distance_threshold: float = 1.4,
        min_rate_visible: float = 0.6,
        res_key: str = "bev_psd_obj",
        local_near_target=None,
    ):
        super().__init__()
        self.global_target = global_target
        self.local_target = local_target
        self.local_near_target = local_near_target
        self.max_objs = max_objs
        self.input_size = input_size
        self.occupancy_dict = {"Y": 0, "N": 1}
        self.slot_type_dict = {
            "Vertical": 0,
            "Parallel": 1,
            "Oblique": 2,
            "Stereo": 0,
            "Not_slot": -2,
            "Not_normal_slot": -1,
            "Handicapped": -1,
        }
        self.point_type_dict = {"Parking_rod": 1, "OJ": 1, "SJ": 0}
        self.scene_dict = {"Indoors": 0, "Outdoors": 1}
        self.ignore_dict = {"Y": 0, "N": 1}
        self.visable_condition = visable_condition
        self.vismask_vcsrange_cfg = vismask_vcsrange_cfg
        self.vcs_range = vcs_range
        self.use_vis_mask = use_vis_mask
        self.res_key = res_key
        self.resolution_m_per_grid = resolution_m_per_grid
        self.lidar_distance_threshold = lidar_distance_threshold
        self.with_rod = with_rod
        self.lidar_blind_spot_range = {
            "Normal": (-4.0, -4.0, 7.0, 4.0),
            "Parallel": (-5.0, -5.0, 8.0, 5.0),
        }
        # Parking Spaces contain the threshold of 3d objects.
        self.has_3d_seg_threshold = {"normal": 20, "strict": 50}
        self.min_rate_visible = min_rate_visible
        self.slot_entrance_line_range = {
            "Vertical": [1.5, 4],
            "Parallel": [4, 7],
            "Oblique": [1.5, 4],
            "Stereo": [1.5, 4],
        }

    def gen_slot_angle(self, point_data_list):

        # point_data_list = torch.tensor(point_data_list)
        # torch.tensor, (4 x 2)
        diff1 = point_data_list[3] - point_data_list[0]
        diff2 = point_data_list[2] - point_data_list[1]
        diff3 = (point_data_list[3] + point_data_list[2]) - (
            point_data_list[1] + point_data_list[0]
        )
        vec0 = (
            diff1 / (np.linalg.norm(diff1, axis=0) + np.finfo(np.float32).eps)
        ).tolist()
        vec1 = (
            diff2 / (np.linalg.norm(diff2, axis=0) + np.finfo(np.float32).eps)
        ).tolist()
        vec2 = (
            -diff2
            / (np.linalg.norm(-diff2, axis=0) + np.finfo(np.float32).eps)
        ).tolist()
        vec3 = (
            -diff1
            / (np.linalg.norm(-diff1, axis=0) + np.finfo(np.float32).eps)
        ).tolist()
        slot_vec = (
            diff3 / (np.linalg.norm(diff3, axis=0) + np.finfo(np.float32).eps)
        ).tolist()
        return vec0, vec1, vec2, vec3, slot_vec

    def gen_slot_center(self, point_data_list):

        point_data_list = np.array(point_data_list)
        center = point_data_list.mean(0)
        return list(center)

    def label_format(self, gt_info: dict):
        """
        Convert label to annotation.

        (TODO)
        .. code-block:: none

            {
                "junct":        []        n*12
                "junct_vec":    []        n*8
                "junct_type":   []        n*4
                "center":       []        n*2
                "center_vec":   []        n*2
                "occupancy":    []        n*1
                "slot_type":    []        n*1
                "ignore_type":  []        n*1

            }

        Args:
            gt_info: label msg.

        Returns:
            Example:

                .. code-block:: none

                    global_label: n x 16
                    ignore or Not_normal_slot:
                        [[x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6,
                        occupancy, len_global, slot_type, ignore_type],
                        [], [], ...
                        ]

                    normal:
                        [[centerx, centery, slot_type, slot_vec_x,
                        slot_vec_y, occupancy,
                        x1, y1, x2, y2, x3, y3, x4, y4, slot_type,
                        ignore_type],
                        [], [], ...
                        ]

                    local_label: n x 22
                    ignore or Not_normal_slot:
                        [[x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6,
                        0, 0, 0, 0, 0, 0, 0, len_local, slot_type,
                        ignore_type],
                        [], [],...
                        ]

                    normal:
                        [[x1, y1, x2, y2, x3, y3, x4, y4,
                        p1_vec_x, p1_vec_y, p2_vec_x, p2_vec_y, p3_vec_x,
                        p3_vec_y, p4_vec_x, p4_vec_y,
                        p1_type, p2_type, p3_type, p4_type, slot_type,
                        ignore_type],
                        [], [],...
                        ]

        """

        total_global_label, total_local_label = [], []
        img_scene = self.scene_dict["Indoors"]
        if "Slot" not in gt_info["gt_bev_parking_obj"].keys():
            total_global_label = np.zeros(
                (self.max_objs, 16), dtype=np.float32
            )
            total_local_label = np.zeros((self.max_objs, 22), dtype=np.float32)

            return [total_global_label, total_local_label, img_scene]
        slot_info_list = gt_info["gt_bev_parking_obj"]["Slot"]

        if len(slot_info_list) == 0:
            total_global_label = np.zeros(
                (self.max_objs, 16), dtype=np.float32
            )
            total_local_label = np.zeros((self.max_objs, 22), dtype=np.float32)

            return [total_global_label, total_local_label, img_scene]

        for slot_info in slot_info_list:
            data_vcs = np.array(slot_info["data_vcs"])
            data_bev = np.zeros_like(data_vcs)
            if len(data_bev) != 4:
                continue
            slot_type = slot_info["attrs"]["Slot_type"]
            if self.slot_type_dict[slot_type] == -2:
                continue
            data_bev[:, 1] = (
                self.vcs_range[2] - data_vcs[:, 0]
            ) / self.resolution_m_per_grid[0]
            data_bev[:, 0] = (
                self.vcs_range[3] - data_vcs[:, 1]
            ) / self.resolution_m_per_grid[1]

            ignore_type = slot_info["attrs"]["ignore"]
            if slot_info["gt_location"] == "trunc":
                ignore_type = "Y"
            global_label, local_label = [], []

            point_data_list = data_bev[:, :2].tolist()

            if (
                self.ignore_dict[ignore_type] == 0
                or self.slot_type_dict[slot_type] == -1
            ):
                for point_data in point_data_list:
                    global_label.extend([float(i) for i in point_data])
                    local_label.extend([float(i) for i in point_data])
                len_global = len(global_label)
                len_local = len(local_label)
                global_label.extend([0.0 for _ in range(16 - len_global - 2)])
                local_label.extend([0.0 for _ in range(22 - len_local - 2)])
            else:
                point_attrs_list = slot_info["point_attrs"]
                if len(point_data_list) < 4:
                    continue
                if len(point_data_list) > 4:
                    last_point = point_data_list.pop()
                    point_data_list = point_data_list[:3]
                    point_data_list.append(last_point)
                    last_point_attr = point_attrs_list.pop()
                    point_attrs_list = point_attrs_list[:3]
                    point_attrs_list.append(last_point_attr)
                occupancy = slot_info["attrs"]["Occupancy"]
                vec0, vec1, vec2, vec3, slot_vec = self.gen_slot_angle(
                    np.array(point_data_list, dtype=np.float32)
                )
                center = self.gen_slot_center(point_data_list)
                center = [float(i) for i in center]
                global_label.extend(center)
                global_label.append(self.slot_type_dict[slot_type])
                global_label.extend(slot_vec)
                global_label.append(self.occupancy_dict[occupancy])
                for point_data in point_data_list:
                    global_label.extend([float(i) for i in point_data])
                    local_label.extend([float(i) for i in point_data])
                local_label.extend(vec0)
                local_label.extend(vec1)
                local_label.extend(vec2)
                local_label.extend(vec3)
                try:
                    point_attr = [
                        self.point_type_dict[i["Junction"]["Attribution"]]
                        for i in point_attrs_list
                    ]
                except Exception:
                    point_attr = [
                        self.point_type_dict[i["point_label"]["Junction"]]
                        for i in point_attrs_list
                    ]
                local_label.extend(point_attr)
            if (
                self.ignore_dict[ignore_type] == 0
                or self.slot_type_dict[slot_type] == -1
            ):
                global_label[-1] = len_global
                local_label[-1] = len_local
            local_label.append(self.slot_type_dict[slot_type])
            local_label.append(self.ignore_dict[ignore_type])
            global_label.append(self.slot_type_dict[slot_type])
            global_label.append(self.ignore_dict[ignore_type])
            # global_label: n x 16
            # local_label: n x 22
            assert (
                len(global_label) == 16 and len(local_label) == 22
            ), f"global_label: {len(global_label)} :{global_label} \n local_label: {len(local_label)} :{local_label}"  # noqa

            total_global_label.append(global_label)
            total_local_label.append(local_label)

        assert (
            len(total_global_label) < self.max_objs
        ), f"len(total_global_label): {len(total_global_label)}, max_objs:{self.max_objs}"  # noqa
        total_global_label = np.array(total_global_label, dtype=np.float32)
        total_local_label = np.array(total_local_label, dtype=np.float32)
        return [total_global_label, total_local_label, img_scene]

    def ignore_gt_with_extension_line(
        self,
        data: dict,
        vis_mask: np.ndarray,
    ):
        """Ignore the groundtruth with extend line.

        The methods is by determining the number of points
        in the visible area of the extend line.

        Args:
            data: Store the dictionary of groundtruth.
            vis_mask: Provide a map with visible&invisible regions.

        Returns: label reserved only in the visible area.
        """
        slot_info_list = data["gt_bev_parking_obj"]["Slot"]
        for slot_info in slot_info_list:
            if slot_info["attrs"]["ignore"] == "Y":
                continue
            data_vcs = np.array(slot_info["data_vcs"])
            data_bev = np.zeros_like(data_vcs)
            if len(data_bev) != 4:
                continue
            data_bev[:, 1] = (
                self.vcs_range[2] - data_vcs[:, 0]
            ) / self.resolution_m_per_grid[0]
            data_bev[:, 0] = (
                self.vcs_range[3] - data_vcs[:, 1]
            ) / self.resolution_m_per_grid[1]
            point_data_list = data_bev[:, :2].tolist()
            _, _, _, _, slot_vec = self.gen_slot_angle(
                np.array(point_data_list, dtype=np.float32)
            )
            center = self.gen_slot_center(point_data_list)
            center, slot_vec = np.array(center), np.array(slot_vec)
            extend_points_len = int(
                self.visable_condition["extend_line_length"]
                // self.resolution_m_per_grid[0]
            )

            extend_points_cnt = 0
            for i in range(extend_points_len):
                center_extend = (center - slot_vec * i).astype(np.int32)
                col = center_extend[0]
                row = center_extend[1]
                col = (
                    col if col < self.input_size[1] else self.input_size[1] - 1
                )
                row = (
                    row if row < self.input_size[0] else self.input_size[0] - 1
                )
                col, row = max(0, col), max(0, row)
                if vis_mask[row, col] == 1:
                    extend_points_cnt += 1
                else:
                    continue
            if extend_points_cnt < self.visable_condition["visable_threshold"]:
                slot_info["attrs"]["ignore"] = "Y"
        return data

    def gt_filter_from_vismask(self, data):
        has_vismask_data = False
        if "bev_occlusion_mask" not in data:
            return data, has_vismask_data
        if self.with_rod:
            ori_vismask = data["bev_occlusion_mask"]
        else:
            ori_vismask = data.pop("bev_occlusion_mask")
        if "Slot" not in data["gt_bev_parking_obj"].keys():
            return data, has_vismask_data
        if ori_vismask:
            has_vismask_data = True
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
                self.vcs_range,
                self.input_size,
                pad_index=0,
            )
        else:
            return data, has_vismask_data
        data = self.ignore_gt_with_extension_line(
            data,
            vis_mask,
        )
        return data, has_vismask_data

    def ignore_with_occupy_vismask(self, data, vismask3d, vismask):
        slot_info_list = data["gt_bev_parking_obj"]["Slot"]
        for slot_info in slot_info_list:
            if slot_info["attrs"]["ignore"] == "Y":
                continue
            data_vcs = np.array(slot_info["data_vcs"])
            data_bev = np.zeros_like(data_vcs)[:, :2]
            if len(data_bev) != 4:
                continue
            data_bev[:, 1] = (
                self.vcs_range[2] - data_vcs[:, 0]
            ) / self.resolution_m_per_grid[0]
            data_bev[:, 0] = (
                self.vcs_range[3] - data_vcs[:, 1]
            ) / self.resolution_m_per_grid[1]

            poly = np.int0(data_bev)
            poly_mask = np.zeros(self.input_size, np.uint8)
            cv2.drawContours(poly_mask, [poly], -1, (1), thickness=-1)
            Area_ori = np.sum(poly_mask > 0)
            bbox_3d_vismask0 = poly_mask[vismask3d == 1]
            bbox_3d_vismask1 = poly_mask[vismask3d == 2]
            bbox_visible = poly_mask[vismask == 1]
            Area_3d0 = np.sum(bbox_3d_vismask0)
            Area_3d1 = np.sum(bbox_3d_vismask1)
            Area_visible = np.sum(bbox_visible)

            rate_visible = Area_visible / Area_ori
            # 3d0 is the segmentation result of the car,
            # but the information is not complete.
            # 3d1 is the covered area including
            # the covered area by the car.
            vismask_3d = (
                Area_3d0 > self.has_3d_seg_threshold["normal"]
                and Area_3d1 > self.has_3d_seg_threshold["normal"]
            )
            vismask_3d = (
                vismask_3d and Area_3d0 < self.has_3d_seg_threshold["strict"]
            )
            vismask_3d_strict = (
                Area_3d0 > self.has_3d_seg_threshold["strict"]
                and Area_3d1 > self.has_3d_seg_threshold["normal"]
            )

            if (
                rate_visible > self.min_rate_visible
                and slot_info["attrs"]["Occupancy"] == "Y"
            ):
                slot_info["attrs"]["Occupancy"] = "N"
            if vismask_3d and slot_info["attrs"]["Occupancy"] == "N":
                slot_info["attrs"]["ignore"] = "Y"
            elif vismask_3d_strict and slot_info["attrs"]["Occupancy"] == "N":
                slot_info["attrs"]["Occupancy"] = "Y"
        return data

    def occupancy_modifly_from_vismask(self, data):
        if "bev_occlusion_mask" not in data:
            return data
        if "Slot" not in data["gt_bev_parking_obj"].keys():
            return data
        vismask_pil = data["bev_occlusion_mask"]
        if vismask_pil:
            # Get 3d properties from vismask
            vismask3d = compute_vismask(
                vismask_pil,
                visible_flag=[0],
                occlusion_flag=[1],
                use_3d_mask=True,
            )
            vismask_size = vismask3d.shape[:2]
            assert vismask_size in self.vismask_vcsrange_cfg
            vis_mask_vcs_range = self.vismask_vcsrange_cfg[vismask_size]
            vismask3d_resize = get_roi_resize_data(
                vismask3d,
                vis_mask_vcs_range,
                self.vcs_range,
                self.input_size,
                pad_index=0,
            )
            # Get visible and invisible areas from vismask
            vismask_ori = compute_vismask(
                vismask_pil,
                visible_flag=[1, 2],
                occlusion_flag=[0, 1, 2, 3],
            )
            vismask_ori_resize = get_roi_resize_data(
                vismask_ori,
                vis_mask_vcs_range,
                self.vcs_range,
                self.input_size,
                pad_index=0,
            )
            data = self.ignore_with_occupy_vismask(
                data,
                vismask3d_resize,
                vismask_ori_resize,
            )
        return data

    @staticmethod
    def set_ignore_type_from_lidar(
        data: dict,
        lidar_blind_range: Sequence[float],
        lidar_distance_threshold: float,
    ):
        """Set the 'ignore' attribute in slot with LIDAR blind range.

        Args:
         data: A dictionary containing parking slot information.
            'is_sheltered': Occlusion information judged by lidar.

         lidar_blind_range: A list of floats specifying the LIDAR blind range.

        Returns:
         data: The updated 'ignore' attribute based lidar strategy.
        """
        if "gt_bev_parking_obj" not in data:
            return data
        if "Slot" not in data["gt_bev_parking_obj"].keys():
            return data
        slot_info_list = data["gt_bev_parking_obj"].pop("Slot")
        new_slot_info_list = []
        for slot_info in slot_info_list:
            slot_loc = np.array(slot_info["data_vcs"])
            # Only used slot 0,1 corner for judgment
            if slot_info["attrs"]["Slot_type"] == "Parallel":
                blind_range = lidar_blind_range["Parallel"]
            else:
                blind_range = lidar_blind_range["Normal"]
            in_bottom = slot_loc[:2, 0] > blind_range[0]
            in_top = slot_loc[:2, 0] < blind_range[2]
            in_left = slot_loc[:2, 1] < blind_range[3]
            in_right = slot_loc[:2, 1] > blind_range[1]
            in_blind_range = np.logical_and(
                np.logical_and(in_bottom, in_top),
                np.logical_and(in_left, in_right),
            )
            count_in_vcs_range = np.sum(in_blind_range)
            if (
                "nearest_point_distance_m" not in slot_info.keys()
                or len(slot_info["nearest_point_distance_m"]) < 2
            ):
                slot_info["attrs"]["ignore"] = "Y"
                new_slot_info_list.append(slot_info)
                continue
            nearest_point_distance = np.array(
                slot_info["nearest_point_distance_m"]
            )
            is_visible = np.all(
                nearest_point_distance[:2] < lidar_distance_threshold
            )
            if count_in_vcs_range == 0 and not is_visible:
                slot_info["attrs"]["ignore"] = "Y"
            new_slot_info_list.append(slot_info)
        data["gt_bev_parking_obj"]["Slot"] = new_slot_info_list
        return data

    def get_slot_vcs_range(self, data, vcs_range):
        """Filter the parking slot information in based on the specified VCS range.

        Args:
         data: A dictionary containing parking slot information.

         vcs_range: A list of floats specifying VCS range to
            filter parking slots.

        Returns:
         data: Save parking slot whose coordinates fall within
            the specified VCS range.
        """
        if "gt_bev_parking_obj" not in data:
            return data
        if "Slot" not in data["gt_bev_parking_obj"].keys():
            return data
        slot_info_list = data["gt_bev_parking_obj"].pop("Slot")
        new_slot_info_list = []
        for slot_info in slot_info_list:
            slot_loc = np.array(slot_info["data_vcs"])
            in_bottom = slot_loc[:, 0] > vcs_range[0]
            in_top = slot_loc[:, 0] < vcs_range[2]
            in_left = slot_loc[:, 1] < vcs_range[3]
            in_right = slot_loc[:, 1] > vcs_range[1]
            in_vcs_range = np.logical_and(
                np.logical_and(in_bottom, in_top),
                np.logical_and(in_left, in_right),
            )
            count_in_vcs_range = np.sum(in_vcs_range)
            if count_in_vcs_range == 0:
                continue
            elif count_in_vcs_range == len(slot_loc):
                new_slot_info_list.append(slot_info)
            else:
                slot_info["gt_location"] = "trunc"
                new_slot_info_list.append(slot_info)
        data["gt_bev_parking_obj"]["Slot"] = new_slot_info_list
        return data

    @staticmethod
    def calculate_angles(points):
        """Calculate quadrangle angles.

        Args:
            points: a list containing four points,
                eg. [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

        Returns:
            The four inner corners of a quadrilateral.
        """

        # computation vector
        v0 = points[1] - points[0]
        v1 = points[2] - points[1]
        v2 = points[3] - points[2]
        v3 = points[0] - points[3]

        # calculated angle (radian)
        angle0 = np.arccos(
            np.dot(v3, v0) / (np.linalg.norm(v3) * np.linalg.norm(v0))
        )
        angle1 = np.arccos(
            np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
        )
        angle2 = np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        )
        angle3 = np.arccos(
            np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3))
        )

        # convert radians to angles (degrees）
        angle0 = np.degrees(angle0)
        angle1 = np.degrees(angle1)
        angle2 = np.degrees(angle2)
        angle3 = np.degrees(angle3)

        return [angle0, angle1, angle2, angle3]

    def check_vertical_slot(self, slot_info):
        """Check the vertical slot annotations.

        1. The length of the entrance should be within the preset range.
        2. The length of the entrance line must not exceed the length of
            the separate line.
        3. Each interior Angle needs to be close to a right Angle.

        Args:
            slot_info: parallel parking slot annotation.

        Returns:
            Clean parallel parking slot annotations.
        """
        slot_type = slot_info["attrs"]["Slot_type"]
        slot_loc = np.array(slot_info["data_vcs"])
        slot_01_distance = np.linalg.norm(slot_loc[0] - slot_loc[1])
        slot_12_distance = np.linalg.norm(slot_loc[1] - slot_loc[2])
        if (
            slot_01_distance < self.slot_entrance_line_range[slot_type][0]
            or slot_01_distance > self.slot_entrance_line_range[slot_type][1]
        ):
            slot_info["attrs"]["ignore"] = "Y"
        if slot_01_distance > slot_12_distance:
            slot_info["attrs"]["ignore"] = "Y"
        angles = self.calculate_angles(slot_loc)
        angles_diff = np.abs(np.array(angles) - 90)
        if angles_diff.max() > 10:
            slot_info["attrs"]["ignore"] = "Y"
        return slot_info

    def check_parallel_slot(self, slot_info):
        """Check the parallel slot annotations.

        1. The length of the entrance should be within the preset range.
        2. The length of the entrance line must exceed the length of the
            separate line.
        3. Each interior Angle needs to be close to a right Angle.

        Args:
            slot_info: parallel parking slot annotation.

        Returns:
            Clean parallel parking slot annotations.
        """
        slot_type = slot_info["attrs"]["Slot_type"]
        slot_loc = np.array(slot_info["data_vcs"])
        slot_01_distance = np.linalg.norm(slot_loc[0] - slot_loc[1])
        slot_12_distance = np.linalg.norm(slot_loc[1] - slot_loc[2])
        if (
            slot_01_distance < self.slot_entrance_line_range[slot_type][0]
            or slot_01_distance > self.slot_entrance_line_range[slot_type][1]
        ):
            slot_info["attrs"]["ignore"] = "Y"
        if slot_01_distance < slot_12_distance:
            slot_info["attrs"]["ignore"] = "Y"
        angles = self.calculate_angles(slot_loc)
        angles_diff = np.abs(np.array(angles) - 90)
        # Check if any of the angle differences exceed 10 degrees
        if angles_diff.max() > 10:
            slot_info["attrs"]["ignore"] = "Y"
        return slot_info

    def check_oblique_slot(self, slot_info):
        """Check the oblique slot annotations.

        1. The length of the entrance should be within the preset range.
        2. The length of the entrance line must not exceed the length
            of the separate line.
        3. The internal Angle of the oblique slot needs to be clearly
            distinguished from the right Angle.

        Args:
            slot_info: oblique parking slot annotation.

        Returns:
            Clean oblique parking slot annotations.
        """
        slot_type = slot_info["attrs"]["Slot_type"]
        slot_loc = np.array(slot_info["data_vcs"])
        slot_01_distance = np.linalg.norm(slot_loc[0] - slot_loc[1])
        slot_12_distance = np.linalg.norm(slot_loc[1] - slot_loc[2])
        if (
            slot_01_distance < self.slot_entrance_line_range[slot_type][0]
            or slot_01_distance > self.slot_entrance_line_range[slot_type][1]
        ):
            slot_info["attrs"]["ignore"] = "Y"
        if slot_01_distance > slot_12_distance:
            slot_info["attrs"]["ignore"] = "Y"
        angles = self.calculate_angles(slot_loc)
        angles_diff = np.abs(np.array(angles) - 90)
        if angles_diff.sum() < 40:
            slot_info["attrs"]["ignore"] = "Y"
        return slot_info

    def check_slot_annotation(self, data):
        """Check the slot annotations in the given data.

        Args:
            data: A dictionary containing the slot annotations.

        Returns:
            A modified data dictionary with clean slot annotations.
        """
        if "gt_bev_parking_obj" not in data:
            return data
        if "Slot" not in data["gt_bev_parking_obj"].keys():
            return data
        slot_info_list = data["gt_bev_parking_obj"].pop("Slot")
        clean_slot_info_list = []
        for slot_info in slot_info_list:
            slot_type = slot_info["attrs"]["Slot_type"]
            if (
                slot_info["attrs"]["ignore"] == "Y"
                or slot_type not in self.slot_entrance_line_range.keys()
            ):
                clean_slot_info_list.append(slot_info)
                continue
            if slot_type in ["Vertical", "Stereo"]:
                slot_info = self.check_vertical_slot(slot_info)
            elif slot_type == "Parallel":
                slot_info = self.check_parallel_slot(slot_info)
            else:
                slot_info = self.check_oblique_slot(slot_info)
            clean_slot_info_list.append(slot_info)
        data["gt_bev_parking_obj"]["Slot"] = clean_slot_info_list
        return data

    def _collect_label(self, data_dict):
        data_dict = self.get_slot_vcs_range(data_dict, self.vcs_range)
        data_dict = self.check_slot_annotation(data_dict)
        data_dict = self.occupancy_modifly_from_vismask(data_dict)
        has_vismask_data = False
        if self.use_vis_mask:
            data_dict, has_vismask_data = self.gt_filter_from_vismask(
                data_dict
            )
        if not has_vismask_data:
            data_dict = self.set_ignore_type_from_lidar(
                data_dict,
                self.lidar_blind_spot_range,
                self.lidar_distance_threshold,
            )
        global_label, local_label, img_scene = self.label_format(data_dict)
        data_dict[f"annos_{self.res_key}"] = {
            "global": global_label,
            "local": local_label,
        }
        data_dict.setdefault(f"gt_{self.res_key}", {})
        return data_dict

    def label_size_align(self, data, max_objs):

        global_label = data[f"annos_{self.res_key}"]["global"]
        local_label = data[f"annos_{self.res_key}"]["local"]
        assert len(global_label) == len(local_label), (
            "They must be the same length, pls check shape with "
            f"global_label:{global_label.shape},global_con:{local_label.shape}"
        )
        if len(local_label) > max_objs:
            data[f"annos_{self.res_key}"]["global"] = global_label[
                :max_objs, :
            ]
            data[f"annos_{self.res_key}"]["local"] = local_label[:max_objs, :]
        else:
            n_con = max_objs - len(local_label)
            global_con = np.zeros((n_con, 16), dtype=np.float32)
            local_con = np.zeros((n_con, 22), dtype=np.float32)
            if n_con == max_objs:
                data[f"annos_{self.res_key}"]["global"] = global_con
                data[f"annos_{self.res_key}"]["local"] = local_con
                return data
            data[f"annos_{self.res_key}"]["global"] = np.concatenate(
                (global_label, global_con), axis=0
            )
            data[f"annos_{self.res_key}"]["local"] = np.concatenate(
                (local_label, local_con), axis=0
            )
        return data

    def __call__(self, data):
        assert isinstance(data, dict), "data is not dict type"
        data = self._collect_label(data)
        data = self.global_target(data)
        data = self.local_target(data)
        if self.local_near_target is not None:
            data = self.local_near_target(data)
        data = self.label_size_align(data, self.max_objs)
        return data


@OBJECT_REGISTRY.register
class ANCBEVPSDGlobalTargetGenerator(object):
    """The global target module for super psd task.

    Args:
        input_size: The size of input feature size.
        out_size: The size global feature.
        slot_weight_dict: The loss weight dict for different slot type.
        dilate_rate: expansion factor for centersampling.
        res_key: result key of output.

    Return:
        global_target_dict: get global target for classify/offset/type...
        global_grad_dict: get grad or mask to local target

        Example:
            if out_size = (48, 32), data[f"gt_{self.res_key}"] will add

            .. code-block:: json

            {
                global_classification_obj : (48, 32, 1)
                global_offset_obj : (48, 32, 8)
                global_occupancy_obj : (48, 32, 1)
                global_slot_type_obj : (48, 32, 3)
                global_direction_obj : (48, 32, 2)
                global_classification_grad : (48, 32, 1)
                global_offset_grad : (48, 32, 8)
                global_occupancy_grad : (48, 32, 1)
                global_slot_type_grad : (48, 32, 3)
                global_direction_grad : (48, 32, 2)

            }
    """

    def __init__(
        self,
        input_size: tuple,
        out_size: tuple,
        slot_weight_dict: dict,
        dilate_rate: int = 1,
        res_key: str = "bev_psd_obj",
    ):
        super().__init__()
        assert (
            input_size[0] / out_size[0] - input_size[1] / out_size[1]
            < np.finfo(np.float16).eps
        )
        self.input_size = input_size
        self.grid_stride = int(input_size[0] / out_size[0])
        self.out_size = out_size
        self.slot_weight_dict = slot_weight_dict
        self.res_key = res_key
        self.dilate_rate = dilate_rate

    def create_grad_mask(self, slot_info):
        mask_size = self.out_size
        mask = np.ones(mask_size, dtype=np.float)
        slot_weight = (
            0 if slot_info[-1] == 0 else self.slot_weight_dict[slot_info[-2]]
        )
        points = (
            np.array(slot_info[: int(slot_info[-3])]).reshape(-1, 2)
            / self.grid_stride
        ).astype(np.int)
        cv2.fillPoly(mask, [points], color=slot_weight)
        return mask

    def __call__(self, data):
        # global_preds: 15*28*28
        # global_labels: [[[centerx, centery, slot_type, slot_vec_x, slot_vec_y, occupancy, x1, y1, x2, y2, x3, y3, x4, y4], []],[[]]] # noqa
        global_labels = data[f"annos_{self.res_key}"]["global"]
        classification_obj = np.zeros(
            (1, self.out_size[0], self.out_size[1]), dtype=np.float32
        )
        offset_obj = np.zeros(
            (8, self.out_size[0], self.out_size[1]), dtype=np.float32
        )
        occupancy_obj = np.zeros(
            (1, self.out_size[0], self.out_size[1]), dtype=np.float32
        )
        slot_type_obj = np.zeros(
            (3, self.out_size[0], self.out_size[1]), dtype=np.float32
        )
        direction_obj = np.zeros(
            (2, self.out_size[0], self.out_size[1]), dtype=np.float32
        )

        classification_grad = np.zeros_like(classification_obj)
        offset_grad = np.zeros_like(offset_obj)
        occupancy_grad = np.zeros_like(occupancy_obj)
        slot_type_grad = np.zeros_like(slot_type_obj)
        direction_grad = np.zeros_like(direction_obj)

        classification_grad.fill(1.0)

        for global_label in global_labels:
            # process Ignore and Not_slot
            if not np.any(global_label):
                continue
            if global_label[-1] == 0 or global_label[-2] == -1:
                grad_mask = self.create_grad_mask(global_label)
                classification_grad = grad_mask * classification_grad
                offset_grad = grad_mask * offset_grad
                occupancy_grad = grad_mask * occupancy_grad
                slot_type_grad = grad_mask * slot_type_grad
                direction_grad = grad_mask * direction_grad
                continue
            origin_H = global_label[1]
            origin_W = global_label[0]
            center_H = origin_H / self.grid_stride
            center_W = origin_W / self.grid_stride
            row, col = int(center_H), int(center_W)
            row = row if row < self.out_size[0] else self.out_size[0] - 1
            col = col if col < self.out_size[1] else self.out_size[1] - 1
            start_row, start_col = max(0, row - self.dilate_rate), max(
                0, col - self.dilate_rate
            )
            end_row, end_col = min(
                self.out_size[0] - 1, row + self.dilate_rate
            ), min(self.out_size[1] - 1, col + self.dilate_rate)
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    slot_poly = (
                        global_label[6:14] / self.grid_stride
                    ).reshape((-1, 2))
                    pt = np.array([col, row], dtype=np.float32)
                    pt_in_slot_flag = cv2.pointPolygonTest(
                        slot_poly, pt, False
                    )
                    if pt_in_slot_flag > 0:
                        # 1. confidence regression
                        classification_obj[0, row, col] = 1.0
                        # 2. slot type classification
                        slot_type_obj[int(global_label[2]), row, col] = 1
                        # 3. direction regression
                        direction_obj[0, row, col] = global_label[3]
                        direction_obj[1, row, col] = global_label[4]
                        # 4. occupancey classifiaction
                        occupancy_obj[0, row, col] = global_label[5]
                        # 5. point offset regression
                        for i in range(4):
                            offset_obj[2 * i, row, col] = (
                                global_label[2 * i + 6] / self.grid_stride
                                - col
                            )
                            offset_obj[2 * i + 1, row, col] = (
                                global_label[2 * i + 7] / self.grid_stride
                                - row
                            )
                        occupancy_grad[:, row, col] = 1.0
                        slot_type_grad[:, row, col] = 1.0
                        direction_grad[:, row, col] = 1.0
                        offset_grad[:, row, col] = 1.0
            classification_grad[:, row, col] = self.slot_weight_dict[
                global_label[2]
            ]

        global_target_dict = {
            "global_classification_obj": classification_obj.transpose(1, 2, 0),
            "global_offset_obj": offset_obj.transpose(1, 2, 0),
            "global_occupancy_obj": occupancy_obj.transpose(1, 2, 0),
            "global_slot_type_obj": slot_type_obj.transpose(1, 2, 0),
            "global_direction_obj": direction_obj.transpose(1, 2, 0),
        }
        global_grad_dict = {
            "global_classification_grad": classification_grad.transpose(
                1, 2, 0
            ),
            "global_offset_grad": offset_grad.transpose(1, 2, 0),
            "global_occupancy_grad": occupancy_grad.transpose(1, 2, 0),
            "global_slot_type_grad": slot_type_grad.transpose(1, 2, 0),
            "global_direction_grad": direction_grad.transpose(1, 2, 0),
        }
        for k in global_target_dict.keys():
            data[f"gt_{self.res_key}"][k] = global_target_dict[k]
        for k in global_grad_dict.keys():
            data[f"gt_{self.res_key}"][k] = global_grad_dict[k]
        return data


@OBJECT_REGISTRY.register
class ANCBEVPSDLocalTargetGenerator(object):
    """Local target module for bev psd task.

    Args:
        input_size: The size of input feature size.
        out_size: The size global feature.
        radius: The radius of guassian kernel.
        slot_weight_dict: The loss weight for different slot type.
        cls_weight: The loss weight for classification.
        offset_weight: The loss weight for offset.
        far_weight: The loss weight of the third and forth corner point.  # noqa
        local_angle_radius: The value representing the radius and
            angle range of a local area centered at a given point. #noqa
        res_key: result key of output.
        gt_postfix: Label name postfix.
        reweight: Re-weight parameters and near range for regresssion, e.g.,
            [all_scale, near_scale, x0, y0, x1, y1].

    Return:
        local_target_dict: get local target for classify/offset/type...
        local_grad_dict: get grad or mask to local target

        Example:
            if out_size = (192, 128), data[f"gt_{self.res_key}"] will add

            .. code-block:: json

            {
                local_classification_obj : (192, 128, 4)
                local_offset_obj : (192, 128, 8)
                local_sline_angle_obj : (192, 128, 8)
                local_point_type_obj : (192, 128, 4)
                local_classification_grad : (192, 128, 4)
                local_offset_grad : (192, 128, 8)
                local_sline_angle_grad : (192, 128, 8)
                local_point_type_grad : (192, 128, 4)

            }

    """

    def __init__(
        self,
        input_size: tuple,
        out_size: tuple,
        radius: int,
        slot_weight_dict: dict,
        cls_weight: float = 1.0,
        offset_weight: float = 1.0,
        far_weight: float = 1.0,
        local_angle_radius: int = 1,
        res_key: str = "bev_psd_obj",
        gt_postfix: str = "",
        reweight: List[float] = None,
    ):
        super().__init__()
        assert (
            input_size[0] / out_size[0] - input_size[1] / out_size[1]
            < np.finfo(np.float16).eps
        )
        self.input_size = input_size
        self.grid_stride = input_size[0] / out_size[0]
        self.out_size = out_size
        self.radius = radius
        self.cls_weight = cls_weight
        self.offset_weight = offset_weight
        self.far_weight = far_weight
        self.slot_weight_dict = slot_weight_dict
        self.local_angle_radius = local_angle_radius
        # A dictionary that maps labels to their corresponding start indices in
        # a point information array. e.g, "point_x" has a starting index of 0.
        # local_labels: [[[x1, y1, x2, y2, x3, y3, x4, y4,
        #                   p1_vec_x, p1_vec_y, p2_vec_x, p2_vec_y,
        #                   p3_vec_x, p3_vec_y, p4_vec_x, p4_vec_y,
        #                  p1_type, p2_type, p3_type, p4_type, slot_type], []]]
        self.label_start_index = {
            "point_x": 0,
            "point_y": 1,
            "shelter": 16,
            "sline_angle_x": 8,
            "sline_angle_y": 9,
        }
        self.res_key = res_key
        self.gt_postfix = gt_postfix
        assert reweight is None or len(reweight) == 6
        self.reweight = reweight

    def gen_label_area(self, target_map, center, radius, target):

        x, y = int(center[0]), int(center[1])
        target = np.array(target)
        target_kernel = np.ones([2 * radius + 1, 2 * radius + 1])
        height, width = target_map.shape[:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        masked_target_map = target_map[
            y - top : y + bottom, x - left : x + right
        ]
        masked_kernel = target_kernel[
            radius - top : radius + bottom, radius - left : radius + right
        ]
        out_map = target_map
        np.maximum(
            masked_target_map,
            masked_kernel,
            out=out_map[y - top : y + bottom, x - left : x + right],
        )
        return out_map

    def create_grad_mask(self, point_info):
        mask_size = self.out_size
        mask = np.ones(mask_size, dtype=np.float)
        slot_weight = (
            0 if point_info[-1] == 0 else self.slot_weight_dict[point_info[-2]]
        )
        points = (
            np.array(point_info[: int(point_info[-3])]).reshape(-1, 2)
            / self.grid_stride
        ).astype(np.int)
        cv2.fillPoly(mask, [points], color=slot_weight)
        return mask

    def __call__(self, data):
        # local_preds: [batch_size, 24, 112, 112]
        # local_labels: [[[x1, y1, x2, y2, x3, y3, x4, y4,
        #                   p1_vec_x, p1_vec_y, p2_vec_x, p2_vec_y,
        #                   p3_vec_x, p3_vec_y, p4_vec_x, p4_vec_y,
        #                  p1_type, p2_type, p3_type, p4_type, slot_type], []]]

        local_label = data[f"annos_{self.res_key}"]["local"]
        classification_obj = np.zeros(
            (4, self.out_size[0], self.out_size[1]),
            dtype=np.float32,
        )
        offset_obj = np.zeros(
            (8, self.out_size[0], self.out_size[1]),
            dtype=np.float32,
        )
        sline_angle_obj = np.zeros(
            (8, self.out_size[0], self.out_size[1]),
            dtype=np.float32,
        )
        point_type_obj = np.zeros(
            (4, self.out_size[0], self.out_size[1]),
            dtype=np.float32,
        )
        classification_grad = np.zeros_like(classification_obj)
        offset_grad = np.zeros_like(offset_obj)
        sline_angle_grad = np.zeros_like(sline_angle_obj)
        point_type_grad = np.zeros_like(point_type_obj)

        classification_grad[0:2].fill(self.cls_weight)
        classification_grad[2:].fill(self.far_weight)

        for point_info in local_label:
            if not np.any(local_label):
                continue
            if point_info[-1] == 0 or point_info[-2] == -1:
                grad_mask = self.create_grad_mask(point_info)
                classification_grad = grad_mask * classification_grad
                offset_grad = grad_mask * offset_grad
                sline_angle_grad = grad_mask * sline_angle_grad
                point_type_grad = grad_mask * point_type_grad
                continue
            for i in range(4):
                origin_H = point_info[
                    2 * i + self.label_start_index["point_y"]
                ]
                origin_W = point_info[
                    2 * i + self.label_start_index["point_x"]
                ]
                corner_H = origin_H / self.grid_stride
                corner_W = origin_W / self.grid_stride
                row, col = int(corner_H), int(corner_W)
                row = row if row < self.out_size[0] else self.out_size[0] - 1
                col = col if col < self.out_size[1] else self.out_size[1] - 1
                # TODO(chunyu.bi): use draw_heatmap replace draw umich&regmap
                # 1. classification
                draw_umich_gaussian(
                    classification_obj[i],
                    [corner_W, corner_H],
                    self.radius,
                )
                # 2. corner offset
                insert_hm = get_ctoff_map(
                    (self.radius * 2, self.radius * 2), (corner_W, corner_H)
                )
                draw_reg_map(
                    offset_obj[2 * i],
                    insert_hm[:, :, 0],
                    (corner_W, corner_H),
                    op="overwrite",
                )
                draw_reg_map(
                    offset_obj[2 * i + 1],
                    insert_hm[:, :, 1],
                    (corner_W, corner_H),
                    op="overwrite",
                )
                # 3. sline_angle
                sline_angle_obj[
                    2 * i,
                    row
                    - self.local_angle_radius : row
                    + self.local_angle_radius
                    + 1,
                    col
                    - self.local_angle_radius : col
                    + self.local_angle_radius
                    + 1,
                ] = point_info[2 * i + self.label_start_index["sline_angle_x"]]
                sline_angle_obj[
                    2 * i + 1,
                    row
                    - self.local_angle_radius : row
                    + self.local_angle_radius
                    + 1,
                    col
                    - self.local_angle_radius : col
                    + self.local_angle_radius
                    + 1,
                ] = point_info[2 * i + self.label_start_index["sline_angle_y"]]
                # 4. point_type
                point_type_obj[0, row, col] = point_info[
                    i + self.label_start_index["shelter"]
                ]
                if point_info[i + self.label_start_index["shelter"]] != 0:
                    self.gen_label_area(
                        classification_grad[i],
                        [corner_W, corner_H],
                        self.radius,
                        self.slot_weight_dict[point_info[-2]],
                    )
                else:
                    classification_grad[i][
                        row - self.radius : row + self.radius + 1,
                        col - self.radius : col + self.radius + 1,
                    ] = 0.0
                sline_angle_grad[
                    i : i + 2,
                    row
                    - self.local_angle_radius : row
                    + self.local_angle_radius
                    + 1,
                    col
                    - self.local_angle_radius : col
                    + self.local_angle_radius
                    + 1,
                ] = 1.0
                self.gen_label_area(
                    offset_grad[2 * i],
                    [corner_W, corner_H],
                    self.radius,
                    self.slot_weight_dict[point_info[-2]],
                )
                self.gen_label_area(
                    offset_grad[2 * i + 1],
                    [corner_W, corner_H],
                    self.radius,
                    self.slot_weight_dict[point_info[-2]],
                )
                point_type_grad[i, row, col] = 1.0

        offset_grad[:4] *= self.offset_weight
        if self.reweight is not None:
            offset_grad[:4] *= self.reweight[0]
            offset_grad[
                :4,
                self.reweight[2] : self.reweight[4],
                self.reweight[3] : self.reweight[5],
            ] *= self.reweight[1]
        local_target_dict = {
            "local_classification_obj": classification_obj.transpose(1, 2, 0),
            "local_offset_obj": offset_obj.transpose(1, 2, 0),
            "local_sline_angle_obj": sline_angle_obj.transpose(1, 2, 0),
            "local_point_type_obj": point_type_obj.transpose(1, 2, 0),
        }
        local_grad_dict = {
            "local_classification_grad": classification_grad.transpose(
                1, 2, 0
            ),
            "local_offset_grad": offset_grad.transpose(1, 2, 0),
            "local_sline_angle_grad": sline_angle_grad.transpose(1, 2, 0),
            "local_point_type_grad": point_type_grad.transpose(1, 2, 0),
        }
        for k in local_target_dict.keys():
            data[f"gt_{self.res_key}"][
                k + self.gt_postfix
            ] = local_target_dict[k]
        for k in local_grad_dict.keys():
            data[f"gt_{self.res_key}"][k + self.gt_postfix] = local_grad_dict[
                k
            ]
        return data
