import json
import os
from typing import Callable, Sequence

import cv2
import numpy as np

from hat.callbacks.task_visualize.bev_multitask import BaseVisualize
from hat.core.virtual_camera import FisheyeCamera, PinholeCamera
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, convert_numpy

slot_occupancy_color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
slot_occupancy_mark_dict = {0: "occ", 1: "Not-occ"}
slot_occupancy_dict = {0: "OFF", 1: "ON"}
slot_type_mark_dict = {0: "V", 1: "P", 2: "O"}
slot_ignore_mark_dict = {0: "ignore", 1: "Not-ignore"}

camera_view_rename = {
    "camera_front_left": "camera_frontleft",
    "camera_front": "camera_front",
    "camera_front_right": "camera_frontright",
    "camera_rear_left": "camera_rearleft",
    "camera_rear": "camera_rear",
    "camera_rear_right": "camera_rearright",
    "fisheye_front": "camera_fisheye_front",
    "fisheye_rear": "camera_fisheye_rear",
    "fisheye_left": "camera_fisheye_left",
    "fisheye_right": "camera_fisheye_right",
    "camera_front_30fov": "camera_front_30fov",
}


@OBJECT_REGISTRY.register
class ANCBevObjVisualize(BaseVisualize):
    """
    Visualize class for bev discrete object.

    Args:
        bev_size: the size of bird eye view(order by h, w).
        vcs_range: visbile range of bev, (bottom, right, top, left) in order.
        camera_view_names: camera view names.
        project_pts_to_cameras: whether project the points to images.
            Defaults to True.
        is_bev_horizon: indicates whether the BEV image occupies the bottom
            row or the rightmost column.
        reformat_fn: the function for reformat the task results.
        camera_layouts: the customized layout of cameras.
        draw_interval: The real world interval size. Defaults to 5m.
        vis_ego: Whether to visualize ego, default False, wide range is True.
        scor_thr: the threshold score.
        vis_img_scale: IPM chart and predict the scale size, default 1.
        color_map: mapping relationship from category to color.
        id2label: mapping relationship from category id to category name.
        anno_name: , for example: ["annos_bev_discrete_obj",].
        vcs range mode, included "wide" and "small".
        res_key2anno_name: a mapping of result key of output to
            annotation name of data, for example:{"bev_discrete_obj":
            "annos_bev_discrete_obj",}.
        visual_interval:  visual frequency.
        task: Only visualize this task if not None.
        output_dir: Pack infer output path.
        prefix: Module output prefix.
    """

    def __init__(
        self,
        bev_size: tuple,
        vcs_range: tuple,
        res_key2anno_name: dict,
        color_map: dict,
        id2label: dict,
        vis_ego: bool = False,
        project_pts_to_cameras: bool = False,
        camera_view_names: Sequence = None,
        is_bev_horizon: bool = False,
        reformat_fn: Callable = None,
        camera_layouts: Sequence = None,
        draw_interval: int = 5,
        scor_thr: float = 0.2,
        vis_img_scale: float = 1.0,
        visual_interval: int = 1,
        range_mode: str = "wide",
        ipm_name: str = "ipm",
        center_shift: tuple = (0, 0),
        # for parkingrod
        roi_vcs_range: tuple = (-12.8, -12.8, 25.6, 12.8),
        ego_vcs_range: tuple = (-1, -1, 4, 1),
        resolution_m_per_grid: float = 0.4,
        task: str = None,
        output_dir: str = None,
        prefix: str = None,
        out_ori_img_size: tuple = (2048 // 2, 1280 // 2),
    ):
        if output_dir is not None:
            super().__init__(output_dir)
        # for parking rod
        self.roi_vcs_range = roi_vcs_range

        self.ego_vcs_range = ego_vcs_range
        self.resolution_m_per_grid = resolution_m_per_grid

        self.reformat_fn = reformat_fn
        self.bev_size = bev_size
        self.project_pts_to_cameras = project_pts_to_cameras
        if project_pts_to_cameras:
            self.project_calib = {}
        self.camera_view_names = camera_view_names
        self.vcs_range = vcs_range
        self.scor_thr = scor_thr
        self.vis_img_scale = vis_img_scale
        self.color_map = color_map
        self.id2label = id2label
        self.res_key2anno_name = res_key2anno_name
        m_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / bev_size[0],
            abs(vcs_range[3] - vcs_range[1]) / bev_size[1],
        )  # bev coord y, x
        self.m_perpixel = [m_p / self.vis_img_scale for m_p in m_perpixel]
        self.range_mode = range_mode
        self.is_bev_horizon = is_bev_horizon
        self.vis_ego = vis_ego
        self.draw_interval = draw_interval
        self.visual_interval = visual_interval
        self.visual_count = 0
        assert self.range_mode in ["wide", "small"]
        self.meta_name = "meta_info"
        self.ipm_name = ipm_name
        self.center_shift = np.array(center_shift, np.int)
        self.task = task
        self.output_dir = output_dir
        self.prefix = prefix
        self.out_ori_img_size = out_ori_img_size

        if camera_layouts is None:
            self.camera_layouts = [
                ["placeholder", "camera_front", "camera_front_30fov"],
                ["camera_front_left", "fisheye_front", "camera_front_right"],
                ["fisheye_left", "fisheye_rear", "fisheye_right"],
                ["camera_rear_left", "camera_rear", "camera_rear_right"],
            ]
        else:
            self.camera_layouts = camera_layouts

    def vcs2bev_coord(self, pt, vcs_range, m_perpixel):
        pt = np.array(pt).reshape((-1, 2))
        u = (vcs_range[3] - pt[:, 1]) / m_perpixel[1]
        v = (vcs_range[2] - pt[:, 0]) / m_perpixel[0]
        u = u.reshape((-1, 1))
        v = v.reshape((-1, 1))
        return np.hstack([u, v]).astype(np.int32)

    def bev2vcs_coord(self, pt, vcs_range, m_perpixel):
        pt = np.array(pt).reshape((-1, 2))
        y = vcs_range[3] - pt[:, 0] * m_perpixel[1]
        x = vcs_range[2] - pt[:, 1] * m_perpixel[0]
        y = y.reshape((-1, 1))
        x = x.reshape((-1, 1))
        return np.hstack([x, y])

    def get_vertices_from_box(
        self, wh: tuple, ct: tuple, yaw: float, is_right_hand_coor: bool
    ) -> list:
        """
        Calculate and return the vertices from a box.

        Args:
            wh: The width and height of the box.
            ct: The center coordinates of the box.
            yaw: The yaw angle of the box.
            is_right_hand_coor: Flag indicating whether the 2D
                coordinate system is right-handed,
                for example: BEV is left-hand, VCS is right-hand.

        Returns:
            vertices: A list of tuples
                representing the vertices of the box in the VCS or BEV.

        Note:
            The VCS is a right-handed coordinate system where the
                x-axis points forward, the y-axis points to the left.
            The BEV is a left-handed coordinate system
                where the x-axis points left, the y-axis points to the down.
            The yaw angle represents the anti-clock rotation of
                the box around the x-axis in radians.
        """
        w, h = wh[:2]
        ctx, cty = ct

        p0 = [-0.5 * w, 0.5 * h, 1]
        p1 = [0.5 * w, 0.5 * h, 1]
        p2 = [0.5 * w, -0.5 * h, 1]
        p3 = [-0.5 * w, -0.5 * h, 1]
        points = np.array([p0, p1, p2, p3])
        if is_right_hand_coor:
            R = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), ctx],
                    [np.sin(yaw), np.cos(yaw), cty],
                    [0, 0, 1],
                ]
            )
        else:
            R = np.array(
                [
                    [np.cos(yaw), np.sin(yaw), ctx],
                    [-np.sin(yaw), np.cos(yaw), cty],
                    [0, 0, 1],
                ]
            )
        points = R.dot(points.T).T
        return points[:, :2]

    def draw_psd_prediction(
        self, det_results, img, color_map, batch_extra_imgs, batch, batch_id
    ):
        # [[junction_locations*8, junction_types*4, junction_visibilities*4,
        # slot_occupancies*1, slot_types*1, junction_orientations*8,
        # slot_orientations*2, slot_score, junction_score*4],...]

        det_results = np.array(det_results)
        if len(det_results) == 0:
            return img
        # point_directions = det_results[:, 18:26].reshape(-1, 4, 2)
        points = det_results[:, 0:8]

        if self.project_pts_to_cameras:
            bev_points = points.reshape(-1, 4, 2) * self.vis_img_scale
            for idx in range(bev_points.shape[0]):
                self.project_pts_to_image(
                    batch_extra_imgs,
                    batch,
                    batch_id,
                    self.bev2vcs_coord(
                        bev_points[idx], self.vcs_range, self.m_perpixel
                    ),
                    (0, 0, 255),
                )
        return ANCBevObjVisualize.gen_slot_map(
            points=points,
            img=img,
            is_pred=True,
            color_map=color_map,
            instance_type=det_results[:, 17],
            available_type=det_results[:, 16],
            ignore_type=None,
            vis_img_scale=self.vis_img_scale,
            color_inline=(255, 0, 255),
        )

    def draw_psd_annotation(
        self,
        slots_global_label,
        img,
        color_map,
        batch_extra_imgs,
        batch,
        batch_id,
    ):
        # global_label: n x 16
        # ignore or Not_normal_slot:
        # [[x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, occupancy, len_global, slot_type, ignore_type],  # noqa
        #     [], [], ...
        # ]
        # normal:
        # [[centerx, centery, slot_type, slot_vec_x, slot_vec_y, occupancy,  # noqa
        #     x1, y1, x2, y2, x3, y3, x4, y4, slot_type, ignore_type],
        #     [], [], ...
        # ]

        # filter ignore labels
        slots_global_label = slots_global_label[slots_global_label[:, -1] != 0]
        # filter Not_normal_slot and Handicapped
        slots_global_label = slots_global_label[
            slots_global_label[:, -2] != -1
        ]
        # draw gt
        ret_img = ANCBevObjVisualize.gen_slot_map(
            points=slots_global_label[:, 6:14],
            img=img,
            is_pred=False,
            color_map=color_map,
            instance_type=slots_global_label[:, -2],
            available_type=slots_global_label[:, 5],
            ignore_type=None,
            vis_img_scale=self.vis_img_scale,
            color_inline=(34, 139, 34),
        )
        if self.project_pts_to_cameras:
            bev_points = (
                slots_global_label[:, 6:14].reshape(-1, 4, 2)
                * self.vis_img_scale
            )
            for idx in range(bev_points.shape[0]):
                self.project_pts_to_image(
                    batch_extra_imgs,
                    batch,
                    batch_id,
                    self.bev2vcs_coord(
                        bev_points[idx], self.vcs_range, self.m_perpixel
                    ),
                    (0, 255, 0),
                )
        return ret_img

    def draw_parkingrod_prediction(
        self, det_results, img, batch_extra_imgs, batch, batch_id
    ):
        # rod_predicts
        # [[endpoint_location*4, parkingrod_type, parkingrod_score], [], ...]

        det_results = np.array(det_results)
        if len(det_results) == 0:
            return img
        points = det_results[:, 0:4]

        if self.project_pts_to_cameras:
            bev_points = points.reshape(-1, 2, 2) * self.vis_img_scale * 2.0
            for idx in range(bev_points.shape[0]):
                self.project_pts_to_image(
                    batch_extra_imgs,
                    batch,
                    batch_id,
                    self.bev2vcs_coord(
                        bev_points[idx], self.vcs_range, self.m_perpixel
                    ),
                    (255, 0, 0),
                )
        return ANCBevObjVisualize.gen_parkingrod_map(
            points=points,
            img=img,
            is_pred=True,
            vis_img_scale=self.vis_img_scale,
            center_shift=self.center_shift,
        )

    def draw_parkingrod_annotation(
        self,
        rod_labels,
        img,
        batch_extra_imgs,
        batch,
        batch_id,
    ):

        # rod_labels
        #      rod_labels_bs: batch * n * 17
        #     ignore:
        #     [[x0, y0, x1, y1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0 len_rod, rod_type, ignore_type], [], ...]  # noqa
        #     normal:
        #     [[x_center, y_center, rod_type, x0, y0, x1, y1, slot_x0, slot_y0, slot_x1, slot_y1,slot_x2, slot_y2, slot_x3, slot_y3, rod_type, ignore_type], [], ...]  # noqa

        idx_valid = np.sum(rod_labels, axis=1) > np.finfo(np.float32).eps
        rod_labels = rod_labels[idx_valid]  # 17 x 17
        # filter ignore labels
        rod_labels = rod_labels[rod_labels[:, -1] != 0]
        # delete labels out of valid range
        idx_in_range = self.is_in_vcs_range(
            self.roi_vcs_range,
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
        normal_rod_label = rod_labels[idx_rod_normal]  # 7 x 17

        # draw gt
        ret_img = ANCBevObjVisualize.gen_parkingrod_map(
            points=normal_rod_label[:, 3:7],
            img=img,
            is_pred=False,
            vis_img_scale=self.vis_img_scale,
            center_shift=self.center_shift,
        )
        if self.project_pts_to_cameras:
            bev_points = (
                normal_rod_label[:, 3:7].reshape(-1, 2, 2)
                * self.vis_img_scale
                * 2.0
            )
            for idx in range(bev_points.shape[0]):
                self.project_pts_to_image(
                    batch_extra_imgs,
                    batch,
                    batch_id,
                    self.bev2vcs_coord(
                        bev_points[idx], self.vcs_range, self.m_perpixel
                    ),
                    (0, 255, 255),
                )
        return ret_img

    @staticmethod
    def gen_slot_map(
        points,
        img,
        is_pred,
        color_map,
        instance_type,
        available_type,
        ignore_type=None,
        color_inline=(0, 0, 255),
        vis_img_scale=1,
    ):
        """Generate a slot map image with annotated slots and their properties.

        Args:
            points: A list of corner points defining slots.
                Represented by four corner points.
            img: The base image on which the slot map will be generated.
            is_pred: Msg is from pred or annotation.
            color_map: color map for slot
            instance_type: List of instance types corresponding each slot.
            available_type: List of available types corresponding each slot.
            ignore_type: List of ignore types corresponding each slot.
            color_inline: color for drawing slot_in_line.

        Returns:
            img: The generated slot map image.
        """
        color_gt = (0, 255, 0)
        # occupancy_idx = (0, 2) if is_pred else (1, 3)
        points = np.array(points)
        if len(points) == 0:
            return img
        junction_locations = points.reshape(-1, 4, 2) * vis_img_scale  # *5
        center_points = junction_locations.mean(1).astype(np.int)
        for junction_location, type_slot, available_slot, center_slot in zip(
            junction_locations, instance_type, available_type, center_points
        ):
            color_boundary = color_map[int(type_slot)] if is_pred else color_gt
            # draw slot line + pred_slot_type
            cv2.line(
                img,
                tuple(junction_location[0].astype(np.int)),
                tuple(junction_location[1].astype(np.int)),
                color_inline,
                2,
            )
            cv2.line(
                img,
                tuple(junction_location[1].astype(np.int)),
                tuple(junction_location[2].astype(np.int)),
                color_boundary,
                2,
            )
            cv2.line(
                img,
                tuple(junction_location[2].astype(np.int)),
                tuple(junction_location[3].astype(np.int)),
                color_boundary,
                2,
            )
            cv2.line(
                img,
                tuple(junction_location[3].astype(np.int)),
                tuple(junction_location[0].astype(np.int)),
                color_boundary,
                2,
            )
            # draw anno_slot_type
            text_occ = slot_occupancy_dict[int(available_slot)]
            text_type = slot_type_mark_dict[int(type_slot)]
            text = f"{text_occ},{text_type}"
            text_pos = [center_slot[0] - 50, center_slot[1]]
            if is_pred:
                text_pos[1] += 20
            else:
                text_pos[1] -= 20
            cv2.putText(
                img,
                text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color_boundary,
                2,
            )
        # draw is_ignore
        if ignore_type is not None:
            for i, ct in enumerate(center_points):
                cv2.putText(
                    img,
                    slot_ignore_mark_dict[int(ignore_type[i])],
                    (ct[0], int(ct[1] + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.1,
                    color_inline,
                    1,
                )
        return img

    def is_in_vcs_range(self, roi_vcs_range, ptx, pty):
        """Determine whether the point is in roi_vcs_range.

        roi_vcs_range: eval vcs range. (bottom, right, top, left)
        ptx: x of point in bev Coordinate.
        pty: y of point in bev Coordinate.
        """

        eval_bev_range = (
            (self.vcs_range[3] - roi_vcs_range[3])
            / self.resolution_m_per_grid,
            (self.vcs_range[2] - roi_vcs_range[2])
            / self.resolution_m_per_grid,
            (self.vcs_range[3] - roi_vcs_range[1])
            / self.resolution_m_per_grid,
            (self.vcs_range[2] - roi_vcs_range[0])
            / self.resolution_m_per_grid,
        )
        in_range_xmin = ptx > eval_bev_range[0]
        in_range_xmax = ptx < eval_bev_range[2]
        in_range_ymin = pty > eval_bev_range[1]
        in_range_ymax = pty < eval_bev_range[3]
        in_range_x = np.logical_and(in_range_xmin, in_range_xmax)
        in_range_y = np.logical_and(in_range_ymin, in_range_ymax)
        in_range = np.logical_and(in_range_x, in_range_y)
        return in_range

    @staticmethod
    def gen_parkingrod_map(
        points,
        img,
        is_pred,
        vis_img_scale,
        center_shift,
    ):
        color_gt = (0, 255, 255)
        color_map = (255, 0, 0)
        points = np.array(points)
        if len(points) == 0:
            return img
        endpoint_locations = (
            points[:, 0:4].reshape(-1, 2, 2) * vis_img_scale * 2.0
        )

        endpoint_locations += center_shift.reshape(1, 1, 2)
        endpoint_locations = endpoint_locations.astype(np.int).tolist()
        color_boundary = color_map if is_pred else color_gt
        for rod in endpoint_locations:
            cv2.line(img, rod[0], rod[1], color_boundary, 2)
        return img

    def gen_parkingrod_result_map(self, det_results, img):
        # [[endpoint_location*4, parkingrod_type, parkingrod_score], [], ...]
        det_results = np.array(det_results)
        if len(det_results) == 0:
            return img
        endpoint_locations = (
            det_results[:, 0:4].reshape(-1, 2, 2) * self.vis_img_scale * 2.0
        )
        endpoint_locations += self.center_shift.reshape(1, 1, 2)
        endpoint_locations = endpoint_locations.astype(np.int).tolist()
        for rod in endpoint_locations:
            cv2.line(img, rod[0], rod[1], (0, 255, 255), 2)
        return img

    def vis_ego_area(self, img_show, color=(255, 255, 255)):
        """Draw ego area.

        Args:
            img_show: base image.
            color: color of ego area, default white.
        """
        center_loc = [
            int(
                self.vcs_range[3]
                / (self.vcs_range[3] - self.vcs_range[1])
                * img_show.shape[1]
            ),
            int(
                self.vcs_range[2]
                / (self.vcs_range[2] - self.vcs_range[0])
                * img_show.shape[0]
            ),
        ]
        vcs_x_y_range = (
            self.vcs_range[2] - self.vcs_range[0],
            self.vcs_range[3] - self.vcs_range[1],
        )
        vcs_max_range_idx = np.argmax(vcs_x_y_range)
        world_width = int(vcs_x_y_range[vcs_max_range_idx])
        bev_pixel_meter = (
            img_show.shape[vcs_max_range_idx] / world_width
        )  # 1 meter -> x pixels
        # draw ego car
        self_lt = (
            int((center_loc[0] - 1.8 / 2 * bev_pixel_meter)),
            int((center_loc[1] - 4 / 2 * bev_pixel_meter)),
        )  # noqa
        self_rb = (
            int((center_loc[0] + 1.8 / 2 * bev_pixel_meter)),
            int((center_loc[1] + 4 / 2 * bev_pixel_meter)),
        )  # noqa
        front_arrow = (center_loc[0], int(center_loc[1] - 4 * bev_pixel_meter))
        cv2.rectangle(img_show, self_lt, self_rb, color, -1)
        cv2.line(img_show, center_loc, front_arrow, color, 2)
        # draw circle
        for meter in range(0, world_width, self.draw_interval):
            color = (200, 200, 200)
            img_show = cv2.circle(
                img_show,
                center_loc,
                int(bev_pixel_meter * meter),
                color,
                1,
            )  # noqa
        # draw stright line
        cv2.line(
            img_show,
            (img_show.shape[1] // 2, 0),
            (img_show.shape[1] // 2, img_show.shape[0]),
            color,
            1,
        )
        cv2.line(
            img_show,
            (0, center_loc[1]),
            (img_show.shape[1], center_loc[1]),
            color,
            1,
        )

    def vis_disc_obj(
        self,
        img,
        vcs_loc,
        vcs_dim,
        yaw,
        vcs_range,
        m_perpixel,
        color=(0, 255, 0),
        line_thicks=3,
        text=None,
        txt_idx=0,
        vis_img_scale=1,
        show_center_point=True,
    ):
        bev_loc = self.vcs2bev_coord(vcs_loc, vcs_range, m_perpixel).flatten()
        bev_dim = [vcs_dim[0] / m_perpixel[0], vcs_dim[1] / m_perpixel[1]]

        box = self.get_vertices_from_box(
            bev_dim, bev_loc, yaw + np.pi / 2, is_right_hand_coor=False
        )
        vcs_points = self.get_vertices_from_box(
            vcs_dim, vcs_loc, yaw, is_right_hand_coor=True
        )
        box = np.int0(box)
        if show_center_point:
            cv2.circle(img, tuple(bev_loc), 3, color, -1)
        cv2.line(
            img, tuple(box[0]), tuple(box[1]), color, line_thicks, cv2.LINE_AA
        )
        cv2.line(
            img, tuple(box[1]), tuple(box[2]), color, line_thicks, cv2.LINE_AA
        )
        cv2.line(
            img, tuple(box[2]), tuple(box[3]), color, line_thicks, cv2.LINE_AA
        )
        cv2.line(
            img, tuple(box[3]), tuple(box[0]), color, line_thicks, cv2.LINE_AA
        )
        if self.vis_ego:
            self.vis_ego_area(img)
        if text is not None:
            left_x_coords = [i[0] for i in box if i[0] > 0]
            left_y_coords = [i[1] for i in box if i[1] > 0]
            if left_x_coords and left_y_coords:
                left_x = min(left_x_coords)
                left_y = min(left_y_coords)
                text_coord = (left_x, left_y)
            else:
                text_coord = tuple(box[txt_idx])
            cv2.putText(
                img,
                text,
                text_coord,
                cv2.FONT_HERSHEY_COMPLEX,
                0.3 * vis_img_scale,
                (225, 225, 225),
                thickness=1,
            )
        return vcs_points

    def get_prefix(self, range_mode, task_name, category):
        if range_mode == "wide":
            return f"{task_name}_bev_stage2_{category}_head_predict_"
        else:
            return f"{task_name}_bev_stage2_{category}_small_head_predict_"

    def project_pts_to_image(
        self,
        batch_extra_imgs,
        batch,
        ind,
        vcs_points,
        label_color,
    ):
        """Save bev 3d vis results.

        Args:
            batch_extra_imgs: origin images.
            batch: items contains input's data and GT annos.
            ind: batch index.
            vcs_points: points in vcs.
            label_color: color
        """
        for idx, img in enumerate(batch_extra_imgs):
            img_h, _ = img.shape[:2]
            camera_name = self.camera_view_names[idx]
            if "fisheye" in camera_name:
                vir_camera = FisheyeCamera()
            else:
                vir_camera = PinholeCamera()

            if "calib_path" not in batch[self.meta_name]:
                meta = batch[self.meta_name]
                intrinsic = meta["intrinsics"][idx][ind][0].cpu().numpy()
                vcs2cam = meta["T_vcs2cam"][idx][ind][0].cpu().numpy()
                image_size = (
                    meta["img_shape"][idx][ind][0].cpu().numpy().tolist()
                )
                distort = (
                    meta["distort_coeffs"][idx][ind].cpu().numpy().tolist()
                )
                if "fisheye" in camera_name:
                    distort = distort[:4]
                vir_camera = vir_camera.init_cam_param_by_matrix(
                    image_size, intrinsic, vcs2cam, distort, is_virtual=False
                )
            else:
                attribute_file = os.path.join(
                    batch[self.meta_name]["calib_path"][ind], "attribute.json"
                )
                if os.path.exists(attribute_file):
                    if attribute_file in self.project_calib:
                        calibration = self.project_calib[attribute_file]
                    else:
                        calibration = json.load(open(attribute_file, "r"))
                        self.project_calib[attribute_file] = calibration
                    camera_clib = calibration["calibration"][camera_name]
                else:
                    attribute_file = os.path.join(
                        batch[self.meta_name]["calib_path"][ind],
                        "calibration.json",
                    )
                    assert os.path.exists(
                        attribute_file
                    ), "Please check the calibration file!"
                    if attribute_file in self.project_calib:
                        calibration = self.project_calib[attribute_file]
                    else:
                        calibration = json.load(open(attribute_file, "r"))
                        self.project_calib[attribute_file] = calibration
                    camera_clib = calibration[
                        camera_view_rename[camera_name] + "_json"
                    ]

                camera_scale = camera_clib["image_height"] // img_h
                if "K" in camera_clib.keys():
                    camera_clib["K"][0][0] /= camera_scale
                    camera_clib["K"][1][1] /= camera_scale
                    camera_clib["K"][0][2] /= camera_scale
                    camera_clib["K"][1][2] /= camera_scale

                camera_clib["center_u"] /= camera_scale
                camera_clib["center_v"] /= camera_scale
                camera_clib["image_height"] /= camera_scale
                camera_clib["image_width"] /= camera_scale
                camera_clib["focal_u"] /= camera_scale
                camera_clib["focal_v"] /= camera_scale
                vir_camera = vir_camera.init_cam_param_by_dict(camera_clib)
            cl_vcs = []
            for pnt_vcs in vcs_points:  # for vcs
                p = [pnt_vcs[0], pnt_vcs[1], 0, 1.0]  # for vcs
                cl_vcs.append(p)
            pnts_pixel = vir_camera.project_vcs2pixel(cl_vcs)
            flag = vir_camera.is_points_in_fov(
                vir_camera.project_vcs2cam(cl_vcs), axis=2
            )
            pnts_pixel_in_fov = pnts_pixel[flag]
            pnts_pixel_in_fov = [
                [int(xy) for xy in p] for p in pnts_pixel_in_fov
            ]
            img_poly = img.copy()
            img_point = img.copy()
            for p_pixel in pnts_pixel_in_fov:
                cv2.circle(img_point, tuple(p_pixel), 5, label_color, -1)
            if len(pnts_pixel_in_fov) == 4:
                pnts_pixel_in_fov = np.array(pnts_pixel_in_fov)
                cv2.fillPoly(img_poly, [pnts_pixel_in_fov], label_color)
                img = cv2.addWeighted(img_poly, 0.3, img_point, 0.7, 0)
            else:
                img = img_point
            batch_extra_imgs[idx] = img
        return batch_extra_imgs

    def class_colorbar(self, task_name):
        """Draw color bar.

        Args:
            task_name: task name, optional crosswalk, stopline, arrow,
                junction, roadmarking, static_obstacle, sod3d.
        """
        assert task_name in self.color_map.keys(), f"unsupport {task_name}!!!"
        color_img = (
            np.ones(
                [50, (max(self.color_map[task_name].keys()) + 2) * 100, 3],
                dtype=np.uint8,
            )
            * 255
        )
        for cls_id in self.color_map[task_name].keys():
            cv2.rectangle(
                color_img,
                (cls_id * 100, 0),
                (100 * cls_id + 100, 20),
                self.color_map[task_name][cls_id],
                -1,
            )
            cv2.putText(
                color_img,
                self.id2label[task_name][cls_id],
                (cls_id * 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
        # draw gt color bar
        cv2.rectangle(
            color_img,
            (cls_id * 100 + 100, 0),
            (100 * cls_id + 200, 20),
            (17, 193, 101),
            -1,
        )
        cv2.putText(
            color_img,
            "GT",
            (cls_id * 100 + 100, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        return color_img

    def format_images(
        self,
        origin_imgs,
        bev_img,
        task_name,
        img_name,
        draw_class_color_bar=True,
    ):
        """Make picture stitching.

        Args:
            origin_imgs: original images.
            bev_img: ipm image.
            task_name: task name.
            img_name: final image name.
            draw_class_color_bar: whether draw class colorbar
                in stitch picture.
        """
        color_imgs_dict = {
            view_name: cv2.resize(img, self.out_ori_img_size)
            for view_name, img in zip(self.camera_view_names, origin_imgs)
        }
        color_imgs_dict["tmp_img"] = np.zeros_like(
            color_imgs_dict[list(color_imgs_dict.keys())[0]]
        )

        def _combine_cams(cam_names):
            """Stitching per view images.

            Args:
                cam_names: per view camera names.
            """
            have_img = False
            output_imgs = []
            for cam_name in cam_names:
                if cam_name not in color_imgs_dict:
                    cam_name = "tmp_img"
                else:
                    have_img = True
                output_imgs.append(color_imgs_dict[cam_name])
            if have_img:
                return np.hstack(output_imgs)
            else:
                return None

        all_imgs = []
        max_columns = max([len(v) for v in self.camera_layouts])
        for row_cams in self.camera_layouts:
            real_row_cams = row_cams + ["placeholder"] * (
                max_columns - len(row_cams)
            )
            row_img = _combine_cams(real_row_cams)
            if row_img is not None:
                all_imgs.append(row_img)

        img_all = np.vstack(all_imgs)
        cv2.putText(
            img_all,
            img_name,
            (0, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 255, 255),  # bgr
            thickness=2,
        )
        if draw_class_color_bar:
            color_bar = self.class_colorbar(task_name)
            resize_color_bar = cv2.resize(
                color_bar,
                (
                    img_all.shape[1],
                    color_bar.shape[0]
                    * img_all.shape[1]
                    // color_bar.shape[1],
                ),
            )
            img_all = np.vstack((resize_color_bar, img_all))
        if self.is_bev_horizon:
            target_bev_size = (
                img_all.shape[1],
                (img_all.shape[1] * bev_img.shape[0]) // bev_img.shape[1],
            )
        else:
            target_bev_size = (
                (img_all.shape[0] * bev_img.shape[1]) // bev_img.shape[0],
                img_all.shape[0],
            )
        bev_img = cv2.resize(
            bev_img,
            target_bev_size,
        )
        if self.is_bev_horizon:
            img_all = np.vstack([img_all, bev_img])
        else:
            img_all = np.hstack([img_all, bev_img])
        return img_all

    def on_batch_end(self, batch, model_outs, **kwargs):
        self.__call__(batch, model_outs, self.task, self.output_dir)

    def __call__(self, batch, results, tasks, save_dir):
        self.visual_count += 1
        if self.visual_count % self.visual_interval != 0:
            return

        if self.reformat_fn:
            results = self.reformat_fn(results)

        results = convert_numpy(results)

        do_pack_infer = True
        if isinstance(batch, tuple):  # multidataloader
            batch = batch[0]
            do_pack_infer = False

        batch_size = batch["img"][0].shape[0]

        for ind in range(batch_size):
            if "img_vis" in batch:
                batch_extra_imgs = [
                    i[ind].cpu().numpy().squeeze().transpose(1, 2, 0)
                    for i in batch["img_vis"][0]
                ]
                name = [
                    int(timestamp * 1000)
                    for timestamp in batch["timestamp"].cpu().numpy()
                ][ind]
            else:
                batch_extra_imgs = []
                if "img" in batch:
                    num_views = 1
                    batch_extra_imgs += [
                        img.detach().cpu().numpy() * 128 + 128
                        for img in batch["img"][0][
                            ind * num_views : (ind + 1) * num_views
                        ]  # noqa
                    ]
                if "side_img" in batch:
                    num_views = 5
                    batch_extra_imgs += [
                        img.detach().cpu().numpy() * 128 + 128
                        for img in batch["side_img"][0][
                            ind * num_views : (ind + 1) * num_views
                        ]  # noqa
                    ]
                if "round_img" in batch:
                    num_views = 4
                    batch_extra_imgs += [
                        img.detach().cpu().numpy() * 128 + 128
                        for img in batch["round_img"][0][
                            ind * num_views : (ind + 1) * num_views
                        ]  # noqa
                    ]
                if "narrow_img" in batch:
                    num_views = 1
                    batch_extra_imgs += [
                        img.detach().cpu().numpy() * 128 + 128
                        for img in batch["narrow_img"][0][
                            ind * num_views : (ind + 1) * num_views
                        ]  # noqa
                    ]
                batch_extra_imgs = [
                    cv2.cvtColor(
                        img.astype(np.uint8).transpose((1, 2, 0)),
                        cv2.COLOR_YUV2BGR,
                    )
                    for img in batch_extra_imgs
                ]
                name = [
                    int(timestamp * 1000)
                    for timestamp in batch["timestamp"].cpu().numpy()
                ][ind]
            # raw_size = [
            #     int(self.bev_size[1] * self.vis_img_scale),
            #     int(self.bev_size[0] * self.vis_img_scale),
            # ]
            # ret_img = batch[self.ipm_name][ind].cpu().numpy().astype("uint8")
            # ret_img_raw = cv2.resize(ret_img, raw_size)
            ret_img_raw = (
                batch[self.ipm_name][ind].cpu().numpy().astype("uint8")
            )
            for task in _as_list(tasks):
                ret_img = ret_img_raw.copy()
                for res_key, anno_name in self.res_key2anno_name.items():
                    if "small" in res_key:
                        res_key = res_key.split("_small")[0]
                    # for pack infer
                    if do_pack_infer and res_key != task:
                        res_key = task
                    category = (
                        res_key.split("bev_")[1]
                        if "bev" in res_key
                        else res_key
                    )
                    if category in ["psd", "parking", "parkingrod"]:
                        # draw psd msg
                        prefix = self.prefix or self.get_prefix(
                            self.range_mode, task, "psd"
                        )
                        # draw psd pred for pack-infer
                        if prefix + "decode_label" in results:
                            pred_slots = results[prefix + "decode_label"][
                                "decode_slots"
                            ][ind]
                            ret_img = self.draw_psd_prediction(
                                pred_slots,
                                ret_img,
                                self.color_map[category],
                                batch_extra_imgs,
                                batch,
                                ind,
                            )
                        # draw psd pred for val-viz
                        if prefix + "decode_label_decode_slots" in results:
                            pred_slots = results[
                                prefix + "decode_label_decode_slots"
                            ][ind]
                            ret_img = self.draw_psd_prediction(
                                pred_slots,
                                ret_img,
                                self.color_map[category],
                                batch_extra_imgs,
                                batch,
                                ind,
                            )
                        # draw psd gt for val-viz
                        if "annos_bev_psd_obj_small" in batch:
                            slots_global_label = (
                                batch["annos_bev_psd_obj_small"]["global"][ind]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            ret_img = self.draw_psd_annotation(
                                slots_global_label,
                                ret_img,
                                self.color_map[category],
                                batch_extra_imgs,
                                batch,
                                ind,
                            )
                        if "annos_bev_psd_obj" in batch:
                            slots_global_label = (
                                batch["annos_bev_psd_obj"]["global"][ind]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            ret_img = self.draw_psd_annotation(
                                slots_global_label,
                                ret_img,
                                self.color_map[category],
                                batch_extra_imgs,
                                batch,
                                ind,
                            )
                        # draw parkingrod msg
                        prefix = self.prefix or self.get_prefix(
                            self.range_mode, task, "parkingrod"
                        )
                        task_key = prefix + "preds_parkingrod"
                        if task_key in results:
                            pred_parkingrods = results[task_key][ind]
                            ret_img = self.draw_parkingrod_prediction(
                                pred_parkingrods,
                                ret_img,
                                batch_extra_imgs,
                                batch,
                                ind,
                            )
                        if "annos_bev_parkingrod_obj" in batch:
                            label_parkingrods = (
                                batch["annos_bev_parkingrod_obj"][
                                    "parking_rod"
                                ][ind]
                                .detach()
                                .cpu()
                                .numpy()
                            )  # label_parkingrods gt B x 13
                            ret_img = self.draw_parkingrod_annotation(
                                label_parkingrods,
                                ret_img,
                                batch_extra_imgs,
                                batch,
                                ind,
                            )
                    else:
                        if self.prefix is None:
                            prefix = self.get_prefix(
                                self.range_mode, task, category
                            )
                        else:
                            prefix = task + "_" + self.prefix + "_"
                        ct = results[prefix + "pred_bev_discobj_ct"][ind]
                        wh = results[prefix + "pred_bev_discobj_wh"][ind]
                        rot = results[prefix + "pred_bev_discobj_rot"][ind]
                        score = results[prefix + "pred_bev_discobj_score"][ind]
                        cls_id = results[prefix + "pred_bev_discobj_cls_id"][
                            ind
                        ]
                        include_box = False
                        for j in range(len(score)):
                            # filter fake data
                            if cls_id[j] == -1:
                                continue
                            score_ = score[j]
                            label_color = self.color_map[category][cls_id[j]]
                            if "arrow" in task:
                                text = self.id2label[category][cls_id[j]]
                            elif "sod3d" in task:
                                text = None
                            else:
                                text = category[0] + "_" + str(int(cls_id[j]))
                            if score_ >= self.scor_thr:
                                include_box = True
                                pred_vcs_points = self.vis_disc_obj(
                                    ret_img,
                                    ct[j],
                                    wh[j],
                                    rot[j],
                                    self.vcs_range,
                                    self.m_perpixel,
                                    color=label_color,
                                    line_thicks=1,
                                    text=text,
                                    vis_img_scale=self.vis_img_scale,
                                    show_center_point=False
                                    if "sod3d" in task
                                    else True,
                                )
                                if self.project_pts_to_cameras:
                                    batch_extra_imgs = (
                                        self.project_pts_to_image(
                                            batch_extra_imgs,
                                            batch,
                                            ind,
                                            pred_vcs_points,
                                            label_color,
                                        )
                                    )
                            if self.range_mode == "wide" and not include_box:
                                self.vis_ego_area(ret_img, (255, 0, 0))

                        if anno_name in batch:
                            annos = batch[anno_name]
                            vcs_discobj_cls = annos["vcs_discobj_cls"][ind]
                            vcs_discobj_loc = annos["vcs_discobj_loc"][ind]
                            vcs_discobj_wh = annos["vcs_discobj_wh"][ind]
                            vcs_discobj_ignore = annos["vcs_discobj_ignore"][
                                ind
                            ]
                            vcs_discobj_yaw = annos["vcs_discobj_yaw"][ind]
                            # filter the placeholder bboxes
                            mask = vcs_discobj_cls != -1
                            if not mask.any():
                                continue
                            vcs_discobj_cls = (
                                vcs_discobj_cls[mask].detach().cpu().numpy()
                            )
                            vcs_discobj_loc = (
                                vcs_discobj_loc[mask].detach().cpu().numpy()
                            )
                            vcs_discobj_wh = (
                                vcs_discobj_wh[mask].detach().cpu().numpy()
                            )
                            vcs_discobj_yaw = (
                                vcs_discobj_yaw[mask].detach().cpu().numpy()
                            )
                            vcs_discobj_ignore = (
                                vcs_discobj_ignore[mask].detach().cpu().numpy()
                            )
                            for j in range(len(vcs_discobj_cls)):
                                color = (17, 193, 101)
                                if "arrow" in task or "sod3d" in task:
                                    gt_text = None
                                else:
                                    gt_text = "gt_" + str(
                                        int(vcs_discobj_cls[j])
                                    )
                                if vcs_discobj_ignore[j]:
                                    if gt_text:
                                        gt_text = gt_text.replace("gt", "ig")
                                    else:
                                        gt_text = "ig"
                                    color = (255, 255, 255)

                                gt_vcs_points = self.vis_disc_obj(
                                    ret_img,
                                    vcs_discobj_loc[j],
                                    vcs_discobj_wh[j],
                                    vcs_discobj_yaw[j],
                                    self.vcs_range,
                                    self.m_perpixel,
                                    color=color,
                                    line_thicks=1,
                                    text=gt_text,
                                    vis_img_scale=self.vis_img_scale,
                                    show_center_point=False
                                    if "sod3d" in task
                                    else True,
                                )
                                if self.project_pts_to_cameras:
                                    batch_extra_imgs = (
                                        self.project_pts_to_image(
                                            batch_extra_imgs,
                                            batch,
                                            ind,
                                            gt_vcs_points,
                                            color,
                                        )
                                    )
                if task in ["bev_parking", "bev_static_obstacle"]:
                    task_name = task.split("bev_")[1]
                else:
                    task_name = category
            ret_img = self.format_images(
                origin_imgs=batch_extra_imgs,
                bev_img=ret_img,
                task_name=task_name,
                img_name=str(name),
                draw_class_color_bar=len(_as_list(tasks)) == 1,
            )
            if len(_as_list(tasks)) > 1:
                task_name = "disc"
            cv2.imwrite(f"{save_dir}/{task_name}_{name}.jpg", ret_img)


def vcs2bev_coord(pt, vcs_range, m_perpixel):
    pt = np.array(pt).reshape((-1, 2))
    u = (vcs_range[3] - pt[:, 1]) / m_perpixel[1]
    v = (vcs_range[2] - pt[:, 0]) / m_perpixel[0]
    u = u.reshape((-1, 1))
    v = v.reshape((-1, 1))
    return np.hstack([u, v]).astype(np.int32)


def get_vertices_from_vcs_box(wh, ct, yaw):
    w, h = wh
    ctx, cty = ct

    p0 = [0.5 * w, 0.5 * h, 1]
    p1 = [0.5 * w, -0.5 * h, 1]
    p2 = [-0.5 * w, -0.5 * h, 1]
    p3 = [-0.5 * w, 0.5 * h, 1]
    points = np.array([p0, p1, p2, p3])
    R = np.array(
        [
            [np.cos(yaw), np.sin(yaw), ctx],
            [-np.sin(yaw), np.cos(yaw), cty],
            [0, 0, 1],
        ]
    )
    points = R.dot(points.T).T
    return points[:, :2]


def vis_disc_obj(
    img,
    vcs_loc,
    vcs_dim,
    yaw,
    vcs_range,
    m_perpixel,
    color=(0, 255, 0),
    line_thicks=3,
):
    bev_loc = vcs2bev_coord(vcs_loc, vcs_range, m_perpixel).flatten()
    bev_dim = [vcs_dim[0] / m_perpixel[0], vcs_dim[1] / m_perpixel[1]]

    box = get_vertices_from_vcs_box(bev_dim, bev_loc, yaw + np.pi / 2)
    box = np.int0(box)
    cv2.line(img, tuple(box[0]), tuple(box[1]), color, line_thicks)
    cv2.line(img, tuple(box[1]), tuple(box[2]), color, line_thicks)
    cv2.line(img, tuple(box[2]), tuple(box[3]), color, line_thicks)
    cv2.line(img, tuple(box[3]), tuple(box[0]), color, line_thicks)
