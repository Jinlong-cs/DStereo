# Copyright (c) Horizon Robotics. All rights reserved.

import json
import logging
import os
import re
from typing import Callable, Dict, Sequence

import cv2
import numpy as np

from hat.callbacks.task_visualize import BaseVisualize
from hat.core.affine import get_vcs2bev_img_mat
from hat.core.virtual_camera import FisheyeCamera, PinholeCamera
from hat.metrics.bev.online_mapping_utils import get_target_categorys
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import convert_numpy
from hat.visualize.bev_3d import Bev3DVisualize, init_bev
from hat.visualize.bev_e2e import ANCBevE2EVisualize
from hat.visualize.online_mapping import (
    draw_online_mapping,
    plot_pts_in_raw_img,
)
from hat.visualize.utils import ANCBEVImgStitcher

__all__ = ["ANCBev3DVisualizeV2"]

logger = logging.getLogger(__name__)

VALID_Z = 0.1
COLOR_MAP = {
    "cyclist": (255, 0, 0),  # blue
    "vehicle": (0, 255, 0),  # green
    "pedestrian": (0, 0, 255),  # red
    "vrumerge": (255, 255, 0),  # yellow
}


@OBJECT_REGISTRY.register
class ANCBev3DVisualizeV2(BaseVisualize):
    """
    Visualize class for bev 3d.

    Args:
        bev_size: the size of bird eye view.
        vcs_range: visbile range of bev, (bottom, right, top, left) in order.
        reformat_fn: the function for reformat the task results.
        extra_img: whether need extra 6v imgs.
        score_threshold: the threshold score.
        color: mapping relationship from category to color.
    """

    def __init__(
        self,
        output_dir: str,
        task: str,
        prefix: str,
        bev_size: tuple,
        vcs_range: tuple,
        extra_img: ANCBEVImgStitcher = None,
        score_threshold: float = 0.2,
        color: dict = COLOR_MAP,
        camera_view_names: list = None,
    ):
        super().__init__(output_dir)
        self.task = task
        self.prefix = prefix
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.extra_img = extra_img
        self.score_threshold = score_threshold
        self.color = color
        self.camera_view_names = camera_view_names

    def on_batch_end(self, batch, model_outs, **kwargs):

        model_outs = convert_numpy(model_outs)

        if isinstance(batch, tuple):  # multidataloader
            batch = batch[0]

        batch_size = batch["img"][0].shape[0]

        for ind in range(batch_size):

            ret_img = np.zeros(
                [self.bev_size[0], self.bev_size[1], 3], dtype=np.uint8
            )
            category = self.task.split("_")[-1]
            task_result = model_outs[self.task]

            bev3d_dim = task_result[self.prefix + "_bev3d_dim"][ind]
            bev3d_ct = task_result[self.prefix + "_bev3d_ct"][ind]
            bev3d_loc_z = task_result[self.prefix + "_bev3d_loc_z"][ind]
            bev3d_rot = task_result[self.prefix + "_bev3d_rot"][ind]
            bev3d_score = task_result[self.prefix + "_bev3d_score"][ind]
            bev3d_cls_id = task_result[self.prefix + "_bev3d_cls_id"][ind]

            pred_bboxes = np.concatenate(
                [
                    bev3d_ct,
                    bev3d_loc_z[:, None],
                    bev3d_dim,
                    bev3d_rot[:, None],
                ],
                axis=1,
            )
            score = bev3d_score[:, None]
            class_id = bev3d_cls_id[:, None]

            ret_img = Bev3DVisualize.draw_bev_boxes(
                bev_img=ret_img,
                pred_bboxes=pred_bboxes,
                score=score,
                bev_size=self.bev_size,
                bev_range=self.vcs_range,
                score_threshold=self.score_threshold,
                thickness=1,
                color=self.color[category],
                class_id=class_id,
            )

            scope_H = self.vcs_range[2] - self.vcs_range[0]
            scope_W = self.vcs_range[3] - self.vcs_range[1]
            bev_origin_coord = (
                int(self.vcs_range[3] / scope_W * self.bev_size[1]),
                int(self.vcs_range[2] / scope_H * self.bev_size[0]),
            )
            cv2.circle(
                ret_img, bev_origin_coord, 3, (255, 255, 255), -1, cv2.LINE_AA
            )

            if self.extra_img:
                assert "img_vis" in batch
                assert self.camera_view_names is not None
                project_calib = {}
                project_calib["local_calibs"] = {}  # K, dist, local2cam
                project_calib["local2chassis"] = {}  # local2vcs
                for i, cam in enumerate(self.camera_view_names):
                    project_calib["local_calibs"][cam] = {}

                    project_calib["local_calibs"][cam]["P2"] = (
                        batch["K"].cpu().numpy()[ind][i]
                    )
                    project_calib["local_calibs"][cam]["disCoeffs"] = (
                        batch["dist"].cpu().numpy()[ind][i]
                    )
                    project_calib["local_calibs"][cam]["Tr_vel2cam"] = (
                        batch["local2cam"].cpu().numpy()[ind][i]
                    )

                    project_calib["local2chassis"][cam] = (
                        batch["local2vcs"].cpu().numpy()[ind][i]
                    )

                batch_extra_imgs = [
                    i[ind].cpu().numpy().squeeze().transpose(1, 2, 0)
                    for i in batch["img_vis"][0]
                ]

                # Project the bev3d pred boxes to 6V cam images.
                camera2index_dict = {}
                for i, view_name in enumerate(self.camera_view_names):
                    camera2index_dict[view_name] = i

                batch_extra_imgs = Bev3DVisualize.draw_camera_boxes(
                    bev3d_dim,
                    bev3d_ct,
                    bev3d_loc_z,
                    bev3d_rot,
                    bev3d_score,
                    project_calib,
                    cameras=self.camera_view_names,
                    color=self.color[category],
                    bev3d_cls_id=bev3d_cls_id,
                    color_imgs=batch_extra_imgs,
                    camera2index=camera2index_dict,
                    score_threshold=0.2,
                    use_lidar=False,
                )

                ret_img = self.extra_img(
                    multiview_imgs=batch_extra_imgs, bev_img=ret_img
                )

            name = int(list(batch["timestamp"].cpu().numpy())[ind] * 1000)
            cv2.imwrite(os.path.join(self.output_dir, f"{name}.png"), ret_img)


slot_occupancy_color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
slot_occupancy_mark_dict = {0: "occ", 1: "Not-occ"}
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
class ANCBevObjVisualizeCB(BaseVisualize):
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
        draw_interval (int): The real world interval size. Defaults to 5m.
        vis_ego: Whether to visualize ego, default False, wide range is True.
        score_threshold: the threshold score.
        vis_img_scale: IPM chart and predict the scale size, default 1.
        color_map: mapping relationship from category to color.
        id2label: mapping relationship from category id to category name.
        anno_name (list): for example:
            ["annos_bev_discrete_obj",].
        res_key2anno_name (dict): a mapping of result key of output to
            annotation name of data.
        visual_interval(int): visual frequency.
    """

    def __init__(
        self,
        output_dir: str,
        task: str,
        prefix: str,
        bev_size: tuple,
        vcs_range: tuple,
        res_key2anno_name: dict,
        color_map: dict,
        id2label: dict,
        vis_ego: bool = False,
        project_pts_to_cameras: bool = False,
        camera_view_names: Sequence = None,
        is_bev_horizon: bool = False,
        camera_layouts: Sequence = None,
        draw_interval: int = 5,
        score_threshold: float = 0.2,
        vis_img_scale: float = 1.0,
        visual_interval: int = 1,
    ):
        super().__init__(output_dir)
        self.task = task
        self.prefix = prefix
        self.bev_size = bev_size
        self.project_pts_to_cameras = project_pts_to_cameras
        if project_pts_to_cameras:
            self.project_calib = {}
        self.camera_view_names = camera_view_names
        self.vcs_range = vcs_range
        self.score_threshold = score_threshold
        self.vis_img_scale = vis_img_scale
        self.color_map = color_map
        self.id2label = id2label
        self.res_key2anno_name = res_key2anno_name
        m_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / bev_size[0],
            abs(vcs_range[3] - vcs_range[1]) / bev_size[1],
        )  # bev coord y, x
        self.m_perpixel = [m_p / self.vis_img_scale for m_p in m_perpixel]
        self.is_bev_horizon = is_bev_horizon
        self.vis_ego = vis_ego
        self.draw_interval = draw_interval
        self.visual_interval = visual_interval
        self.visual_count = 0
        self.meta_name = "meta_info"
        self.ipm_name = "ipm"

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

    def get_vertices_from_box(self, wh, ct, yaw, is_right_hand_coor):
        """
        Calculate and return the vertices from a box.

        Args:
            wh (tuple): The width and height of the box.
            ct (tuple): The center coordinates of the box.
            yaw (float): The yaw angle of the box.
            is_right_hand_coor (bool): Flag indicating whether 2D coordinate
                system is right-handed, for example, BEV
                is left-hand, VCS is right-hand.

        Returns:
            vertices (list): A list of tuples
                representing the vertices of the box in the VCS or BEV.

        Note:
            The VCS is a right-handed coordinate system
                where the x-axis points forward, the y-axis points to the left.
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

    def draw_psd_prediction(self, det_results, img, color_map):
        # [[junction_locations*8, junction_types*4, junction_visibilities*4,
        # slot_occupancies*1, slot_types*1, junction_orientations*8,
        # slot_orientations*2, slot_score, junction_score*4],...]

        det_results = np.array(det_results)
        if len(det_results) == 0:
            return img
        # point_directions = det_results[:, 18:26].reshape(-1, 4, 2)
        points = det_results[:, 0:8]
        return ANCBevObjVisualizeCB.gen_slot_map(
            points=points,
            img=img,
            is_pred=True,
            color_map=color_map,
            instance_type=det_results[:, 17],
            available_type=det_results[:, 16],
            ignore_type=None,
        )

    def draw_psd_annotation(self, slots_global_label, img, color_map):
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
        ret_img = ANCBevObjVisualizeCB.gen_slot_map(
            points=slots_global_label[:, 6:14],
            img=img,
            is_pred=False,
            color_map=color_map,
            instance_type=slots_global_label[:, -2],
            available_type=slots_global_label[:, 5],
            ignore_type=None,
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
    ):
        """
        Generate a slot map image with annotated slots and their properties.

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
        color_gt = (17, 193, 101)
        occupancy_idx = (0, 2) if is_pred else (1, 3)
        points = np.array(points)
        if len(points) == 0:
            return img
        junction_locations = points.reshape(-1, 4, 2)  # *5
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
                1,
            )
            cv2.line(
                img,
                tuple(junction_location[1].astype(np.int)),
                tuple(junction_location[2].astype(np.int)),
                color_boundary,
                1,
            )
            cv2.line(
                img,
                tuple(junction_location[2].astype(np.int)),
                tuple(junction_location[3].astype(np.int)),
                color_boundary,
                1,
            )
            cv2.line(
                img,
                tuple(junction_location[3].astype(np.int)),
                tuple(junction_location[0].astype(np.int)),
                color_boundary,
                1,
            )
            # draw anno_slot_type
            if not is_pred:
                cv2.putText(
                    img,
                    slot_type_mark_dict[int(type_slot)],
                    (center_slot[0], center_slot[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color_inline,
                    2,
                )
            # draw is_occupancy
            if not available_slot:
                cv2.line(
                    img,
                    tuple(junction_location[occupancy_idx[0]].astype(np.int)),
                    tuple(junction_location[occupancy_idx[1]].astype(np.int)),
                    color_boundary,
                    1,
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

    def gen_parkingrod_result_map(self, det_results, img):
        # [[endpoint_location*4, parkingrod_type, parkingrod_score], [], ...]
        img1, img2 = img.copy(), img.copy()
        det_results = np.array(det_results)
        if len(det_results) == 0:
            return img
        endpoint_locations = (
            det_results[:, 0:4].reshape(-1, 2, 2) * self.vis_img_scale * 2.0
        )
        for i, endpoint_location in enumerate(endpoint_locations):  # noqa
            cv2.line(
                img1,
                tuple(endpoint_location[0].astype(np.int)),
                tuple(endpoint_location[1].astype(np.int)),
                (0, 255, 255),
                1,
            )
        img1 = cv2.addWeighted(img1, 0.8, img2, 0.2, 0)
        return img1

    def vis_ego_area(self, img_show, color=(255, 255, 255)):
        """
        Draw ego area.

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

    def project_pts_to_image(
        self,
        batch_extra_imgs,
        batch,
        ind,
        vcs_points,
        label_color,
    ):
        """
        Save bev 3d vis results.

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
            batch_extra_imgs[idx] = img
        return batch_extra_imgs

    def class_colorbar(self, task_name):
        """
        Draw color bar.

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
        """
        Make picture stitching.

        Args:
            origin_imgs: original images.
            bev_img: ipm image.
            task_name: task name.
            img_name: final image name.
            draw_class_color_bar: whether draw class colorbar
                in stitch picture.
        """
        color_imgs_dict = {
            view_name: cv2.resize(img, (2048 // 2, 1280 // 2))
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
            (0, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3,
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
        self.visual_count += 1
        if self.visual_count % self.visual_interval != 0:
            return

        model_outs = convert_numpy(model_outs)

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
            raw_size = [
                int(self.bev_size[1] * self.vis_img_scale),
                int(self.bev_size[0] * self.vis_img_scale),
            ]
            ret_img = batch[self.ipm_name][ind].cpu().numpy().astype("uint8")
            ret_img_raw = cv2.resize(ret_img, raw_size)
            ret_img = ret_img_raw.copy()

            for res_key, anno_name in self.res_key2anno_name.items():
                # for pack infer
                if res_key != self.task:
                    res_key = self.task
                task_result = model_outs[res_key]
                category = res_key.split("bev_")[1]
                if category in ["psd", "parking", "parkingrod"]:
                    # draw psd msg
                    # draw psd pred for pack-infer
                    if self.prefix + "_decode_label" in task_result:
                        pred_slots = task_result[
                            self.prefix + "_decode_label"
                        ]["decode_slots"][ind]
                        ret_img = self.draw_psd_prediction(
                            pred_slots,
                            ret_img,
                            color_map=self.color_map[category],
                        )
                    # draw psd pred for val-viz
                    if (
                        self.prefix + "_decode_label_decode_slots"
                        in task_result
                    ):
                        pred_slots = task_result[
                            self.prefix + "_decode_label_decode_slots"
                        ][ind]
                        ret_img = self.draw_psd_prediction(
                            pred_slots,
                            ret_img,
                            color_map=self.color_map[category],
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
                            color_map=self.color_map[category],
                        )
                    # draw parkingrod msg
                    task_key = self.prefix + "_preds_parkingrod"
                    if task_key in task_result:
                        pred_parkingrods = task_result[task_key][ind]
                        ret_img = self.gen_parkingrod_result_map(
                            pred_parkingrods, ret_img
                        )
                else:
                    ct, wh, rot, score, cls_id = None, None, None, None, None
                    for key, val in task_result.items():
                        if re.match("^.*_pred_bev_discobj_ct", key):
                            ct = val[ind]
                        if re.match("^.*_pred_bev_discobj_wh", key):
                            wh = val[ind]
                        if re.match("^.*_pred_bev_discobj_rot", key):
                            rot = val[ind]
                        if re.match("^.*_pred_bev_discobj_score", key):
                            score = val[ind]
                        if re.match("^.*_pred_bev_discobj_cls_id", key):
                            cls_id = val[ind]
                    assert (
                        ct is not None
                        and wh is not None
                        and rot is not None
                        and score is not None
                        and cls_id is not None
                    )

                    include_box = False
                    for j in range(len(score)):
                        # filter fake data
                        if cls_id[j] == -1:
                            continue
                        score_ = score[j]
                        label_color = self.color_map[category][cls_id[j]]
                        if "arrow" in self.task:
                            text = self.id2label[category][cls_id[j]]
                        elif "sod3d" in self.task:
                            text = None
                        else:
                            text = category[0] + "_" + str(int(cls_id[j]))
                        if score_ >= self.score_threshold:
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
                                if "sod3d" in self.task
                                else True,
                            )
                            if self.project_pts_to_cameras:
                                batch_extra_imgs = self.project_pts_to_image(
                                    batch_extra_imgs,
                                    batch,
                                    ind,
                                    pred_vcs_points,
                                    label_color,
                                )
                        if not include_box:
                            self.vis_ego_area(ret_img, (255, 0, 0))

                    if anno_name in batch:
                        annos = batch[anno_name]
                        vcs_discobj_cls = annos["vcs_discobj_cls"][ind]
                        vcs_discobj_loc = annos["vcs_discobj_loc"][ind]
                        vcs_discobj_wh = annos["vcs_discobj_wh"][ind]
                        vcs_discobj_ignore = annos["vcs_discobj_ignore"][ind]
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
                            if "arrow" in self.task or "sod3d" in self.task:
                                gt_text = None
                            else:
                                gt_text = "gt_" + str(int(vcs_discobj_cls[j]))
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
                                if "sod3d" in self.task
                                else True,
                            )
                            if self.project_pts_to_cameras:
                                batch_extra_imgs = self.project_pts_to_image(
                                    batch_extra_imgs,
                                    batch,
                                    ind,
                                    gt_vcs_points,
                                    color,
                                )
            if self.task in ["bev_parking", "bev_static_obstacle"]:
                task_name = self.task.split("bev_")[1]
            else:
                task_name = category

            ret_img = self.format_images(
                origin_imgs=batch_extra_imgs,
                bev_img=ret_img,
                task_name=task_name,
                img_name=str(name),
                draw_class_color_bar=True,
            )
            cv2.imwrite(f"{self.output_dir}/{task_name}_{name}.jpg", ret_img)


@OBJECT_REGISTRY.register
class ANCBevOccupancyVisualize(BaseVisualize):
    """
    Visualize class for bev Occupancy.

    Args:
        vcs_origin_coord: vcs origin coordinates (H, W).
        color_map: mapping relationship from category to color, shape (N, 3).
        reformat_fn: the function for reformat the task results.
        img_stitcher: whether need stitch extra camera view imgs.
        range_mode: vcs range mode, included "wide" and "small".
    """

    def __init__(
        self,
        output_dir: str,
        prefix: str,
        color_map: dict,
        vcs_origin_coord: Sequence[int],
        img_stitcher: ANCBEVImgStitcher = None,
    ):
        super().__init__(output_dir)
        self.prefix = prefix
        self.color_map = color_map
        self.vcs_origin_coord = vcs_origin_coord
        self.img_stitcher = img_stitcher

    def on_batch_end(self, batch, model_outs, **kwargs):
        model_outs = convert_numpy(model_outs)

        batch_size = batch["img"][0].shape[0]
        for sub_task, sub_color_map in self.color_map.items():
            if sub_task not in model_outs:
                continue
            task_result = model_outs[sub_task]
            for ind in range(batch_size):
                bev_pred = (
                    task_result[self.prefix + f"_pred_{sub_task}_frame0"][0][
                        ind
                    ]
                    .squeeze()
                    .astype("uint8")
                )  # pred label, (h, w)

                ret_img = sub_color_map[bev_pred]
                cv2.circle(
                    ret_img,
                    (self.vcs_origin_coord[1], self.vcs_origin_coord[0]),
                    3,
                    (255, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )

                if self.img_stitcher:
                    assert "img_vis" in batch
                    batch_extra_imgs = [
                        i[ind].cpu().numpy().squeeze().transpose(1, 2, 0)
                        for i in batch["img_vis"][0]
                    ]
                    ret_img = self.img_stitcher(
                        multiview_imgs=batch_extra_imgs, bev_img=ret_img
                    )

                name = int(list(batch["timestamp"].cpu().numpy())[ind] * 1000)

                if sub_task in self.output_dir:
                    cv2.imwrite(f"{self.output_dir}/{name}.png", ret_img)
                else:
                    cv2.imwrite(
                        f"{self.output_dir}/{sub_task}_{name}.png", ret_img
                    )


@OBJECT_REGISTRY.register
class ANCOMVisualize(BaseVisualize):
    """
    Visualize class for online mapping.

    Args:
        head_groups: output heads config info.
        out_size: model output size.
        view_bev_size: the size of bird eye view.
        vcs_range: visbile range of bev, (bottom, right, top, left) in order.
        reformat_fn: the function for reformat the task results.
        cluster_alg: the algorithm of cluster.
        extra_img: whether need extra 6v imgs.
        vis_in_img: whether draw pred points in 6v imgs.

    """

    def __init__(
        self,
        output_dir: str,
        task: str,
        prefix: str,
        head_groups: Dict,
        out_size: tuple,
        view_bev_size: tuple,
        vcs_range: tuple,
        reformat_fn: Callable = None,
        cluster_alg: str = "ogc",
        extra_img: ANCBEVImgStitcher = None,
        vis_in_img: bool = False,
    ):
        super().__init__(output_dir)
        self.task = task
        self.prefix = prefix
        self.out_size = out_size
        self.head_groups = head_groups
        self.reformat_fn = reformat_fn
        self.view_bev_size = view_bev_size
        self.vcs_range = vcs_range
        self.extra_img = extra_img
        self.cluster_alg = cluster_alg
        self._mat_vcs2bev = None
        self.vis_in_img = vis_in_img

        self.target_categorys = get_target_categorys(head_groups)

        # get group info
        self.group_sub_heads = {}
        for head, head_infos in head_groups.items():
            multi_head = head_infos.get("multi", False)
            group = head_infos["group"]
            if group not in self.group_sub_heads:
                self.group_sub_heads[group] = []
            if not multi_head:
                self.group_sub_heads[group].append(head)

        self.main_attr_list = [
            "prob",
            "cls",
            "instance",
            "r",
            "sin",
            "cos",
            "embedding",
            "offset",
            "direction",
        ]

    @property
    def mat_vcs2bev(self):
        if self._mat_vcs2bev is None:
            self._mat_vcs2bev = get_vcs2bev_img_mat(
                self.vcs_range, self.view_bev_size
            )
        return self._mat_vcs2bev

    def on_batch_end(self, batch, model_outs, **kwargs):

        batch_size = batch["img"][0].shape[0]

        # convert output
        if isinstance(model_outs[self.task], list):
            task_result = model_outs[self.task][0][self.prefix]
        else:
            task_result = model_outs[self.task][self.prefix]

        for ind in range(batch_size):
            ret_img = self.draw_ret_img(task_result, ind)

            if self.extra_img:
                assert "img_vis" in batch
                if self.vis_in_img:
                    img_vis = self.draw_ori_img(task_result, batch, ind)
                else:
                    img_vis = [
                        i[ind].cpu().numpy().squeeze().transpose(1, 2, 0)
                        for i in batch["img_vis"][0]
                    ]
                ret_img = self.extra_img(
                    multiview_imgs=img_vis, bev_img=ret_img
                )

            name = int(list(batch["timestamp"].cpu().numpy())[ind] * 1000)
            cv2.imwrite(f"{self.output_dir}/{name}.png", ret_img)

    def draw_ret_img(self, pred_stats_batch, batch_id):

        scope_h = self.vcs_range[2] - self.vcs_range[0]
        scope_w = self.vcs_range[3] - self.vcs_range[1]
        out_h, out_w = self.out_size
        res_h = scope_h / out_h
        res_w = scope_w / out_w

        pred_lanes_raw = pred_stats_batch[batch_id]["pred_lanes_raw"]

        ret_img = draw_online_mapping(
            pred_lanes_raw,
            self.view_bev_size[0],
            self.view_bev_size[1],
            self.mat_vcs2bev,
            draw_pt=True,
            res_h=res_h,
            res_w=res_w,
        )
        return ret_img

    def draw_ori_img(self, pred_stats_batch, data_batch, batch_id):
        pred_lanes_raw = pred_stats_batch[batch_id]["pred_lanes_raw"]
        ori_img = data_batch["img_vis"][0]
        ret_img_list = []
        for idx, img in enumerate(ori_img):
            new_img = (
                img[batch_id].cpu().contiguous().numpy().transpose(1, 2, 0)
            )
            meta_info = data_batch["meta_info"]
            T_vcs2cam = meta_info["T_vcs2cam"][idx][batch_id][0].cpu().numpy()
            K = meta_info["intrinsics"][idx][batch_id][0].cpu().numpy()
            distort_coeffs = (
                meta_info["distort_coeffs"][idx][batch_id].cpu().numpy()
            )
            ret_img = plot_pts_in_raw_img(
                new_img, pred_lanes_raw, T_vcs2cam, K, distort_coeffs
            )
            ret_img_list.append(ret_img)
        return ret_img_list


category2vis_name = {
    "merge_start": "m_st",
    "merge_stop": "m_sp",
    "split_start": "s_st",
    "split_stop": "s_sp",
    "u_turn": "u_t",
    "other": "o",
    "changepoint": "cp",
}


def get_bev_pts(cross_pts, mat_vcs2bev):
    """Convert crosspoint from vcs to bev.

    Args:
        cross_pts: crosspoint result, [x, y, cls_id, score].
        mat_vcs2bev: transfer matrix from vcs coord to bev pixel.

    """
    if not cross_pts:
        return []
    cross_pts = np.array(cross_pts, np.float)
    pts = np.array([[pt[0], pt[1], 1] for pt in cross_pts]).transpose()
    bev_pts = (mat_vcs2bev @ pts).transpose()
    bev_pts[:, 0] = bev_pts[:, 0] / bev_pts[:, 2]
    bev_pts[:, 1] = bev_pts[:, 1] / bev_pts[:, 2]
    cross_pts[:, :2] = bev_pts[:, :2]
    return cross_pts


def image_cover(cover_mask, img, alpha):
    img_copy = img.copy()
    mask1 = cover_mask[:, :, 0] != 0
    mask2 = cover_mask[:, :, 1] != 0
    mask3 = cover_mask[:, :, 2] != 0
    mask = mask1 | mask2 | mask3
    img_copy[mask, :] = (1.0 - alpha) * img_copy[mask, :] + alpha * cover_mask[
        mask, :
    ]
    return img_copy


@OBJECT_REGISTRY.register
class ANCCrossPointVisualize(BaseVisualize):
    """
    Visualize class for Bev-crosspoint.

    Args:
        stride: the stride of module output.
        cls_group_map: Output categories of each group.
        bev_size: the size of bird eye view.
        vcs_range: visbile range of bev, (bottom, right, top, left) in order.
        reformat_fn: the function for reformat the task results.
        extra_img: whether need extra 6v imgs.
        vcs range mode, included "wide" and "small".
        prefix_name: prefix name of model output.
    """

    def __init__(
        self,
        output_dir: str,
        task: str,
        prefix: str,
        stride: int,
        target_categorys: Dict,
        bev_size: tuple,
        vcs_range: tuple,
        reformat_fn: Callable = None,
        extra_img: ANCBEVImgStitcher = None,
    ):
        super().__init__(output_dir)
        self.task = task
        self.prefix = prefix
        self.stride = stride
        self.target_categorys = target_categorys
        self.reformat_fn = reformat_fn
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.extra_img = extra_img
        self._mat_vcs2bev = None

    @property
    def mat_vcs2bev(self):
        if self._mat_vcs2bev is None:
            self._mat_vcs2bev = get_vcs2bev_img_mat(
                self.vcs_range, self.bev_size
            )
        return self._mat_vcs2bev

    @staticmethod
    def draw_crosspoint(
        cross_pts,
        bev_h,
        bev_w,
        mat_vcs2bev,
        target_categorys,
        text=None,
        thickness=1,
        pt_thickness=None,
        img_bev=None,
        is_gt=False,
        ignore_mask=None,
    ):
        """Crosspoint result visulization.

        Args:
            cross_pts: crosspoint result, [x, y, cls_id, score].
            bev_h: bev image height.
            bev_w: bev image width.
            mat_vcs2bev: transfer matrix from vcs coord to bev pixel.
            target_categorys: crosspoint output category.
            text: annotation text.
            thickness: draw pt thickness for backup.
            pt_thickness: draw pt thickness.
            img_bev: image in bev view.
            is_gt: is ground truth or prediction.
            ignore_mask: ignore mask in gt.

        """
        pt_thickness = thickness * 2 if pt_thickness is None else pt_thickness
        if img_bev is None:
            img_bev = np.zeros((bev_h, bev_w, 3))
        else:
            h, w, _ = img_bev.shape
            assert h == bev_h and w == bev_w
        if text:
            cv2.putText(
                img_bev,
                text,
                (50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=2,
            )

        bev_ego_pts = (mat_vcs2bev @ np.array([[0], [0], [1]])).transpose()
        bev_ego_pts[:, 0] = bev_ego_pts[:, 0] / bev_ego_pts[:, 2]
        bev_ego_pts[:, 1] = bev_ego_pts[:, 1] / bev_ego_pts[:, 2]
        bev_ego_pts = bev_ego_pts[0, :2]
        cv2.rectangle(
            img_bev,
            (int(bev_ego_pts[0]) - 5, int(bev_ego_pts[1]) - 10),
            (int(bev_ego_pts[0]) + 5, int(bev_ego_pts[1]) + 10),
            color=(0, 255, 0),
            thickness=thickness,
        )

        if ignore_mask is not None:
            img_bev = image_cover(ignore_mask, img_bev, alpha=0.8)
        bev_pts = get_bev_pts(cross_pts, mat_vcs2bev)
        for pt in bev_pts:
            color = [0, 0, 255] if not is_gt else [0, 255, 0]
            cv2.circle(
                img_bev,
                (int(round(pt[0])), int(round(pt[1]))),
                radius=1,
                color=color,
                thickness=pt_thickness,
            )
            cv2.circle(
                img_bev,
                (int(round(pt[0])), int(round(pt[1]))),
                radius=7,
                color=color,
                thickness=1,
            )

            # find key from value
            cls_name = [k for k, v in target_categorys.items() if v == pt[2]][
                0
            ]

            font_scale = 0.5
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            cls_text = category2vis_name[cls_name]
            size, _ = cv2.getTextSize(
                cls_text, font_face, font_scale, pt_thickness
            )
            height_offset = 3
            cv2.putText(
                img_bev,
                cls_text,
                (
                    int(round(pt[0]) - size[0] / 2),
                    int(round(pt[1])) + size[1] + height_offset,
                ),
                fontFace=font_face,
                fontScale=font_scale,
                color=color,
                thickness=pt_thickness,
            )
        return img_bev

    def on_batch_end(self, batch, model_outs, **kwargs):

        task_result = model_outs[self.task]

        batch_size = batch["img"][0].shape[0]
        for ind in range(batch_size):
            pred_pts = task_result[self.prefix][ind]
            ret_img = self.draw_crosspoint(
                pred_pts,
                self.bev_size[0],
                self.bev_size[1],
                self.mat_vcs2bev,
                self.target_categorys,
            )

            if self.extra_img:
                assert "img_vis" in batch
                batch_extra_imgs = [
                    i[ind].cpu().numpy().squeeze().transpose(1, 2, 0)
                    for i in batch["img_vis"][0]
                ]
                ret_img = self.extra_img(
                    multiview_imgs=batch_extra_imgs, bev_img=ret_img
                )

            name = int(list(batch["timestamp"].cpu().numpy())[ind] * 1000)
            cv2.imwrite(f"{self.output_dir}/{name}.png", ret_img)


@OBJECT_REGISTRY.register
class ANCBevE2EVisualizeV2(BaseVisualize):
    """
    Visualize callback for pack infer of e2e dynamic.

    Args:
        bev_size: bev image map size.
        vcs_range: vcs visible range.
        reformat_fn: the reformat function for the output.
        vis_velocity: whether draw the predicted velo. default False.
        vis_trajectory: whether draw the predicted traj. default False,
        max_draw_traj_num: Only used when vis_trajectory is True.
            default 5.
        extra_img: whether need origin imgs.
        camera_view_names: camera view names.
        color: mapping relationship from category to color.
    """

    def __init__(
        self,
        output_dir,
        prefix,
        bev_size,
        vcs_range,
        task: str = "e2e_dynamic",
        reformat_fn: Callable = None,
        vis_velocity: bool = False,
        vis_trajectory: bool = False,
        max_draw_traj_num: int = 5,
        extra_img: ANCBEVImgStitcher = None,
        camera_view_names: list = None,
        color: dict = COLOR_MAP,
    ):
        super().__init__(output_dir)
        self.prefix = prefix
        self.task = task
        self.reformat_fn = reformat_fn
        self.pack_path = None
        self.trajectory_frame_num = {}
        self.CUR_FRAME_IDX = -1
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.extra_img = extra_img
        self.camera_view_names = camera_view_names
        self.color = color
        self.center_loc = [
            int(vcs_range[3] / (vcs_range[3] - vcs_range[1]) * bev_size[1]),
            int(vcs_range[2] / (vcs_range[2] - vcs_range[0]) * bev_size[0]),
        ]
        vcs_x_y_range = (
            vcs_range[2] - vcs_range[0],
            vcs_range[3] - vcs_range[1],
        )
        vcs_max_range_idx = np.argmax(vcs_x_y_range)
        self.world_width = int(vcs_x_y_range[vcs_max_range_idx])
        self.bev_pixel_meter = (
            bev_size[vcs_max_range_idx] / self.world_width
        )  # 1 meter -> x pixels
        self.vis_velocity = vis_velocity
        self.vis_trajectory = vis_trajectory
        self.max_draw_traj_num = max_draw_traj_num

    def clear(self):

        self.pack_path = None
        self.trajectory_frame_num = {}
        self.CUR_FRAME_IDX = -1

    def on_batch_end(self, batch, model_outs, **kwargs):
        model_outs = convert_numpy(model_outs)

        task_result = model_outs[self.task][self.prefix]
        pack_path = batch["pack_path"][0]

        if pack_path != self.pack_path:
            self.clear()
            self.pack_path = pack_path

        clip_size = batch["img"][0].shape[0]

        for ind in range(clip_size):

            self.CUR_FRAME_IDX = self.CUR_FRAME_IDX + 1

            timestamp = int(list(batch["timestamp"].cpu().numpy())[ind] * 1000)

            # initial the bev map,
            bev_img = init_bev(
                world_width=self.world_width,
                init_bev_size=self.bev_size,
                center_loc=self.center_loc,
                bev_pixel_meter=self.bev_pixel_meter,
            )

            pred_bev_img = bev_img.copy()  # 448 x 512 x 3

            pred_img, pred_img_traj = ANCBevE2EVisualize.draw_bev_boxes(
                pred_bev_img,
                task_result[ind],
                self.vcs_range,
                self.bev_size,
                self.vis_velocity,
                self.vis_trajectory,
                self.max_draw_traj_num,
            )

            bev_boxes_img = np.hstack(
                (
                    pred_img,
                    pred_img_traj,
                )
            )
            task_str = "velo_pred / traj_pred"

            if self.extra_img:
                assert "img_vis" in batch
                assert self.camera_view_names is not None
                project_calib = {}
                project_calib["local_calibs"] = {}  # K, dist, local2cam
                project_calib["local2chassis"] = {}  # local2vcs
                for i, cam in enumerate(self.camera_view_names):
                    project_calib["local_calibs"][cam] = {}

                    project_calib["local_calibs"][cam]["P2"] = (
                        batch["K"].cpu().numpy()[ind][i]
                    )
                    project_calib["local_calibs"][cam]["disCoeffs"] = (
                        batch["dist"].cpu().numpy()[ind][i]
                    )
                    project_calib["local_calibs"][cam]["Tr_vel2cam"] = (
                        batch["local2cam"].cpu().numpy()[ind][i]
                    )

                    project_calib["local2chassis"][cam] = (
                        batch["local2vcs"].cpu().numpy()[ind][i]
                    )
                batch_extra_imgs = [
                    i[ind].cpu().numpy().squeeze().transpose(1, 2, 0)
                    for i in batch["img_vis"][0]
                ]

                # Project the bev3d pred boxes to 6V cam images.
                camera2index_dict = {}
                for i, view_name in enumerate(self.camera_view_names):
                    camera2index_dict[view_name] = i

                bev3d_dim = np.concatenate(
                    (
                        task_result[ind]["heights"][:, None],
                        task_result[ind]["boxes"][:, [3, 2]],
                    ),
                    axis=-1,
                )
                bev3d_ct = task_result[ind]["boxes"][:, :2]
                bev3d_loc_z = task_result[ind]["bev_loc_z"]
                bev3d_rot = np.arctan2(
                    task_result[ind]["yaws"][:, 1],
                    task_result[ind]["yaws"][:, 0],
                )
                bev3d_score = task_result[ind]["scores"]
                bev3d_cls_id = task_result[ind]["labels"]
                batch_extra_imgs = Bev3DVisualize.draw_camera_boxes(
                    bev3d_dim,
                    bev3d_ct,
                    bev3d_loc_z,
                    bev3d_rot,
                    bev3d_score,
                    project_calib,
                    cameras=self.camera_view_names,
                    color=self.color,
                    bev3d_cls_id=bev3d_cls_id,
                    color_imgs=batch_extra_imgs,
                    camera2index=camera2index_dict,
                    score_threshold=0,
                    use_lidar=False,
                )

                bev_boxes_img = self.extra_img(
                    multiview_imgs=batch_extra_imgs, bev_img=bev_boxes_img
                )

            cv2.putText(
                bev_boxes_img,
                f"{timestamp}_{task_str}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                color=(255, 255, 255),
                thickness=2,
            )

            savefile = os.path.join(self.output_dir, f"{timestamp}.jpg")
            cv2.imwrite(savefile, bev_boxes_img)
