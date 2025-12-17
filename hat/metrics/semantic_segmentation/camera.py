# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import math

import cv2
import numpy as np

from hat.core.camera import Camera

logger = logging.getLogger(__name__)


class FisheyeCamera(Camera):
    def __init__(self, camera_param: dict, is_fisheye_camera: bool = False):
        self.scale = 1
        self.is_fisheye_camera = is_fisheye_camera
        self.m = np.asarray(camera_param["matVcsgnd2img"]).reshape(-1, 3)
        self.minv = np.asarray(camera_param["matImg2vcsgnd"]).reshape(-1, 3)
        self.distort = np.asarray(camera_param["distort"])
        self.focal_u = np.asarray(camera_param["focalU"])
        self.focal_v = np.asarray(camera_param["focalV"])
        self.center_u = np.asarray(camera_param["centerU"])
        self.center_v = np.asarray(camera_param["centerV"])
        self.camera_x = np.asarray(camera_param["cameraX"])
        self.camera_y = np.asarray(camera_param["cameraY"])
        self.camera_z = np.asarray(camera_param["cameraZ"])
        self.pitch = np.asarray(camera_param["pitch"])
        self.yaw = np.asarray(camera_param["yaw"])
        self.roll = np.asarray(camera_param["roll"])
        super(FisheyeCamera, self).__init__(
            rx=self.roll,
            ry=self.pitch,
            rz=self.yaw,
            tx=self.camera_x,
            ty=self.camera_y,
            tz=self.camera_z,
            fu=self.focal_u,
            fv=self.focal_v,
            cu=self.center_u,
            cv=self.center_v,
            version=1,
        )
        self.D = np.array(self.distort).reshape(-1, 1)
        self.local2img = self.gnd2img
        self.invalid_camera = False
        try:
            self.img2local = np.linalg.inv(self.local2img)
        except BaseException:
            self.invalid_camera = True
            logger.warning("camera parameter is invalid!")
        self.vp_x = int(self.local2img[0][0] / self.local2img[2][0])
        self.vp_y = int(self.local2img[1][0] / self.local2img[2][0])
        (
            self.distort_vp_x,
            self.distort_vp_y,
        ) = self.get_vanishing_point_distort()

    def get_vanishing_point_undistort(self):
        return self.vp_x, self.vp_y

    def get_vanishing_point_distort(self):
        self.distort_vp_x, self.distort_vp_y = self.get_distort_points(
            self.vp_x, self.vp_y
        )
        return self.distort_vp_x, self.distort_vp_y

    def cvtUndistortImageToVCSGround(self, i_points):
        enlarge_i_points = np.concatenate(
            (i_points, np.ones((len(i_points), 1))), axis=1
        )
        gs_ = np.dot(enlarge_i_points, np.transpose(self.minv))
        g_x = gs_[:, 0] / gs_[:, 2]
        g_y = gs_[:, 1] / gs_[:, 2]
        g_points = np.concatenate(
            (g_x.reshape(g_x.shape[0], 1), g_y.reshape(g_y.shape[0], 1)),
            axis=1,
        )
        return g_points

    def cvtVCSGroundToUndistortImage(self, g_points):
        enlarge_g_points = np.concatenate(
            (g_points, np.ones((len(g_points), 1))), axis=1
        )
        uis_ = np.dot(enlarge_g_points, np.transpose(self.m))
        ui_x = uis_[:, 0] / uis_[:, 2]
        ui_y = uis_[:, 1] / uis_[:, 2]
        ui_points = np.concatenate(
            (ui_x.reshape(ui_x.shape[0], 1), ui_y.reshape(ui_y.shape[0], 1)),
            axis=1,
        )
        ui_points = np.round(ui_points).astype(np.int32)
        return ui_points

    def cvtLocalToUndistortImage(self, g_points):
        enlarge_g_points = np.concatenate(
            (g_points, np.ones((len(g_points), 1))), axis=1
        )
        uis_ = np.dot(enlarge_g_points, np.transpose(self.local2img))
        ui_x = uis_[:, 0] / uis_[:, 2]
        ui_y = uis_[:, 1] / uis_[:, 2]
        ui_points = np.concatenate(
            (ui_x.reshape(ui_x.shape[0], 1), ui_y.reshape(ui_y.shape[0], 1)),
            axis=1,
        )
        ui_points = np.round(ui_points).astype(np.int32)
        return ui_points

    def get_image_height_vcs(self, front_dis_thresh):
        vcspoint = np.array([[front_dis_thresh, 0]])
        i_points = self.cvtVCSGroundToUndistortImage(vcspoint)
        image_h_thresh = i_points[0][1]
        return image_h_thresh

    def get_image_height_local(self, front_dis_thresh):
        vcspoint = np.array([[front_dis_thresh, 0]])
        i_points = self.cvtLocalToUndistortImage(vcspoint)
        image_h_thresh = i_points[0][1]
        return image_h_thresh

    def get_image_point_vcs2image(self, front_dis_thresh, lateral_dis):
        vcspoint = np.array([[front_dis_thresh, lateral_dis]])
        i_points = self.cvtVCSGroundToUndistortImage(vcspoint)
        return i_points[0]

    def get_image_point_local2image(self, front_dis_thresh, lateral_dis):
        vcspoint = np.array([[front_dis_thresh, lateral_dis]])
        i_points = self.cvtLocalToUndistortImage(vcspoint)
        return i_points[0]

    def get_image_points_vcs2image(self, vcs_points):
        vcspoints = np.array(vcs_points)
        i_points = self.cvtVCSGroundToUndistortImage(vcspoints)
        return i_points

    def get_distort_points(self, x_in, y_in):
        fx = self.focal_u
        fy = self.focal_v
        cx = self.center_u
        cy = self.center_v
        distort = np.zeros((8,))
        num_params = len(self.distort)
        distort[:num_params] = self.distort
        k1, k2, p1, p2, k3, k4, k5, k6 = distort

        xd = x_in
        yd = y_in
        xd = (xd - cx) / fx
        yd = (yd - cy) / fy
        xp = 0.0
        yp = 0.0
        if self.is_fisheye_camera:
            r2 = xd * xd + yd * yd
            r = math.sqrt(r2)
            theta = math.atan(r)
            theda_d = theta * (
                1 + k1 * math.pow(theta, 2) + k2 * math.pow(theta, 4)
            )
            xp = (theda_d / r) * xd
            yp = (theda_d / r) * yd
            xp = fx * xp + cx
            yp = fy * yp + cy
        else:
            r2 = xd * xd + yd * yd
            r4 = r2 * r2
            r6 = r2 * r4
            coeff = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (
                1 + k4 * r2 + k5 * r4 + k6 * r6
            )
            xp = coeff * xd + 2 * p1 * xd * yd + p2 * (r2 + 2 * xd * xd)
            yp = coeff * yd + p1 * (r2 + 2 * yd * yd) + 2 * p2 * xd * yd
            xp = xp * fx + cx
            yp = yp * fy + cy
        return int(xp), int(yp)

    def get_undistort_map(self, image_height_valid, image_width):
        K = self.K
        D = self.D
        Knew = K.copy()
        Knew[0, 2] = K[0, 2] * self.scale
        Knew[1, 2] = K[1, 2] * self.scale
        if self.is_fisheye_camera:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K,
                D,
                np.eye(3),
                Knew,
                (self.scale * image_width, self.scale * image_height_valid),
                cv2.CV_16SC2,
            )
        else:
            map1, map2 = cv2.initUndistortRectifyMap(
                K,
                D,
                np.eye(3),
                Knew,
                (self.scale * image_width, self.scale * image_height_valid),
                cv2.CV_16SC2,
            )
        return map1, map2

    def get_undistorted_label_map(
        self, label_array, image_height, image_width
    ):
        map1, map2 = self.get_undistort_map(image_height, image_width)
        undistorted_label = cv2.remap(
            label_array,
            map1,
            map2,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return undistorted_label

    def get_undistorted_image(self, image_array, image_height, image_width):
        map1, map2 = self.get_undistort_map(image_height, image_width)
        undistorted_image = cv2.remap(
            image_array,
            map1,
            map2,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return undistorted_image

    def draw_vp(self, image, size=20, color=(0, 0, 255), thickness=2):
        cv2.line(
            image,
            (int(self.vp_x - size / 2), self.vp_y),
            (int(self.vp_x + size / 2), self.vp_y),
            color=color,
            thickness=thickness,
        )
        cv2.line(
            image,
            (self.vp_x, int(self.vp_y - size / 2)),
            (self.vp_x, int(self.vp_y + size / 2)),
            color=color,
            thickness=thickness,
        )
        return

    def draw_camera_grid(
        self,
        image,
        forward_list=range(20, 101, 10),  # noqa: B008
        lateral_list=range(-8, 9, 4),  # noqa: B008
        line_color=(0, 255, 0),
        thickness=1,
        camera_ype_=cv2.LINE_AA,
    ):
        def _pt_img(mat, pt):
            pt_img = np.dot(mat, np.array(pt))
            return (pt_img[0] / pt_img[2], pt_img[1] / pt_img[2])

        horizon_lines = [
            [
                _pt_img(self.local2img, [forward, lateral_list[0], 1]),
                _pt_img(self.local2img, [forward, lateral_list[-1], 1]),
                (forward, lateral_list[0]),
                (forward, lateral_list[-1]),
            ]
            for forward in forward_list
        ]
        vertical_lines = [
            [
                _pt_img(self.local2img, [forward_list[0], lateral, 1]),
                _pt_img(self.local2img, [forward_list[-1], lateral, 1]),
                (forward_list[0], lateral),
                (forward_list[-1], lateral),
            ]
            for lateral in lateral_list
        ]
        lines = horizon_lines + vertical_lines
        for line in lines:
            cv2.line(
                image,
                (int(line[0][0]), int(line[0][1])),
                (int(line[1][0]), int(line[1][1])),
                color=line_color,
                lineType=camera_ype_,
                thickness=thickness,
            )
            cv2.putText(
                image,
                "{}".format(line[2]),
                (int(line[0][0]), int(line[0][1])),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                color=line_color,
            )
            cv2.putText(
                image,
                "{}".format(line[3]),
                (int(line[1][0]), int(line[1][1])),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                color=line_color,
            )

        cv2.line(
            image,
            (self.vp_x - 10, self.vp_y),
            (self.vp_x + 10, self.vp_y),
            color=line_color,
            lineType=camera_ype_,
            thickness=thickness,
        )
        cv2.line(
            image,
            (self.vp_x, self.vp_y - 10),
            (self.vp_x, self.vp_y + 10),
            color=line_color,
            lineType=camera_ype_,
            thickness=thickness,
        )
        return image
