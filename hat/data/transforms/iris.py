# Copyright (c) Horizon Robotics. All rights reserved.

import math
import random
from typing import Tuple

import cv2
import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY

__all__ = ["IrisMtlTrans"]


@OBJECT_REGISTRY.register
class IrisMtlTrans(object):
    """Align gaze mutil-task transform, ensure to have the same input.

    Args:
        input_size : Size of input image.
        cam_hw : Height and weight of camera.
        standard_focal: Standard Camera focal.
        to_yuv420sp: Whether transform to yuv420sp.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (192, 320),
        cam_hw: Tuple[int, int] = (720, 1280),
        standard_focal: float = 600,
        to_yuv420sp: bool = False,
    ):
        self._default_cam_intrin = [
            900,
            0.0,
            640.0,
            0.0,
            900,
            360.0,
            0.0,
            0.0,
            1.0,
        ]
        self._default_cam_hw = cam_hw
        self._default_standard_focal = standard_focal
        self._default_to_yuv420sp = to_yuv420sp
        self._input_size = input_size
        (
            self.horizon_pos_map,
            self.vertical_pos_map,
        ) = self.position_generate_image()

    def position_generate_image(self):
        cam_intrin = np.array(self._default_cam_intrin).reshape((3, 3))
        cam_height, cam_width = self._default_cam_hw
        std_focal = self._default_standard_focal
        to_yuv420sp = self._default_to_yuv420sp

        std_intrin = np.array(
            [
                [std_focal, 0, cam_width / 2],
                [0, std_focal, cam_height / 2],
                [0, 0, 1],
            ]
        )
        cam_intrin_inv = np.linalg.inv(cam_intrin)
        transform_mat = np.matmul(std_intrin, cam_intrin_inv)
        width_range = np.linspace(0, cam_width - 1, cam_width)
        height_range = np.linspace(0, cam_height - 1, cam_height)

        width_grid, height_grid = np.meshgrid(width_range, height_range)
        normal_grid = np.ones((cam_height, cam_width))
        raw_uv = np.dstack((width_grid, height_grid, normal_grid))
        raw_uv = raw_uv.reshape((cam_height * cam_width, -1)).transpose()
        convert_uv = np.matmul(transform_mat, raw_uv)
        convert_uv = convert_uv.transpose().reshape(
            (cam_height, cam_width, -1)
        )

        horizon_pos_map = ((convert_uv[:, :, 0] / cam_width) * 255).astype(
            np.uint8
        )
        vertical_pos_map = ((convert_uv[:, :, 1] / cam_height) * 255).astype(
            np.uint8
        )

        if to_yuv420sp:
            horizon_pos_map = convert_channel_to_420sp(horizon_pos_map)
            vertical_pos_map = convert_channel_to_420sp(vertical_pos_map)

        return horizon_pos_map, vertical_pos_map

    def preprocess_net_input(self, raw, dst_size=(160, 96), return_chw=False):
        yuv = cv2.cvtColor(raw, cv2.COLOR_BGR2YUV)
        y_hist_equalized = yuv[:, :, 0]
        # Global Histogram Equalization
        y_hist_equalized = cv2.equalizeHist(yuv[:, :, 0])
        # Local Histogram Equalization
        hori_eye_map, vert_eye_map = (
            self.horizon_pos_map,
            self.vertical_pos_map,
        )
        new_yuv = np.zeros((dst_size[1], dst_size[0], 3))

        posmap_w = random.randint(5, hori_eye_map.shape[1] - 1)
        posmap_h = random.randint(5, hori_eye_map.shape[0] - 1)

        posmap_x = random.randint(0, hori_eye_map.shape[1] - posmap_w - 1)
        posmap_y = random.randint(0, hori_eye_map.shape[0] - posmap_h - 1)

        hori_eye_map_chose = hori_eye_map[
            posmap_y : posmap_y + posmap_h, posmap_x : posmap_x + posmap_w
        ]
        vert_eye_map_chose = vert_eye_map[
            posmap_y : posmap_y + posmap_h, posmap_x : posmap_x + posmap_w
        ]

        new_yuv[:, :, 0] = cv2.resize(y_hist_equalized, dst_size)
        new_yuv[:, :, 1] = cv2.resize(hori_eye_map_chose, dst_size)
        new_yuv[:, :, 2] = cv2.resize(vert_eye_map_chose, dst_size)

        new_yuv = new_yuv / 128.0 - 1

        if return_chw:
            return new_yuv.transpose(2, 0, 1).astype(np.float32)
        else:
            return new_yuv.astype(np.float32)

    def __call__(self, data):
        img = data["img"]
        img = img.numpy() if isinstance(img, torch.Tensor) else img
        img = self.preprocess_net_input(
            img.astype(np.uint8), dst_size=tuple(self._input_size[::-1])
        )
        data["img"] = img
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"input_size={self._input_size}"
        repr_str += f"cam_hw={self._default_cam_hw}"
        repr_str += f"standard_focal={self._default_standard_focal}"
        repr_str += f"to_yuv420sp={self._default_to_yuv420sp}"
        return repr_str


def convert_channel_to_420sp(image):
    """Transfer image to align with YUV420SP.

    Image will be downsampled to YUV420 resolution (0.5 x 0.5)
    and upsampled to original size
    """
    img_h, img_w = image.shape[:2]
    downsampled_w = int(math.ceil(img_w / 2.0))
    downsmapled_h = int(math.ceil(img_h / 2.0))
    img = cv2.resize(image, (downsampled_w, downsmapled_h))
    img = np.repeat(img, 2, axis=0)
    img = np.repeat(img, 2, axis=1)
    return img[:img_h, :img_w]
