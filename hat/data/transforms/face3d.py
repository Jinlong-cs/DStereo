# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import logging
import math
import random
from typing import List, Optional, Tuple, Union

import cv2

try:
    import kornia
except ImportError:
    kornia = None
import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages
from .pupil_segmentation import Ellipse

logger = logging.getLogger(__name__)


__all__ = [
    "RandomRotateCrop",
    "VirtualCameraNorm",
    "PositionEncoding",
    "BboxEncodingCLIFF",
    "SimpleNormGenGridMap",
    "SimpleNormPositionEncoding",
    "CropRoIJitter",
    "ImageCLAHE",
]


def _expand_bbox(
    bbox: Union[List, np.ndarray],
    img_shape: Tuple,
    expand_ratio_hw: Union[Tuple[int, int], float],
    norm_method="longside_square",
    exceed_type="clip",
):
    if isinstance(expand_ratio_hw, Tuple):
        expand_type = "size"
    else:
        expand_type = "ratio"
    ori_w = bbox[2] - bbox[0]
    ori_h = bbox[3] - bbox[1]
    if norm_method == "longside_ratio":
        assert expand_type == "ratio"
        exp_w = ori_w * (expand_ratio_hw - 1.0) / 2
        exp_h = ori_h * (expand_ratio_hw - 1.0) / 2
    elif norm_method == "longside_square":
        if expand_type == "ratio":
            length = max(ori_h, ori_w)
            exp_w = (expand_ratio_hw * length - ori_w) / 2.0
            exp_h = (expand_ratio_hw * length - ori_h) / 2.0
        elif expand_type == "size":
            exp_w = (expand_ratio_hw[1] - ori_w) / 2.0
            exp_h = (expand_ratio_hw[0] - ori_h) / 2.0
    else:
        raise ValueError(f"Not supported norm_method: {norm_method}.")
    x1 = bbox[0] - exp_w
    x2 = bbox[2] + exp_w
    y1 = bbox[1] - exp_h
    y2 = bbox[3] + exp_h
    if exceed_type == "clip":
        img_h, img_w = img_shape[:2]
        x1 = max(x1, 0)
        x2 = min(x2, img_w)
        y1 = max(y1, 0)
        y2 = min(y2, img_h)
    elif exceed_type != "raw":
        raise ValueError(f"Not supported exceed_type: {exceed_type}")
    # TODO: @yuhao.dou transform to int or not?
    new_bbox = (
        np.array(list(map(int, [x1, y1, x2, y2])))
        .reshape(2, 2)
        .astype(np.float32)
    )
    return new_bbox


def _resize_fix_ratio(
    img: np.ndarray,
    dst_size: Tuple[int, int],
):
    scale_h = dst_size[1] / img.shape[0]
    scale_w = dst_size[0] / img.shape[1]
    scale = min(scale_h, scale_w)
    return cv2.warpAffine(
        img,
        np.array([[scale, 0, 0], [0, scale, 0]]),
        dst_size,
        flags=cv2.INTER_NEAREST,
    )


def get_transform(
    point, scale, output_size, rot=0, base_len=1, point_type="center"
):
    # Generate transformation matrix
    h = base_len * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    if point_type == "center":
        t[0, 2] = output_size[1] * (-float(point[0]) / h + 0.5)
        t[1, 2] = output_size[0] * (-float(point[1]) / h + 0.5)
    elif point_type == "lefttop":
        t[0, 2] = output_size[1] * (-float(point[0]) / h)
        t[1, 2] = output_size[0] * (-float(point[1]) / h)
    else:
        raise ValueError(f"Not supported point_type: {point_type}.")
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1] / 2
        t_mat[1, 2] = -output_size[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform_keypoints(kps, meta, invert=False):
    keypoints = kps.copy()
    if invert:
        meta = np.linalg.inv(meta)
    keypoints[:, :2] = np.dot(keypoints[:, :2], meta[:2, :2].T) + meta[:2, 2]
    return keypoints


def get_union_ldmk_roi(rects, lmks, expansion):
    ldmk_upper_left = lmks.min(axis=0)
    ldmk_low_right = lmks.max(axis=0)
    rects[:2] = np.min([rects[:2], ldmk_upper_left], axis=0)
    rects[2:] = np.max([rects[2:], ldmk_low_right], axis=0)
    center = rects.reshape(2, 2).mean(axis=0)
    wh_max = max(rects[2] - rects[0], rects[3] - rects[1]) + expansion
    return (center[0], center[1]), wh_max, wh_max


@OBJECT_REGISTRY.register
class CropRoIJitter(object):
    """Jitter RoI in face3d dataset.

    Args:
        jitter_prob: Apply prob of this transform. Defaults to 0.0.
        exp_ratio: Ratio of the expansion of box. Defaults to 1.0.
        exp_jitter: Jitter of expansion ratio . Defaults to 0.0.
        center_shift: Box center shift range. Defaults to 0.0.
    """

    def __init__(
        self,
        jitter_prob: float = 0.0,
        exp_ratio: float = 1.0,
        exp_jitter: float = 0.0,
        center_shift: float = 0.0,
    ):
        self.jitter_prob = jitter_prob
        self.exp_ratio = exp_ratio
        self.exp_jitter = exp_jitter
        self.center_shift = center_shift

    def __call__(self, data):
        if self.jitter_prob < 1e-6 or random.random() > self.jitter_prob:
            return data

        img_shape = data["raw_img_shape"]
        boxes = data["gt_bboxes"]

        boxes = self._box_jitter(boxes, img_shape)
        data["gt_bboxes"] = boxes
        return data

    def _box_jitter(self, boxes, img_shape):
        img_h, img_w = img_shape[:2]
        # calcluate expand ratio
        scale = self.exp_ratio + random.uniform(
            -self.exp_jitter, self.exp_jitter
        )

        center_x = 0.5 * (boxes[0] + boxes[2])
        center_y = 0.5 * (boxes[1] + boxes[3])
        box_h = boxes[3] - boxes[1]
        box_w = boxes[2] - boxes[0]

        # shift center
        shifts = np.clip(
            np.random.normal(0, 0.1, (2,)),
            -self.center_shift,
            self.center_shift,
        )  # noqa
        center_x = center_x + shifts[0] * box_w
        center_y = center_y + shifts[1] * box_h

        # expand the box
        x1 = np.clip(center_x - scale * 0.5 * box_w, 0, img_w)
        x2 = np.clip(center_x + scale * 0.5 * box_w, 0, img_w)
        y1 = np.clip(center_y - scale * 0.5 * box_h, 0, img_h)
        y2 = np.clip(center_y + scale * 0.5 * box_h, 0, img_h)
        new_boxes = np.array([x1, y1, x2, y2], dtype=np.float32)

        return new_boxes


@OBJECT_REGISTRY.register
class RandomRotateCrop(object):
    """Randomly crop the roi with scaling, rotation and center shifting.

    .. note::
        Affected keys: 'img', 'gt_bboxes', 'gt_ldmk', 'gt_mask'.

    Args:
        net_input_size: network input image size. Generally the layout of
            `image size` is WH, while the counterpart of `image shape` is
            HW(C) or (C)HW.
        rot_prob: rotation probability. Defaults to 0.0.
        rot_angle_range: max rotation range, in [-180.0, 180.0].
            Defaults to 30.0.
        center_shift_prob: crop center shifting probability. Defaults to 0.0.
        center_shift_range: crop center shifting max range normalized by
            img size. Defaults to 0.01.
        norm_ratio: bbox expand ratio. The finally expand ratio is the summary
            of a fix norm_ratio and a random norm_jitter.Defaults to 1.0.
        norm_method: bbox expand method. Only `longside ratio` and
            `longsie_square` are supported. `longside_ratio` means expanding
            bbox directly without padding. Resize the roi to square input may
            bring about deformation. `longside_square` means expanding bbox to
            a square area via padding the shorter side with pixels outside of
            the roi (NOT ALL ZEROS). If the expanded roi transcends the image
            area, it will be clipped to the image border. Defaults to
            "longside_square".
        norm_jitter_range: bbox expand ratio jitter range. Keep it to be 0
            during evaluation. Defaults to 0.0.
        keep_ldmk_complt_ratio: whether to keep whole ldmk when croping images.
            Defaults to 0.0.
        keep_ldmk_complt_expansion: expansion length (pix) of keep roi relative
            to ldmk. Defaults to 1.0.
        return_normalized_ldmk: whether to normalize ldmk by target imgsize.
            Defaults to True.
        net_target_size: resized targert image, a part of supervision info.
            This param is only available for face3d task for reconstructed
            face supervision. Defaults to None.
        base_len: base length for coordinates. Defaults to 1.
        interpolation: interpolation method of scaling. Defaults to bilinear.
    """

    def __init__(
        self,
        net_input_size: Tuple[int, int],
        rot_prob: Optional[float] = 0.0,
        rot_angle_range: Optional[float] = 30.0,
        center_shift_prob: Optional[float] = 0.0,
        center_shift_range: Optional[float] = 0.01,
        norm_ratio: Optional[float] = 1.0,
        norm_method: Optional[str] = "longside_square",
        norm_jitter_range: Optional[float] = 0.0,
        keep_ldmk_complt_ratio: Optional[float] = 0.0,
        keep_ldmk_complt_expansion: Optional[float] = 1.0,
        return_normalized_ldmk: Optional[bool] = True,
        net_target_size: Optional[Tuple] = None,
        base_len: Optional[float] = 1,
        interpolation: str = "bilinear",
    ):
        self.net_input_size = net_input_size
        self.rot_prob = rot_prob
        self.rot_angle_range = rot_angle_range
        self.center_shift_prob = center_shift_prob
        self.center_shift_range = center_shift_range
        self.norm_ratio = norm_ratio
        self.norm_method = norm_method
        self.norm_jitter_range = norm_jitter_range
        self.keep_ldmk_complt_ratio = keep_ldmk_complt_ratio
        self.keep_ldmk_complt_expansion = keep_ldmk_complt_expansion
        self.return_normalized_ldmk = return_normalized_ldmk
        self.net_target_size = net_target_size
        self.base_len = base_len
        assert norm_method.lower() in ["longside_ratio", "longside_square"]
        self.cv2_interp_codes = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        self.interpolation = interpolation

    def _get_crop_params(self, data):
        if "gt_bboxes" in data:
            bbox = data["gt_bboxes"]
            center_x = (bbox[0] + bbox[2]) * 0.5
            center_y = (bbox[1] + bbox[3]) * 0.5
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
        elif "roi_scale" in data:
            # smoke cls model, using roi_scale of image as bbox
            img = data["img"]
            img_h, img_w = img.shape[0], img.shape[1]
            center_x, center_y = img_w // 2, img_h // 2
            width = img_w * data["roi_scale"]
            height = img_h * data["roi_scale"]
        else:
            raise ValueError("'gt_bboxes' must be in data.")

        # center shift
        do_center_shift = np.random.choice(
            [False, True],
            p=[1 - self.center_shift_prob, self.center_shift_prob],
        )
        if do_center_shift:
            # TODO: (yuhao.dou) Compare gaussian and uniform.
            shifts = np.random.uniform(
                -self.center_shift_range, self.center_shift_range, (2,)
            )
            center_x = center_x + shifts[0] * width
            center_y = center_y + shifts[1] * height
        center = (center_x, center_y)

        # keep_ldmk_complt
        if (
            "gt_ldmk" in data
            and random.random() <= self.keep_ldmk_complt_ratio
        ):
            crop_bbox = np.array(
                [
                    center[0] - width // 2,
                    center[1] - height // 2,
                    center[0] + width // 2,
                    center[1] + height // 2,
                ]
            )
            center, height, width = get_union_ldmk_roi(
                crop_bbox,
                data["gt_ldmk"][:, :2],
                self.keep_ldmk_complt_expansion,
            )

        # rot
        do_rotation = np.random.choice(
            [False, True],
            p=[1 - self.rot_prob, self.rot_prob],
        )
        rot = 0
        if do_rotation:
            rot = np.random.randint(
                -self.rot_angle_range, self.rot_angle_range
            )

        # bbox expand
        rand_num = np.clip(np.random.normal(0, 1 / 3), -1.0, 1.0)
        norm_ratio = rand_num * self.norm_jitter_range + self.norm_ratio
        if self.norm_method == "longside_square":
            scale = max(height, width) / self.base_len * norm_ratio
        elif self.norm_method == "longside_ratio":
            raise NotImplementedError("Not supported yet.")
        else:
            raise ValueError(f"Not supported norm_method: {self.norm_method}.")

        self.center = center
        self.rot = rot
        self.scale = scale

    def _crop_input_img(self, data):
        img = data["img"]
        meta = get_transform(
            self.center,
            self.scale,
            self.net_input_size,
            self.rot,
            self.base_len,
        )
        net_input_img = cv2.warpAffine(
            img,
            meta[:2],
            self.net_input_size,
            flags=self.cv2_interp_codes[self.interpolation],
        )
        data["img"] = net_input_img
        return data, meta

    def _crop_ldmk(self, data):
        ldmk = data["gt_ldmk"]
        meta = self.crop_info
        # rotate and crop
        target_ldmk = transform_keypoints(ldmk, meta)
        # Normalize ldmk by target img size.
        # WARN: target[0] != target[1] may cause bias.
        if self.return_normalized_ldmk:
            target_ldmk[..., :2] /= self.net_input_size[0]
        data["gt_ldmk"] = target_ldmk
        return data

    def _crop_target_img(self, data):
        """Crop target img and normlaize it to 0~1."""
        img = data["gt_img"]
        meta = get_transform(
            self.center,
            self.scale,
            self.net_target_size,
            self.rot,
            self.base_len,
        )
        net_target_img = cv2.warpAffine(
            img, meta[:2], self.net_target_size, flags=cv2.INTER_LINEAR
        )
        net_target_img = np.transpose(net_target_img, (2, 0, 1))
        net_target_img = net_target_img.astype(np.float32) / 255.0
        data["gt_img"] = net_target_img
        return data, meta

    def _crop_mask(self, data):
        """Crop gt mask and normalize it to 0~1."""
        mask = data["gt_mask"].copy()
        meta = self.crop_info
        target_mask = cv2.warpAffine(
            mask, meta[:2], self.net_target_size, flags=cv2.INTER_NEAREST
        )
        if target_mask.ndim == 2:
            target_mask = np.expand_dims(target_mask, axis=2)
        target_mask = np.transpose(target_mask, (2, 0, 1))
        target_mask = target_mask.astype(np.float32) / 255.0
        data["gt_mask"] = target_mask
        return data

    def _crop_ellipse_param(self, data):
        ellipse_param = data["gt_pupil_ellipse_param"].copy()
        ellipse_param[-1] = np.deg2rad(ellipse_param[-1])
        meta = self.crop_info
        ellipse_param = Ellipse(ellipse_param).transform(meta)[0][:-1]
        ellipse_param[-1] = np.rad2deg(ellipse_param[-1])
        data["gt_pupil_ellipse_param"] = ellipse_param

    def __call__(self, data):
        self._get_crop_params(data)
        data, crop_info = self._crop_input_img(data)
        self.crop_info = crop_info
        if "gt_ldmk" in data:
            data = self._crop_ldmk(data)
        if "gt_img" in data:
            """Only for face3d task."""
            data, crop_info = self._crop_target_img(data)
            self.crop_info = crop_info
        if "gt_mask" in data:
            data = self._crop_mask(data)
        if "gt_pupil_ellipse_param" in data:
            self._crop_ellipse_param(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"net_input_size={self.net_input_size}"
        repr_str += f"rot_prob={self.rot_prob}"
        repr_str += f"rot_angle={self.rot_angle_range}"
        repr_str += f"center_shift_prob={self.center_shift_prob}"
        repr_str += f"center_shift_range={self.center_shift_range}"
        repr_str += f"norm_ratio={self.norm_ratio}"
        repr_str += f"norm_method={self.norm_method}"
        repr_str += f"norm_jitter_range={self.norm_jitter_range}"
        repr_str += f"net_target_size={self.net_target_size}"
        repr_str += f"base_len={self.base_len}"
        return repr_str


@OBJECT_REGISTRY.register
class VirtualCameraNorm(object):
    """Normalize ROI with virtual camera.

    For norm details, refer to 'PCLs, Geometry-aware Neural Reconstruction of
    3D Pose with Perspective Crop Layers'. For position map details, refer
    to 'An intriguing failing of convolutional neural networks and the
    CoordConv solution'.

    Args:
        net_input_size: network input size (W, H). Defaults to 128.
        net_target_size: target image size (W, H). Defaults to 256.
        norm_ratio: Refer to RandomRotateCrop. Defaults to 1.2.
        norm_method: Refer to RandomRotateCrop. Defaults to "longside_square".
        use_dist: use distortion coeffcient or not. Defaults to False.
        virtual_intrinsic: virtual camera intrinsic. It will be set to be the
            same as the real camera if it is None. Defaults to None.
    """

    def __init__(
        self,
        net_input_size: Union[Tuple[int, int], int] = 128,
        net_target_size: Union[Tuple[int, int], int] = 256,
        norm_ratio: float = 1.2,
        norm_method: str = "longside_square",
        norm_center: str = "face",
        use_dist: bool = False,
        virtual_intrinsic: Optional[np.ndarray] = None,
    ):
        if isinstance(net_input_size, Tuple):
            self.net_input_size = net_input_size
        elif isinstance(net_input_size, int):
            self.net_input_size = (net_input_size, net_input_size)
        else:
            raise ValueError(f"Not supported net_input_size: {net_input_size}")
        if isinstance(net_target_size, Tuple):
            self.net_target_size = net_target_size
        elif isinstance(net_target_size, int):
            self.net_target_size = (net_target_size, net_target_size)
        else:
            raise ValueError(
                f"Not supported net_target_size: {net_target_size}"
            )
        self.norm_ratio = norm_ratio
        self.norm_method = norm_method
        self.norm_center = norm_center
        self.use_dist = use_dist
        self.virtual_intrinsic = virtual_intrinsic
        self.eps = 1e-6

    def cal_real2virtual_rot_mat(
        self,
        rot_center: np.ndarray,
        intri: np.ndarray,
        dist: np.ndarray,
    ) -> np.ndarray:
        """Rotate the camera to a virtual direction.

        Face bbox center locates at the optical center after rotation.
        Return the matrix from the virtual optical direction to the original.
        The matrix transfer real camera coordinates to the virtual.
        """
        rot_center = cv2.undistortPoints(rot_center, intri, dist)
        optical_center_unit_vec = np.array([0, 0, 1], np.float32)
        rot_center_ex = np.array([0, 0, 1], np.float32)
        rot_center_ex[:2] = rot_center
        rot_center_unit_vec = rot_center_ex / np.linalg.norm(
            rot_center_ex, ord=2
        )
        theta = np.arccos(rot_center_unit_vec[-1])  # simplified dot prod
        unit_vec = np.cross(rot_center_unit_vec, optical_center_unit_vec)
        unit_vec = unit_vec / (np.linalg.norm(unit_vec, ord=2) + self.eps)
        rvec = theta * unit_vec
        rot_mat = cv2.Rodrigues(rvec)[0]
        return rot_mat

    def get_real2virtual_mat(
        self,
        rot_mat: np.ndarray,
        real_intrinsic: np.ndarray,
        virtual_intrinsic: np.ndarray,
    ):
        real_intrinsic_inv = np.linalg.inv(real_intrinsic)
        if rot_mat is None:
            real2virtual_mat = real_intrinsic_inv.T @ virtual_intrinsic.T
        else:
            real2virtual_mat = (
                real_intrinsic_inv.T @ rot_mat.T @ virtual_intrinsic.T
            )
        virtual2real_mat = np.linalg.inv(real2virtual_mat)
        return real2virtual_mat, virtual2real_mat

    def transform_points(
        self,
        points_2d: np.ndarray,
        matrix: np.ndarray,
        intri: np.ndarray,
        dist: np.ndarray,
    ):
        """Transform 2D image points to another view.

        Args:
            points_2d: points on source image plane.
            matrix: transform matrix with a shape of (3, 3). Use A @ matrix
                to transform (N, 3) array A to another coordinate space.
                The matrix is composed of inverse of intrinsic on source image,
                rotation matrix between source and target camera space and
                intrinsic on target image.
            intri: camera intrinsic of source image plane. Only used for
                undistortion.
            dist: camera distortion of source image plane. Only used for
                undistortion

        Returns:
            Transformed 2D points on the image plane of the target view.
        """
        assert points_2d.shape[1] == 2 and points_2d.ndim == 2
        assert matrix.shape == (3, 3)
        # If use_dist, the matrix is obtained on undistorted image plane.
        if self.use_dist and intri is not None and dist is not None:
            points_2d = cv2.undistortPoints(points_2d, intri, dist, P=intri)
            points_2d = points_2d.squeeze(1)
        points_h = np.concatenate(
            (points_2d, np.ones_like(points_2d)[:, :1]), axis=1
        )
        points_h = np.einsum("ij,jk->ik", points_h, matrix)
        points_2d = points_h[:, :2] / points_h[:, 2:]
        return points_2d

    @require_packages("kornia")
    def warp_crop_image(
        self,
        image: np.ndarray,
        roi_offset: List,
        virtual_bbox: Union[List, np.ndarray],
        virtual2real_mat: np.ndarray,
        intr: np.ndarray,
        dist: np.ndarray,
        rand_inter_type: bool = False,
    ):
        left, top, right, bottom = virtual_bbox
        w, h = int(right - left), int(bottom - top)
        range_x = np.linspace(left, right, w, endpoint=False, dtype=np.float32)
        range_y = np.linspace(top, bottom, h, endpoint=False, dtype=np.float32)
        # index on virutal image plane
        map_ori = np.transpose(
            np.meshgrid(range_x, range_y), (1, 2, 0)
        ).reshape(-1, 2)
        map_trans = self.transform_points(
            map_ori, virtual2real_mat, None, None
        )
        if self.use_dist:
            map_trans = kornia.geometry.calibration.distort_points(
                torch.tensor(map_trans[None, ...]),
                torch.tensor(intr[None, ...]),
                torch.tensor(dist[None, ...]),
            ).numpy()[0, ...]

        roi_offset = np.array([roi_offset]).reshape((1, 2))
        map_trans = map_trans - roi_offset
        map_trans = map_trans.reshape(h, w, 2).astype(np.float32)
        inter = random.randint(0, 2) if rand_inter_type else 1
        image_crop = cv2.remap(
            image, map_trans, None, inter, cv2.BORDER_REFLECT
        )
        h, w = np.array(image_crop.shape[:2]) // 2 * 2
        return image_crop[:h, :w]

    def get_meta(self, new_bbox):
        xmin, ymin, xmax, ymax = new_bbox
        scale = max(xmax - xmin, ymax - ymin)
        meta = get_transform(
            new_bbox[:2],
            scale,
            self.net_input_size,
            point_type="lefttop",
        )
        return meta

    def get_gts(self, data, real2virtual_mat, virtual2real_mat, rot_mat):
        virtual_bbox = data["virtual_bbox"]
        meta = self.get_meta(virtual_bbox)
        data["meta"] = meta[:2].astype(np.float32)
        # gt image in virtual camera.
        if data.get("gt_img") is not None:
            img_crop = self.warp_crop_image(
                data["gt_img"],
                data["roi_offset"],
                virtual_bbox,
                virtual2real_mat,
                data["intrinsic"],
                data["distortion"],
            )
            net_target_img = _resize_fix_ratio(img_crop, self.net_target_size)
            if net_target_img.ndim == 2:
                net_target_img = np.expand_dims(net_target_img, 2)
            net_target_img = np.transpose(net_target_img, (2, 0, 1)) / 255.0
            data["gt_img"] = net_target_img.astype(np.float32)
        # gt mask in virtual camera.
        if data.get("gt_mask", None) is not None:
            # TODO: gt_img & gt_mask share the same warp_crop?
            target_mask = self.warp_crop_image(
                data["gt_mask"],
                data["roi_offset"],
                virtual_bbox,
                virtual2real_mat,
                data["intrinsic"],
                data["distortion"],
                False,
            )
            target_mask = _resize_fix_ratio(target_mask, self.net_target_size)
            if target_mask.ndim == 2:
                target_mask = np.expand_dims(target_mask, 2)
            target_mask = np.transpose(target_mask, (2, 0, 1)) / 255.0
            data["gt_mask"] = target_mask.astype(np.float32)
        # gt ldmk in virtual camera.
        if data.get("gt_ldmk", None) is not None:
            gt_ldmk = data["gt_ldmk"].reshape((-1, 3))
            gt_ldmk[:, :2] = self.transform_points(
                gt_ldmk[:, :2], real2virtual_mat, None, None
            )
            data["gt_ldmk"] = gt_ldmk
        # gt translation in virtual camera.
        if data.get("transl", None) is not None:
            data["transl"] = (
                data["real2vir_rotmat"] @ data["transl"][..., None]
            ).reshape(-1)

        # gt 3d joints in virtual camera space.
        if data.get("gt_ldmk3d", None) is not None:
            gt_ldmk3d = data["gt_ldmk3d"].reshape((-1, 3))
            gt_ldmk3d = np.dot(rot_mat, gt_ldmk3d.swapaxes(1, 0)).swapaxes(
                1, 0
            )
            data["gt_ldmk3d"] = gt_ldmk3d

            root_idx = data.get("root_idx", 0)
            data["gt_root"] = copy.deepcopy(gt_ldmk3d[root_idx : root_idx + 1])
            data["ldmk3d_relat"] = (
                gt_ldmk3d - gt_ldmk3d[root_idx, None, :]
            )  # root-relative

        # gt 3d verts in virtual camera space.
        if data.get("gt_verts", None) is not None:
            gt_verts = data["gt_verts"].reshape((-1, 3))
            gt_verts = np.dot(rot_mat, gt_verts.swapaxes(1, 0)).swapaxes(1, 0)
            data["gt_verts"] = gt_verts

        return data

    def __call__(self, data):
        bbox = data["gt_bboxes"]
        gt_ldmk = data["gt_ldmk"]
        intrinsic = data["intrinsic"]
        distortion = data["distortion"] if self.use_dist else np.zeros(5)
        if data.get("roi_offset") is None:
            data["roi_offset"] = np.array([[0, 0]])
        # TODO: @yuhao.dou bbox jitter.
        real_bbox = _expand_bbox(
            copy.deepcopy(bbox),
            data["raw_img_shape"],
            self.norm_ratio,
            self.norm_method,
        )
        data["real_bbox"] = real_bbox.copy()
        data["virtual_intrinsic"] = virtual_intrinsic = (
            intrinsic
            if self.virtual_intrinsic is None
            else self.virtual_intrinsic
        )
        if self.norm_center == "face":
            center_x = (bbox[0] + bbox[2]) * 0.5
            center_y = (bbox[1] + bbox[3]) * 0.5
            rot_center = np.array([[center_x, center_y]])
        elif self.norm_center == "eye":
            rot_center = (gt_ldmk[39] + gt_ldmk[42]) / 2.0
        else:
            raise ValueError("Not supported norm center.")
        rot_mat = self.cal_real2virtual_rot_mat(
            rot_center, intrinsic, distortion
        )
        data["real2vir_rotmat"] = rot_mat
        data["vir2real_rotmat"] = rot_mat.T
        real2virtual_mat, virtual2real_mat = self.get_real2virtual_mat(
            rot_mat, intrinsic, virtual_intrinsic
        )
        virtual_bbox = self.transform_points(
            real_bbox, real2virtual_mat, intrinsic, distortion
        ).reshape(-1)
        data["virtual_bbox"] = virtual_bbox
        # data["face_center"] = face_center
        # TODO: input norm or crop directly.
        img_crop = self.warp_crop_image(
            data["img"],
            data["roi_offset"],
            virtual_bbox,
            virtual2real_mat,
            intrinsic,
            distortion,
        )
        img_input = _resize_fix_ratio(img_crop, self.net_input_size)
        data["img"] = img_input.astype(np.float32)
        data["img_shape"] = img_input.shape
        data["net_input_size"] = np.array(self.net_input_size)

        data = self.get_gts(data, real2virtual_mat, virtual2real_mat, rot_mat)
        return data


@OBJECT_REGISTRY.register
class SimpleNormGenGridMap(VirtualCameraNorm):
    """Simple and Fast implementation of generating img_for_grid and gird_map.

    SimpleNormGenGridMap is used when `F.grid_sample` used in network forward.
    This process generates `img` and `grid_map` for remap basically the same
    as halo_perception_app.
    `img` is pre-croped from origin img without cv.resize to save resources.

    .. Note::
        Required keys: 'gt_bboxes', 'intrinsic', 'distortion', 'img_shape'
            'img'
        Affected keys: 'raw_img_shape', 'virtual_intrisic', 'net_input_size',
            'real2vir_rotmat', 'vir2real_rotmat', 'virtual_bbox', 'grid_map'
            'img_shape', 'img'

    Args:
        net_input_size: network input size (W, H). Defaults to 128.
        net_target_size: target image size (W, H). Defaults to 256.
        norm_ratio: Normed img expand ratio. Refer to RandomRotateCrop.
            Defaults to 1.2.
        expand_crop_hw: Crop img for grid sample. Defaults to 640.
        norm_method: Refer to RandomRotateCrop. Defaults to "longside_square".
        virtual_intrinsic: virtual camera intrinsic. It will be set to be the
            same as the real camera if it is None. Defaults to None.
        use_dist: use distortion or not when calculate rot mat. Defaults to
            False.
    """

    def __init__(
        self,
        net_input_size: Union[Tuple[int, int], int] = 128,
        net_target_size: Union[Tuple[int, int], int] = 256,
        norm_ratio: float = 1.2,
        expand_crop_hw: Union[Tuple[int, int], int] = 640,
        norm_method: str = "longside_square",
        norm_center: str = "face",
        virtual_intrinsic: Optional[np.ndarray] = None,
        use_dist: bool = False,
    ):
        if isinstance(net_input_size, Tuple):
            self.net_input_size = net_input_size
        elif isinstance(net_input_size, int):
            self.net_input_size = (net_input_size, net_input_size)
        else:
            raise ValueError(f"Not supported net_input_size: {net_input_size}")

        if isinstance(net_target_size, Tuple):
            self.net_target_size = net_target_size
        elif isinstance(net_target_size, int):
            self.net_target_size = (net_target_size, net_target_size)
        else:
            raise ValueError(
                f"Not supported net_target_size: {net_target_size}"
            )

        if isinstance(expand_crop_hw, Tuple):
            self.expand_crop_hw = expand_crop_hw
        elif isinstance(expand_crop_hw, int):
            self.expand_crop_hw = (expand_crop_hw, expand_crop_hw)
        else:
            raise ValueError(f"Not supported expand_crop_hw: {expand_crop_hw}")

        self.norm_ratio = norm_ratio
        self.norm_method = norm_method
        self.norm_center = norm_center
        self.virtual_intrinsic = virtual_intrinsic
        self.use_dist = use_dist
        self.eps = 1e-6

        grid_pt_mat = np.transpose(
            np.meshgrid(
                np.arange(self.net_input_size[0]),
                np.arange(self.net_input_size[1]),
            ),
            (1, 2, 0),
        ).reshape(-1, 2)
        self.grid_pt_mat = np.concatenate(
            (grid_pt_mat, np.ones_like(grid_pt_mat)[:, :1]), axis=1
        ).reshape(-1, 3)

    def __call__(self, data):
        data["raw_img_shape"] = data.get("raw_img_shape", data["img_shape"])
        data["roi_offset"] = data.get("roi_offset", np.array([[0, 0]]))
        data["save_crop"] = data.get("save_crop", False)
        # base info
        bbox = data["gt_bboxes"]
        gt_ldmk = data["gt_ldmk"]
        intrinsic = data["intrinsic"]
        distortion = data["distortion"] if self.use_dist else np.zeros(5)

        real_bbox = _expand_bbox(
            copy.deepcopy(bbox),
            data["raw_img_shape"],
            self.norm_ratio,
            self.norm_method,
            "raw",
        )
        data["virtual_intrinsic"] = virtual_intrinsic = (
            intrinsic
            if self.virtual_intrinsic is None
            else self.virtual_intrinsic
        )
        data["net_input_size"] = np.array(self.net_input_size)
        if self.norm_center == "face":
            center_x = (bbox[0] + bbox[2]) * 0.5
            center_y = (bbox[1] + bbox[3]) * 0.5
            rot_center = np.array([[center_x, center_y]])
        elif self.norm_center == "eye":
            rot_center = (gt_ldmk[39] + gt_ldmk[42]) / 2.0
        else:
            raise ValueError("Not supported norm center.")
        rot_mat = self.cal_real2virtual_rot_mat(
            rot_center, intrinsic, distortion
        )
        data["real2vir_rotmat"] = rot_mat
        data["vir2real_rotmat"] = rot_mat.T
        real2virtual_mat, virtual2real_mat = self.get_real2virtual_mat(
            rot_mat, intrinsic, virtual_intrinsic
        )
        virtual_bbox = self.transform_points(
            real_bbox, real2virtual_mat, None, None
        ).reshape(-1)
        data["virtual_bbox"] = virtual_bbox

        if not data["save_crop"]:
            # input crop img
            expand_img_bbox = _expand_bbox(
                copy.deepcopy(bbox),
                data["raw_img_shape"],
                self.expand_crop_hw,
                self.norm_method,
                "clip",
            )
            expand_left, expand_top, expand_right, expand_bottom = list(
                map(int, expand_img_bbox.reshape(-1))
            )
            if expand_bottom - expand_top > self.expand_crop_hw[0]:
                expand_bottom = expand_top + self.expand_crop_hw[0]
            if expand_right - expand_left > self.expand_crop_hw[1]:
                expand_right = expand_left + self.expand_crop_hw[1]
            img_expand_crop = data["img"][
                expand_top:expand_bottom, expand_left:expand_right
            ]
            if img_expand_crop.ndim == 2:
                img_expand_crop = img_expand_crop[..., None]
            assert img_expand_crop.ndim == 3
            img_for_grid = np.zeros(
                self.expand_crop_hw + (img_expand_crop.shape[2],)
            )
            img_for_grid[
                : expand_bottom - expand_top, : expand_right - expand_left
            ] = img_expand_crop
            data["img"] = img_for_grid.astype(np.float32)
            data["img_shape"] = img_for_grid.shape
            data["expand_img_bbox"] = expand_img_bbox
        else:
            data["img"] = data["img"].astype(np.float32)
            expand_left, expand_top = data["roi_offset"][0]
            assert self.expand_crop_hw[:2] == data["img_shape"][:2]

        # grid
        (
            vir_bbox_left,
            vir_bbox_top,
            vir_bbox_right,
            vir_bbox_bottom,
        ) = virtual_bbox
        vir_bbox_width = vir_bbox_right - vir_bbox_left
        vir_bbox_height = vir_bbox_bottom - vir_bbox_top
        warp_scale_x = self.net_input_size[0] / vir_bbox_width
        warp_scale_y = self.net_input_size[1] / vir_bbox_height
        warp_scale = min(warp_scale_x, warp_scale_y)
        trans_mat = np.array(
            [
                [1 / warp_scale, 0, 0],
                [0, 1 / warp_scale, 0],
                [vir_bbox_left, vir_bbox_top, 1],
            ]
        )
        data["trans_mat"] = trans_mat
        point_in_src = self.grid_pt_mat @ trans_mat @ virtual2real_mat
        grid_sample_map_xy = point_in_src[:, :2] / point_in_src[:, 2:]

        grid_sample_map_xy_local = grid_sample_map_xy - np.array(
            [expand_left, expand_top]
        ).reshape((1, 2))
        grid_sample_map_xy_norm = (
            grid_sample_map_xy_local
            * 2
            / np.array([self.expand_crop_hw[1], self.expand_crop_hw[0]])
            - 1
        )
        grid_sample_map_xy_norm = grid_sample_map_xy_norm.reshape(
            self.net_input_size[1], self.net_input_size[0], 2
        ).astype(np.float32)
        grid_sample_map_xy_norm = np.transpose(
            grid_sample_map_xy_norm, (2, 0, 1)
        )
        data["grid_map"] = torch.from_numpy(grid_sample_map_xy_norm)

        # gt
        data = self.get_gts(data, real2virtual_mat, virtual2real_mat, rot_mat)

        return data


@OBJECT_REGISTRY.register
class SimpleNormPositionEncoding(object):
    """Simple and Fast implementation of normed position map.

    .. Note::
        Required keys: 'virtual_bbox', 'virtual_intrinsic'
        Affected keys: 'position map'

    Args:
        net_input_size: network input size (W, H). Defaults to 128.
    """

    def __init__(
        self,
        net_input_size: Union[Tuple[int, int], int] = 128,
    ):
        if isinstance(net_input_size, Tuple):
            self.net_input_size = net_input_size
        elif isinstance(net_input_size, int):
            self.net_input_size = (net_input_size, net_input_size)
        else:
            raise ValueError(f"Not supported net_input_size: {net_input_size}")

        range_x = np.linspace(-1, 1, self.net_input_size[0])
        range_y = np.linspace(-1, 1, self.net_input_size[1])
        base_position_mat_x_, base_position_mat_y_ = np.meshgrid(
            range_x, range_y
        )
        self.base_position_mat_x_ = base_position_mat_x_.astype(np.float32)
        self.base_position_mat_y_ = base_position_mat_y_.astype(np.float32)

    def __call__(self, data):
        virtual_bbox = data.get("virtual_bbox")
        virtual_intri = data.get("virtual_intrinsic")
        if virtual_bbox is None or virtual_intri is None:
            raise ValueError(
                "Make sure `virtual_bbox` and `virtual_intrinsic` are valid."
            )
        cx, cy = virtual_intri[:2, 2]
        (
            vir_bbox_left,
            vir_bbox_top,
            vir_bbox_right,
            vir_bbox_bottom,
        ) = virtual_bbox
        vir_bbox_width = vir_bbox_right - vir_bbox_left
        vir_bbox_height = vir_bbox_bottom - vir_bbox_top
        scale_x = vir_bbox_width / (2 * cx)
        scale_y = vir_bbox_height / (2 * cy)
        pos_uv_map_x = self.base_position_mat_x_ * scale_x
        pos_uv_map_y = self.base_position_mat_y_ * scale_y
        position_map = np.stack([pos_uv_map_x, pos_uv_map_y], axis=0)
        data["position_map"] = torch.from_numpy(position_map)

        return data


@OBJECT_REGISTRY.register
class PositionEncoding(object):
    """Generate position map on virtual camera space and concat with image.

    Args:
        net_input_size: input size of the network, (W, H)
        to_yuv420sp: transfer color space to yuv420sp or not. Defaults to True.
        concat_img: concat position map with image or not. Defaults to True.
    """

    def __init__(
        self,
        net_input_size: Union[Tuple[int, int], int],
        to_yuv420sp: bool = True,
        concat_img: bool = True,
    ):
        if isinstance(net_input_size, Tuple):
            self.net_input_size = net_input_size
        elif isinstance(net_input_size, int):
            self.net_input_size = (net_input_size, net_input_size)
        else:
            raise ValueError(f"Not supported net_input_size: {net_input_size}")
        self.to_yuv420sp = to_yuv420sp
        self.concat_img = concat_img

    def generate_pos_map(
        self,
        bbox: Union[List, np.ndarray],
        dst_size: Tuple,
        cx: float,
        cy: float,
        to_yuv420sp: bool = True,
    ):
        w, h = dst_size[:2]
        left, top, right, bottom = bbox
        range_x = np.linspace(left, right, w, endpoint=False)
        range_y = np.linspace(top, bottom, h, endpoint=False)
        pos_map = np.transpose(
            np.meshgrid(range_x, range_y), (1, 2, 0)
        ).reshape(-1, 2)
        pos_map = pos_map / np.array([2 * cx, 2 * cy]) * 255
        pos_map = pos_map.reshape(h, w, 2).astype(np.uint8)
        horizon_pos_map, vertical_pos_map = pos_map[..., 0], pos_map[..., 1]
        if to_yuv420sp:
            horizon_pos_map = self.convert_channel_to_420sp(horizon_pos_map)
            vertical_pos_map = self.convert_channel_to_420sp(vertical_pos_map)
        return horizon_pos_map, vertical_pos_map

    def convert_channel_to_420sp(self, image):
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

    def __call__(self, data):
        img = data["img"]
        bbox = data["virtual_bbox"]
        if data.get("virtual_intrinsic", None) is not None:
            intrinsic = data["virtual_intrinsic"]
        else:
            intrinsic = data["intrinsic"]
        pos_map_h, pos_map_v = self.generate_pos_map(
            bbox,
            self.net_input_size,
            cx=intrinsic[0, 2],
            cy=intrinsic[1, 2],
            to_yuv420sp=self.to_yuv420sp,
        )
        pos_map_h = _resize_fix_ratio(pos_map_h, self.net_input_size)
        pos_map_v = _resize_fix_ratio(pos_map_v, self.net_input_size)
        data["pos_map_h"] = pos_map_h
        data["pos_map_v"] = pos_map_v
        if self.concat_img:
            data["img"] = np.concatenate(
                (img, pos_map_h[..., None], pos_map_v[..., None]), 2
            )
        return data


@OBJECT_REGISTRY.register
class BboxEncodingCLIFF(object):
    """Encode bbox infomation into a vector.

    Refer to ECCV2022 paper: `CLIFF: Carrying Location Information in Full
    Frames into Human Pose and Shape Estimation`.

    Args:
        focal_length_norm: focal length in pixel.
    """

    def __init__(self, focal_length_norm: float = 2000):
        self.focol_length_nrom = focal_length_norm

    def __call__(self, data):
        intrinsic = data["intrinsic"]
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        f_cliff = 0.5 * (fx + fy)
        # NOTE: input bbox should be a square.
        input_bbox = data["input_bbox"]
        cx_cliff = (0.5 * (input_bbox[0] + input_bbox[2]) - cx) / f_cliff
        cy_cliff = (0.5 * (input_bbox[1] + input_bbox[3]) - cy) / f_cliff
        b_cliff = (
            max(
                input_bbox[2] - input_bbox[0],
                input_bbox[3] - input_bbox[1],
            )
            / f_cliff
        )
        # NOTE: f_norm is not included in the original paper.
        f_norm = f_cliff / self.focol_length_nrom
        data["cliff_info"] = (
            np.array([cx_cliff, cy_cliff, b_cliff, f_norm])
            .reshape(4, 1, 1)
            .astype(np.float32)
        )
        return data


@OBJECT_REGISTRY.register
class ImageCLAHE(object):
    """Do CLAHE (Contrast Limited Adaptive Histogram Equalization) on image.

    Args:
        clipLimit: thresh of contrast. Default to 5.0
        tileGridSize: size of patch for histogram equalization. Default to 8.

    """

    def __init__(
        self,
        clipLimit: float = 5.0,
        tileGridSize: Union[Tuple[int, int], int] = 8,
    ):
        self.clipLimit = clipLimit
        if isinstance(tileGridSize, Tuple):
            self.tileGridSize = tileGridSize
        elif isinstance(tileGridSize, int):
            self.tileGridSize = (tileGridSize, tileGridSize)
        else:
            raise ValueError(f"Not supported tileGridSize: {tileGridSize}")

    def __call__(self, data):
        channel_num = data["img"].shape[2] if data["img"].ndim == 3 else 0
        if channel_num > 0:
            img = cv2.cvtColor(data["img"], cv2.COLOR_RGB2GRAY)
        else:
            img = data["img"]
        clahe = cv2.createCLAHE(
            clipLimit=self.clipLimit, tileGridSize=self.tileGridSize
        )
        img_cla = clahe.apply(img)

        if channel_num > 0:
            img_cla_all = np.stack([img_cla] * channel_num, axis=2)
        assert img_cla_all.shape == data["img_shape"]
        data["img"] = img_cla_all
        if data.get("gt_img") is not None:
            data["gt_img"] = img_cla_all

        return data
