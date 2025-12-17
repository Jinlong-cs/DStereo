import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from dataclasses_json import DataClassJsonMixin

from hat.data.transforms.affine import (
    AffineAugMat,
    AffineMat2DGenerator,
    ImageAffineTransform,
    LabelAffineTransform,
)
from hat.data.transforms.bbox import clip_bbox
from hat.data.transforms.detection import _transform_bboxes
from hat.registry import OBJECT_REGISTRY


@dataclass
class CropParm(DataClassJsonMixin):
    norm_len: int = 116
    norm_method: str = "height"
    output_wh: List = None
    min_crop_scale: float = 0.8
    max_crop_scale: float = 1.2
    max_coord_jitter_ratio: float = 0.05
    img_min_scale: float = 0.01
    img_max_scale: float = 100
    padd_val: float = 0
    random_roi_ratio: float = 0
    restrict_roi_in_center: bool = False
    input_wh: Optional[List] = None
    flip_ratio: float = 0.0


@dataclass
class ImgCropParm(DataClassJsonMixin):
    target_wh: List = None
    inter_method: int = 10
    use_pyramid: bool = True
    pyramid_min_step: float = 0.7
    pyramid_max_step: float = 0.8
    pixel_center_aligned: bool = False


@dataclass
class BBoxTransformParm(DataClassJsonMixin):
    clip: bool = False
    min_valid_area: float = 100
    min_valid_clip_area_ratio: float = 0.02
    min_edge_size: float = 10
    label_type: str = "classification"


@OBJECT_REGISTRY.register
class RoiTransformer:
    """
    Iterable transformer base on rois for object detection.

    Parameters
    ----------
    resize_wh : list/tuple of 2 int, optional
        Resize input image to target size, by default None
    **kwargs :
        Please see :py:class:`AffineMatFromROIBoxGenerator` and
        :py:class:`ImageAffineTransform`
    """

    def __init__(
        self,
        roi_crop_parm: CropParm or dict = CropParm,
        img_crop_parm: ImgCropParm or dict = ImgCropParm,
        bbox_ts_parm: BBoxTransformParm or dict = BBoxTransformParm,
        convert_roi: bool = False,
    ):
        if isinstance(img_crop_parm, dict):
            img_crop_parm = ImgCropParm.from_dict(img_crop_parm)
        if isinstance(roi_crop_parm, dict):
            roi_crop_parm = CropParm.from_dict(roi_crop_parm)
        if isinstance(bbox_ts_parm, dict):
            bbox_ts_parm = BBoxTransformParm.from_dict(bbox_ts_parm)

        self._img_ts = ImageAffineTransform(
            dst_wh=img_crop_parm.target_wh,
            inter_method=img_crop_parm.inter_method,
            border_value=0,
            use_pyramid=img_crop_parm.use_pyramid,
            pyramid_min_step=img_crop_parm.pyramid_min_step,
            pyramid_max_step=img_crop_parm.pyramid_max_step,
            pixel_center_aligned=img_crop_parm.pixel_center_aligned,
        )
        self._bbox_ts_kwargs = bbox_ts_parm.to_dict()
        self._roi_crop_parm = roi_crop_parm
        self.label_type = self._bbox_ts_kwargs.pop("label_type")
        self.convert_roi = convert_roi
        assert self.label_type in ["detection", "classification"]

    def __call__(self, results):
        img = results["img"]
        gt_boxes = results.get("gt_bboxes", None)
        ig_regions = results.get("ig_bboxes", None)
        roi = results["crop_roi"].copy()

        src_h, src_w, _ = img.shape
        self._roi_crop_parm.input_wh = [src_w, src_h]

        affine_aug_param = self.get_affine_transform(
            self._roi_crop_parm, roi[:4], return_affine_mat_only=True
        )
        ts_img = self._img_ts(img, affine_aug_param.mat)
        ts_img_wh = ts_img.shape[:2][::-1]

        if gt_boxes is not None:
            gt_classes = results.get("gt_classes")
            gt_boxes = np.concatenate([gt_boxes, gt_classes[:, None]], -1)
            ts_gt_boxes, ts_ig_regions = _transform_bboxes(
                gt_boxes,
                ig_regions,
                (0, 0, ts_img_wh[0], ts_img_wh[1]),
                affine_aug_param,
                **self._bbox_ts_kwargs,
            )
            results["gt_bboxes"] = ts_gt_boxes[:, :-1]
            results["gt_classes"] = ts_gt_boxes[:, -1].astype(np.int64)
            results["ig_bboxes"] = ts_ig_regions

        if self.convert_roi:
            bbox_ts = LabelAffineTransform(label_type="box")
            ts_crop_roi = bbox_ts(
                roi, affine_aug_param.mat, flip=affine_aug_param.flipped
            )
            results["ts_crop_roi"] = clip_bbox(
                ts_crop_roi, (0, 0, ts_img_wh[0], ts_img_wh[1]), need_copy=True
            )

            results["affine_aug_param"] = affine_aug_param.mat
        results["img"] = ts_img
        results["img_shape"] = ts_img.shape
        return results

    @staticmethod
    def norm_method2norm_dist_func(norm_method):
        if norm_method == "width":  # width

            def roi_norm_func(roi):
                return np.sqrt((roi[2] - roi[0]) ** 2)

        elif norm_method == "height":  # height

            def roi_norm_func(roi):
                return np.sqrt((roi[3] - roi[1]) ** 2)

        elif norm_method == "diagonal":  # diagonal

            def roi_norm_func(roi):
                return np.sqrt((roi[3] - roi[1]) ** 2 + (roi[2] - roi[0]) ** 2)

        elif norm_method == "mean_width_height":

            def roi_norm_func(roi):
                return (roi[3] - roi[1] + roi[2] - roi[0]) * 0.5

        elif norm_method == "sqrt_width_height":

            def roi_norm_func(roi):
                return np.sqrt((roi[3] - roi[1]) * (roi[2] - roi[0]))

        elif norm_method == "max_width_height":

            def roi_norm_func(roi):
                return max(roi[3] - roi[1], roi[2] - roi[0])

        else:
            raise NotImplementedError
        return roi_norm_func

    @staticmethod
    def calculate_scale(crop_parm, roi):
        # from densebox/src/densebox/roi_transform/roi_transform_executor.cpp::CalculateScale  # noqa
        bs = np.random.uniform(
            crop_parm.min_crop_scale, crop_parm.max_crop_scale
        )
        inst_norm_dist = RoiTransformer.norm_method2norm_dist_func(  # noqa
            norm_method=crop_parm.norm_method
        )(
            roi.tolist()
        )  # noqa
        bs *= crop_parm.norm_len / inst_norm_dist
        bs = max(crop_parm.img_min_scale, min(bs, crop_parm.img_max_scale))
        # assert (0.01 <= bs <= 100)
        return bs

    @staticmethod
    def calculate_ori_roi_center(crop_parm: CropParm, base_scale, roi):
        # make a copy
        roi_x1, roi_y1, roi_x2, roi_y2 = roi.tolist()

        # from densebox/src/densebox/roi_transform/roi_transform_executor.cpp::CalcualteOriROICenter  # noqa
        def get_min_img_scale(src_h, src_w, min_size: float):
            return float(min_size / min(src_h, src_w))

        def get_rand_pyramid_scale(scale_begin, scale_end):
            return math.sqrt(
                scale_end * scale_end
                - np.random.uniform(0, 1)
                * (scale_end * scale_end - scale_begin * scale_begin)
            )

        scale = max(
            [
                base_scale,
                get_min_img_scale(
                    crop_parm.input_wh[1], crop_parm.input_wh[0], 16
                ),
            ]
        )
        rand_center = np.random.uniform(0, 1) < crop_parm.random_roi_ratio
        center = [crop_parm.input_wh[0] / 2.0, crop_parm.input_wh[1] / 2.0]

        if not rand_center:
            center = [(roi_x1 + roi_x2) / 2.0, (roi_y1 + roi_y2) / 2.0]
            center[0] += (
                crop_parm.norm_len
                / scale
                * crop_parm.max_coord_jitter_ratio
                * np.random.uniform(-1, 1)
            )
            center[1] += (
                crop_parm.norm_len
                / scale
                * crop_parm.max_coord_jitter_ratio
                * np.random.uniform(-1, 1)
            )
        else:
            min_scale = max(
                [
                    get_min_img_scale(
                        crop_parm.input_wh[1], crop_parm.input_wh[0], 16
                    ),
                    crop_parm.img_min_scale,
                ]
            )
            scale = get_rand_pyramid_scale(min_scale, crop_parm.img_max_scale)
            center[0] = crop_parm.input_wh[0] * np.random.uniform(0, 1)
            center[1] = crop_parm.input_wh[1] * np.random.uniform(0, 1)

        # TODO impl restrict_roi_in_center logic
        assert not crop_parm.restrict_roi_in_center
        return center, scale

    @staticmethod
    def generate_affine_aug_parm(crop_parm: CropParm, center, scale):
        tri_points_src = [
            center,
            [center[0] + 100, center[1]],
            [center[0], center[1] + 100],
        ]
        tri_points_dst = [
            [crop_parm.output_wh[0] / 2.0, crop_parm.output_wh[1] / 2.0],
            [
                crop_parm.output_wh[0] / 2.0 + 100 * scale,
                crop_parm.output_wh[1] / 2.0,
            ],  # noqa
            [
                crop_parm.output_wh[0] / 2.0,
                crop_parm.output_wh[1] / 2.0 + 100 * scale,
            ],  # noqa
        ]

        affine_mat = cv2.getAffineTransform(
            np.array(tri_points_src).astype(np.float32),  # noqa
            np.array(tri_points_dst).astype(np.float32),
        )  # noqa
        affine_mat = np.concatenate([affine_mat, [[0, 0, 1]]], axis=0)
        flip = False
        if np.random.uniform(0, 1) < crop_parm.flip_ratio:
            flip_mat = AffineMat2DGenerator.flip_with_axis(
                center_xy=np.array(crop_parm.output_wh) / 2
            )
            affine_mat = AffineMat2DGenerator.stack_affine_transform(
                *[affine_mat, flip_mat]
            )[0:2]
            flip = True

        affine_aug_parm = AffineAugMat(mat=affine_mat, flipped=flip)
        return affine_aug_parm

    def get_affine_transform(
        self, crop_parm: CropParm, roi: np.ndarray, return_affine_mat_only=True
    ):
        base_scale = self.calculate_scale(crop_parm, roi)

        center, scale = self.calculate_ori_roi_center(
            crop_parm, base_scale, roi
        )
        affine_aug_parm = self.generate_affine_aug_parm(
            crop_parm, center, scale
        )

        if return_affine_mat_only:
            return affine_aug_parm
        # go back to src roi with context
        # to homo coord
        affine_mat = affine_aug_parm.mat
        affine_mat = np.row_stack((affine_mat, np.array([[0, 0, 1]])))
        affine_mat_inv = np.linalg.inv(affine_mat)
        dst_lt = [0, 0, 1]
        dst_rb = [crop_parm.output_wh[0], crop_parm.output_wh[1], 1]
        src_roi_lt = np.dot(affine_mat_inv, np.array(dst_lt))
        src_roi_rb = np.dot(affine_mat_inv, np.array(dst_rb))
        # generate crop roi
        roi_x1 = src_roi_lt[0] / src_roi_lt[2]
        roi_y1 = src_roi_lt[1] / src_roi_lt[2]
        roi_x2 = src_roi_rb[0] / src_roi_rb[2]
        roi_y2 = src_roi_rb[1] / src_roi_rb[2]
        return affine_mat, [roi_x1, roi_y1, roi_x2, roi_y2]
