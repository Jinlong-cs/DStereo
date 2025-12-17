# Copyright (c) Horizon Robotics. All rights reserved.
import random

import numpy as np

from hat.data.datasets.densebox_dataset_with_roilist import RoiTransformCroper
from hat.data.transforms.affine import LabelAffineTransform
from hat.data.transforms.bbox import (
    clip_bbox,
    filter_bbox,
    remap_bbox_label_by_area,
    remap_bbox_label_by_clip_area_ratio,
)
from hat.data.transforms.detection import pad_detection_data
from hat.data.transforms.roi_crop import (
    BBoxTransformParm,
    CropParm,
    ImgCropParm,
)
from hat.registry import OBJECT_REGISTRY

__all__ = ["SliceTime", "MaskOutsideRegion", "RoiTransformCroperWithMask"]


@OBJECT_REGISTRY.register
class SliceTime(object):
    def __init__(self, key="multi_label"):
        self.key = key

    def _slice_time(self, time):
        # 4个label，【位数、百位数、十位数、个位数】
        if time == -1:
            return [-1, -1, -1, -1]
        if time == 0:
            return [0, 0, 0, 0]
        res = []
        cnt = 0
        for _ in range(3):
            if time == 0:
                res.append(0)
                continue
            res.append(time % 10)
            time = time // 10
            cnt += 1
        return [cnt - 1] + res[::-1]

    def __call__(self, data):
        if self.key not in data.keys():
            time = data["gt_classes"][0]
        else:
            time = data[self.key]
        gt_time = self._slice_time(time)
        data[self.key] = np.array(gt_time, dtype=np.int64)
        return data


@OBJECT_REGISTRY.register
class MaskOutsideRegion(object):
    def __init__(
        self,
        input_wh: list,
        mask_out_gt_crop_flag=True,
        extend_ratio=0.0,
        mask_in_bpu=False,
        transpose_hw=False,
    ):
        self.input_wh = input_wh
        self.mask_out_gt_crop_flag = mask_out_gt_crop_flag
        self.extend_ratio = extend_ratio
        self.mask_in_bpu = mask_in_bpu
        self.transpose_hw = transpose_hw

    def __call__(self, data):
        if not self.mask_out_gt_crop_flag:
            return data
        assert "ts_crop_roi" in data.keys()
        x_1, y_1, x_2, y_2 = data["ts_crop_roi"][0]
        h, w = y_2 - y_1, x_2 - x_1
        img_h, img_w, _ = data["img_shape"]
        xnew_1 = max(int(x_1 - w * self.extend_ratio), 0)
        ynew_1 = max(int(y_1 - h * self.extend_ratio), 0)
        xnew_2 = min(int(x_2 + w * self.extend_ratio), img_w)
        ynew_2 = min(int(y_2 + h * self.extend_ratio), img_h)
        if not self.mask_in_bpu:
            tmp_img = np.zeros_like(data["img"])
            tmp_img[ynew_1:ynew_2, xnew_1:xnew_2, :] = data["img"][
                ynew_1:ynew_2, xnew_1:xnew_2, :
            ]
            data["img"] = tmp_img
        x_axis_mask = np.zeros((1, self.input_wh[0]))
        y_axis_mask = np.zeros((1, self.input_wh[1]))
        x_axis_mask[:, xnew_1:xnew_2] = 1
        y_axis_mask[:, ynew_1:ynew_2] = 1
        if self.transpose_hw:
            y_axis_mask = y_axis_mask.transpose(1, 0)
        data["mask_width"] = np.expand_dims(
            x_axis_mask.astype(np.float32), axis=0
        )
        data["mask_height"] = np.expand_dims(
            y_axis_mask.astype(np.float32), axis=0
        )
        return data


@OBJECT_REGISTRY.register
class RoiTransformCroperWithMask(RoiTransformCroper):
    """transformer base on rois for traffic lens object detection.

    Unlike RoiTransformCroper, there are two main differences:
    a. In the gt sampling mode, the converted GT box will be returned at the
    same time.
    b. Support masking the crop region based on the converted GT Bbox.
    To solve the problem of adjacent lights.

    Args:
        mask_out_gt_crop_flag: Whether to mask the parts in the crop area and
            outside the GT box.
        mask_in_bpu: Whether the operation of the mask is placed on the bpu.
        transpose_hw: Whether the dimension of the height mask is transposed
            in advance.
        extend_ratio: The proportion of scale-out at mask.
        **kwargs :
            Please see :py:class:`RoiTransformCroper`.
    """

    def __init__(
        self,
        min_sample_num,
        max_sample_num,
        roi_crop_parm: CropParm or dict,
        img_crop_parm: ImgCropParm or dict,
        bbox_ts_parm: BBoxTransformParm or dict,
        from_roi_ratio: float = 1,
        hard_sample_select_ratio: float = 0,
        transforms: list or dict = None,
        mask_out_gt_crop_flag: bool = False,
        mask_in_bpu: bool = False,
        transpose_hw: bool = False,
        extend_ratio: float = 0.0,
    ):
        super(RoiTransformCroperWithMask, self).__init__(
            min_sample_num,
            max_sample_num,
            roi_crop_parm,
            img_crop_parm,
            bbox_ts_parm,
            from_roi_ratio,
            hard_sample_select_ratio,
            transforms,
        )
        if isinstance(img_crop_parm, dict):
            img_crop_parm = ImgCropParm.from_dict(img_crop_parm)
        self._img_crop_parm = img_crop_parm
        self.mask_out_gt_crop_flag = mask_out_gt_crop_flag
        self.mask_in_bpu = mask_in_bpu
        self.transpose_hw = transpose_hw
        self.extend_ratio = extend_ratio

    def __call__(self, data):
        img, gt_boxes = data["img"], data["gt_boxes"]
        ig_regions, gt_cropes, roi_lists = (
            data["ig_regions"],
            data["gt_cropes"],
            data["roi_list"],
        )
        src_h, src_w, _ = img.shape
        self._roi_crop_parm.input_wh = [src_w, src_h]

        has_roi_list = False
        if roi_lists is not None and len(roi_lists.rois) > 0:
            has_roi_list = True
        from_roi = False
        if random.random() < self.from_roi_ratio:
            from_roi = True

        roi_idxs = []
        use_roi_list_crop_img = False
        if has_roi_list and from_roi:
            rois_num = len(roi_lists.rois)
            if rois_num < self.min_sample_num:
                num = np.int64(self.min_sample_num / rois_num) + 1
                roi_idxs = np.arange(num) % rois_num
            elif rois_num <= self.max_sample_num:
                roi_idxs = np.arange(rois_num)
            else:
                roi_idxs = np.arange(rois_num)
                random.shuffle(roi_idxs)
                roi_idxs = roi_idxs[: self.max_sample_num]
            use_roi_list_crop_img = True
        elif len(gt_cropes) > 0:
            rois_num = len(gt_cropes)
            if rois_num < self.min_sample_num:
                num = np.int64(self.min_sample_num / rois_num) + 1
                roi_idxs = np.arange(num) % rois_num
            elif rois_num < self.max_sample_num:
                roi_idxs = np.arange(rois_num)
            else:
                roi_idxs = np.arange(rois_num)
                random.shuffle(roi_idxs)
                roi_idxs = roi_idxs[: self.max_sample_num]
        gt_cropes_tmp = gt_cropes.copy()
        results = []
        for roi_idx in roi_idxs:
            affine_aug_param = None
            if use_roi_list_crop_img:
                roi = roi_lists.rois[roi_idx]
                assert roi.pt1 != [-1, -1]
                affine_aug_param = (
                    self.get_roi_transform_mat_wrt_instance_from_roi_list(
                        self._roi_crop_parm, roi.pt1, roi.pt2, roi.center
                    )
                )
                gt_crop = None
            else:
                roi = gt_cropes_tmp[roi_idx]
                affine_aug_param = self.get_roi_transform_mat_wrt_instance(
                    self._roi_crop_parm, roi[:4]
                )
                gt_crop = roi[:-1]

            (
                ts_img,
                ts_gt_boxes,
                ts_ig_regions,
                ts_gt_crop,
            ) = self.affine_img_and_bboxes(
                img, gt_boxes, ig_regions, affine_aug_param, gt_crop
            )
            if self.mask_out_gt_crop_flag:
                ts_img_tmp = np.zeros_like(ts_img)
                x_1, y_1, x_2, y_2 = ts_gt_crop[0]
                w_extend, h_extend = (x_2 - x_1) * self.extend_ratio, (
                    y_2 - y_1
                ) * self.extend_ratio
                xnew_1 = max(0, int(x_1 - w_extend))
                ynew_1 = max(0, int(y_1 - h_extend))
                target_w, target_h = self._img_crop_parm.target_wh
                xnew_2 = min(target_w, int(x_2 + w_extend))
                ynew_2 = min(target_h, int(y_2 + h_extend))
                if not self.mask_in_bpu:
                    ts_img_tmp[ynew_1:ynew_2, xnew_1:xnew_2, :] = ts_img[
                        ynew_1:ynew_2, xnew_1:xnew_2, :
                    ]
                    ts_img = ts_img_tmp
                x_axis_mask = np.zeros((1, self._roi_crop_parm.output_wh[0]))
                y_axis_mask = np.zeros((1, self._roi_crop_parm.output_wh[1]))
                x_axis_mask[:, xnew_1:xnew_2] = 1
                y_axis_mask[:, ynew_1:ynew_2] = 1
                if self.transpose_hw:
                    y_axis_mask = y_axis_mask.transpose(1, 0)
                # calu ignore region
                mask_regions = []
                if xnew_1 != 0:
                    mask_regions.append([0, 0, xnew_1, target_h - 1, 1])
                if xnew_2 != target_w - 1:
                    mask_regions.append(
                        [xnew_2, 0, target_w - 1, target_h - 1, 1]
                    )
                if ynew_1 != 0:
                    mask_regions.append([0, 0, target_w - 1, ynew_1, 1])
                if ynew_2 != target_h - 1:
                    mask_regions.append(
                        [0, ynew_2, target_w - 1, target_h - 1, 1]
                    )
                mask_regions = (
                    np.array(mask_regions, dtype=ts_ig_regions.dtype)
                    if len(mask_regions) > 0
                    else np.zeros((0, 5), dtype=ts_ig_regions.dtype)
                )  # noqa
                ig_regions = np.concatenate(
                    [ts_ig_regions, mask_regions], axis=0
                )

            result = pad_detection_data(
                ts_img,
                ts_gt_boxes,
                ts_ig_regions,
            )
            if self.transforms is not None:
                result = self.transforms(result)
            result["img"] = result["img"].transpose(2, 0, 1)
            if self.mask_out_gt_crop_flag:
                result["mask_width"] = np.expand_dims(
                    x_axis_mask.astype(np.float32), axis=0
                )
                result["mask_height"] = np.expand_dims(
                    y_axis_mask.astype(np.float32), axis=0
                )
            results.append(result)
        return results

    def affine_img_and_bboxes(
        self,
        img,
        gt_boxes,
        ig_regions,
        affine_aug_param,
        gt_crop=None,
    ):
        ts_img = self._img_ts(img, affine_aug_param.mat)
        ts_img_wh = ts_img.shape[:2][::-1]

        if self.label_type == "detection":
            boxes = gt_boxes
            ig_region = ig_regions
        else:
            raise ValueError(f"Unknown label_type:{self.label_type}")
        ts_gt_boxes, ts_ig_regions, ts_gt_crop = self._transform_bboxes(
            boxes,
            ig_region,
            gt_crop,
            (0, 0, ts_img_wh[0], ts_img_wh[1]),
            affine_aug_param,
            **self._bbox_ts_kwargs,
        )
        return ts_img, ts_gt_boxes, ts_ig_regions, ts_gt_crop

    def _transform_bboxes(
        self,
        gt_boxes,
        ig_regions,
        gt_crop,
        img_roi,
        affine_aug_param,
        clip=True,
        min_valid_area=8,
        min_valid_clip_area_ratio=0.5,
        min_edge_size=2,
    ):
        bbox_ts = LabelAffineTransform(label_type="box")

        ts_gt_boxes = bbox_ts(
            gt_boxes, affine_aug_param.mat, flip=affine_aug_param.flipped
        )
        if clip:
            clip_gt_boxes = clip_bbox(ts_gt_boxes, img_roi, need_copy=True)
        else:
            clip_gt_boxes = ts_gt_boxes
        clip_gt_boxes = remap_bbox_label_by_area(clip_gt_boxes, min_valid_area)
        clip_gt_boxes = remap_bbox_label_by_clip_area_ratio(
            ts_gt_boxes, clip_gt_boxes, min_valid_clip_area_ratio
        )
        clip_gt_boxes = filter_bbox(
            clip_gt_boxes,
            img_roi,
            allow_outside_center=True,
            min_edge_size=min_edge_size,
        )

        if ig_regions is not None:
            ts_ig_regions = bbox_ts(
                ig_regions, affine_aug_param.mat, flip=affine_aug_param.flipped
            )
            if clip:
                clip_ig_regions = clip_bbox(ts_ig_regions, img_roi)
            else:
                clip_ig_regions = ts_ig_regions
            clip_ig_regions = filter_bbox(
                clip_ig_regions,
                img_roi,
                allow_outside_center=True,
                min_edge_size=min_edge_size,
            )
        else:
            clip_ig_regions = None

        if gt_crop is not None:
            ts_gt_crop = bbox_ts(
                gt_crop, affine_aug_param.mat, flip=affine_aug_param.flipped
            )
            if clip:
                ts_gt_crop = clip_bbox(ts_gt_crop, img_roi)
            else:
                ts_gt_crop = ts_gt_crop
        else:
            ts_gt_crop = None

        return clip_gt_boxes, clip_ig_regions, ts_gt_crop
