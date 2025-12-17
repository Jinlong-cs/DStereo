import copy
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY
from .affine import (
    AffineAugMat,
    AffineMat2DGenerator,
    AffineMatFromROIBoxGenerator,
    ImageAffineTransform,
    LabelAffineTransform,
    Point2DAffineTransform,
    _pad_array,
    resize_affine_mat,
)


@OBJECT_REGISTRY.register
class FlankCornersDetRoiTransform:
    def __init__(
        self,
        target_wh,
        img_scale_range=(0.5, 2.0),
        roi_scale_range=(0.8, 1.0 / 0.8),
        min_sample_num=1,
        max_sample_num=5,
        center_aligned=True,
        inter_method=10,
        use_pyramid=False,
        pyramid_min_step=0.45,
        pyramid_max_step=0.8,
        pixel_center_aligned=True,
        min_valid_area=8,
        min_valid_clip_area_ratio=0.5,
        min_edge_size=2,
        rand_translation_ratio=0,
        rand_aspect_ratio=0,
        rand_rotation_angle=0,
        flip_prob=0.5,
        rand_sampling_bbox=False,
        resize_wh=None,
        keep_aspect_ratio=False,
        min_flank_width=5,
        point_clip_mode=None,
        is_directed=False,
    ):
        super(FlankCornersDetRoiTransform, self).__init__()
        self._roi_ts = AffineMatFromROIBoxGenerator(
            target_wh=target_wh,
            scale_range=img_scale_range,
            min_sample_num=min_sample_num,
            max_sample_num=max_sample_num,
            min_valid_edge=min_edge_size,
            min_valid_area=min_valid_area,
            center_aligned=center_aligned,
            rand_scale_range=roi_scale_range,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=rand_aspect_ratio,
            rand_rotation_angle=rand_rotation_angle,
            flip_prob=flip_prob,
            rand_sampling_bbox=rand_sampling_bbox,
        )
        self._img_ts = ImageAffineTransform(
            dst_wh=target_wh,
            inter_method=inter_method,
            border_value=0,
            use_pyramid=use_pyramid,
            pyramid_min_step=pyramid_min_step,
            pyramid_max_step=pyramid_max_step,
            pixel_center_aligned=pixel_center_aligned,
        )
        self.min_valid_area = (min_valid_area,)
        self.min_valid_clip_area_ratio = min_valid_clip_area_ratio
        self._resize_wh = resize_wh
        self._keep_aspect_ratio = keep_aspect_ratio
        self.min_flank_width = min_flank_width
        self.point_clip_mode = point_clip_mode
        self._bbox_ts = LabelAffineTransform(label_type="box")
        self._kps_ts = Point2DAffineTransform()
        self.is_directed = is_directed

    def __call__(self, data):
        img = data["img"]
        ret = data["anno"]
        orgin_wh = img.shape[:2][::-1]
        bboxes = np.asarray(ret["bboxes"], dtype=np.float32)
        # (num_instance,box_dimension)
        bboxes = _clip_bboxes(
            bboxes.reshape(-1, bboxes.shape[-1]),
            (0, 0, orgin_wh[0], orgin_wh[1]),
            True,
        )
        keypoints = np.asarray(ret["keypoints"], dtype=np.float32)
        # (num_instance,num_keypoints,point_dimension)
        keypoints = keypoints.reshape(len(bboxes), -1, keypoints.shape[-1])
        # (num_instacne,)
        flank_classes = np.asarray(ret["flank_classes"], dtype=np.float32)
        # list with shape:
        # (num_instance,num_interpolation,num_sample_points,poitn_dimension)
        inp_points = ret.get("interpolation_points", [])
        inp_indices = ret.get("interpolation_indices", [])

        # self._debug_vis(img, bboxes, keypoints, flank_clses=flank_classes)

        rois = copy.deepcopy(bboxes)  # [n,4]
        if self._keep_aspect_ratio and self._resize_wh:
            resize_wh_ratio = float(self._resize_wh[0]) / float(
                self._resize_wh[1]
            )
            orgin_wh_ratio = float(orgin_wh[0]) / float(orgin_wh[1])
            affine = np.array([[1.0, 0, 0], [0, 1.0, 0]])

            if resize_wh_ratio > orgin_wh_ratio:
                new_wh = (int(orgin_wh[1] * resize_wh_ratio), orgin_wh[1])
                img = cv2.warpAffine(img, affine, new_wh, 0)
            elif resize_wh_ratio < orgin_wh_ratio:
                new_wh = (orgin_wh[0], int(orgin_wh[0] / resize_wh_ratio))
                img = cv2.warpAffine(img, affine, new_wh, 0)

        if self._resize_wh is None:
            img_wh = img.shape[:2][::-1]
            affine_mat = AffineMat2DGenerator.identity()
        else:
            img_wh = self._resize_wh
            affine_mat = resize_affine_mat(
                img.shape[:2][::-1], self._resize_wh
            )
            rois = self._bbox_ts(rois, affine_mat, flip=False)

        for affine_aug_param in self._roi_ts(rois, img_wh):
            cur_affine_mat = AffineMat2DGenerator.stack_affine_transform(
                affine_mat, affine_aug_param.mat
            )[:2]
            affine_aug_param = AffineAugMat(
                mat=cur_affine_mat,
                flipped=affine_aug_param.flipped,
            )

            ts_img = self._img_ts(img, affine_aug_param.mat)
            ts_img_wh = ts_img.shape[:2][::-1]

            (
                ts_bboxes,
                ts_points,
                cp_points_cls,
                ts_inp_points,
            ) = self.transform_rois_and_kps(
                affine_aug_param,
                bboxes,
                keypoints,
                flank_classes,
                inp_points,
                inp_indices,
                (0, 0, ts_img_wh[0], ts_img_wh[1]),
                self.point_clip_mode,
            )
            ts_points = np.concatenate([ts_points, cp_points_cls], axis=-1)
            tmp_shape = list(ts_bboxes.shape)
            tmp_shape[-1] = 1
            ts_bboxes_cls = np.ones(tmp_shape, dtype=ts_bboxes.dtype)
            ts_bboxes = np.concatenate([ts_bboxes, ts_bboxes_cls], axis=-1)

            # self._debug_vis(ts_img, ts_bboxes, ts_points, ts_inp_points)

            return {"img": ts_img, "gt_boxes": ts_bboxes, "points": ts_points}

    def transform_rois_and_kps(
        self,
        affine_aug_param: AffineAugMat,
        bboxes: np.ndarray,
        keypoints: np.ndarray,
        flank_classes: np.ndarray,
        inp_points: list,
        inp_indices: list,
        img_roi: Tuple[float, float, float, float] = None,
        point_clip_mode: Optional[str] = None,
    ):
        # (num_instance,box_dimension)
        ts_bboxes = self._bbox_ts(
            bboxes, affine_aug_param.mat, flip=affine_aug_param.flipped
        )
        # (num_instance,num_keypoints,point_dimension)
        ts_keypoints = self._kps_ts(keypoints, affine_aug_param.mat)
        ts_keypoints = ts_keypoints.reshape(keypoints.shape)
        # (num_instance,num_interpolation,num_sample_points,poitn_dimension)
        ts_inp_points = []
        for instance_points in inp_points:
            # (num_interpolation,num_sample_points,poitn_dimension)
            instance_points = np.asarray(instance_points, dtype=np.float64)
            instance_points = self._kps_ts(
                instance_points, affine_aug_param.mat
            )
            ts_inp_points.append(instance_points)

        # clip bboxes
        cp_bboxes = _clip_bboxes(ts_bboxes, img_roi, True)
        cp_bboxes = np.asarray(cp_bboxes, dtype=np.float32)
        cp_bboxes = cp_bboxes.reshape(-1, cp_bboxes.shape[-1])
        bboxes_mask = self._filter_bboxes(ts_bboxes, cp_bboxes)

        # clip keypoints with interpolation
        cp_keypoints, cp_points_cls = _clip_points(
            ts_keypoints, ts_inp_points, inp_indices, img_roi, point_clip_mode
        )
        cp_keypoints = np.asarray(cp_keypoints, dtype=np.float32)
        cp_keypoints = cp_keypoints.reshape(
            cp_bboxes.shape[0], -1, cp_keypoints.shape[-1]
        )
        cp_points_cls = np.asarray(cp_points_cls, dtype=np.float32)
        cp_points_cls = cp_points_cls.reshape(
            cp_points_cls.shape[0], -1, cp_points_cls.shape[-1]
        )
        keypoints_mask = self._filter_keypoints(
            ts_keypoints,
            cp_keypoints,
            cp_points_cls,
            inp_indices,
            flank_classes,
        )
        cp_points_cls[flank_classes == 0] = 0

        mask = np.logical_and(bboxes_mask, keypoints_mask)

        cp_bboxes = cp_bboxes[mask]
        cp_keypoints = cp_keypoints[mask]
        cp_points_cls = cp_points_cls[mask]
        ts_inp_points = [
            inps for valid, inps in zip(mask, ts_inp_points) if valid
        ]

        if affine_aug_param.flipped and not self.is_directed:
            left_pts = cp_keypoints[..., 0:1, :]
            right_pts = cp_keypoints[..., 1:2, :]
            cp_keypoints = np.concatenate([right_pts, left_pts], axis=1)

        return cp_bboxes, cp_keypoints, cp_points_cls, ts_inp_points

    def _filter_bboxes(
        self,
        org_bboxes,
        cp_bboxes,
    ):
        org_wh = org_bboxes[..., 2:4] - org_bboxes[..., 0:2]
        org_wh = np.maximum(org_wh, 0)
        org_area = org_wh[..., 0] * org_wh[..., 1]

        cp_wh = cp_bboxes[..., 2:4] - cp_bboxes[..., 0:2]
        cp_wh = np.maximum(cp_wh, 0)
        cp_area = cp_wh[..., 0] * cp_wh[..., 1]

        area_ratio = np.where(
            org_area > 0,
            cp_area / org_area,
            np.zeros_like(org_area),
        )

        mask = np.logical_and(
            area_ratio > self.min_valid_clip_area_ratio,
            cp_area > self.min_valid_area,
        )

        return mask

    def _filter_keypoints(
        self,
        org_keypoints,
        cp_keypoints,
        cp_points_cls,
        inp_indices,
        flank_classes,
    ):
        inp_indices = np.asarray(inp_indices, dtype=np.int64).reshape(-1, 2)
        if len(inp_indices) != 1:
            raise NotImplementedError("Just support interpolate with one edge")

        mask = np.full([len(org_keypoints)], True)
        for idx, keypoints in enumerate(cp_keypoints):
            width = 0
            for one_idx, other_idx in inp_indices:
                one_pt = keypoints[one_idx]
                other_pt = keypoints[other_idx]
                width = abs(other_pt[0] - one_pt[0])
            mask[idx] = width > self.min_flank_width

        num_inst = len(cp_points_cls)
        # 对于点clip之后，只保留可以正确clip的点
        cls_mask = np.all(cp_points_cls.reshape(num_inst, -1) > 0, axis=-1)
        mask = np.logical_and(mask, cls_mask)

        # flank negative
        mask = np.logical_or(mask, flank_classes.flatten() == 0)

        return mask

    def _debug_vis(
        self,
        img,
        bboxes,
        keypoints,
        inp_points=None,
        flank_clses=None,
        outfile="output/debug.jpg",
    ):
        _red = (0, 0, 255)
        _green = (0, 255, 0)
        _sky_blue = (235, 206, 135)
        _yellow = (0, 255, 255)
        _white = (250, 250, 250)

        img = cv2.cvtColor(copy.deepcopy(img), cv2.COLOR_RGB2BGR)
        if flank_clses is None:
            flank_clses = keypoints[:, :, -1]
            flank_clses = np.all(flank_clses, axis=-1)

        for bbox, kps, cls in zip(bboxes, keypoints, flank_clses):
            x1, y1, x2, y2 = map(int, bbox[0:4])
            if cls == 0:
                # negative: just render the bbox
                img = cv2.rectangle(img, (x1, y1), (x2, y2), _yellow, 2)
                continue

            img = cv2.rectangle(img, (x1, y1), (x2, y2), _sky_blue, 2)
            for idx, point in enumerate(kps):
                x, y = map(int, point[0:2])
                color = _red if idx % 2 == 0 else _white
                img = cv2.circle(img, (x, y), 6, color, 6)

        if inp_points is not None:
            for cls, inps in zip(flank_clses, inp_points):
                if cls == 0:
                    continue
                inps = np.asarray(inps)
                inps = inps.reshape(-1, inps.shape[-1])
                for point in inps:
                    x, y = map(int, point[0:2])
                    img = cv2.circle(img, (x, y), 2, _green, 2)

        outdir = os.path.dirname(outfile)
        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(outfile, img)


def _clip_bboxes(bboxes: np.ndarray, img_roi: tuple, copy_src=True):
    if copy_src:
        bboxes = copy.deepcopy(bboxes)
    x1, y1, x2, y2 = img_roi
    bboxes[..., 0:4:2] = np.clip(bboxes[..., 0:4:2], x1, x2)
    bboxes[..., 1:4:2] = np.clip(bboxes[..., 1:4:2], y1, y2)

    return bboxes


def _clip_points(
    points: np.ndarray,
    inp_points: list,
    inp_indices: list,
    img_roi: tuple,
    clip_mode: Optional[str] = None,
    padding_value: Optional[float] = 0,
):
    """
    Clip points.

    Args:
        points (np.ndarray):
            (num_instance,box_dimension)
        inp_points (list):
            (num_instance,num_interpolation,num_sample_points,poitn_dimension)
        inp_indices (list):
            (num_interpolation,2)
        img_roi (tuple):
            (4)
        clip_mode (Optional[str], optional):
        padding_value (Optional[float]):

    """
    inp_indices = np.asarray(inp_indices, dtype=np.int64).reshape(-1, 2)
    if len(inp_indices) != 1:
        raise NotImplementedError("Just support interpolate with one edge")

    cp_points = copy.deepcopy(points)
    # 0: negative; 1: origin; 2: clipped
    # (num_instance,num_keypoints)
    points_cls = np.full(points.shape[:-1], 0, np.float32)
    for i, (one_idx, other_idx) in enumerate(inp_indices):
        for idx, inps in enumerate(inp_points):
            one_pt = points[idx][one_idx]
            other_pt = points[idx][other_idx]
            one_pt, one_clip = _clip_point(one_pt, img_roi, inps[i], clip_mode)
            if one_pt is None:
                cp_points[idx][one_idx] = padding_value
                points_cls[idx][one_idx] = 0
            else:
                cp_points[idx][one_idx] = one_pt
                points_cls[idx][one_idx] = 2 if one_clip else 1

            other_pt, other_clip = _clip_point(
                other_pt, img_roi, inps[i], clip_mode
            )
            if other_pt is None:
                cp_points[idx][other_idx] = padding_value
                points_cls[idx][other_idx] = 0
            else:
                cp_points[idx][other_idx] = other_pt
                points_cls[idx][other_idx] = 2 if other_clip else 1

    return cp_points, points_cls[..., None]


def _clip_point(
    point,
    img_roi: tuple,
    inp_points: list,
    clip_mode: Optional[str] = None,
):
    """
    Clip point.

    Args:
        point:
        img_roi:
        inp_points:
        clip_mode: str
            value in ['vertical', 'horizon', 'inside']
        copy_src: bool

    """
    if clip_mode is None:
        clip_mode = "inside"
    min_x, min_y, max_x, max_y = img_roi
    x, y = point[0:2]
    is_clipped = False
    is_inside = (min_x <= x <= max_x) and (min_y <= y <= max_y)
    if clip_mode.lower() == "vertical":
        if not is_inside:
            is_clipped = True
            inp_points = sorted(inp_points, key=lambda val: val[1])
            if y > max_y:
                # clip the bottom
                clip_y = max_y
            else:
                # clip the top
                clip_y = min_y
            up_point = None
            down_point = None
            find_success = False
            for idx in range(len(inp_points) - 1):
                up_point = inp_points[idx]
                down_point = inp_points[idx + 1]
                if up_point[1] <= clip_y <= down_point[1]:
                    find_success = True
                    break
            if find_success:
                ratio = (clip_y - up_point[1]) / (down_point[1] - up_point[1])
                clip_x = up_point[0] + (down_point[1] - up_point[1]) * ratio
                point = [clip_x, clip_y]
            else:
                point = None
    elif clip_mode.lower() == "horizon":
        if not is_inside:
            is_clipped = True
            inp_points = sorted(inp_points, key=lambda val: val[0])
            if x > max_x:
                # clip the right
                clip_x = max_x
            else:
                # clip the left
                clip_x = min_x
            left_point = None
            right_point = None
            find_success = False
            for idx in range(len(inp_points) - 1):
                left_point = inp_points[idx]
                right_point = inp_points[idx + 1]
                if left_point[0] <= clip_x <= right_point[0]:
                    find_success = True
                    break
            if find_success:
                ratio = (clip_x - left_point[0]) / (
                    right_point[0] - left_point[0]
                )
                clip_y = (
                    left_point[1] + (right_point[1] - left_point[1]) * ratio
                )
                point = [clip_x, clip_y]
            else:
                point = None
    elif clip_mode.lower() == "inside":
        if not is_inside:
            is_clipped = True
            inp_points = sorted(inp_points, key=lambda val: val[0])
            tmpt, _ = _clip_point(point, img_roi, inp_points, "horizon")
            if tmpt is None or not _point_in_roi(tmpt, img_roi):
                tmpt, _ = _clip_point(point, img_roi, inp_points, "vertical")
            point = tmpt
    else:
        raise NotImplementedError(f"Unsupport clip mode: {clip_mode}")

    return point, is_clipped


def _point_in_roi(point, roi):
    min_x, min_y, max_x, max_y = roi
    x, y = point[0:2]
    return (min_x <= x <= max_x) and (min_y <= y <= max_y)


@OBJECT_REGISTRY.register
class PadFlankCornersData(object):
    def __init__(
        self, target_wh, max_gt_boxes_num=100, max_ig_regions_num=100
    ):
        self.target_wh = target_wh
        self.max_gt_boxes_num = max_gt_boxes_num
        self.max_ig_regions_num = max_ig_regions_num

    def __call__(self, data):
        img = data["img"]
        bboxes = data["gt_boxes"]
        points = data["points"]
        ig_regions = None
        if len(img.shape) == 3:
            assert img.shape[2] in [1, 3], "img should with shape HWC"

        # pad image
        pad_shape = list(img.shape)
        pad_shape[0] = self.target_wh[1]
        pad_shape[1] = self.target_wh[0]
        im_hw = np.array(img.shape[:2]).reshape((2,)).astype(np.float32)
        img = _pad_array(img, pad_shape, "img")

        # pad bboxes
        pad_shape = list(bboxes.shape)
        pad_shape[0] = self.max_gt_boxes_num
        bboxes_num = np.array(bboxes.shape[0]).reshape((1,)).astype(np.float32)
        bboxes = _pad_array(bboxes, pad_shape, "bboxes")

        # pad points
        pad_shape = list(points.shape)
        pad_shape[0] = self.max_gt_boxes_num
        points_num = np.array(points.shape[0]).reshape((1,)).astype(np.float32)
        points = _pad_array(points, pad_shape, "points")

        # pad ignore regions
        if ig_regions is None:
            ig_regions = np.zeros([0, 5], dtype=np.float32)
        pad_shape = list(ig_regions.shape)
        pad_shape[0] = self.max_ig_regions_num
        ig_regions_num = (
            np.array(ig_regions.shape[0]).reshape((1,)).astype(np.float32)
        )
        ig_regions = _pad_array(ig_regions, pad_shape, "ig_regions")

        return {
            "img": img.transpose(2, 0, 1),
            "im_hw": im_hw,
            "gt_boxes": bboxes,
            "gt_boxes_num": bboxes_num,
            "gt_flanks": points,
            "gt_flanks_num": points_num,
            "ig_regions": ig_regions,
            "ig_regions_num": ig_regions_num,
        }
