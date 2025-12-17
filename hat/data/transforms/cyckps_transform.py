# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict, List, Union

import cv2
import numpy as np
import torch

from hat.data.transforms.roi_crop import CropParm, RoiTransformer
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = [
    "CysKpsLabelTransform",
    "ReshapeBatchSize",
    "SimpleCropForEval",
]


def filter_bboxes(data, filter_key, keep_value):
    gt_filter = data[filter_key]
    keep = np.where(abs(gt_filter - keep_value) < 1e-5)[0]
    assert len(keep) > 0
    for k, v in data.items():
        if k in [
            "img",
        ]:
            continue
        if isinstance(v, np.ndarray):
            data[k] = v[keep]
    return data


def sample_inds(num_ori, num_dst):
    assert num_ori > 0
    if num_ori == num_dst:
        all_inds = range(num_ori)
    elif num_ori > num_dst:
        all_inds = np.random.choice(
            range(num_ori), size=num_dst, replace=False
        )
    else:
        all_inds = np.array(range(num_ori))
        while all_inds.shape[0] < num_dst:
            ex_size = np.minimum(num_ori, num_dst - all_inds.shape[0])
            ex_inds = np.random.choice(
                range(num_ori), size=ex_size, replace=False
            )
            all_inds = np.append(all_inds, ex_inds)
    return all_inds


def kps_compute_label_and_pos_offset(
    bbox: np.ndarray,
    gt_kps: np.ndarray,
    kps_width: int,
    kps_height: int,
    kps_loss_type: str,
    keep_outside: bool,
    keep_invis: bool,
    kps_pos_distance_x: float,
    kps_pos_distance_y: float,
    kps_feat_stride: int,
    keep_outside_as_negative: bool = False,
    kps_in_roi: bool = True,
    scale_by_stride: bool = True,
    kps_gauss_sigma: Union[int, float] = None,
) -> Dict:
    """
    Generate kps label.

    Args:
        bbox: the input box
        gt_kps: the input kps
        kps_height: the target height size of kps label
        kps_width: the target width size of kps label
        kps_loss_type: the loss type must have one_hot or pixel,
            ont_hot means generate, one-dim vector label,
            pixel means generate two-dims label
        keep_outside: true, keep the outside point to detect; otherwise not
        keep_outside_as_negative: true, keep the outside point to detect
            and set it as negative point; otherwise not detect it
        kps_pos_distance_x: the radius_x of heatmap in input image
        kps_pos_distance_y: the radius_y of heatmap in input image
        kps_feat_stride: the stride is equal to
            input image shape/target label shape
        kps_gauss_sigma: if use gauss heatmap,
            need this param to generate gauss radius
        scale_by_stride: true, scale the kps to feature map
    """

    num_kps = gt_kps.shape[0]
    feat_width = kps_width
    feat_height = kps_height
    kps_loss_type = kps_loss_type  # pixel_reg_fixed_smooth_L1

    if "one_hot" in kps_loss_type:
        kps_label = np.full((num_kps,), fill_value=-1, dtype=np.float32)
    else:
        kps_label = np.full(
            (num_kps, feat_height * feat_width),
            fill_value=-1,
            dtype=np.float32,
        )

    kps_label_weight = np.zeros(
        (num_kps, feat_height * feat_width), dtype=np.float32
    )
    kps_pos_offset = np.zeros(
        (num_kps, 2, feat_height * feat_width), dtype=np.float32
    )
    kps_pos_offset_weight = np.zeros(
        (num_kps, 2, feat_height * feat_width), dtype=np.float32
    )

    x = gt_kps[:, 0] - bbox[0]
    y = gt_kps[:, 1] - bbox[1]
    if scale_by_stride:
        scale_x = feat_width / (bbox[2] - bbox[0] + 1)
        scale_y = feat_height / (bbox[3] - bbox[1] + 1)
        x *= scale_x
        y *= scale_y
    x_int = np.floor(x)
    y_int = np.floor(y)

    vis = np.logical_or(gt_kps[:, 2] == 2, gt_kps[:, 2] == 4)
    if keep_outside or keep_outside_as_negative:
        vis = np.logical_or(gt_kps[:, 2] == 0, vis)
    if keep_invis:
        vis = np.logical_or(gt_kps[:, 2] == 1, vis)

    # # for kps in roi
    if kps_in_roi:
        valid = np.logical_and(
            np.logical_and(x_int >= 0, y_int >= 0),
            np.logical_and(x_int < feat_width, y_int < feat_height),
        )

        valid = np.logical_and(valid, vis)
    else:
        valid = vis

    keep = np.where(valid == 1)[0]
    if len(keep) > 0:
        if "one_hot" in kps_loss_type:
            x_offset = x - x_int
            y_offset = y - y_int
            pos = y_int * feat_width + x_int
            keep_pos = pos[keep].astype(np.int32)
            kps_label[keep] = keep_pos
            kps_pos_offset[keep, 0, keep_pos] = x_offset[keep]
            kps_pos_offset[keep, 1, keep_pos] = y_offset[keep]
            kps_pos_offset_weight[keep, 0, keep_pos] = 1
            kps_pos_offset_weight[keep, 1, keep_pos] = 1
            assert kps_pos_offset.min() >= 0 and kps_pos_offset.max() <= 1
        elif "pixel" in kps_loss_type:
            kps_label[:, :] = 0
            kps_label_weight[:, :] = 1
            ignore_kps = np.where(gt_kps[:, 2] == 3)[0]
            kps_label[ignore_kps, :] = -1
            kps_label_weight[ignore_kps, :] = 0

            feat_x_int = np.arange(0, feat_width)
            feat_y_int = np.arange(0, feat_height)
            feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
            feat_x_int = feat_x_int.reshape((-1,))
            feat_y_int = feat_y_int.reshape((-1,))
            kps_pos_distance_x = kps_pos_distance_x / kps_feat_stride
            kps_pos_distance_y = kps_pos_distance_y / kps_feat_stride

            for keep_i in keep:
                x_offset = (x[keep_i] - feat_x_int) / kps_pos_distance_x
                y_offset = (y[keep_i] - feat_y_int) / kps_pos_distance_y
                if keep_i in (17, 18):
                    # for point outside bbox, keep kps_label inside roi,
                    # kps_pos_offset from kps_label to bbox outside
                    x[keep_i] = min(x[keep_i], feat_width - 1)
                    x[keep_i] = max(x[keep_i], 0)
                    y[keep_i] = min(y[keep_i], feat_height - 1)
                    y[keep_i] = max(y[keep_i], 0)
                    x_offset_dis = (
                        x[keep_i] - feat_x_int
                    ) / kps_pos_distance_x  # noqa
                    y_offset_dis = (
                        y[keep_i] - feat_y_int
                    ) / kps_pos_distance_y  # noqa
                    dis = x_offset_dis ** 2 + y_offset_dis ** 2
                else:
                    dis = x_offset ** 2 + y_offset ** 2

                keep_pos = np.where((dis <= 1) & (dis >= 0))[0]
                if "cls" in kps_loss_type or "reg_fixed" in kps_loss_type:
                    kps_label[keep_i, keep_pos] = 1
                else:
                    sigma = kps_gauss_sigma
                    kps_label[keep_i, keep_pos] = np.exp(
                        dis[keep_pos] / (-2.0 * sigma * sigma)
                    )

                kps_pos_offset[keep_i, 0, keep_pos] = x_offset[keep_pos]
                kps_pos_offset[keep_i, 1, keep_pos] = y_offset[keep_pos]
                kps_pos_offset_weight[keep_i, 0, keep_pos] = 1
                kps_pos_offset_weight[keep_i, 1, keep_pos] = 1
            # add outside points
            if keep_outside_as_negative:
                outside_kps = np.where(gt_kps[:, 2] == 0)[0]
                kps_label[outside_kps, :] = 0
                kps_label_weight[outside_kps, :] = 1
                kps_pos_offset_weight[outside_kps, 0, :] = 0
                kps_pos_offset_weight[outside_kps, 1, :] = 0
        else:
            raise ValueError("unknown kps loss type {}".format(kps_loss_type))

    kps_pos_offset = kps_pos_offset.reshape((num_kps * 2, -1))
    kps_pos_offset_weight = kps_pos_offset_weight.reshape((num_kps * 2, -1))
    res = {
        "kps_label": [kps_label, kps_label_weight],
        "kps_pos_offset": [kps_pos_offset, kps_pos_offset_weight],
    }

    return res


def limit_roi(roi: List, im_height: int, im_width: int) -> List:
    """Limit roi to image border.

    Args:
        roi: bbox.
        im_height: image height.
        im_width: image width.
    """
    left = max(0, roi[0])
    top = max(0, roi[1])
    right = min(im_width - 1, roi[2])
    bottom = min(im_height - 1, roi[3])
    return [left, top, right, bottom]


def crop_image(im: np.ndarray, roi: List, roi_shape: List) -> np.ndarray:
    """Crop im patches in roi to roi_shape, padding when roi exceed."""
    roi = [int(v) for v in roi]
    cut_roi = limit_roi(roi, im.shape[0], im.shape[1])

    if len(im.shape) == 3:
        im_roi = im[
            cut_roi[1] : cut_roi[3] + 1, cut_roi[0] : cut_roi[2] + 1, :
        ]
    else:
        im_roi = im[cut_roi[1] : cut_roi[3] + 1, cut_roi[0] : cut_roi[2] + 1]

    im_roi = cv2.copyMakeBorder(
        im_roi,
        cut_roi[1] - roi[1],
        roi[3] - cut_roi[3],
        cut_roi[0] - roi[0],
        roi[2] - cut_roi[2],
        cv2.BORDER_CONSTANT,
    )  # pad black wrt the rgb space

    roi_shape = (roi_shape[-1], roi_shape[-2])
    im_roi = cv2.resize(im_roi, roi_shape, interpolation=cv2.INTER_LINEAR)

    im_roi = im_roi.astype(np.float32)
    return im_roi


@OBJECT_REGISTRY.register
class CysKpsLabelTransform(object):
    """Label transform for cyclist wheel kps training.

    Args:
        roi_batch_size: the number of select boxes
        crop_params: params of crop
        kps_height: the target height size of kps label
        kps_width: the target width size of kps label
        kps_loss_type: the loss type must have one_hot or pixel,
            ont_hot means generate, one-dim vector label,
            pixel means generate two-dims label
        keep_outside: true, keep the outside point to detect; otherwise not
        keep_invis: true, keep the invisible point to detect; otherwise not
        kps_pos_distance_x: the radius_x of heatmap in input image
        kps_pos_distance_y: the radius_y of heatmap in input image
        kps_feat_stride: the stride is equal to
            input image shape/target label shape
        kps_input_height: input image height
        kps_input_width: input image width
    """

    def __init__(
        self,
        roi_batch_size: int,
        crop_params: Dict,
        kps_height: int = 8,
        kps_width: int = 8,
        kps_loss_type: str = "pixel_reg_fixed_smooth_L1",
        keep_outside: bool = True,
        keep_invis: bool = True,
        kps_pos_distance_x: float = 12.5,
        kps_pos_distance_y: float = 12.5,
        kps_feat_stride: float = 16,
        kps_input_height: int = 128,
        kps_input_width: int = 128,
    ):
        if isinstance(crop_params, dict):
            crop_params = CropParm.from_dict(crop_params)
        self.roi_batch_size = roi_batch_size
        self.crop_params = crop_params
        self.kps_height = int(kps_height)
        self.kps_width = int(kps_width)
        self.kps_loss_type = kps_loss_type
        self.keep_outside = keep_outside
        self.keep_invis = keep_invis
        self.kps_pos_distance_x = kps_pos_distance_x
        self.kps_pos_distance_y = kps_pos_distance_y
        self.kps_feat_stride = kps_feat_stride

        self.kps_input_height = kps_input_height
        self.kps_input_width = kps_input_width

    def choose_and_change_box(self, boxes, kps):
        # batch roi
        all_inds = sample_inds(boxes.shape[0], self.roi_batch_size)
        all_kps = kps[all_inds, :]
        all_boxes = boxes[all_inds, :]
        # norm scale crop
        all_crop_scale = np.ones((self.roi_batch_size, 1), dtype=np.float32)
        for i in range(self.roi_batch_size):
            roi_box = boxes[all_inds[i], :]
            all_crop_scale[i, 0] = RoiTransformer.calculate_scale(
                self.crop_params, roi_box
            )
        dst_wh = np.array(self.crop_params.output_wh, dtype=np.float32)
        crop_wh = np.hstack(
            [dst_wh[0] / all_crop_scale, dst_wh[1] / all_crop_scale]
        )
        center_roi = np.hstack(
            [
                (all_boxes[:, 0:1] + all_boxes[:, 2:3]) / 2.0,
                (all_boxes[:, 1:2] + all_boxes[:, 3:4]) / 2.0,
            ]
        )
        assert center_roi.shape == crop_wh.shape
        all_normed_boxes = np.zeros((self.roi_batch_size, 4), dtype=np.float32)
        all_normed_boxes[:, 0] = center_roi[:, 0] - crop_wh[:, 0] / 2.0
        all_normed_boxes[:, 1] = center_roi[:, 1] - crop_wh[:, 1] / 2.0
        all_normed_boxes[:, 2] = center_roi[:, 0] + crop_wh[:, 0] / 2.0
        all_normed_boxes[:, 3] = center_roi[:, 1] + crop_wh[:, 1] / 2.0
        return all_normed_boxes, all_kps

    def gen_kps_labels(self, all_kps, all_normed_boxes, num_kps):
        # 逐个roi处理kps label
        kps_label_list = []
        kps_label_weight_list = []
        kps_pos_offset_list = []
        kps_pos_offset_weight_list = []
        all_kps_split = all_kps.reshape((-1, num_kps, 3))
        for i in range(self.roi_batch_size):
            cur_label = kps_compute_label_and_pos_offset(
                bbox=all_normed_boxes[i, :],
                gt_kps=all_kps_split[i, :, :],
                kps_width=self.kps_width,
                kps_height=self.kps_height,
                kps_loss_type=self.kps_loss_type,
                keep_outside=self.keep_outside,
                keep_invis=self.keep_invis,
                kps_pos_distance_x=self.kps_pos_distance_x,
                kps_pos_distance_y=self.kps_pos_distance_y,
                kps_feat_stride=self.kps_feat_stride,
            )
            kps_label, kps_label_weight = cur_label["kps_label"]
            kps_pos_offset, kps_pos_offset_weight = cur_label["kps_pos_offset"]
            kps_label_list.append(kps_label)
            kps_label_weight_list.append(kps_label_weight)
            kps_pos_offset_list.append(kps_pos_offset)
            kps_pos_offset_weight_list.append(kps_pos_offset_weight)

        if "one_hot" in self.kps_loss_type:
            kps_labels = np.hstack(kps_label_list).reshape(
                (self.roi_batch_size, num_kps)
            )
        else:
            kps_labels = np.vstack(kps_label_list).reshape(
                (self.roi_batch_size, num_kps, self.kps_height, self.kps_width)
            )

        kps_labels_weights = np.vstack(kps_label_weight_list).reshape(
            (self.roi_batch_size, num_kps, self.kps_height, self.kps_width)
        )
        kps_pos_offsets = np.vstack(kps_pos_offset_list).reshape(
            (self.roi_batch_size, num_kps * 2, self.kps_height, self.kps_width)
        )
        kps_pos_offset_weights = np.vstack(kps_pos_offset_weight_list).reshape(
            (self.roi_batch_size, num_kps * 2, self.kps_height, self.kps_width)
        )

        return (
            kps_labels,
            kps_labels_weights,
            kps_pos_offsets,
            kps_pos_offset_weights,
        )

    def __call__(self, data):
        data = filter_bboxes(data, "gt_classes", 1)
        boxes = data["boxes"]
        kps = data["keypoints"]
        assert boxes.shape[0] == kps.shape[0]
        assert kps.shape[1] % 3 == 0, "%d vs. 0" % kps.shape[1] % 3
        num_kps = int(kps.shape[1] // 3)
        all_normed_boxes, all_kps = self.choose_and_change_box(boxes, kps)
        (
            kps_labels,
            kps_labels_weights,
            kps_pos_offsets,
            kps_pos_offset_weights,
        ) = self.gen_kps_labels(all_kps, all_normed_boxes, num_kps)

        saved_data = {}
        saved_data["kps_cls_label"] = kps_labels  # 4,2,8,8
        saved_data["kps_cls_label_weight"] = kps_labels_weights  # 4,2,8,8
        saved_data["kps_pos_offset"] = kps_pos_offsets  # 4,4,8,8
        saved_data["kps_pos_offset_weight"] = kps_pos_offset_weights  # 4,4,8,8

        img_list = []
        for i in range(self.roi_batch_size):
            tmp_roi_img = crop_image(
                data["img"],
                all_normed_boxes[i, :].tolist(),
                [self.kps_input_height, self.kps_input_width],
            )
            img_list.append(tmp_roi_img)
        img_bs = np.stack(img_list, 0).astype(np.float32)
        saved_data["img"] = torch.from_numpy(img_bs).permute(
            0, 3, 1, 2
        )  # 4,3,128,128

        return saved_data


@OBJECT_REGISTRY.register
class ReshapeBatchSize(object):
    """Label transform for cyclist wheel kps training.

    Args:
        reshape_keys_list: list of reshape variable key

    """

    def __init__(self, reshape_keys_list: List = None):
        self.reshape_keys_list = reshape_keys_list

    def __call__(self, data):
        for k, v in data.items():
            if self.reshape_keys_list is None or k in self.reshape_keys_list:
                new_shape = [v.shape[0] * v.shape[1]] + list(v.shape[2:])
                data[k] = v.reshape(new_shape)
        return data


@OBJECT_REGISTRY.register
class SimpleCropForEval(object):
    """Crop image for eval.

    Args:
        norm_method:
            Normalization method, supported norm_method:
                ['resize', 'width', 'height', 'diagonal',
                'mean_width_height', 'sqrt_width_height',
                'max_width_height', 'max_2width_height']
        norm_len: Normalization length.
        output_wh: output image size.
        kps_input_height: input image height
        kps_input_width: input image width
    """

    def __init__(
        self,
        norm_method: str,
        norm_len: int,
        output_wh: List,
        kps_input_height: int = 128,
        kps_input_width: int = 128,
    ):
        self.norm_method = norm_method
        self.norm_len = norm_len
        self.output_wh = output_wh
        self.kps_input_height = kps_input_height
        self.kps_input_width = kps_input_width

    def __call__(self, data):
        roi = data["bbox"]
        center_roi = [(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0]
        inst_norm_dist = RoiTransformer.norm_method2norm_dist_func(
            norm_method=self.norm_method
        )(roi.tolist())
        scale = self.norm_len / inst_norm_dist
        crop_wh = [self.output_wh[0] / scale, self.output_wh[1] / scale]
        normed_box = [
            center_roi[0] - crop_wh[0] / 2.0,
            center_roi[1] - crop_wh[1] / 2.0,
            center_roi[0] + crop_wh[0] / 2.0,
            center_roi[1] + crop_wh[1] / 2.0,
        ]
        roi_img = crop_image(
            data["img"],
            normed_box,
            [self.kps_input_height, self.kps_input_width],
        )
        crop_offset = [
            center_roi[0] - crop_wh[0] / 2.0,
            center_roi[1] - crop_wh[1] / 2.0,
        ]
        data["crop_scale"] = np.array([[scale]], np.float32)
        data["crop_offset"] = np.array([crop_offset, crop_offset], np.float32)
        data["normed_box"] = np.array(normed_box, np.float32)
        data["img"] = roi_img
        return data
