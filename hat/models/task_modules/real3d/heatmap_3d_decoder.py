# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from hat.core.data_struct.base_struct import DetBoxes2D3D
from hat.core.eq_focal_length import cal_equivalent_focal_length_uv_mat
from hat.core.undistort_lut import get_undistort_points
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["HeatMap3DDecoder"]


def maxpool_nms(heatmap, k=3):
    maxp = heatmap
    pad = (k - 1) // 2
    maxp = F.max_pool2d(maxp, (k, k), stride=1, padding=(pad, pad))
    keep = (maxp == heatmap).float()
    heatmap = keep * heatmap
    return heatmap


def get_3rd_point(point_a, point_b):
    direct = point_a - point_b
    return point_b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(
    center,
    scale,
    rot,
    output_size,
    shift=np.array([0, 0], dtype=np.float32),  # noqa B008
    inv=0,
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0] = center + scale_tmp * shift
    src[1] = center + src_dir + scale_tmp * shift
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:] = get_3rd_point(src[0], src[1])
    dst[2:] = get_3rd_point(dst[0], dst[1])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    pt = np.array(pt)
    new_pt = np.concatenate([pt, np.ones([pt.shape[0], 1])], axis=1).T
    new_pt = np.dot(t, new_pt).T
    return new_pt[:, :2]


def decoding_2d(
    output,
    meta,
    down_stride,
    topk,
    input_padding,
    undistort_2dcenter,
    shift=np.array([0, 0], dtype=np.float32),  # noqa B008
):
    height, width, num_classes = output["hm"].shape
    for k, v in output.items():
        output[k] = v.reshape(-1, v.shape[-1])
    hm = output.pop("hm")
    hm = hm.reshape(-1)
    argsort_inds = np.argsort(-hm)[:topk]
    class_inds = argsort_inds % num_classes
    scores = hm[argsort_inds]
    argsort_inds = argsort_inds // num_classes

    for k, v in output.items():
        output[k] = v[argsort_inds]

    xs = (argsort_inds % width).astype(np.float32)
    ys = (argsort_inds // width).astype(np.float32)
    center = np.concatenate([xs[:, np.newaxis], ys[:, np.newaxis]], axis=-1)
    if "center_offset" in output:
        center += output.pop("center_offset")
    else:
        center += 0.5

    # trans_matrix: from heatmap to original image.
    # width_offset: from padding heatmap width to original heatmap width.
    width_offset = (input_padding[2] + input_padding[3]) // down_stride
    trans_matrix = get_affine_transform(
        meta["center"],
        meta["ori_img_wh"],
        0,
        (width - width_offset, height),
        inv=1,
        shift=shift,
    )  # noqa

    # wh = affine_transform(output.pop('wh'), trans_matrix)
    wh = output.pop("wh")
    bbox = np.concatenate([center - wh / 2, center + wh / 2], axis=1)
    # convert bbox to input image scale.
    bbox[:, :2] = bbox[:, :2] * down_stride
    bbox[:, 2:] = bbox[:, 2:] * down_stride
    center = affine_transform(center, trans_matrix)
    # if pad left, then subtract left padding.
    center[:, 0] -= input_padding[2] * down_stride
    if undistort_2dcenter:
        # undistort
        center = get_undistort_points(
            center,
            meta["calib"][:3, :3],
            meta["dist_coeffs"],
            meta["ori_img_wh"],
        )  # noqa
    output["class_idx"] = class_inds
    output["score"] = scores
    output["bbox"] = bbox
    # center is at original image scale.
    output["center"] = center

    return output


def convert_pred_rot_to_alpha(rot):
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def convert_pred_rot_to_alpha_simplified(rot):
    alpha = np.arctan2(rot[:, 0], rot[:, 1])
    alpha -= np.pi / 2
    return alpha


def convert_gt_rot_to_alpha(rot):
    idx = rot[:, 0] > rot[:, 1]
    alpha1 = np.arctan2(np.sin(rot[:, 2]), np.cos(rot[:, 2])) + (-0.5 * np.pi)
    alpha2 = np.arctan2(np.sin(rot[:, 3]), np.cos(rot[:, 3])) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def unproject_2d_to_3d(pt_2d, depth, P):
    z = depth - P[2, 3]
    x = (pt_2d[:, 0:1] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
    y = (pt_2d[:, 1:2] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
    pt_3d = np.concatenate([x, y, z], axis=-1)

    return pt_3d


def alpha2rot_y(alpha, x, cx, fx):
    """Get rotation_y by alpha + theta - 180.

    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    rot_y[rot_y > np.pi] -= 2 * np.pi
    rot_y[rot_y < -np.pi] += 2 * np.pi
    return rot_y


def ddd2locrot(center, alpha, dim, depth, calib):
    locations = unproject_2d_to_3d(center, depth, calib)
    locations[:, 1] += dim[:, 0] / 2
    rotation_y = alpha2rot_y(alpha, center[:, 0], calib[0, 2], calib[0, 0])
    return locations, rotation_y


def decoding_3d(
    output,
    meta,
    focal_length_default=None,
    is_gt=False,
    undistort_depth_uv=False,
):
    dim = output["dim"]
    valid_mask = np.any(dim > 0, axis=-1)
    for k, v in output.items():
        output[k] = v[valid_mask]

    calib = meta["calib"]
    if not is_gt:
        pre_rot = output.pop("rot")
        if pre_rot.shape[-1] == 8:
            alpha = convert_pred_rot_to_alpha(pre_rot)
        else:
            alpha = convert_pred_rot_to_alpha_simplified(pre_rot)
    else:
        rot_bin, rot_res = output.pop("rot_bin"), output.pop("rot_res")
        rot = np.concatenate([rot_bin, rot_res], axis=-1)
        alpha = convert_gt_rot_to_alpha(rot)

    if undistort_depth_uv is False:
        if focal_length_default is not None:
            output["dep"] *= calib[0, 0] / focal_length_default
    else:
        eq_fu, eq_fv = cal_equivalent_focal_length_uv_mat(
            int(meta["ori_img_wh"][0]),
            int(meta["ori_img_wh"][1]),
            calib,
            meta["dist_coeffs"],
        )

        dist_ct = (output["bbox"][:, :2] + output["bbox"][:, 2:4]) / 2.0
        dist_ct[:, 0] *= meta["ori_img_wh"][0] / meta["img_wh"][0]
        dist_ct[:, 1] *= meta["ori_img_wh"][1] / meta["img_wh"][1]
        dist_ct[:, 0] = dist_ct[:, 0].clip(
            min=0, max=meta["ori_img_wh"][0] - 1
        )
        dist_ct[:, 1] = dist_ct[:, 1].clip(
            min=0, max=meta["ori_img_wh"][1] - 1
        )
        dist_ct = dist_ct.astype(int)
        dist_ct = dist_ct[:, ::-1].transpose()
        eq_fu = eq_fu[dist_ct[0], dist_ct[1]]
        eq_fv = eq_fv[dist_ct[0], dist_ct[1]]

        if focal_length_default is not None:
            output["dep"][:, 0] *= eq_fu / focal_length_default
            output["dep"][:, 1] *= eq_fv / focal_length_default
        else:
            output["dep"][:, 0] *= eq_fu / calib[0, 0]
            output["dep"][:, 1] *= eq_fv / calib[0, 0]
        output["dep"] = (output["dep"][:, 0] + output["dep"][:, 1]) / 2
        output["dep"] = output["dep"][:, None]

    locations, rotation_y = ddd2locrot(
        output["center"], alpha, output["dim"], output["dep"], calib
    )
    if "loc_offset" in output:
        loc_offset = output.pop("loc_offset")
        if focal_length_default is not None:
            loc_offset *= calib[0, 0] / focal_length_default
        locations[:, :2] += loc_offset
        if pre_rot.shape[-1] != 8:
            rotation_y = alpha2rot_y(
                alpha, locations[:, 0], 0.0, locations[:, 2]
            )

    output["rotation_y"] = rotation_y
    output["location"] = locations
    output["alpha"] = alpha

    return output


def compute_box_3d(dim, location, rotation_y):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(
        3, 1
    )
    return corners_3d.transpose(1, 0)


def bev_nms(dets, nms_iou_thresh):
    # only support one class
    assert isinstance(dets, list)
    # convert det to 3d bev box
    bev_boxes = []
    for det in dets:
        dim = det["dim"]
        loc = det["location"]
        roty = det["rotation_y"]
        cid = det["class_idx"]
        score = det["score"]

        # cid = det.idx
        bev_box = compute_box_3d(dim, loc, roty)[:4, [0, 2]]
        min_x, min_y = np.min(bev_box, axis=0)
        max_x, max_y = np.max(bev_box, axis=0)
        bev_boxes.append((min_x, min_y, max_x, max_y, score, cid))
    bev_boxes = np.array(bev_boxes)
    # nms
    x1 = bev_boxes[:, 0]
    y1 = bev_boxes[:, 1]
    x2 = bev_boxes[:, 2]
    y2 = bev_boxes[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = bev_boxes[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        keep.append(i)
        min_x1 = np.maximum(x1[i], x1[index[1:]])
        min_y1 = np.maximum(y1[i], y1[index[1:]])
        max_x2 = np.minimum(x2[i], x2[index[1:]])
        max_y2 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, max_x2 - min_x1 + 1)
        h = np.maximum(0, max_y2 - min_y1 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= nms_iou_thresh)[0]
        index = index[idx + 1]

    return [dets[i] for i in keep]


def get_bbox3d(
    output, use_bev_nms, nms_iou_thresh, num_classes=None, cls_ids_save=None
):
    # all_classes_dets = [[] for i in range(num_classes)]
    predictions = []
    for i in range(output["class_idx"].shape[0]):
        if (
            cls_ids_save is not None
            and output["class_idx"][i] not in cls_ids_save
        ):
            continue
        det = {k: v[i] for k, v in output.items()}
        predictions.append(det)
    if len(predictions) == 0:
        return None
    if use_bev_nms:
        predictions = _as_list(bev_nms(predictions, nms_iou_thresh))

    rotation_y = torch.stack(
        [torch.tensor(det["rotation_y"]) for det in predictions]
    ).unsqueeze(1)
    bbox = torch.stack([torch.tensor(det["bbox"]) for det in predictions])
    locations = torch.stack(
        [torch.tensor(det["location"]) for det in predictions]
    )
    dim = torch.stack([torch.tensor(det["dim"]) for det in predictions])
    dep = torch.stack([torch.tensor(det["dep"]) for det in predictions])
    score = torch.stack(
        [torch.tensor(det["score"]) for det in predictions]
    ).unsqueeze(1)

    result = torch.cat(
        [
            rotation_y,
            locations,
            dim,
            dep,
            score,
            bbox,
        ],
        dim=1,
    )

    return result


@OBJECT_REGISTRY.register
class HeatMap3DDecoder(nn.Module):
    # heatmap base 3d decoder.
    def __init__(
        self,
        focal_length_default: float,
        scale_wh: Tuple[float, float],
        center,
        down_stride,
        use_bev_nms,
        nms_iou_thresh,
        image_hw,
        undistort_2dcenter=False,
        undistort_depth_uv=False,
        input_padding=(0, 0, 0, 0),  # noqa B008
        topk=100,
        shift=np.array([0, 0]),  # noqa B008
        cls_ids_save=None,
    ):

        super().__init__()

        self.topk = topk
        self.shift = shift
        self.cls_ids_save = cls_ids_save
        self.center = center
        self.use_bev_nms = use_bev_nms
        self.nms_iou_thresh = nms_iou_thresh
        self.down_stride = down_stride
        self.undistort_2dcenter = undistort_2dcenter

        self.undistort_depth_uv = undistort_depth_uv
        self.focal_length_default = focal_length_default
        assert len(input_padding) == 4
        self.input_padding = input_padding
        if image_hw is not None:
            image_hw = np.array(image_hw)
        self.image_hw = image_hw
        affine_mat = torch.tensor(
            [[scale_wh[0], 0, 0], [0, scale_wh[1], 0], [0, 0, 1]]
        )  # noqa

        affine_inv = torch.linalg.pinv(affine_mat)
        self.register_buffer("affine_mat_inv", affine_inv, persistent=False)

    def forward(
        self,
        head_out: Dict[str, List[torch.Tensor]],
        calib: torch.Tensor,
        dist_coeffs: torch.Tensor,
        im_hw: Optional[torch.Tensor] = None,
    ):

        calib = calib.cpu().numpy()
        dist_coeffs = dist_coeffs.cpu().numpy()

        meta = {"center": self.center}

        if self.undistort_depth_uv:
            head_out["dep"] = torch.cat(
                [head_out.pop("dep_u"), head_out.pop("dep_v")],
                dim=1,
            )

        head_out["hm"] = maxpool_nms(head_out["hm"].sigmoid())
        head_out["dep"] = 1.0 / (head_out["dep"].sigmoid() + 1e-6) - 1.0

        # calculate padding info
        hm_l = self.input_padding[2] // self.down_stride
        hm_t = self.input_padding[0] // self.down_stride
        hm_r = self.input_padding[3] // self.down_stride
        hm_b = self.input_padding[1] // self.down_stride

        batch_size = None
        for k, v in head_out.items():
            v_array = v.detach().cpu().numpy().transpose(0, 2, 3, 1)
            v_shape = v_array.shape
            head_out[k] = v_array[
                :, hm_t : v_shape[1] - hm_b, hm_l : v_shape[2] - hm_r
            ]  # noqa
            if batch_size is None:
                batch_size = v_array.shape[0]
            else:
                assert batch_size == v_array.shape[0]

        if im_hw is None:
            assert self.image_hw is not None
            im_hw = [self.image_hw for _ in range(batch_size)]
        else:
            im_hw = im_hw.cpu().numpy()

        det_results = []
        for i in range(batch_size):
            # meta info
            ori_img_wh = (
                int(im_hw[i][1] * self.affine_mat_inv[0, 0]),
                int(im_hw[i][0] * self.affine_mat_inv[1, 1]),
            )
            ori_img_wh = np.array(ori_img_wh)
            meta.update(
                {
                    "calib": calib[i],
                    "dist_coeffs": dist_coeffs[i],
                    "ori_img_wh": ori_img_wh,
                    "img_wh": (im_hw[i][1], im_hw[i][0]),
                }
            )
            output = {k: v[i] for k, v in head_out.items()}
            output = decoding_2d(
                output,
                meta,
                self.down_stride,
                self.topk,
                input_padding=self.input_padding,
                undistort_2dcenter=self.undistort_2dcenter,
                shift=self.shift,
            )

            output = decoding_3d(
                output,
                meta,
                is_gt=False,
                focal_length_default=self.focal_length_default,
                undistort_depth_uv=self.undistort_depth_uv,
            )

            result = get_bbox3d(
                output,
                self.use_bev_nms,
                self.nms_iou_thresh,
                cls_ids_save=self.cls_ids_save,
            )
            if result is not None:
                det_results.append(
                    DetBoxes2D3D(
                        yaw=result[..., 0],
                        x=result[..., 1],
                        y=result[..., 2],
                        z=result[..., 3],
                        h=result[..., 4],
                        w=result[..., 5],
                        l=result[..., 6],
                        scores=result[..., 8],
                        cls_idxs=torch.zeros_like(result[..., 8]),
                        boxes=result[..., 9:],
                    )
                )
            else:
                det_results.append(None)

        return det_results
