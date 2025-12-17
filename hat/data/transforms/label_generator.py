import copy
import math

import cv2
import numpy as np

from hat.core.affine import point_affine_transform
from hat.core.box_utils import xywh_to_x1y1x2y2
from hat.core.eq_focal_length import cal_equivalent_focal_length_uv_mat
from hat.core.heatmap import draw_heatmap, draw_heatmap_min
from hat.core.undistort_lut import get_undistort_points
from hat.core.utils_3d import (
    affine_transform,
    convert_alpha,
    get_affine_transform,
    get_dense_locoffset,
    get_gaussian2D,
    get_reg_map,
)
from hat.utils.package_helper import check_packages_available

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None


__all__ = [
    "roi_heatmap_label_encoding",
    "roi_heatmap_label_encoding_undistort_uv_depth",
    "label_encoding",
    "dense3d_pad_after_label_generator",
]


def roi_heatmap_label_encoding(
    img,
    label,
    meta,
    num_classes,
    classid_map,
    normalize_depth,
    focal_length_default,
    max_gt_boxes_num,
    filtered_name,
    use_bbox2d,
    shift,
    max_depth,
    use_project_bbox2d,
    crop_roi=None,
):

    calib = copy.deepcopy(meta["calib"])
    if crop_roi is not None:
        calib[0, 2] -= crop_roi[0]  # center_u
        calib[1, 2] -= crop_roi[1]  # center_v
    # get trans_mat
    trans_mat = meta["trans_matrix"]

    depth = np.zeros([max_gt_boxes_num, 1], dtype=np.float32)
    proj_2d_bboxes = np.zeros([max_gt_boxes_num, 4], dtype=np.float32)
    ig_regions = np.zeros([max_gt_boxes_num, 4], dtype=np.float32)

    dimensions = np.zeros([max_gt_boxes_num, 3], dtype=np.float32)
    locations = np.zeros([max_gt_boxes_num, 3], dtype=np.float32)
    rotation_y = np.zeros([max_gt_boxes_num, 1], dtype=np.float32)
    location_offsets = np.zeros([max_gt_boxes_num, 2], dtype=np.float32)

    if "image_key" in meta.keys():
        meta["file_name"] = meta["image_key"]

    if "file_name" not in meta or filtered_name not in meta["file_name"]:
        gt_boxes_num = 0
        for ann in label:
            in_camera = ann["in_camera"] if "in_camera" in ann.keys() else ann
            if crop_roi is not None:
                ann = copy.deepcopy(ann)
                ann["bbox"][0] -= crop_roi[0]
                ann["bbox"][1] -= crop_roi[1]
                if "bbox_2d" in ann and ann["bbox_2d"] is not None:
                    ann["bbox_2d"][0] -= crop_roi[0]
                    ann["bbox_2d"][1] -= crop_roi[1]
            # set id
            cls_id = int(classid_map[ann["category_id"]])
            # only vehicle
            if cls_id < 0:
                continue
            if in_camera["depth"] > max_depth:
                continue
            # set box3D
            locs = np.array(in_camera["location"])
            rot_y = np.array(in_camera["rotation_y"])

            if use_bbox2d and "bbox_2d" in ann and ann["bbox_2d"] is not None:
                bbox = xywh_to_x1y1x2y2(ann["bbox_2d"])
                a_bbox = ann["bbox_2d"].copy()
                bbox_cx = a_bbox[0] + a_bbox[2] / 2.0
                bbox_cy = a_bbox[1] + a_bbox[3] / 2.0
                proj_p = get_undistort_points(
                    np.array([[bbox_cx, bbox_cy]]),
                    np.array(calib[:3, :3]),
                    np.array(meta["distCoeffs"]),
                    img_wh=meta["orgin_wh"],
                )
                proj_p = proj_p.reshape(-1, 2)
                proj_p = proj_p.astype(np.int64)
                bbox_cx = proj_p[0, 0]
                bbox_cy = proj_p[0, 1]
                cx, cy = calib[0, 2], calib[1, 2]
                location_offset = np.array(
                    [
                        (bbox_cx - cx) * in_camera["depth"] / calib[0, 0],
                        (bbox_cy - cy) * in_camera["depth"] / calib[1, 1],
                        in_camera["depth"],
                    ]
                ).astype(np.float64)
                location_offset[1] += in_camera["dim"][0] / 2.0
                location_offset = in_camera["location"] - location_offset
                location_offset = location_offset.tolist()
            elif use_project_bbox2d:
                bbox = xywh_to_x1y1x2y2(ann["bbox"])
                location_offset = in_camera["location_offset"]
            else:
                continue
            # origin_ctx = (bbox[0] + bbox[2]) / 2
            # origin_cty = (bbox[1] + bbox[3]) / 2
            bbox[:2] = point_affine_transform(bbox[:2], trans_mat)
            bbox[2:] = point_affine_transform(bbox[2:], trans_mat)

            _wh = bbox[2:] - bbox[:2]
            # bbox = np.hstack([bbox, [cls_id]])
            if np.any(_wh <= 0):
                continue
            proj_2d_bboxes[gt_boxes_num] = bbox

            depth[gt_boxes_num] = (
                np.array(in_camera["depth"])
                * focal_length_default
                / calib[0, 0]
            )
            location_offsets[gt_boxes_num] = (
                np.array(location_offset)[:2]
                * focal_length_default
                / calib[0, 0]
            )

            dimensions[gt_boxes_num] = np.array(in_camera["dim"])
            locations[gt_boxes_num] = locs
            rotation_y[gt_boxes_num] = rot_y
            # alpha_x[gt_boxes_num] = alpha
            gt_boxes_num += 1
            if gt_boxes_num >= max_gt_boxes_num:
                break

    gt_boxes = np.hstack(
        (
            proj_2d_bboxes,  # 4
            location_offsets,  # 2
            depth,  # 1
            dimensions,  # 3
            locations,  # 3
            rotation_y,  # 1
        )
    )
    ret = {
        "trans_mat": np.array(trans_mat, np.float32),
        "calib": np.array(calib, np.float32),
        "gt_boxes": gt_boxes.astype(np.float32),
        "gt_boxes_num": np.array(gt_boxes_num, np.float32),
        "ig_regions_num": np.array(0.0, np.float32),
        "ig_regions": ig_regions.astype(np.float32),
    }
    del label

    return ret


def roi_heatmap_label_encoding_undistort_uv_depth(
    img,
    label,
    meta,
    num_classes,
    classid_map,
    normalize_depth,
    focal_length_default,
    max_gt_boxes_num,
    filtered_name,
    use_bbox2d,
    shift,
    max_depth,
    use_project_bbox2d,
    crop_roi=None,
):

    calib = copy.deepcopy(meta["calib"])
    if crop_roi is not None:
        calib[0, 2] -= crop_roi[0]  # center_u
        calib[1, 2] -= crop_roi[1]  # center_v
    # get trans_mat
    trans_mat = meta["trans_matrix"]

    depth_u = np.zeros([max_gt_boxes_num, 1], dtype=np.float32)
    depth_v = np.zeros([max_gt_boxes_num, 1], dtype=np.float32)
    proj_2d_bboxes = np.zeros([max_gt_boxes_num, 4], dtype=np.float32)
    ig_regions = np.zeros([max_gt_boxes_num, 4], dtype=np.float32)

    dimensions = np.zeros([max_gt_boxes_num, 3], dtype=np.float32)
    locations = np.zeros([max_gt_boxes_num, 3], dtype=np.float32)
    rotation_y = np.zeros([max_gt_boxes_num, 1], dtype=np.float32)
    location_offsets = np.zeros([max_gt_boxes_num, 2], dtype=np.float32)

    # undistort uv
    center, size = meta["center"], meta["size"]
    width, height = meta["img_wh"]
    trans_output = get_affine_transform(
        center, size, 0, [width, height], shift=shift
    )

    eq_fu_mat, eq_fv_mat = cal_equivalent_focal_length_uv_mat(
        size[0], size[1], calib, meta["distCoeffs"]
    )
    resized_eq_fu = cv2.warpAffine(eq_fu_mat, trans_output, (width, height))
    resized_eq_fv = cv2.warpAffine(eq_fv_mat, trans_output, (width, height))
    meta["eq_fu"] = resized_eq_fu
    meta["eq_fv"] = resized_eq_fv

    if "image_key" in meta.keys():
        meta["file_name"] = meta["image_key"]
    if filtered_name not in meta["file_name"]:
        gt_boxes_num = 0
        for _, ann in enumerate(label):
            in_camera = ann["in_camera"] if "in_camera" in ann.keys() else ann
            if crop_roi is not None:
                ann = copy.deepcopy(ann)
                ann["bbox"][0] -= crop_roi[0]
                ann["bbox"][1] -= crop_roi[1]
                if "bbox_2d" in ann and ann["bbox_2d"] is not None:
                    ann["bbox_2d"][0] -= crop_roi[0]
                    ann["bbox_2d"][1] -= crop_roi[1]
            # set id
            cls_id = int(classid_map[ann["category_id"]])
            # only vehicle
            if cls_id < 0:
                continue
            if in_camera["depth"] > max_depth:
                continue
            # set box3D
            locs = np.array(in_camera["location"])
            rot_y = np.array(in_camera["rotation_y"])

            if use_bbox2d and "bbox_2d" in ann and ann["bbox_2d"] is not None:
                bbox = xywh_to_x1y1x2y2(ann["bbox_2d"])
                a_bbox = ann["bbox_2d"].copy()
                bbox_cx = a_bbox[0] + a_bbox[2] / 2.0
                bbox_cy = a_bbox[1] + a_bbox[3] / 2.0
                proj_p = get_undistort_points(
                    np.array([[bbox_cx, bbox_cy]]),
                    np.array(calib[:3, :3]),
                    np.array(meta["distCoeffs"]),
                    img_wh=meta["orgin_wh"],
                )
                proj_p = proj_p.reshape(-1, 2)
                proj_p = proj_p.astype(np.int64)
                bbox_cx = proj_p[0, 0]
                bbox_cy = proj_p[0, 1]
                cx, cy = calib[0, 2], calib[1, 2]
                location_offset = np.array(
                    [
                        (bbox_cx - cx) * in_camera["depth"] / calib[0, 0],
                        (bbox_cy - cy) * in_camera["depth"] / calib[1, 1],
                        in_camera["depth"],
                    ]
                ).astype(np.float64)
                location_offset[1] += in_camera["dim"][0] / 2.0
                location_offset = in_camera["location"] - location_offset
                location_offset = location_offset.tolist()
            elif use_project_bbox2d:
                bbox = xywh_to_x1y1x2y2(ann["bbox"])
                location_offset = in_camera["location_offset"]
            else:
                continue
            origin_ctx = (bbox[0] + bbox[2]) / 2
            origin_cty = (bbox[1] + bbox[3]) / 2
            bbox[:2] = point_affine_transform(bbox[:2], trans_mat)
            bbox[2:] = point_affine_transform(bbox[2:], trans_mat)

            _wh = bbox[2:] - bbox[:2]
            # bbox = np.hstack([bbox, [cls_id]])
            if np.any(_wh <= 0):
                continue
            proj_2d_bboxes[gt_boxes_num] = bbox

            eq_fu = eq_fu_mat[int(origin_cty), int(origin_ctx)]
            eq_fv = eq_fv_mat[int(origin_cty), int(origin_ctx)]
            depth_u[gt_boxes_num] = (
                np.array(in_camera["depth"]) * focal_length_default / eq_fu
            )
            depth_v[gt_boxes_num] = (
                np.array(in_camera["depth"]) * focal_length_default / eq_fv
            )

            location_offsets[gt_boxes_num] = (
                np.array(location_offset)[:2]
                * focal_length_default
                / calib[0, 0]
            )

            dimensions[gt_boxes_num] = np.array(in_camera["dim"])
            locations[gt_boxes_num] = locs
            rotation_y[gt_boxes_num] = rot_y
            # alpha_x[gt_boxes_num] = alpha
            gt_boxes_num += 1
            if gt_boxes_num >= max_gt_boxes_num:
                break

    gt_boxes = np.hstack(
        (
            proj_2d_bboxes,  # 4
            location_offsets,  # 2
            depth_u,  # 1
            depth_v,  # 1
            dimensions,  # 3
            locations,  # 3
            rotation_y,  # 1
        )
    )
    ret = {
        "trans_mat": np.array(trans_mat, np.float32),
        "calib": np.array(calib, np.float32),
        "gt_boxes": gt_boxes.astype(np.float32),
        "gt_boxes_num": np.array(gt_boxes_num, np.float32),
        "ig_regions_num": np.array(0.0, np.float32),
        "ig_regions": ig_regions.astype(np.float32),
    }
    del label

    return ret


def _convert_alpha(alpha, alpha_in_degree):
    return math.radians(alpha + 45) if alpha_in_degree else alpha


def label_encoding(
    label,
    meta,
    num_classes,
    classid_map,
    normalize_depth,
    focal_length_default,
    alpha_in_degree,
    down_stride,
    use_bbox2d,
    enable_ignore_area,
    depth_min_option,
    shift,
    filtered_name,
    min_box_edge,
    max_depth,
    max_objs,
    use_project_bbox2d,
    undistort_2dcenter,
    undistort_depth_uv,
    crop_roi=None,
):

    calib = copy.deepcopy(meta["calib"])
    if crop_roi is not None:
        calib[0, 2] -= crop_roi[0]  # center_u
        calib[1, 2] -= crop_roi[1]  # center_v
    center, size = meta["center"], meta["size"]
    width, height = meta["img_wh"]
    output_width, output_height = [width // down_stride, height // down_stride]
    trans_output = get_affine_transform(
        center, size, 0, [output_width, output_height], shift=shift
    )

    if enable_ignore_area:
        if "ignore_mask" not in meta:
            ignore_mask = np.zeros((output_height, output_width, 1))
        else:
            ignore_mask = meta["ignore_mask"]
            if not isinstance(ignore_mask, np.ndarray):
                if coco_mask is None:
                    check_packages_available("pycocotools")
                ignore_mask = coco_mask.decode(ignore_mask)
            ignore_mask = ignore_mask.astype(np.uint8)
            if crop_roi is not None:
                x1, y1, x2, y2 = crop_roi
                ignore_mask = ignore_mask[y1:y2, x1:x2]
            ignore_mask = cv2.warpAffine(
                ignore_mask,
                trans_output,
                (output_width, output_height),
                flags=cv2.INTER_NEAREST,
            )
            ignore_mask = ignore_mask.astype(np.float32)[:, :, np.newaxis]

    hm = np.zeros((output_height, output_width, num_classes), dtype=np.float32)
    wh = np.zeros((output_height, output_width, 2), dtype=np.float32)
    depth = np.zeros((output_height, output_width), dtype=np.float32)
    dim = np.zeros((output_height, output_width, 3), dtype=np.float32)
    loc_offset = np.zeros((output_height, output_width, 2), dtype=np.float32)
    weight_hm = np.zeros((output_height, output_width), dtype=np.float32)
    weight_hm_min = 10000 * np.ones(
        (output_height, output_width), dtype=np.float32
    )
    point_pos_mask = np.zeros((output_height, output_width), dtype=np.float32)
    # sin cos
    alpha_x = np.zeros((max_objs, 1), dtype=np.float32)
    ind_ = np.zeros((max_objs), dtype=np.int64)
    ind_mask_ = np.zeros((max_objs), dtype=np.float32)
    rot_y_ = np.zeros((max_objs, 1), dtype=np.float32)
    loc_ = np.zeros((max_objs, 3), dtype=np.float32)
    dim_ = np.zeros((max_objs, 3), dtype=np.float32)
    if "image_key" in meta.keys():
        meta["file_name"] = meta["image_key"]
    if filtered_name not in meta["file_name"]:
        ann_idx = -1
        for ann in label:
            in_camera = ann["in_camera"] if "in_camera" in ann.keys() else ann
            if crop_roi is not None:
                ann = copy.deepcopy(ann)
                ann["bbox"][0] -= crop_roi[0]
                ann["bbox"][1] -= crop_roi[1]
                if "bbox_2d" in ann and ann["bbox_2d"] is not None:
                    ann["bbox_2d"][0] -= crop_roi[0]
                    ann["bbox_2d"][1] -= crop_roi[1]
            if use_bbox2d and "bbox_2d" in ann and ann["bbox_2d"] is not None:
                # use image 2d bbox
                bbox = xywh_to_x1y1x2y2(ann["bbox_2d"])
                if undistort_2dcenter:
                    a_bbox = ann["bbox_2d"]
                    bbox_cx = a_bbox[0] + a_bbox[2] / 2.0
                    bbox_cy = a_bbox[1] + a_bbox[3] / 2.0
                    proj_p = get_undistort_points(
                        np.array([[bbox_cx, bbox_cy]]),
                        np.array(calib[:3, :3]),
                        np.array(meta["distCoeffs"]),
                        img_wh=meta["orgin_wh"],
                    )
                    proj_p = proj_p.reshape(-1, 2)
                    proj_p = proj_p.astype(np.int64)
                    bbox_cx = proj_p[0, 0]
                    bbox_cy = proj_p[0, 1]
                    cx, cy = calib[0, 2], calib[1, 2]
                    location_offset = np.array(
                        [
                            (bbox_cx - cx) * in_camera["depth"] / calib[0, 0],
                            (bbox_cy - cy) * in_camera["depth"] / calib[1, 1],
                            in_camera["depth"],
                        ]
                    ).astype(np.float64)
                    location_offset[1] += in_camera["dim"][0] / 2.0
                    location_offset = in_camera["location"] - location_offset
                else:
                    location_offset = in_camera["location_offset_2d"]
                alpha = convert_alpha(in_camera["alpha_2d"], alpha_in_degree)
            elif use_project_bbox2d:
                # use lidar 3d projection 2d bbox
                # this branch is actually not used, deleting is suggested
                bbox = xywh_to_x1y1x2y2(ann["bbox"])
                location_offset = in_camera["location_offset"]
                alpha = convert_alpha(in_camera["alpha"], alpha_in_degree)
            else:
                continue
            cls_id = int(classid_map[ann["category_id"]])
            if cls_id < 0:
                continue

            bbox[:2] = affine_transform([bbox[:2]], trans_output)
            bbox[2:] = affine_transform([bbox[2:]], trans_output)

            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_width - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_height - 1)
            _wh = bbox[2:] - bbox[:2]
            if np.any(_wh <= 0):
                continue
            # filter bbox
            if (
                ann["bbox_2d"][2] < min_box_edge
                or ann["bbox_2d"][3] < min_box_edge
            ):
                ignore_mask[
                    int(bbox[1]) : int(bbox[3] + 1),
                    int(bbox[0]) : int(bbox[2] + 1),
                    :,
                ] = 1.0
                continue
            # filter by depth
            if in_camera["depth"] > max_depth:
                ignore_mask[
                    int(bbox[1]) : int(bbox[3] + 1),
                    int(bbox[0]) : int(bbox[2] + 1),
                    :,
                ] = 1.0
                continue
            # w, h = _wh
            ct = (bbox[:2] + bbox[2:]) / 2
            ct_int = tuple(ct.astype(np.int32).tolist())

            ann_idx += 1
            if ann_idx >= max_objs:
                break
            alpha_x[ann_idx] = alpha  # alpha will be delete, next version
            ind_[ann_idx] = ct_int[1] * output_width + ct_int[0]
            ind_mask_[ann_idx] = 1
            loc_[ann_idx] = in_camera["location"]
            rot_y_[ann_idx] = in_camera["rotation_y"]
            dim_[ann_idx] = in_camera["dim"]
            # ttfnet style
            insert_hm = get_gaussian2D(_wh)

            insert_hm_wh = insert_hm.shape[:2][::-1]

            if not depth_min_option:
                insert_reg_map_list = [
                    get_reg_map(insert_hm_wh, in_camera["depth"]),
                    get_reg_map(insert_hm_wh, _wh),
                    get_reg_map(insert_hm_wh, in_camera["dim"]),
                    get_dense_locoffset(
                        insert_hm_wh,
                        ct_int,
                        location_offset[:2],
                        in_camera["location"],
                        in_camera["dim"],
                        calib,
                        trans_output,
                        meta["distCoeffs"],
                        undistort_2dcenter,
                    ),
                ]
                reg_map_list = [
                    depth,
                    wh,
                    dim,
                    loc_offset,  # rotbin, rotres
                ]
                draw_heatmap(hm[:, :, cls_id], insert_hm, ct_int)
                draw_heatmap(
                    weight_hm,
                    insert_hm,
                    ct_int,
                    reg_map_list,
                    insert_reg_map_list,
                )

            else:
                insert_reg_map_list = [
                    get_reg_map(insert_hm_wh, _wh),
                    get_reg_map(insert_hm_wh, in_camera["dim"]),
                    get_dense_locoffset(
                        insert_hm_wh,
                        ct_int,
                        location_offset[:2],
                        in_camera["location"],
                        in_camera["dim"],
                        calib,
                        trans_output,
                        meta["distCoeffs"],
                        undistort_2dcenter,
                    ),
                ]
                reg_map_list = [
                    wh,
                    dim,
                    loc_offset,  # rotbin, rotres
                ]
                draw_heatmap(hm[:, :, cls_id], insert_hm, ct_int)
                draw_heatmap(
                    weight_hm,
                    insert_hm,
                    ct_int,
                    reg_map_list,
                    insert_reg_map_list,
                )

                insert_reg_map_list_min = [
                    get_reg_map(insert_hm_wh, in_camera["depth"]),
                ]

                reg_map_list_min = [
                    depth,
                ]

                insert_hm_depth = in_camera["depth"] * np.ones(
                    (insert_hm.shape[0], insert_hm.shape[1]), dtype=np.float64
                )

                draw_heatmap_min(
                    weight_hm_min,
                    insert_hm_depth,
                    ct_int,
                    reg_map_list_min,
                    insert_reg_map_list_min,
                )

            point_pos_mask[ct_int[1], ct_int[0]] = 1

    if normalize_depth:
        if undistort_depth_uv:
            eq_fu, eq_fv = cal_equivalent_focal_length_uv_mat(
                size[0], size[1], calib, meta["distCoeffs"]
            )
            resized_eq_fu = cv2.warpAffine(
                eq_fu, trans_output, (output_width, output_height)
            )
            resized_eq_fv = cv2.warpAffine(
                eq_fv, trans_output, (output_width, output_height)
            )
            depth_u = depth * focal_length_default / resized_eq_fu
            depth_v = depth * focal_length_default / resized_eq_fv
            depth = np.stack([depth_u, depth_v], axis=-1)
        else:
            depth *= focal_length_default / calib[0, 0]
            depth = depth[:, :, np.newaxis]
        loc_offset *= focal_length_default / calib[0, 0]

    ret = {
        "heatmap": hm,
        "box2d_wh": wh,
        "dimensions": dim,
        "location_offset": loc_offset,
        "depth": depth,
        "heatmap_weight": weight_hm[:, :, np.newaxis],
        "point_pos_mask": point_pos_mask[:, :, np.newaxis],
    }
    if enable_ignore_area:
        ret["ignore_mask"] = ignore_mask

    for k in ret.keys():
        ret[k] = ret[k].transpose(2, 0, 1)
    ret.update(
        {
            "index": ind_,
            "index_mask": ind_mask_,
            "alpha_x": alpha_x,
            "location": loc_,
            "rotation_y": rot_y_,
            "dimensions_": dim_,
        }
    )
    del label

    return ret


def dense3d_pad_after_label_generator(gt, input_padding, down_stride):
    # heatmap (1, 80,120) -> (1, 80, 128)
    output_width = gt["heatmap"].shape[2]
    left_w = input_padding[0] // down_stride
    right_w = input_padding[1] // down_stride
    gt["heatmap"] = np.pad(
        gt["heatmap"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["box2d_wh"] = np.pad(
        gt["box2d_wh"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["dimensions"] = np.pad(
        gt["dimensions"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["location_offset"] = np.pad(
        gt["location_offset"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["depth"] = np.pad(
        gt["depth"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["heatmap_weight"] = np.pad(
        gt["heatmap_weight"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["point_pos_mask"] = np.pad(
        gt["point_pos_mask"], ((0, 0), (0, 0), (left_w, right_w)), "constant"
    )
    gt["ignore_mask"] = np.pad(
        gt["ignore_mask"],
        ((0, 0), (0, 0), (left_w, right_w)),
        "constant",
        constant_values=(1, 1),
    )

    center_x = gt["index"] % output_width
    center_y = gt["index"] // output_width

    gt["index"] = (
        center_y * (output_width + left_w + right_w) + center_x + left_w
    )
    return gt
