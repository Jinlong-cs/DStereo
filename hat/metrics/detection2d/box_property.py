# Copyright (c) Horizon Robotics. All rights reserved.

import math

import numpy as np

from hat.utils.package_helper import require_packages

try:
    from hatbc.message.structure import BBox2D
except ImportError:
    BBox2D = None


def get_pixel_distance(bbox, image_x1, image_x2):
    pixel_distance = min(bbox.x1 - image_x1, image_x2 - bbox.x2)
    return pixel_distance


@require_packages("hatbc")
def get_region_distance(bbox, bbox_type, dynamic_region):
    if bbox_type == "BBox2D":
        pixel_distance = min(
            abs(bbox["x1"] - dynamic_region[0]),
            abs(bbox["y1"] - dynamic_region[1]),
            abs(bbox["x2"] - dynamic_region[2]),
            abs(bbox["y2"] - dynamic_region[3]),
        )
    else:
        raise NotImplementedError(
            "Bbox Type not Implemented yet: {}".format(bbox_type)
        )
    return pixel_distance


def get_left(bbox, bbox_type):
    if bbox_type == "BBox2D":
        left = bbox["x1"]
    elif bbox_type == "BBOX_rotation":
        left = min(bbox["x1"], bbox["x2"], bbox["x3"], bbox["x4"])
    else:
        raise NotImplementedError(
            "Bbox Type not Implemented yet: {}".format(bbox_type)
        )
    return left


def get_right(bbox, bbox_type):
    if bbox_type == "BBox2D":
        right = bbox["x2"]
    elif bbox_type == "BBOX_rotation":
        right = max(bbox["x1"], bbox["x2"], bbox["x3"], bbox["x4"])
    else:
        raise NotImplementedError(
            "Bbox Type not Implemented yet: {}".format(bbox_type)
        )
    return right


def get_top(bbox, bbox_type):
    if bbox_type == "BBox2D":
        top = bbox["x1"]
    elif bbox_type == "BBOX_rotation":
        top = min(bbox["y1"], bbox["y2"], bbox["y3"], bbox["y4"])
    else:
        raise NotImplementedError(
            "Bbox Type not Implemented yet: {}".format(bbox_type)
        )
    return top


def get_bottom(bbox, bbox_type):
    if bbox_type == "BBox2D":
        bottom = bbox["x1"]
    elif bbox_type == "BBOX_rotation":
        bottom = max(bbox["y1"], bbox["y2"], bbox["y3"], bbox["y4"])
    else:
        raise NotImplementedError(
            "Bbox Type not Implemented yet: {}".format(bbox_type)
        )
    return bottom


def get_shortside(bbox, bbox_type):
    if bbox_type == "BBox2D":
        w = bbox["x2"] - bbox["x1"]
        h = bbox["y2"] - bbox["y1"]
        shortside = min(w, h)
    else:
        raise NotImplementedError(
            "Bbox Type not Implemented yet: {}".format(bbox_type)
        )
    return shortside


def get_longside(bbox, bbox_type):
    if bbox_type == "BBox2D":
        w = bbox["x2"] - bbox["x1"]
        h = bbox["y2"] - bbox["y1"]
        longside = max(w, h)
    else:
        raise NotImplementedError(
            "Bbox Type not Implemented yet: {}".format(bbox_type)
        )
    return longside


def get_height(bbox, bbox_type):
    if bbox_type == "BBox2D":
        h = bbox["y2"] - bbox["y1"]
    else:
        raise NotImplementedError(
            "Bbox Type not Implemented yet: {}".format(bbox_type)
        )
    return h


def get_width(bbox, bbox_type):
    if bbox_type == "BBox2D":
        w = bbox["x2"] - bbox["x1"]
    else:
        raise NotImplementedError(
            "Bbox Type not Implemented yet: {}".format(bbox_type)
        )
    return w


def get_whr(bbox, bbox_type):
    if bbox_type == "BBox2D":
        w = bbox["x2"] - bbox["x1"]
        h = bbox["y2"] - bbox["y1"]
        whr = w / (h + 1e-6)
    else:
        raise NotImplementedError(
            "Bbox Type not Implemented yet: {}".format(bbox_type)
        )
    return whr


def get_area(bbox, bbox_type):
    if bbox_type == "BBox2D":
        w = bbox["x2"] - bbox["x1"]
        h = bbox["y2"] - bbox["y1"]
        area = w * h
    else:
        raise NotImplementedError(
            "Bbox Type not Implemented yet: {}".format(bbox_type)
        )
    return area


def get_euclidean_distance(x, y):
    return math.sqrt(x ** 2 + y ** 2)


def get_iou(gt, det, iou_type="iou"):
    bi = [
        max(gt[0], det[0]),
        max(gt[1], det[1]),
        min(gt[2], det[2]),
        min(gt[3], det[3]),
    ]
    gt_w = gt[2] - gt[0] + 1
    gt_h = gt[3] - gt[1] + 1
    det_w = det[2] - det[0] + 1
    det_h = det[3] - det[1] + 1
    iw = max(0, bi[2] - bi[0] + 1)
    ih = max(0, bi[3] - bi[1] + 1)
    gt_area = gt_w * gt_h
    det_area = det_w * det_h
    inter_area = iw * ih
    union_area = gt_area + det_area - inter_area
    if iou_type == "iou":
        iou = inter_area / union_area
    elif iou_type == "iod":
        iou = inter_area / det_area
    elif iou_type == "iog":
        iou = inter_area / gt_area
    else:
        raise NotImplementedError(
            "Iou Type not Implemented yet: {}".format(iou_type)
        )
    return iou


def bbox_errors(gt_bbox: "BBox2D", det_bbox: "BBox2D"):
    bi = [
        max(det_bbox.x1, gt_bbox.x1),
        max(det_bbox.y1, gt_bbox.y1),
        min(det_bbox.x2, gt_bbox.x2),
        min(det_bbox.y2, gt_bbox.y2),
    ]
    det_bbox_w = det_bbox.x2 - det_bbox.x1
    det_bbox_h = det_bbox.y2 - det_bbox.y1
    gt_bbox_w = gt_bbox.x2 - gt_bbox.x1
    gt_bbox_h = gt_bbox.y2 - gt_bbox.y1
    det_w = det_bbox_w + 1
    det_h = det_bbox_h + 1
    gt_w = gt_bbox_w + 1
    gt_h = gt_bbox_h + 1
    iw = max(0, bi[2] - bi[0] + 1)
    ih = max(0, bi[3] - bi[1] + 1)
    det_center_x = (det_bbox.x1 + det_bbox.x2) / 2.0
    gt_center_x = (gt_bbox.x1 + gt_bbox.x2) / 2.0
    det_center_y = (det_bbox.y1 + det_bbox.y2) / 2.0
    gt_center_y = (gt_bbox.y1 + gt_bbox.y2) / 2.0
    uw = det_w + gt_w - iw
    uh = det_h + gt_h - ih
    det_area = det_w * det_h
    gt_area = gt_w * gt_h
    inter_area = iw * ih
    union_area = det_area + gt_area - inter_area
    errors = {
        "iou_err": 1 - inter_area / union_area,
        "iou_horizontal_err": 1 - iw / uw if inter_area > 0 else 0,
        "iou_vertical_err": 1 - ih / uh if inter_area > 0 else 0,
        "width_err": abs(det_bbox_w - gt_bbox_w) / gt_bbox_w,
        "width_err_signed": (det_bbox_w - gt_bbox_w) / gt_bbox_w,
        "width_diff": (det_bbox_w - gt_bbox_w),
        "height_err": abs(det_bbox_h - gt_bbox_h) / gt_bbox_h,
        "height_err_signed": (det_bbox_h - gt_bbox_h) / gt_bbox_h,
        "height_diff": (det_bbox_h - gt_bbox_h),
        "center_x_err": abs(det_center_x - gt_center_x) / gt_bbox_w,
        "center_x_err_signed": (det_center_x - gt_center_x) / gt_bbox_w,
        "center_y_err": abs(det_center_y - gt_center_y) / gt_bbox_h,
        "center_y_err_signed": (det_center_y - gt_center_y) / gt_bbox_h,
        "top_y_err": abs(det_bbox.y1 - gt_bbox.y1) / gt_bbox_h,
        "top_y_err_signed": (det_bbox.y1 - gt_bbox.y1) / gt_bbox_h,
        "bottom_y_err": abs(det_bbox.y2 - gt_bbox.y2) / gt_bbox_h,
        "bottom_y_err_signed": (det_bbox.y2 - gt_bbox.y2) / gt_bbox_h,
        "left_x_err": abs(det_bbox.x1 - gt_bbox.x1) / gt_bbox_w,
        "left_x_err_signed": (det_bbox.x1 - gt_bbox.x1) / gt_bbox_w,
        "right_x_err": abs(det_bbox.x2 - gt_bbox.x2) / gt_bbox_w,
        "right_x_err_signed": (det_bbox.x2 - gt_bbox.x2) / gt_bbox_w,
    }
    return errors


def lower_overlap_for_small(bbox: "BBox2D", width: int = 10, height: int = 10):
    """Overlap for small bbox.

    Args:
        bbox: bounding box.
        width: width of small box.
        height: height of small box.
    """
    w = bbox.x2 - bbox.x1
    h = bbox.y2 - bbox.y1
    return (w + 1) * (h + 1) / (w + 1 + width) / (h + 1 + width)


def bbox_critical(bbox: "BBox2D", critical_label: np.array) -> bool:
    # if det in critical area, return true
    bottom = np.zeros_like(critical_label)
    bottom[int(bbox.y2), int(bbox.x1) : int(bbox.x2)] = 1
    if np.sum(bottom * critical_label) > 0:
        return True
    else:
        return False
