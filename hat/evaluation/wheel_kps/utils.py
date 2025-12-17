from typing import List, Union

import numpy as np

__all__ = [
    "check_bbox_dimension",
    "compute_norm_distance_error_wheel",
    "compute_angle_error_xwheel",
]


def check_bbox_dimension(
    bbox: List,
    dimension_threshold_max: float,
    dimension_threshold_min: float,
    check_dimension_type: str,
) -> bool:
    """Check bbox dimension.

    Args:
        bbox: bbox(xyxy)
        dimension_threshold_max: bbox dimension max threshold.
        dimension_threshold_min: bbox dimension min threshold.
        check_dimension_type: bbox dimension type.
    """

    bbox_width, bbox_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if check_dimension_type == "long_side":
        longer_side = max(bbox_width, bbox_height)
        # no max limits
        if not dimension_threshold_max:
            return longer_side >= dimension_threshold_min
        else:
            return (longer_side >= dimension_threshold_min) & (
                longer_side <= dimension_threshold_max
            )

    elif check_dimension_type == "short_side":
        shorter_side = min(bbox_width, bbox_height)
        # no max limits
        if not dimension_threshold_max:
            return shorter_side >= dimension_threshold_min
        else:
            return (shorter_side >= dimension_threshold_min) & (
                shorter_side <= dimension_threshold_max
            )

    elif check_dimension_type == "height":
        # no max limits
        if not dimension_threshold_max:
            return bbox_height >= dimension_threshold_min
        else:
            return (bbox_height >= dimension_threshold_min) & (
                bbox_height <= dimension_threshold_max
            )

    elif check_dimension_type == "width":
        # no max limits
        if not dimension_threshold_max:
            return bbox_width >= dimension_threshold_min
        else:
            return (bbox_width >= dimension_threshold_min) & (
                bbox_width <= dimension_threshold_max
            )

    else:
        raise Exception(
            f"unknown check_dimension_type: {check_dimension_type}"
        )


def compute_norm_distance_error_wheel(
    bbox: List,
    gt_wheel_kps: Union[List, np.ndarray],
    pred_wheel_kps: Union[List, np.ndarray],
) -> List:
    """Compute norm distance (per landmark).

    Args:
        bbox: bbox(xyxy)
        gt_wheel_kps: groundtruth of wheel keypoints
        pred_wheel_kps: prediction of wheel keypoints
    """

    x_min, y_min, x_max, y_max = bbox
    diagonal = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

    assert len(gt_wheel_kps) == len(
        pred_wheel_kps
    ), "gt and pred does not have the same num_kps!"

    num_kps = len(gt_wheel_kps)
    errors = []
    for kps_i in range(num_kps):
        error_kpi = (
            np.sqrt(
                (gt_wheel_kps[kps_i][0] - pred_wheel_kps[kps_i][0]) ** 2
                + (gt_wheel_kps[kps_i][1] - pred_wheel_kps[kps_i][1]) ** 2
            )
        ) / diagonal
        errors.append(error_kpi)

    return errors


def compute_angle_error_xwheel(
    gt_wheel_kps: Union[List, np.ndarray],
    pred_wheel_kps: Union[List, np.ndarray],
) -> float:
    """Compute angle error(kp_back_gt, kp_front_gt).

    Args:
        gt_wheel_kps: groundtruth of wheel keypoints
        pred_wheel_kps: prediction of wheel keypoints
    """
    assert len(gt_wheel_kps) == len(
        pred_wheel_kps
    ), "gt's num_kps should be the same as pred's"
    kps_pair_map = {
        2: [[0, 1]],
        4: [[0, 1], [3, 2]],
        8: [[0, 1], [3, 2], [4, 5], [7, 6]],
        12: [[0, 1], [3, 2], [4, 5], [7, 6], [8, 9], [11, 10]],
    }
    num_kps = len(gt_wheel_kps)
    kps_pair = kps_pair_map[num_kps]
    angle = 0
    for pair_i in kps_pair:
        back = pair_i[0]
        front = pair_i[1]
        gt_v = np.array(
            [
                gt_wheel_kps[front][0] - gt_wheel_kps[back][0],
                gt_wheel_kps[front][1] - gt_wheel_kps[back][1],
            ]
        )
        pred_v = np.array(
            (
                pred_wheel_kps[front][0] - pred_wheel_kps[back][0],
                pred_wheel_kps[front][1] - pred_wheel_kps[back][1],
            )
        )
        angle += angleCompute(gt_v, pred_v)[0]
    return angle


def angleCompute(v1, v2):
    """Compute vector angle."""
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    angle_val = np.arctan2(sinang, cosang)

    # denote the pose orientation flag
    pos_ori = True

    return angle_val * 57.2957795, pos_ori
