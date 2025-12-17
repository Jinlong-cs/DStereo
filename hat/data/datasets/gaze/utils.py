import math
from typing import Dict, Tuple

import numpy as np

__all__ = [
    "R2xyz",
    "xyz2R",
    "parse_gaze_mtl_label",
    "eye_ldmk_transform",
    "calculate_rotation_matrix",
]


def R2xyz(R):
    # ORDER_ZXY
    x = math.asin(-R[1][2])
    y = math.atan2(R[0][2], R[2][2])
    z = math.atan2(R[1][0], R[1][1])
    return x, y, z


def xyz2R(x, y, z):
    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(x), -math.sin(x)],
            [0, math.sin(x), math.cos(x)],
        ]
    )
    Ry = np.array(
        [
            [math.cos(y), 0, math.sin(y)],
            [0, 1, 0],
            [-math.sin(y), 0, math.cos(y)],
        ]
    )
    Rz = np.array(
        [
            [math.cos(z), -math.sin(z), 0],
            [math.sin(z), math.cos(z), 0],
            [0, 0, 1],
        ]
    )
    Rn = np.dot(Ry, np.dot(Rx, Rz))
    return Rn


def parse_gaze_mtl_label(
    meta: Dict, rotate_3d_augm: bool = False, angle_form: str = "degree"
):
    """Parse labels for gaze_mtl training or validating.

    Labels include head pose, gaze, eye landmarks, eye bounding box,
    face landmarks, face bounding box(if exists) and loss weight.
    For each label, there could be missing values. Missing values are
    replaced by -1000.

    Args:
        meta: current sample info from rec
        rotate_3d_augm: flag for 3d roate augm
        angle_form: type for gaze angle, ["degree", "radian", "degree_exp"]
    """
    headpose_num = 3
    gaze_num = 4
    per_eye_gaze_num = 2
    per_eye_ldmk_num = 21
    face_ldmk_num = 68 if rotate_3d_augm else 27
    face_bbox_num = 4
    g_pitch_thresh = 180
    g_yaw_thresh = 180

    label = {}

    assert angle_form in [
        "degree",
        "radian",
        "degree_exp",
    ], f"invalid angle form: {angle_form}!"

    # For headpose
    if (
        isinstance(meta["headpose"], float)
        or meta["headpose"] is None
        or meta["headpose"] == "INVALID"
    ):
        head_pose = np.ones((headpose_num), dtype=np.float32) * -1000
    else:
        head_pose = np.array(meta["headpose"]).astype(np.float32)

    # For gaze
    if isinstance(meta["left_eye"], float):
        gaze = np.ones((gaze_num), dtype=np.float32) * -1000
    elif (
        len(meta["left_eye"]) < per_eye_gaze_num
        or len(meta["right_eye"]) < per_eye_gaze_num
    ):
        gaze = np.ones((gaze_num), dtype=np.float32) * -1000
    elif (
        (-g_yaw_thresh < meta["left_eye"][1] < g_yaw_thresh)
        and (-g_pitch_thresh < meta["left_eye"][0] < g_pitch_thresh)
        and (-g_yaw_thresh < meta["right_eye"][1] < g_yaw_thresh)
        and (-g_pitch_thresh < meta["right_eye"][0] < g_pitch_thresh)
    ):
        gaze = np.hstack([meta["left_eye"], meta["right_eye"]]).astype(
            np.float32
        )
        if angle_form == "radian":
            gaze = gaze * np.pi / 180
        if angle_form == "degree_exp":
            gaze = np.log(np.abs(gaze))
    else:
        gaze = np.ones((gaze_num), dtype=np.float32) * -1000

    # For eye_ldmk:
    if (
        isinstance(meta["left_eye_ldmk"], float)
        or meta["left_eye_ldmk"] == "INVALID"
        or meta["left_eye_ldmk"] is None
        or meta["left_eye_ldmk"] == "None"
    ):
        left_eye_ldmk = (
            np.ones((per_eye_ldmk_num, 2), dtype=np.float32) * -1000
        )
    else:
        left_eye_ldmk = (
            np.ones((per_eye_ldmk_num, 2), dtype=np.float32) * -1000
        )
        left_eye_ldmk[: len(meta["left_eye_ldmk"])] = np.array(
            meta["left_eye_ldmk"]
        )
    if (
        isinstance(meta["right_eye_ldmk"], float)
        or meta["right_eye_ldmk"] == "INVALID"
        or meta["right_eye_ldmk"] is None
        or meta["right_eye_ldmk"] == "None"
    ):
        right_eye_ldmk = (
            np.ones((per_eye_ldmk_num, 2), dtype=np.float32) * -1000
        )
    else:
        right_eye_ldmk = (
            np.ones((per_eye_ldmk_num, 2), dtype=np.float32) * -1000
        )
        right_eye_ldmk[: len(meta["right_eye_ldmk"])] = np.array(
            meta["right_eye_ldmk"]
        )
    eye_ldmks = np.vstack([left_eye_ldmk, right_eye_ldmk]).astype(np.float32)
    eye_ldmks[eye_ldmks < 0] = -1000

    # For eye_bbox
    eye_bbox = np.array(meta["eye_bbox"])

    # For face_ldmks
    face_ldmks = np.array(meta["face_ldmk"])[:face_ldmk_num].astype(np.float32)

    # For face_bbox
    # face bounding box only exists when packing face images
    if "face_bbox" in meta:
        face_bbox = np.array(meta["face_bbox"])
    else:
        face_bbox = np.ones((face_bbox_num,)) * -1000

    # For sample_weight
    if "gaze_sample_weight" in meta:
        loss_weight = meta["gaze_sample_weight"]
    else:
        loss_weight = 1
    loss_weight = np.array([loss_weight]).astype(np.float32)

    if rotate_3d_augm:
        intrinsics_K = np.array(meta["intrinsics"]["K"]).reshape(3, 3)
        if "origin_image_shape" in meta.keys():
            origin_image_shape = np.array(meta["origin_image_shape"])
        else:
            origin_image_shape = np.array([-1000, -1000])
        label["gt_head_pose"] = head_pose
        label["gt_gaze"] = gaze
        label["gt_eye_ldmk"] = eye_ldmks
        label["gt_eye_bbox"] = eye_bbox
        label["gt_face_ldmks"] = face_ldmks
        label["gt_face_bbox"] = face_bbox
        label["gt_loss_weight"] = loss_weight
        label["intrinsics_K"] = intrinsics_K
        label["origin_image_shape"] = origin_image_shape
        return label

    label["gt_head_pose"] = head_pose
    label["gt_gaze"] = gaze
    label["gt_eye_ldmk"] = eye_ldmks
    label["gt_eye_bbox"] = eye_bbox
    label["gt_face_ldmks"] = face_ldmks
    label["gt_face_bbox"] = face_bbox
    label["gt_loss_weight"] = loss_weight
    label["image_id"] = meta["image_id"]
    return label


def eye_ldmk_transform(
    eye_ldmk: np.ndarray, eye_bbox: np.ndarray, resize_shape: Tuple
):
    """Convert raw eye landmarks from camera frame to ratio in input region.

    This transform is used for directly normalzie eye_ldmk acording to input
    size, normed eye_ldmk and raw eye_ldmk would be all contained in label

    Args:
        eye_ldmk: Raw eye landmarks coordinates in camera frame,
                  -1000 for invalid eye landmarks.
        eye_bbox: Original eye bounding box.
        resize_shape: Resize shape
    """
    bbox_w = eye_bbox[2] - eye_bbox[0]
    bbox_h = eye_bbox[3] - eye_bbox[1]
    resize_x = resize_shape[0]
    resize_y = resize_shape[1]

    eye_bbox_upperleft = np.array([[eye_bbox[0], eye_bbox[1]]])
    eye_bbox_shape = np.array([[bbox_w, bbox_h]])
    resize_shape = np.array([resize_x, resize_y]).reshape(1, -1)
    crop_upperleft = np.array([0, 0]).reshape(1, -1)

    # eye_ldmk ratio
    eye_ldmk = (
        (eye_ldmk - eye_bbox_upperleft) / eye_bbox_shape * resize_shape
        - crop_upperleft
    ) / resize_shape

    # filter invalid points
    mask = (1 - (eye_ldmk > 0) * (eye_ldmk < 1)).astype("bool")
    eye_ldmk[mask] = -1000
    return eye_ldmk.astype(np.float32)


def calculate_rotation_matrix(gaze):
    # Functions to calculate relative rotation matrices for
    # gaze dir and head pose

    # assume no roll
    # pitch
    def R_x(theta):
        sin_ = np.sin(theta)
        cos_ = np.cos(theta)
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_, -sin_], [0.0, sin_, cos_]]
        ).astype(np.float32)

    # yaw
    def R_y(phi):
        sin_ = np.sin(phi)
        cos_ = np.cos(phi)
        return np.array(
            [[cos_, 0.0, sin_], [0.0, 1.0, 0.0], [-sin_, 0.0, cos_]]
        ).astype(np.float32)

    # unvalid gaze not to generate rotation matrix
    if gaze[0] > -1000 or gaze[1] > -1000:
        return np.matmul(R_y(gaze[1]), R_x(gaze[0]))
    else:
        return np.eye(3, dtype=np.float32)
