# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from hat.core.box3d_utils import (
    compute_box_3d,
    project_3d_to_bird,
    project_to_image,
)
from hat.core.data_struct.base_struct import DetBoxes3D
from .bbox2d import draw_bbox

__all__ = [
    "draw_bbox3d",
]


def cal_truncation(box_2d, im_hw):
    bbox2 = np.array(
        [
            box_2d[:, 0].min(),
            box_2d[:, 1].min(),
            box_2d[:, 0].max(),
            box_2d[:, 1].max(),
        ]
    )
    bbox1 = np.array([0, 0, im_hw[1], im_hw[0]])

    # area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    inter = max(
        min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) + 1, 0
    ) * max(min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]) + 1, 0)

    truncation = 1.0 - inter / area2
    return truncation


def draw_box_3d(image, corners, c=(0, 0, 255), show_arrow=True, thickness=1):
    face_idx = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(
                image,
                (corners[f[j], 0], corners[f[j], 1]),
                (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]),
                c,
                thickness,
                lineType=cv2.LINE_AA,
            )

        if not show_arrow:
            if ind_f == 0:
                cv2.line(
                    image,
                    (corners[f[0], 0], corners[f[0], 1]),
                    (corners[f[2], 0], corners[f[2], 1]),
                    c,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
                cv2.line(
                    image,
                    (corners[f[1], 0], corners[f[1], 1]),
                    (corners[f[3], 0], corners[f[3], 1]),
                    c,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        # show an arrow to indicate 3D orientation of the object
        if show_arrow:
            # 4,5,6,7
            p1 = (
                corners[0, :] + corners[1, :] + corners[2, :] + corners[3, :]
            ) / 4
            p2 = (corners[0, :] + corners[1, :]) / 2
            p3 = p2 + (p2 - p1) * 0.5

            p1 = p1.astype(np.int32)
            p2 = p2.astype(np.int32)
            p3 = p3.astype(np.int32)

            cv2.line(
                image,
                (p1[0], p1[1]),
                (p3[0], p3[1]),
                c,
                thickness,
                lineType=cv2.LINE_AA,
            )
    return image


def draw_bev(
    location,
    dimension,
    yaw,
    color,
    canvas=None,
    out_size=384,
    world_size=64,
):
    if canvas is None:
        canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)

    pts_3d = compute_box_3d(dimension, location, yaw)
    rect = pts_3d[:4, [0, 2]]
    rect = project_3d_to_bird(rect, out_size, world_size)
    cv2.polylines(
        canvas,
        [rect.reshape(-1, 1, 2).astype(np.int32)],
        True,
        color,
        1,
        lineType=cv2.LINE_AA,
    )
    p1 = np.mean(rect, axis=0)
    p2 = (rect[0] + rect[1]) / 2
    p3 = p2 + (p2 - p1) / 2
    p1, p3 = p1.astype(np.int32), p3.astype(np.int32)

    cv2.line(
        canvas,
        (p1[0], p1[1]),
        (p3[0], p3[1]),
        color,
        1,
        lineType=cv2.LINE_AA,
    )

    return canvas


def draw_bbox3d(
    img,
    location,
    dimension,
    yaw,
    color,
    thickness,
    calib,
    distCoeffs,
    fisheye=False,
    truncation_thresh=0.3,
    min_depth_dist=0.8,
    bbox=None,
    draw_2d=False,
):
    img_wh = img.shape[:2][::-1]
    pts_3d = compute_box_3d(dimension, location, yaw)
    pts_2d = project_to_image(
        img_wh, pts_3d, calib, dist_coeff=distCoeffs, fisheye=fisheye
    )
    truncation = cal_truncation(pts_2d, img.shape[:2])

    if (
        truncation > truncation_thresh
        or (pts_3d[:, 2] <= min_depth_dist).any()
    ) and bbox is not None:
        img = draw_bbox(img, bbox, color, thickness)
    else:
        pts_2d = pts_2d.astype(np.int32)
        draw_box_3d(img, pts_2d, color, thickness=thickness)
        if draw_2d:
            bbox2d = [
                pts_2d[:, 0].min(),
                pts_2d[:, 1].min(),
                pts_2d[:, 0].max(),
                pts_2d[:, 1].max(),
            ]
            img = draw_bbox(img, bbox2d, color, thickness)

    return img


def blend_top_right(img, blend, blend_factor=0.8):
    height, width = blend.shape[:2]
    img[:height, -width:] = (
        img[:height, -width:] * (1 - blend_factor) + blend * blend_factor
    )
    return img.astype(np.uint8)


def vis_det_boxes_3d(
    vis_image: np.ndarray,
    calib: np.ndarray,
    distCoeffs: np.ndarray,
    det_boxes_3d: DetBoxes3D,
    vis_configs: Dict,
):
    color = vis_configs["color"]
    thickness = vis_configs["thickness"]

    for det_box_3d in iter(det_boxes_3d):
        vis_image = draw_bbox3d(
            vis_image,
            det_box_3d.location.numpy(),
            det_box_3d.dimension.numpy(),
            det_box_3d.yaw.item(),
            color,
            thickness,
            calib,
            distCoeffs,
        )
    return vis_image


def vis_det_boxes_3d_bev(
    det_boxes_3d: DetBoxes3D,
    vis_configs: Dict,
):
    canvas = None
    for det_box_3d in iter(det_boxes_3d):
        canvas = draw_bev(
            det_box_3d.location.numpy(),
            det_box_3d.dimension.numpy(),
            det_box_3d.yaw.item(),
            vis_configs.get("color", vis_configs["color"]),
            canvas=canvas,
        )

    return canvas


def draw_box_represented_by_corners(
    img: np.ndarray,
    points: np.ndarray,
    color: Optional[Sequence[int]] = (0, 0, 255),
    score: Optional[float] = None,
    draw_arrow: Optional[bool] = False,
    font_scale: Optional[float] = 0.5,
):
    """Draw either 2D (4 points) or 3D (8 points) box represented by corners.

    There are two assumptions:
        1. First two points are ground front, i.e., they indicate
            the object's heading direction
        2. If the are eight corners indicate 3D box, then the
            first four points should be on the bottom surface,
            and last four should be on the top surface. And the bottom
            surface points should have the same order with the ones on the
            top surface.
        A example is:
                5 -------- 4
               /|         /|
              6 -------- 7 .
              | |     ^  | |
              . 1 ---/---- 0
              |/    /   |/
              2 ---/---- 3
                  / direction of the heading
    Args:
        img : ndarray
            Input image
        points : ndarray, shape [8, 2] or [4, 2]
            Eight points represented 3D box
        color : array like
            The color to draw the 3D box
        score : float
            Score of the 3d object
        draw_arrow : bool
            Draw a arrow to indicate direction
        font_scale : font
            font size
    """
    N = points.shape[0]
    points = points.astype(np.int32)
    for i in range(4):
        s = points[i, :]
        e = points[(i + 1) % 4, :]
        cv2.line(img, (s[0], s[1]), (e[0], e[1]), color=color, thickness=1)

    if N > 4:
        for i in range(4, N):
            s = points[i, :]
            e = points[(i + 1) % 4 + 4, :]
            cv2.line(img, (s[0], s[1]), (e[0], e[1]), color=color, thickness=1)
            # draw vertical line
            e = points[i - 4, :]
            cv2.line(img, (s[0], s[1]), (e[0], e[1]), color=color, thickness=1)

    # draw arrow indicate direction
    if draw_arrow:
        center = np.mean(points[:4, :], axis=0, dtype=np.int32)
        s = (points[0, :] + points[1, :]) // 2
        p = s + (s - center) // 2
        cv2.line(
            img, (center[0], center[1]), (p[0], p[1]), color=color, thickness=1
        )

    # draw score if not None
    if score is not None:
        cv2.putText(
            img,
            "%.2f" % score,
            (points[2, 0], points[2, 1]),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color,
            1,
        )


def draw_dense_box(
    img: np.ndarray,
    points: np.ndarray,
    color: Optional[Sequence[int]] = (0, 0, 255),
    text: Optional[str] = None,
    vis_cfg: Optional[dict] = None,
):
    """Draw 3D box represented by dense points.

    Args:
        img : ndarray
            Input image
        points : ndarray, shape [N, 2]
            Dense points of 3D box
        color : array like
            The color to draw the 3D box
        text : str
            text of the 3d object
        vis_cfg: dict
            params of opencv plot
    """
    nums = points.shape[0]
    points = points.astype(np.int32)
    if vis_cfg is None:
        vis_cfg = {}
    thickness = vis_cfg.get("thickness", -1)
    font_scale = vis_cfg.get("front_scale", 1.5)
    font_thickness = vis_cfg.get("font_thickness", 2)
    for i in range(nums):
        cv2.circle(
            img,
            (points[i, 0], points[i, 1]),
            2,
            color=color,
            thickness=thickness,
        )

    cx = np.mean(points[:, 0]).astype(np.int32)
    cy = np.min(points[:, 1]).astype(np.int32)
    if text is not None:
        cv2.putText(
            img,
            text,
            (cx, cy),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color,
            font_thickness,
        )
