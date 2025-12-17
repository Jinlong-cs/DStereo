# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import os
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch

from hat.models.task_modules.bev.spatial_transfomer import (
    SpatialTransfomerWithOffset,
)
from hat.utils.apply_func import img_array2tensor

__all__ = [
    "arrange_imgs",
    "draw_raw_img",
    "draw_single_task",
    "draw_multi_head",
    "draw_single_head",
    "get_cls_color_table",
    "draw_grid",
    "draw_label",
    "draw_reg",
    "draw_online_mapping",
    "get_bev_pts",
]

general_color_list = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "yellow": (0, 255, 255),
    "orange": (255, 165, 0),
}

category_color_dict = {
    "background": (0, 0, 0),
    "solid_lanes": (255, 255, 255),
    "roadedges": (0, 0, 255),
    "crosswalks": (0, 255, 0),
    "TP_GT": (0, 255, 0),
    "TP_PRED": (255, 0, 0),
    "FN_GT": (0, 255, 255),
    "FP_PRED": (0, 0, 255),
    "ignores": (0, 255, 255),
    "255.0": (0, 255, 255),
    "lane_solid_single": (255, 255, 255),
    "lane_solid_double": (0, 255, 0),
    "lane_dotted_single": (255, 0, 0),
    "lane_dotted_double": (255, 255, 0),
    "lane_solid": (255, 255, 255),
    "lane_dotted": (255, 0, 0),
    "lane_dashed": (255, 0, 0),
    "lane_other": (0, 255, 0),
    "roadedge_roadside": (0, 0, 255),
    "lane_lane": (255, 255, 255),
}
instance_colors = np.random.rand(256, 3) * 0.75 + 0.25
instance_colors = (instance_colors * 256).astype(np.uint8).tolist()


def put_text_on_image(
    img,
    info: List[Dict[str, float]],
    start_x=10,
    start_y=20,
    vertical_margin=20,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.5,
    thickness=1,
    colors=None,
):
    """Put text on image."""
    if not colors:
        colors = [(255, 255, 255) for _ in range(len(info))]
    for idx, line in enumerate(info):
        _str = ""
        for k, v in line.items():
            if isinstance(v, float):
                _str += f" {k}: {v:.3f}"
            else:
                _str += f" {k}: {v}"
        pos = (int(start_x), int(start_y + vertical_margin * idx))
        cv2.putText(
            img,
            _str,
            pos,
            fontFace=fontFace,
            fontScale=fontScale,
            color=colors[idx],
            thickness=thickness,
        )


def draw_grid(
    img,
    pxstep_x=8,
    pxstep_y=8,
    line_color=(0, 255, 0),
    thickness=1,
    type_=cv2.LINE_AA,
):
    """Draw grid on image."""
    x = pxstep_x - 1
    y = pxstep_y - 1
    while x < img.shape[1]:
        cv2.line(
            img,
            (x, 0),
            (x, img.shape[0]),
            color=line_color,
            lineType=type_,
            thickness=thickness,
        )
        x += pxstep_x

    while y < img.shape[0]:
        cv2.line(
            img,
            (0, y),
            (img.shape[1], y),
            color=line_color,
            lineType=type_,
            thickness=thickness,
        )
        y += pxstep_y
    return img


def draw_label(
    label,
    prob=None,
    text=None,
    out_width=512,
    out_height=512,
    color_by_category=False,
    target_categorys=None,
):
    """Draw label on image."""
    if prob is None:
        prob = np.ones_like(label)
    if color_by_category:
        assert target_categorys is not None
        idx2category = {v: k for k, v in target_categorys.items()}
        idx2category.update({0: "background"})
    h, w = label.shape
    img = np.zeros((h, w, 3))
    ids = np.unique(label)
    for idx in ids:
        color = (
            category_color_dict[idx2category[idx]]
            if color_by_category
            else list(np.random.random(size=3) * 256)
        )
        mask = label == idx
        prob_ = prob[mask]
        for i in range(3):
            img[mask, i] = prob_ * color[i]
    img = img.astype(np.uint8)
    img = cv2.resize(img, (out_width, out_height))
    if text:
        fontScale = min(max(out_width / 512.0, 0.4), 1.0)
        thickness = max(int(fontScale * 2), 1)
        cv2.putText(
            img,
            text,
            (30, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontScale,
            color=(255, 255, 255),
            thickness=thickness,
        )
    return img


def draw_reg(
    reg, direction=False, prob=None, text=None, out_width=512, out_height=512
):
    """Draw reg."""
    if prob is None:
        prob = np.ones_like(reg)
    colormap = cv2.COLORMAP_HSV if direction else cv2.COLORMAP_RAINBOW
    img = cv2.applyColorMap((reg * 255).astype(np.uint8), colormap)
    img = img * np.expand_dims(prob, axis=2)
    img = img.astype(np.uint8)
    img = cv2.resize(img, (out_width, out_height))
    if text:
        fontScale = min(max(out_width / 512.0, 0.4), 1.0)
        thickness = max(int(fontScale * 2), 1)
        cv2.putText(
            img,
            text,
            (30, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontScale,
            color=(255, 255, 255),
            thickness=thickness,
        )
    return img


def draw_direction(
    sin,
    cos,
    prob=None,
    text=None,
    out_width=512,
    out_height=512,
    normalize=False,
):
    """Draw direction map.

    Args:
        sin (_type_): sin value map
        cos (_type_): cos value map
        prob (_type_, optional): equal view mask. Defaults to None.
        text (_type_, optional): descriptive text. Defaults to None.
        out_width (int, optional): Defaults to 512.
        out_height (int, optional): Defaults to 512.
        normalize (bool, optional): true, angle alpha and alpha+180 are equal.

    Returns:
       direction view color img
    """
    if prob is None:
        prob = np.ones_like(sin)
    init_h, init_w = sin.shape
    ang = np.degrees(np.arctan2(sin, cos)) + 180.0
    if normalize:
        ang = ang % 180 * 2
    hsv = np.zeros((init_h, init_w, 3))
    hsv[..., 0] = ang
    hsv[..., 1] = 255.0
    hsv[..., 2] = 255.0
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = img * np.expand_dims(prob, axis=2)
    img = img.astype(np.uint8)
    img = cv2.resize(img, (out_width, out_height))
    if text:
        fontScale = min(max(out_width / 512.0, 0.4), 1.0)
        thickness = max(int(fontScale * 2), 1)
        cv2.putText(
            img,
            text,
            (30, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontScale,
            color=(255, 255, 255),
            thickness=thickness,
        )
    return img


def draw_online_mapping(
    online_mapping,
    bev_h,
    bev_w,
    mat_vcs2bev,
    text=None,
    draw_polyline_crosswalk=False,
    draw_polyline=False,
    draw_pt_crosswalk=False,
    draw_pt=False,
    draw_line_crosswalk=False,
    draw_line=False,
    draw_segment=False,
    draw_idx=False,
    draw_start_end_pt=False,
    thickness=1,
    pt_thickness=None,
    img_bev=None,
    color_by_category=False,
    color_table=None,
    res_h=1.6,
    res_w=1.6,
):
    """Draw online mapping."""
    delta_t = max(res_h, res_w) / 4.0
    if color_table is None:
        color_table = category_color_dict
    pt_thickness = thickness * 2 if pt_thickness is None else pt_thickness
    if img_bev is None:
        img_bev = np.zeros((bev_h, bev_w, 3))
    else:
        h, w, _ = img_bev.shape
        assert h == bev_h and w == bev_w
    # set fontscale according to the img size
    font_scale = min(max(bev_w / 512.0, 0.4), 1.0)
    if text:
        cv2.putText(
            img_bev,
            text,
            (30, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=2,
        )

    bev_ego_pts = (mat_vcs2bev @ np.array([[0], [0], [1]])).transpose()
    bev_ego_pts[:, 0] = bev_ego_pts[:, 0] / bev_ego_pts[:, 2]
    bev_ego_pts[:, 1] = bev_ego_pts[:, 1] / bev_ego_pts[:, 2]
    bev_ego_pts = bev_ego_pts[0, :2]
    cv2.rectangle(
        img_bev,
        (int(bev_ego_pts[0]) - 5, int(bev_ego_pts[1]) - 10),
        (int(bev_ego_pts[0]) + 5, int(bev_ego_pts[1]) + 10),
        color=(0, 255, 0),
        thickness=thickness * 2,
    )
    for category, lanes in online_mapping.items():
        for idx, lane in enumerate(lanes):
            color = (
                color_table[category]
                if color_by_category
                else list(np.random.random(size=3) * 256)
            )
            if isinstance(lane, tuple):
                color = instance_colors[lane[1] % 256]
                lane = lane[0]
            bev_pts = get_bev_pts(lane[:, :2], mat_vcs2bev)
            if draw_idx:
                img_bev = cv2.putText(
                    img_bev,
                    str(idx),
                    (int(bev_pts[0, 0]), int(bev_pts[0, 1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4 * font_scale,
                    color=color,
                    thickness=1,
                )
                img_bev = cv2.putText(
                    img_bev,
                    str(idx),
                    (int(bev_pts[-1, 0]), int(bev_pts[-1, 1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4 * font_scale,
                    color=color,
                    thickness=1,
                )
            if draw_polyline and (
                category != "crosswalks" or draw_polyline_crosswalk
            ):
                img_bev = cv2.polylines(
                    img_bev,
                    np.int32([bev_pts]),
                    isClosed=False,
                    color=color,
                    thickness=thickness,
                )
            if draw_line and (category != "crosswalks" or draw_line_crosswalk):
                bev_pts_last = bev_pts[:-1]
                bev_pts_next = bev_pts[1:]
                for bev_pt_last, bev_pt_next in zip(
                    bev_pts_last, bev_pts_next
                ):
                    img_bev = cv2.line(
                        img_bev,
                        (int(bev_pt_last[0]), int(bev_pt_last[1])),
                        (int(bev_pt_next[0]), int(bev_pt_next[1])),
                        color,
                        thickness=thickness * 2,
                    )
            if draw_pt or (category == "crosswalks" and draw_pt_crosswalk):
                for pt in bev_pts:
                    cv2.circle(
                        img_bev,
                        (int(round(pt[0])), int(round(pt[1]))),
                        radius=1,
                        color=color,
                        thickness=pt_thickness,
                    )
            if draw_start_end_pt:
                cv2.circle(
                    img_bev,
                    (int(round(bev_pts[0][0])), int(round(bev_pts[0][1]))),
                    radius=3,
                    color=(0, 255, 255),
                    thickness=thickness * 2,
                )  # start point yellow
                cv2.circle(
                    img_bev,
                    (int(round(bev_pts[-1][0])), int(round(bev_pts[-1][1]))),
                    radius=3,
                    color=(255, 0, 0),
                    thickness=thickness * 2,
                )  # end point blue
            if draw_segment:
                vcs_pts_plus = np.hstack(
                    [
                        lane[:, 0:1] - delta_t * lane[:, 6:7],
                        lane[:, 1:2] + delta_t * lane[:, 7:8],
                    ]
                )
                vcs_pts_minus = np.hstack(
                    [
                        lane[:, 0:1] + delta_t * lane[:, 6:7],
                        lane[:, 1:2] - delta_t * lane[:, 7:8],
                    ]
                )
                bev_pts_plus = get_bev_pts(vcs_pts_plus, mat_vcs2bev)
                bev_pts_minus = get_bev_pts(vcs_pts_minus, mat_vcs2bev)
                for pt_plus, pt_minus in zip(bev_pts_plus, bev_pts_minus):
                    img_bev = cv2.line(
                        img_bev,
                        (int(pt_plus[0] + 0.5), int(pt_plus[1] + 0.5)),
                        (int(pt_minus[0] + 0.5), int(pt_minus[1] + 0.5)),
                        color,
                        thickness=2,
                    )
    return img_bev


def get_bev_pts(lane, mat_vcs2bev):
    """Get bev points."""
    lane = np.array([[pt[0], pt[1], 1] for pt in lane]).transpose()
    bev_pts = (mat_vcs2bev @ lane).transpose()
    bev_pts[:, 0] = bev_pts[:, 0] / bev_pts[:, 2]
    bev_pts[:, 1] = bev_pts[:, 1] / bev_pts[:, 2]
    bev_pts = bev_pts[:, :2]
    return bev_pts


def draw_ego(img_bev, mat_vcs2bev, dx=0, dy=0, thickness=1):
    """Draw ego car on bev img.

    Args:
        lane: lane pts
        mat_vcs2bev: transformation matrix
        dx, dy: the offset of ploted point
        thickness: line thickness
    """
    bev_ego_pts = (mat_vcs2bev @ np.array([[0], [0], [1]])).transpose()
    bev_ego_pts[:, 0] = bev_ego_pts[:, 0] / bev_ego_pts[:, 2]
    bev_ego_pts[:, 1] = bev_ego_pts[:, 1] / bev_ego_pts[:, 2]
    bev_ego_pts = bev_ego_pts[0, :2]
    cv2.rectangle(
        img_bev,
        (int(bev_ego_pts[0]) - 5 + dx, int(bev_ego_pts[1]) - 10 + dy),
        (int(bev_ego_pts[0]) + 5 + dx, int(bev_ego_pts[1]) + 10 + dy),
        color=(0, 255, 0),
        thickness=thickness,
    )
    return img_bev


def trans_lane(lane, mat_vcs2bev):
    """Transform vcs lane to bev grids.

    Args:
        lane: lane pts
        mat_vcs2bev: transformation matrix
    """
    lane = np.array([[pt[0], pt[1], 1] for pt in lane]).transpose()
    bev_pts = (mat_vcs2bev @ lane).transpose()
    bev_pts[:, 0] = bev_pts[:, 0] / bev_pts[:, 2]
    bev_pts[:, 1] = bev_pts[:, 1] / bev_pts[:, 2]
    bev_pts = bev_pts[:, :2].astype(np.int)
    return bev_pts


def arrange_imgs(img_list, col, img_h, img_w):
    """Fuse multi img into one img.

    Args:
        img_list: image data list.
        col: the number of images of each row.
        img_h: the height of target img.
        img_w: the width of target img.
    """

    num = len(img_list)
    fuse_view_imgs = []
    for i in range((num + col - 1) // col):
        row_img = np.zeros((img_h, img_w * col, 3), dtype=np.uint8)
        for j in range(col):
            index = i * col + j
            if index >= num:
                continue
            target_img = img_list[index]
            target_h, target_w = target_img.shape[:2]
            if target_h != img_h or target_w != img_w:
                target_img = cv2.resize(
                    target_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST
                )
            row_img[:, img_w * j : img_w * (j + 1), :] = target_img
        fuse_view_imgs.append(row_img)
    return np.vstack(fuse_view_imgs)


def draw_raw_img(
    image_files,
    origin_imgs,
    image_size,
    view_cols=3,
):
    """Draw raw imgs.

    Args:
        image_files: image files list.
        origin_imgs: origin image from lmdb.
        image_size: the target size of show raw img.
        view_cols: the column of view img.
    """

    tmp = image_files[0].split("/")
    car_name = tmp[-4]
    pack_name = tmp[-3]
    image_name = os.path.splitext(tmp[-1])[0]
    image_height, image_width = image_size
    all_img_orders = {
        "camera_front_left": 0,
        "camera_front": 1,
        "camera_front_right": 2,
        "camera_rear_left": 3,
        "camera_rear": 4,
        "camera_rear_right": 5,
        "fisheye_left": 6,
        "fisheye_front": 7,
        "fisheye_rear": 8,
        "fisheye_right": 9,
        "camera_front_30fov": 10,
    }
    img_list = []
    img_orders = []
    for idx, image_file in enumerate(image_files):
        if origin_imgs is not None:
            img = cv2.cvtColor(
                np.array(origin_imgs[idx], np.uint8), cv2.COLOR_RGB2BGR
            )
        else:
            continue
        img = cv2.resize(img, (image_width, image_height))
        img_type = image_file.split("/")[-2]
        if img_type not in all_img_orders:
            continue
        cv2.putText(
            img,
            img_type.replace("camera_", ""),
            (image_width // 2 - 80, image_height - 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
        )
        img_list.append(np.copy(img))
        img_orders.append(all_img_orders[img_type])

    # resort image list
    img_list = [x for _, x in sorted(zip(img_orders, img_list))]
    if img_list:
        fuse_img = arrange_imgs(img_list, view_cols, image_height, image_width)
    else:
        fuse_img = np.zeros(
            (image_height, image_width * view_cols, 3), dtype=np.uint8
        )
    fontScale = min(max(image_width / 512.0, 0.4), 1.0)
    thickness = max(int(fontScale * 2), 1)
    cv2.putText(
        fuse_img,
        f"{car_name}/{pack_name}/{image_name}",
        (30, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(0, 0, 255),
        thickness=thickness,
    )
    return fuse_img


def draw_raw_gt(
    gt_online_mapping_ori,
    bev_size,
    mat_vcs2bev,
    target_category=None,
    show_id=False,
):
    """Draw raw gt data.

    Args:
        gt_online_mapping_ori: origin gt data
        bev_size: the size of bev map
        mat_vcs2bev: transformation matrix from vcs to bev
    """
    bev_h, bev_w = bev_size
    left_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    right_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    curb_status_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    # plot self car
    left_img = draw_ego(left_img, mat_vcs2bev, thickness=3)
    right_img = draw_ego(right_img, mat_vcs2bev, thickness=3)
    curb_status_img = draw_ego(curb_status_img, mat_vcs2bev, thickness=3)
    id = 0
    for category, lanes in gt_online_mapping_ori.items():
        if category not in category_color_dict:
            continue
        if target_category is not None and category not in target_category:
            continue
        is_curb = True if "roadedge" in category else False
        for i in range(len(lanes)):
            category_color = category_color_dict[category]
            raw_pts = lanes[i]["pts"]
            lane = np.concatenate([raw_pts[:, :2], raw_pts[-1:, 3:5]], axis=0)
            instance_color = instance_colors[i % 256]
            bev_pts = trans_lane(lane, mat_vcs2bev)
            for j in range(len(lane) - 1):
                # plot left img
                start_pt = tuple(bev_pts[j])
                end_pt = tuple(bev_pts[j + 1])
                left_img = cv2.line(
                    left_img, start_pt, end_pt, category_color, 2
                )
                left_img = cv2.circle(left_img, end_pt, 3, category_color, 1)
                if category != "ignores":
                    # plot right img
                    right_img = cv2.line(
                        right_img, start_pt, end_pt, instance_color, 2
                    )
                    right_img = cv2.circle(
                        right_img, end_pt, 3, instance_color, 1
                    )
                # plot curb status
                if is_curb:
                    # use color to show curb status
                    curb_status_color = (255, 255, 255)
                    if "curb_status" in lanes[i]:
                        if lanes[i]["curb_status"] == "closed":
                            curb_status_color = (0, 0, 255)
                        elif lanes[i]["curb_status"] == "open":
                            curb_status_color = (0, 255, 0)
                    curb_status_img = cv2.line(
                        curb_status_img, start_pt, end_pt, curb_status_color, 2
                    )
                    curb_status_img = cv2.circle(
                        curb_status_img, end_pt, 3, curb_status_color, 1
                    )
                # plot lane id on right img
                id += 1
                if show_id:
                    right_img = cv2.putText(
                        right_img,
                        str(id),
                        tuple(bev_pts[len(lane) // 2]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75,
                        color=instance_color,
                        thickness=1,
                    )
            # plot start pt
            left_img = cv2.circle(
                left_img, tuple(bev_pts[0]), 4, category_color, 2
            )
            if category != "ignores":
                right_img = cv2.circle(
                    right_img, tuple(bev_pts[0]), 4, instance_color, 2
                )

    fontScale = min(max(bev_w / 512.0, 0.4), 1.0)
    thickness = max(int(fontScale * 2), 1)

    left_img = cv2.putText(
        left_img,
        "origin gt category",
        (30, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(255, 255, 255),
        thickness=thickness,
    )

    right_img = cv2.putText(
        right_img,
        "origin gt instance",
        (30, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(255, 255, 255),
        thickness=thickness,
    )

    curb_status_img = cv2.putText(
        curb_status_img,
        "origin gt curb status",
        (30, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(255, 255, 255),
        thickness=thickness,
    )

    return [left_img, right_img, curb_status_img]


def draw_dilate_mask(
    gt_stats,
    bev_size,
    mat_vcs2bev,
):
    """Draw om dilate mask.

    Args:
        gt_stats: om gt data.
        bev_size: the size of bev map.
        mat_vcs2bev: transformation matrix from vcs to bev.
    """
    bev_h, bev_w = bev_size
    view_list = []
    for group in gt_stats:
        mask = gt_stats[group]["dilate"]
        mask_c, mask_h, mask_w = mask.shape
        for c in range(mask_c):
            view_mask = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
            # target region
            target_msk = mask[c] == 2
            view_mask[:, :, 2][target_msk] = 255
            # dilated region
            dilate_msk = mask[c] == 1
            view_mask[:, :, 0][dilate_msk] = 255
            view_mask = cv2.resize(view_mask, (bev_w, bev_h))
            # plot self car
            view_mask = draw_ego(view_mask, mat_vcs2bev, thickness=3)
            # plot text
            fontScale = min(max(bev_w / 512.0, 0.4), 1.0)
            thickness = max(int(fontScale * 2), 1)
            view_mask = cv2.putText(
                view_mask,
                "{} mask {}".format(group, c),
                (30, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=fontScale,
                color=(255, 255, 255),
                thickness=thickness,
            )
            view_list.append(view_mask)

    return view_list


def draw_clip_raw_gt(
    online_mapping_gt,
    bev_size,
    mat_vcs2bev,
    target_category=None,
    show_id=False,
):
    """Draw raw gt data.

    Args:
        online_mapping_gt: origin gt data after clip.
        bev_size: the size of bev map
        mat_vcs2bev: transformation matrix from vcs to bev
    """
    bev_h, bev_w = bev_size
    fontScale = min(max(bev_w / 512.0, 0.4), 1.0)
    thickness = max(int(fontScale * 2), 1)
    left_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    right_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    # plot self car
    left_img = draw_ego(left_img, mat_vcs2bev, thickness=3)
    right_img = draw_ego(right_img, mat_vcs2bev, thickness=3)
    id = 0
    for category, instances in online_mapping_gt.items():
        if category == "ignores":
            continue
        if target_category is not None and category not in target_category:
            continue
        for i in range(len(instances)):
            raw_pts = instances[i]["pts"]
            instance_color = instance_colors[i % 256]
            bev_pts = trans_lane(raw_pts, mat_vcs2bev)
            for j in range(len(raw_pts) - 1):
                # skip out-region line segments
                category_color = category_color_dict[category]
                if raw_pts[j][-1] < 1:
                    category_color = (128, 128, 128)
                # plot left img
                start_pt = tuple(bev_pts[j])
                end_pt = tuple(bev_pts[j + 1])
                left_img = cv2.line(
                    left_img, start_pt, end_pt, category_color, 2
                )
                left_img = cv2.circle(left_img, end_pt, 3, category_color, 1)
                if raw_pts[j][-1] < 1:
                    continue
                # plot right img
                right_img = cv2.line(
                    right_img, start_pt, end_pt, instance_color, 2
                )
                right_img = cv2.circle(right_img, end_pt, 3, instance_color, 1)
                # plot lane id on right img
                id += 1
                if show_id:
                    right_img = cv2.putText(
                        right_img,
                        str(id),
                        tuple(bev_pts[len(raw_pts) // 2]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75 * fontScale,
                        color=instance_color,
                        thickness=thickness,
                    )
            # plot start pt
            category_color = category_color_dict[category]
            if raw_pts[0][-1] < 1:
                category_color = (128, 128, 128)
            left_img = cv2.circle(
                left_img, tuple(bev_pts[0]), 4, category_color, 2
            )
            right_img = cv2.circle(
                right_img, tuple(bev_pts[0]), 4, instance_color, 2
            )

    left_img = cv2.putText(
        left_img,
        "cliped gt category",
        (30, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(255, 255, 255),
        thickness=thickness,
    )

    right_img = cv2.putText(
        right_img,
        "cliped gt instance",
        (30, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(255, 255, 255),
        thickness=thickness,
    )

    return [left_img, right_img]


def draw_multi_head(lanes, head, bev_size, mat_vcs2bev, color_table):
    """Draw raw gt data of multi group head.

    Args:
        lanes: lane instances
        head: head name
        grids: grids center position values of output map
        bev_size: the size of bev map
        mat_vcs2bev: transformation matrix from vcs to bev
        color_table: used to show cls info.
    """
    bev_h, bev_w = bev_size
    left_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    right_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    # plot self car
    left_img = draw_ego(left_img, mat_vcs2bev, thickness=3)
    right_img = draw_ego(right_img, mat_vcs2bev, thickness=3)

    # plot pts
    for lane in lanes:
        bev_pts = trans_lane(lane, mat_vcs2bev)
        instance_color = list(np.random.random(size=3) * 256)
        for i in range(len(bev_pts)):
            pt = tuple(bev_pts[i])
            label = int(lane[i, -1])
            left_img = cv2.circle(left_img, pt, 2, color_table[label], 2)
            right_img = cv2.circle(right_img, pt, 2, instance_color, 2)

    fontScale = min(max(bev_w / 512.0, 0.4), 1.0)
    thickness = max(int(fontScale * 2), 1)
    left_img = cv2.putText(
        left_img,
        head + " gt category",
        (30, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(255, 255, 255),
        thickness=thickness,
    )

    right_img = cv2.putText(
        right_img,
        head + " gt instance",
        (30, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(255, 255, 255),
        thickness=thickness,
    )

    return [left_img, right_img]


def draw_single_head(
    lanes, head, label_map, bev_size, mat_vcs2bev, color_table
):
    """Draw raw gt data of single group head.

    Args:
        lanes: lane instances
        head: head name
        label_map: label gt data of current head.
        bev_size: the size of bev map
        mat_vcs2bev: transformation matrix from vcs to bev
        color_table: used to show cls info.
    """
    bev_h, bev_w = bev_size
    view_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    # plot self car
    view_img = draw_ego(view_img, mat_vcs2bev, thickness=3)

    # plot pts
    for lane in lanes:
        bev_pts = trans_lane(lane, mat_vcs2bev)
        for i in range(len(bev_pts)):
            # plot left img
            pt = tuple(bev_pts[i])
            h, w, ch = lane[i, 2:5].astype(np.int)
            label = int(label_map[ch, h, w])
            view_img = cv2.circle(view_img, pt, 2, color_table[label], 2)

    fontScale = min(max(bev_w / 512.0, 0.4), 1.0)
    thickness = max(int(fontScale * 2), 1)
    view_img = cv2.putText(
        view_img,
        head + " gt category",
        (30, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(255, 255, 255),
        thickness=thickness,
    )

    return [view_img]


def get_cls_color_table(head_infos: dict):
    """Get cls colro table.

    Args:
        head_infos: head config info.
    """

    color_table = {}
    color_array = head_infos["cls_colors"]
    num_class = max([v for _, v in head_infos["cls_remap"].items()]) + 1
    if num_class > len(color_array):
        color_array.insert(0, "gray")
    assert num_class == len(
        color_array
    ), "The cls color list must match head cls info: {}".format(head_infos)
    color_table[-1] = general_color_list["gray"]
    for i in range(num_class):
        color_table[i] = general_color_list[color_array[i]]
    return color_table


def draw_single_task(
    pred_lanes, index, sub_head, head_groups, bev_size, mat_vcs2bev, text=None
):
    """Draw pred lanes of a task.

    Args:
        pred_lanes: om instances with category info.
        index: the index of head label in a lane pt.
        head: current head name.
        head_groups: head group info.
        bev_size: the size of bev map.
        mat_vcs2bev: transformation matrix from vcs to bev.
        attr_offset: the offset of base attrs of a lane point.
        text: the text to showed.
    """
    bev_h, bev_w = bev_size
    view_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    # plot self car
    view_img = draw_ego(view_img, mat_vcs2bev)

    # get valid category
    valid_category = []
    group = head_groups[sub_head]["group"]
    for head, head_infos in head_groups.items():
        multi_head = head_infos.get("multi", False)
        if multi_head and head_infos["group"] == group:
            # get cls list
            cls_list = head_infos.get("cls_list", None)
            if cls_list is None:
                cls_list = head_infos["cls_remap"].keys()
            cls_list = list(cls_list)
            for i in range(len(cls_list)):
                cls_name = cls_list[i]
                valid_category.append(f"{head}_{cls_name}")
    # get color table
    color_table = get_cls_color_table(head_groups[sub_head])

    # plot pts
    for category, lanes in pred_lanes.items():
        if category not in valid_category:
            continue
        for lane in lanes:
            bev_pts = trans_lane(lane, mat_vcs2bev)
            for i in range(len(bev_pts)):
                # plot left img
                pt = tuple(bev_pts[i])
                label = int(lane[i, index]) - 1
                view_img = cv2.circle(view_img, pt, 2, color_table[label], 2)

    if text is not None:
        fontScale = min(max(bev_w / 512.0, 0.4), 1.0)
        thickness = max(int(fontScale * 2), 1)
        view_img = cv2.putText(
            view_img,
            text,
            (30, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontScale,
            color=(255, 255, 255),
            thickness=thickness,
        )

    return view_img


def draw_ipm_img(
    gt_online_mapping_ori,
    imgs,
    ipm_img_sizes,
    homo_offset,
    bev_size,
    mat_vcs2bev,
    target_category,
    block_warp_padding,
):
    """Draw eye birds img and visualize om gt in img.

    Args:
        gt_online_mapping_ori: origin gt data
        imgs: origin input img list
        ipm_img_sizes: ipm img size list
        homo_offset: corresponging homo-offset list
        bev_size: the size of bev map
        mat_vcs2bev: transformation matrix from vcs to bev
        target_category: given target category
        block_warp_padding: order is (left,right,up,bottom)
    """
    bev_h, bev_w = bev_size
    homo_offset = torch.tensor(homo_offset)
    homo_offset_list = []
    imgs_list = []
    imgs = imgs[0]
    for idx, homo_offset_i in enumerate(homo_offset[:6]):
        homo_offset_list.append(
            torch.from_numpy(np.expand_dims(homo_offset_i, axis=0)).float()
        )
        resize_w = int(ipm_img_sizes[idx][1])
        resize_h = int(ipm_img_sizes[idx][0])
        cur_img = imgs[idx].resize((resize_w, resize_h))  # (w, h)
        cur_img = np.array(cur_img)
        img_torch = img_array2tensor(cur_img)
        imgs_list.append(img_torch)
    left_img = visualize_ipm(
        imgs_list, homo_offset_list, block_warp_padding=block_warp_padding
    )
    left_img = cv2.resize(left_img, (bev_w, bev_h))
    left_img = left_img.astype(np.uint8)
    right_img = copy.deepcopy(left_img)
    right_img = draw_ego(right_img, mat_vcs2bev, thickness=3)
    for category, lanes in gt_online_mapping_ori.items():
        if category not in category_color_dict:
            continue
        if target_category is not None and category not in target_category:
            continue
        for i in range(len(lanes)):
            category_color = category_color_dict[category]
            raw_pts = lanes[i]["pts"]
            lane = np.concatenate([raw_pts[:, :2], raw_pts[-1:, 3:5]], axis=0)
            bev_pts = trans_lane(lane, mat_vcs2bev)
            for j in range(len(lane) - 1):
                # plot left img
                start_pt = tuple(bev_pts[j])
                end_pt = tuple(bev_pts[j + 1])
                right_img = cv2.line(
                    right_img, start_pt, end_pt, category_color, 2
                )
                right_img = cv2.circle(right_img, end_pt, 3, category_color, 1)

            # plot start pt
            right_img = cv2.circle(
                right_img, tuple(bev_pts[0]), 4, category_color, 1
            )

    fontScale = min(max(bev_w / 512.0, 0.4), 1.0)
    thickness = max(int(fontScale * 2), 1)

    left_img = cv2.putText(
        left_img,
        "ipm img",
        (30, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(255, 255, 255),
        thickness=thickness,
    )

    right_img = cv2.putText(
        right_img,
        "ipm with gt",
        (30, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=fontScale,
        color=(255, 255, 255),
        thickness=thickness,
    )
    return [left_img, right_img]


def visualize_ipm(
    img_list: Sequence,
    homo_offset_list: Sequence,
    block_warp_padding: Sequence,
):
    """Generate eye bird view image.

    Args:
        img_list: images list.
        homo_offfset_list: homo_offset list.
        block_warp_padding: block_warp_padding list.
    """

    bird_eye_view = np.zeros(
        (
            homo_offset_list[0].shape[1],
            homo_offset_list[0].shape[2],
            3,
        ),
        np.uint8,
    )

    for idx, img_torch in enumerate(img_list):
        st = SpatialTransfomerWithOffset(
            height=homo_offset_list[idx].shape[1],
            width=homo_offset_list[idx].shape[2],
            block_warp_padding=block_warp_padding[idx],
        )
        ipm = st(img_torch, homo_offset_list[idx])
        ipm = ipm.permute(0, 2, 3, 1).squeeze().numpy()
        bird_eye_view = np.maximum(bird_eye_view, ipm.astype(np.uint8))
    return bird_eye_view


def plot_pts_in_raw_img(
    img: np.array,
    gt: dict,
    chassis2cam: np.array,
    K: np.array,
    dist_coeff: np.array,
    ratio: float = 0.5,
) -> np.array:
    """
    Plot points in the raw image using ground truth (gt) data.

    Args:
        img: Raw image
        gt: Ground truth data
        chassis2cam: Chassis to camera transformation matrix
        K: Camera intrinsic matrix
        dist_coeff: Distortion coefficients
        ratio: Points retain scope in undistortion image

    Returns:
        Raw image with points plotted
    """
    R = chassis2cam[:3, :3]
    t = chassis2cam[:3, 3]
    img = img.copy()
    h = img.shape[0]
    w = img.shape[1]
    for category in gt.keys():
        lanes = gt[category]
        for _, lane in enumerate(lanes):
            pts = copy.deepcopy(lane[:, :2])
            pts_h = np.zeros((pts.shape[0], 1))
            pts = np.hstack((pts, pts_h))
            pts = pts.transpose()
            camera_pts = (np.dot(R, pts) + np.array([t]).T).T
            camera_pts = camera_pts[camera_pts[:, 2] > 0, :]
            if camera_pts.shape[0] == 0:
                continue
            rvec, _ = cv2.Rodrigues(np.identity(3, np.float32))
            tvec = np.zeros(shape=(3, 1), dtype=np.float32)
            image_pts = cv2.projectPoints(
                camera_pts,
                np.array(rvec),
                tvec,
                K,
                np.zeros((4,), dtype=np.float32),
            )[0]
            image_pts = np.squeeze(image_pts).reshape((-1, 2))
            valid_index = np.logical_and(
                np.logical_and(
                    image_pts[:, 0] > -ratio * w,
                    image_pts[:, 0] < (1 + ratio) * w,
                ),
                np.logical_and(
                    image_pts[:, 1] > -ratio * h,
                    image_pts[:, 1] < (1 + ratio) * h,
                ),
            )
            camera_pts = camera_pts[valid_index, :]
            if len(camera_pts > 1):
                image_pts = cv2.projectPoints(
                    camera_pts, np.array(rvec), tvec, K, dist_coeff
                )[0]
                image_pts = np.array(image_pts).squeeze()
                if len(image_pts.shape) < 2:
                    image_pts = np.expand_dims(image_pts, 0)
            for pt in image_pts:
                if (
                    int(round(pt[0])) < w
                    and int(round(pt[0])) > 0
                    and int(round(pt[1])) < h
                    and int(round(pt[1])) > 0
                ):
                    cv2.circle(
                        img,
                        (int(round(pt[0])), int(round(pt[1]))),
                        radius=int(round(w / 400)),
                        color=category_color_dict[category],
                        thickness=-1,
                    )
    return img
