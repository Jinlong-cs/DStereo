import cv2
import numpy as np

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

from hat.core.box3d_utils import (
    camera2image_pinhole,
    compute_box_3d,
    project_3d_to_bird,
    project_to_image,
)
from hat.utils.package_helper import require_packages
from hat.visualize.bbox3d import draw_box_3d


def _score_filter(results, viz_cfg):
    if "score_threshold" in viz_cfg and len(results) > 0:
        score_thresh = viz_cfg["score_threshold"]
        if "score" in results[0]:
            keep = []
            for _, res in enumerate(results):
                class_id = res["class"]
                score = res["score"]
                t = (
                    score_thresh
                    if isinstance(score_thresh, float)
                    else score_thresh[class_id]
                )
                if score >= t:
                    keep += [res]
            results = keep
    return results


def _depth_filter(results, viz_cfg):
    if "dep_threshold" in viz_cfg and len(results) > 0:
        dep_thresh = viz_cfg["dep_threshold"]
        if "dep" in results[0]:
            results = [
                res
                for res in results
                if res["dep"] >= 0 and res["dep"] <= dep_thresh
            ]  # noqa
    return results


def draw_bbox_text(
    img,
    text,
    bbox,
    color=(255, 255, 255),  # noqa [B006]
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.75,
    thickness=1,
):  # noqa
    text_size, baseline = cv2.getTextSize(text, font_face, scale, thickness)
    font_height = text_size[1]
    pos = [int(bbox[[0, 2]].min()), int(bbox[[1, 3]].min()) - font_height]
    pos[0] = np.clip(pos[0], 15, img.shape[1] - 15)
    pos[1] = np.clip(pos[1], 15, img.shape[0] - 15)
    # if np.any(np.array(pos) < 0):
    #     return
    img = cv2.putText(
        img, text, tuple(pos), font_face, scale, color, thickness
    )
    return img


def draw_bbox3d(
    img,
    dets,
    colors_map,
    calib,
    dist_coeff=None,
    class_names=None,
    thickness=2,
    truncation_thresh=0.3,
    fisheye=False,
    min_depth_dist=0.8,
    draw_truncation_2d=True,
    camera=None,
):

    for det in dets:
        dim, loc, roty = (
            det["dimensions"],
            det["location"],
            det["rotation_y"],
        )  # noqa
        cid = det["class"]
        if "eval_type" in det.keys():
            color = colors_map[det["eval_type"]]
        else:
            color = colors_map["ignore"] if det["ignore"] else colors_map["gt"]
        pts_3d = compute_box_3d(dim, loc, roty)

        if camera is not None:
            pts_2d = camera.project_cam2pixel(pts_3d)
        else:
            if fisheye:
                pts_2d = project_to_image(
                    img.shape[:2][::-1],
                    pts_3d,
                    calib,
                    dist_coeff=dist_coeff,
                    fisheye=fisheye,
                )

            else:
                pts_2d = camera2image_pinhole(
                    pts_3d, calib, dist_coeff, img.shape[1], img.shape[0]
                )

        label_text = []
        if "score" in det:
            label_text += ["{:.02f}".format(det["score"])]
        if class_names:
            label_text += [class_names[cid]]
        label_text = "|".join(label_text)

        if pts_2d is not None:
            pts_2d = pts_2d.astype(np.int32)
            draw_box_3d(img, pts_2d, color, thickness=thickness)
            bbox2d = np.array(
                [
                    pts_2d[:, 0].min(),
                    pts_2d[:, 1].min(),
                    pts_2d[:, 0].max(),
                    pts_2d[:, 1].max(),
                ]
            )
            draw_bbox_text(img, label_text, bbox2d, color)
    return img


def draw_bbox3d_with_filter(
    img, results, viz_cfg, meta, apply_dist_coeff, camera=None
):
    results = _score_filter(results, viz_cfg)
    results = _depth_filter(results, viz_cfg)
    dist_coeffs = meta["distCoeffs"] if apply_dist_coeff else None
    img = draw_bbox3d(
        img,
        results,
        viz_cfg["colors_map"],
        meta["calib"],
        dist_coeffs,
        viz_cfg["class_names"],
        viz_cfg["thickness"],
        draw_truncation_2d=viz_cfg["draw_truncation_2d"],
        camera=camera,
    )
    return img


@require_packages("pycocotools")
def draw_mask(img, ignore_mask):
    if ignore_mask is None:
        return img
    ignore_mask = coco_mask.decode(ignore_mask).astype(np.uint8)
    ignore_mask = ignore_mask.astype(np.uint8)[:, :]
    mask = np.where(ignore_mask > 0)
    img[mask] = img[mask] * 0.7 + 0.3 * np.array([255, 0, 0])

    return img


def draw_bird_eye_view(
    img,
    dets,
    colors_map,
    class_names=None,
    out_size=384,
    world_size=64,
    blend_factor=0.8,
):
    bev = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    for det in dets:
        dim, loc, roty = det["dimensions"], det["location"], det["rotation_y"]
        if "eval_type" in det.keys():
            color = colors_map[det["eval_type"]]
        else:
            # det is gt.
            color = colors_map["ignore"] if det["ignore"] else colors_map["gt"]
        rect = compute_box_3d(dim, loc, roty)[:4, [0, 2]]
        rect = project_3d_to_bird(rect, out_size, world_size)

        cv2.polylines(
            bev,
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
            bev, (p1[0], p1[1]), (p3[0], p3[1]), color, 1, lineType=cv2.LINE_AA
        )

    height, width = bev.shape[:2]
    img[:height, -width:] = (
        img[:height, -width:] * (1 - blend_factor) + bev * blend_factor
    )
    img = img.astype(np.uint8)
    return img


def init_bev():
    world_width = 100
    bev = np.zeros((1280, 1280, 3), dtype=np.uint8)
    bev_pixel_meter = bev.shape[1] / world_width
    # draw ego car
    center_bev = (bev.shape[1] // 2, bev.shape[0] // 2)
    self_lt = (
        int((center_bev[0] - 1.8 / 2 * bev_pixel_meter)),
        int((center_bev[0] - 4 / 2 * bev_pixel_meter)),
    )  # noqa
    self_rb = (
        int((center_bev[0] + 1.8 / 2 * bev_pixel_meter)),
        int((center_bev[0] + 4 / 2 * bev_pixel_meter)),
    )  # noqa
    front_arrow = (center_bev[1], int(center_bev[0] - 4 * bev_pixel_meter))
    cv2.rectangle(bev, self_lt, self_rb, (0, 255, 0), 2)
    cv2.line(bev, center_bev, front_arrow, (0, 255, 0), 2)
    # draw circle
    for meter in range(0, world_width, 10):
        color = (200, 200, 200)
        bev = cv2.circle(
            bev,
            (int(bev.shape[1] // 2), bev.shape[0] // 2),
            int(bev_pixel_meter * meter),
            color,
            1,
        )  # noqa
    # draw circle
    color = (255, 255, 255)
    for split_line in [2, 4]:
        x = bev.shape[0] // 2 + int(bev_pixel_meter * split_line)
        bev = cv2.line(bev, (x, 0), (x, bev.shape[1]), color, 1)
        x = bev.shape[0] // 2 - int(bev_pixel_meter * split_line)
        bev = cv2.line(bev, (x, 0), (x, bev.shape[1]), color, 1)
    return bev


def add_bev(
    bbox_bev,
    bev,
    score,
    out_size=[200, 400],  # noqa [B006]
    world_width=240,
    color=[0, 0, 255],  # noqa [B006]
    thresh=0.4,
    within_fov=True,
    rescale=True,
    show_arrow=True,
):
    if rescale:
        out_w = out_size[1]
        out_h = out_size[0]
        world_w = world_width
        world_h = float(world_w) / out_w * out_h
        bbox_bev[:, 1] += world_h / 2
        bbox_bev[:, 0] = int(world_w * 0.5) - bbox_bev[:, 0]
        bbox_bev[:, 1] *= out_h / world_h
        bbox_bev[:, 0] *= out_w / world_w
        bbox_bev[:, 1] = out_h - bbox_bev[:, 1]
        bbox_bev[:, 0] = out_w - bbox_bev[:, 0]

    if score < thresh:
        return
    lc = color
    if not within_fov:
        lc = [255, 255, 255]
    bbox_bev = bbox_bev.astype(np.int32)
    cv2.polylines(bev, [bbox_bev], True, lc, 5, lineType=cv2.LINE_AA)
    if show_arrow:
        p1 = (
            bbox_bev[0, :] + bbox_bev[1, :] + bbox_bev[2, :] + bbox_bev[3, :]
        ) / 4
        p2 = (bbox_bev[0, :] + bbox_bev[1, :]) / 2
        p3 = p2 + (p2 - p1) * 0.5
        cv2.line(
            bev,
            (p1[0].astype(int), p1[1].astype(int)),
            (p3[0].astype(int), p3[1].astype(int)),
            color,
            2,
            lineType=cv2.LINE_AA,
        )


def draw_bird_eye_view_lidar(
    img,
    dets,
    colors_map,
    class_names=None,
    out_size=384,
    world_size=64,
    blend_factor=0.8,
):
    bev = init_bev()
    for det in dets:
        if "eval_type" in det.keys():
            color = colors_map[det["eval_type"]]
        else:
            # det is gt.
            color = colors_map["ignore"] if det["ignore"] else colors_map["gt"]
        bev_bbox = np.array(det["bev_bbox"])
        bev_x = bev_bbox[:, 0].copy()
        bev_bbox[:, 0] = -bev_bbox[:, 1]
        bev_bbox[:, 1] = bev_x
        add_bev(
            bev_bbox, bev, 1.0, bev.shape[:2], world_width=100, color=color
        )  # noqa
    bev = cv2.resize(bev, (img.shape[0], img.shape[0]))
    img = img.astype(np.uint8)
    new_img = np.hstack((img, bev))
    return new_img


def draw_bird_eye_view_with_filter(img, results, viz_cfg, meta):
    results = _score_filter(results, viz_cfg)
    results = _depth_filter(results, viz_cfg)
    if len(results) > 0:
        if "bev_bbox" not in results[0].keys():
            img = draw_bird_eye_view(
                img,
                results,
                viz_cfg["colors_map"],
                viz_cfg["class_names"],
                viz_cfg["out_size"],
                viz_cfg["world_size"],
            )
        else:
            img = draw_bird_eye_view_lidar(
                img,
                results,
                viz_cfg["colors_map"],
                viz_cfg["class_names"],
                viz_cfg["out_size"],
                viz_cfg["world_size"],
            )
    return img
