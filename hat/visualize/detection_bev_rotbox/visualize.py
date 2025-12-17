from typing import Any, List, Sequence, Union

import cv2
import numpy as np

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

from hat.callbacks.task_visualize.bev_discrete_obj import vis_disc_obj
from hat.core.data_struct.base_struct import DetRotBoxes2D
from hat.core.virtual_camera import CameraModelType
from hat.utils.package_helper import require_packages


@require_packages("pycocotools")
def draw_mask(img: np.ndarray, ignore_mask: Any) -> np.ndarray:
    if ignore_mask is None:
        return img
    ignore_mask = coco_mask.decode(ignore_mask).astype(np.uint8)
    ignore_mask = ignore_mask.astype(np.uint8)[:, :]
    mask = np.where(ignore_mask > 0)
    img[mask] = img[mask] * 0.7 + 0.3 * np.array([255, 0, 0])

    return img


def get_virtual_params(
    bev_size: Sequence[Union[int, float]],
    vcs_range: Sequence[Union[int, float]],
) -> Sequence[float]:
    x1, y1, x2, y2 = vcs_range
    vf = bev_size[0] / (x2 - x1 + 5)
    cv = bev_size[0] - abs(int(vf * x1))
    cu = bev_size[1] // 2
    return cu, cv, vf


def draw_multi_camera_imgs(
    images_list: List[np.ndarray],
    bev_map: np.ndarray,
    calib_list: List[dict],
    data: DetRotBoxes2D,
    vis_configs: Any,
    vis_ipm_scale: float,
    bev_size: List[Union[float, int]] = None,
    vcs_range: List[Union[float, int]] = None,
    additional_info: Any = None,
) -> Sequence[Union[List, np.ndarray]]:
    # if vis_bird_eye_size is None:
    # vis_bird_eye_size = vis_configs["vis_bird_eye_size"]
    if bev_size is None:
        bev_size = vis_configs["bev_size"]
    if vcs_range is None:
        vcs_range = vis_configs["vcs_range"]
    # vis_img_scale = vis_bird_eye_size[0] /  bev_size[0]
    m_perpixel = (
        (vcs_range[2] - vcs_range[0]) / bev_size[0],
        (vcs_range[3] - vcs_range[1]) / bev_size[1],
    )
    m_perpixel = [m_p / vis_ipm_scale for m_p in m_perpixel]
    # resize_size = [
    #     int(bev_map.shape[1] * vis_img_scale),
    #     int(bev_map.shape[0] * vis_img_scale),
    # ]
    # bev_map = cv2.resize(bev_map, resize_size)
    colors_map = vis_configs["colors_map"]
    score_threshold = vis_configs.get("score_threshold", 0.0)
    if additional_info is not None and "camera_model" in additional_info:
        camera_model = additional_info["camera_model"]
    else:
        camera_model = ["PinholeCamera"] * len(images_list)
    for obj_idx, _ in enumerate(data):
        loc = data.locations[obj_idx]  # [cx, cy]
        dim = data.dimensions[obj_idx]  # [w, h]
        yaw = data.yaw[obj_idx]
        score = data.scores
        vcs_points = data.vcs_points[obj_idx]
        if score is not None:
            score = score[obj_idx]
        bbox_type = data._cls_name_mapping[int(data.cls_idxs[obj_idx])]
        if score is not None and score < score_threshold:
            if bbox_type != "TP":
                continue
        # rendar on bev
        vis_disc_obj(
            bev_map,
            loc,
            dim,
            yaw,
            vcs_range,
            m_perpixel,
            color=colors_map[bbox_type],
            line_thicks=1,
        )
        # rendar on img
        for idx in range(len(images_list)):
            img = images_list[idx]
            img_h, img_w = img.shape[:2]
            camera_tool = CameraModelType[camera_model[idx]].value()
            calib = calib_list[idx]
            scale = calib["image_height"] // img_h
            if "K" in calib.keys():
                calib["K"][0][0] /= scale
                calib["K"][1][1] /= scale
                calib["K"][0][2] /= scale
                calib["K"][1][2] /= scale

            calib["center_u"] /= scale
            calib["center_v"] /= scale
            calib["image_height"] /= scale
            calib["image_width"] /= scale
            calib["focal_u"] /= scale
            calib["focal_v"] /= scale
            camera_tool = camera_tool.init_cam_param_by_dict(calib)
            cl_vcs = []
            for pnt_vcs in vcs_points:  # for vcs
                p = [pnt_vcs[0], pnt_vcs[1], 0, 1.0]  # for vcs
                cl_vcs.append(p)
            pnts_pixel = camera_tool.project_vcs2pixel(cl_vcs)
            flag = camera_tool.is_points_in_fov(
                camera_tool.project_vcs2cam(cl_vcs), axis=2
            )
            pnts_pixel_in_fov = pnts_pixel[flag]
            pnts_pixel_in_fov = [
                [int(xy) for xy in p] for p in pnts_pixel_in_fov
            ]
            for p_pixel in pnts_pixel_in_fov:
                cv2.circle(img, tuple(p_pixel), 3, colors_map[bbox_type], -1)
            images_list[idx] = img
    return images_list, bev_map
