from typing import Any, List, Sequence, Union

import numpy as np
import torch

try:
    from hatbc.message import Frame
except ImportError:
    Frame = None

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

from hat.core.box3d_utils import get_3dbox_corners
from hat.core.data_struct.base_struct import DetBoxes3D
from hat.core.utils_3d import get_3dbox_dense_points
from hat.core.virtual_camera import CameraModelType
from hat.utils.package_helper import require_packages
from hat.visualize.bbox3d import (
    draw_box_represented_by_corners,
    draw_dense_box,
)


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
    data: Union[DetBoxes3D, Frame],
    vis_configs: Any,
    bird_eye_size: List[Union[float, int]] = None,
    vcs_range: List[Union[float, int]] = None,
    additional_info: Any = None,
) -> Sequence[Union[List, np.ndarray]]:
    if bird_eye_size is None:
        bird_eye_size = vis_configs["bird_eye_size"]
    if vcs_range is None:
        vcs_range = vis_configs["vcs_range"]
    cu, cv, vf = get_virtual_params(bird_eye_size, vcs_range)
    colors_map = vis_configs["colors_map"]
    score_threshold = vis_configs.get("score_threshold", 0.0)
    category = None
    if additional_info is not None and "category" in additional_info:
        category = additional_info["category"]
    if additional_info is not None and "camera_model" in additional_info:
        camera_model = additional_info["camera_model"]
    else:
        camera_model = ["PinholeCamera"] * len(images_list)
    is_msg = False
    if isinstance(data, Frame):
        data = data.perceptions
        is_msg = True
    for obj_idx, res in enumerate(data):
        if is_msg:
            bbox3d = res.bbox3ds[0]
            bbox_type = res.attributes[0].value
            loc = bbox3d.loc.as_list()
            dim_whl = bbox3d.dim.as_list()
            dim = [dim_whl[2], dim_whl[0], dim_whl[1]]
            yaw = bbox3d.yaw
            score = bbox3d.score
        else:
            loc = res.location
            # default as [h, w, l]
            dim = res.dimension
            # reset as [l, w, h]
            dim = [float(dim[2]), float(dim[1]), float(dim[0])]
            yaw = res.yaw
            score = res.score
            bbox_type = res.cls_name

        if score is not None and score < score_threshold:
            if bbox_type != "TP":
                continue
        # rendar on bev
        points_3d = get_3dbox_corners(loc, dim, yaw)
        # convert vcs to bev
        _point = points_3d[:4, :]
        points_bev = np.zeros_like(_point)
        points_bev[:, 0] = -_point[:, 1] * vf + cu
        points_bev[:, 1] = -_point[:, 0] * vf + cv
        draw_box_represented_by_corners(
            bev_map,
            points_bev,
            color=colors_map[bbox_type],
            score=score,
            draw_arrow=True,
            font_scale=0.3,
        )

        # rendar on img
        points_dense_3d = get_3dbox_dense_points(
            loc, dim, yaw, pts_per_line=200, front_lines=True
        )
        for idx in range(len(images_list)):
            img = images_list[idx]
            img_h, img_w = img.shape[:2]
            camera_tool = CameraModelType[camera_model[idx]].value()
            calib = calib_list[idx]
            calib.update(
                {
                    "image_width": img_w,
                    "image_height": img_h,
                }
            )
            camera_tool = camera_tool.init_cam_param_by_dict(calib)

            points_3d = camera_tool.project_vcs2cam(points_dense_3d)
            is_valid = camera_tool.is_points_in_fov(points_3d)
            if np.any(is_valid):
                points_img = camera_tool.project_cam2pixel(points_3d[is_valid])
                color = colors_map[bbox_type]
                text = []
                if category is not None and isinstance(category[obj_idx], str):
                    text.append(category[obj_idx])
                if bbox_type is not None and isinstance(bbox_type, str):
                    text.append(bbox_type)
                if score is not None and isinstance(
                    score, (float, torch.Tensor, str)
                ):
                    text.append(str(round(float(score), 2)))
                draw_dense_box(
                    img,
                    points_img,
                    color,
                    "|".join(text),
                    {"font_scale": 1, "font_thickness": 1},
                )

    return images_list, bev_map
