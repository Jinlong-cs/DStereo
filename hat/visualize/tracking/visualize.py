from typing import Any, List, Sequence, Union

import numpy as np

try:
    from hatbc.message import CameraFrame, Frame
except ImportError:
    CameraFrame, Frame = None, None

from hat.core.box3d_utils import get_3dbox_corners
from hat.core.utils_3d import get_3dbox_dense_points
from hat.core.virtual_camera import CameraModelType
from hat.utils.package_helper import require_packages
from hat.visualize.bbox3d import (
    draw_box_represented_by_corners,
    draw_dense_box,
)
from hat.visualize.detection_bev.visualize import get_virtual_params


@require_packages("hatbc")
def draw_multi_camera_imgs(
    images_list: List[np.ndarray],
    bev_map: np.ndarray,
    meta_list: List[CameraFrame],
    data: Frame,
    vis_configs: Any,
    bird_eye_size: List[Union[float, int]] = None,
    vcs_range: List[Union[float, int]] = None,
) -> Sequence[Union[List, np.ndarray]]:
    if bird_eye_size is None:
        bird_eye_size = vis_configs["bird_eye_size"]
    if vcs_range is None:
        vcs_range = vis_configs["vcs_range"]
    cu, cv, vf = get_virtual_params(bird_eye_size, vcs_range)
    colors_map = vis_configs["colors_map"]
    score_threshold = vis_configs.get("score_threshold", 0.0)
    camera_model = [meta.camera_type for meta in meta_list]
    calib_list = []
    for meta in meta_list:
        camera_param = meta.camera_param
        vcs = camera_param.vcs
        calib = {
            "focal_u": camera_param.focal_u,
            "focal_v": camera_param.focal_v,
            "center_u": camera_param.center_u,
            "center_v": camera_param.center_v,
            "image_height": camera_param.image_height,
            "image_width": camera_param.image_width,
            "distort": np.expand_dims(np.array(camera_param.distort), axis=0),
            "vcs": {
                "rotation": vcs.rotation,
                "translation": vcs.translation,
            },
            "camera_x": camera_param.camera_x,
            "camera_y": camera_param.camera_y,
            "camera_z": camera_param.camera_z,
            "roll": camera_param.roll,
            "pitch": camera_param.pitch,
            "yaw": camera_param.yaw,
        }
        calib_list.append(calib)
    for inst in data.perceptions:
        loc = inst.bbox3ds[0].loc
        # default as [w, h, l]
        dim = inst.bbox3ds[0].dim
        # reset as [l, w, h]
        dim = [float(dim[2]), float(dim[0]), float(dim[1])]
        yaw = inst.bbox3ds[0].yaw
        score = inst.get_attributes(topics="tracking")[0].score
        # frames with no gt will skip matching, in which all preds are fp
        if len(inst.get_attributes(topics="asso")) == 0:
            bbox_type = "FP"
            vis_id = None
        else:
            bbox_type = inst.get_attributes(topics="asso")[0].value.get(
                "match_type", "FP"
            )
            vis_id = inst.get_attributes(topics="asso")[0].value.get("vis_id")
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
            score=score if score > 0 else None,
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
                if bbox_type is not None:
                    text.append(bbox_type)
                if vis_id is not None:
                    text.append(vis_id)
                draw_dense_box(
                    img,
                    points_img,
                    color,
                    "|".join(text),
                    {"font_scale": 1, "font_thickness": 1},
                )

    return images_list, bev_map
