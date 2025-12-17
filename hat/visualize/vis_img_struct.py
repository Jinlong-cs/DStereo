import logging
from copy import deepcopy
from dataclasses import fields
from typing import Dict, Optional

import numpy as np

from hat.core.data_struct.app_struct import DetObjects
from hat.core.data_struct.base import BaseData
from hat.core.data_struct.base_struct import (
    DetBox2D,
    DetBoxes2D,
    DetBoxes2D3D,
    DetBoxes3D,
    Lines2D,
    Mask,
    MultipleBoxes2D,
    Points2D_2,
)
from hat.core.data_struct.img_structures import ImgBase
from hat.utils.apply_func import convert_numpy
from .bbox2d import draw_text, vis_det_boxes_2d, vis_multiple_boxes_2d
from .bbox3d import blend_top_right, vis_det_boxes_3d, vis_det_boxes_3d_bev
from .kps import vis_lines_2d, vis_n_points_2d

logger = logging.getLogger(__name__)


def vis_img_struct(
    img_struct: ImgBase, data: BaseData, vis_configs, vis_image=None
):
    if vis_image is None:
        vis_image = convert_numpy(img_struct.ori_img)[:, :, ::-1].copy()
    calib = img_struct.calib
    if calib is not None:
        calib = calib.cpu().numpy()
    distCoeffs = img_struct.distCoeffs
    if distCoeffs is not None:
        distCoeffs = distCoeffs.cpu().numpy()

    if isinstance(data, DetObjects):
        vis_image = vis_det_objects(
            vis_image,
            data,
            vis_configs,
            calib=calib,
            distCoeffs=distCoeffs,
        )
    elif isinstance(data, DetBoxes2D3D):
        vis_image = vis_det_2d_3d(
            vis_image,
            data,
            vis_configs,
            calib=calib,
            distCoeffs=distCoeffs,
        )
    elif isinstance(data, Mask):
        vis_image = vis_mask(
            vis_image,
            data,
            vis_configs,
        )

    return vis_image


def vis_det_2d_3d(
    vis_image: np.ndarray,
    det_2d_3d_boxes: DetBoxes2D3D,
    vis_configs: Dict,
    calib: Optional[np.ndarray] = None,
    distCoeffs: Optional[np.ndarray] = None,
):

    bbox2d_vis_configs = deepcopy(vis_configs)
    bbox2d_vis_configs.update(vis_configs.get("box2d", {}))

    det_2d_3d_boxes = det_2d_3d_boxes.with_scores_ge(0.2)
    vis_image = vis_det_boxes_2d(
        vis_image, det_2d_3d_boxes, bbox2d_vis_configs
    )

    if calib is None or distCoeffs is None:
        logger.info("calib or distCoeffs not provided, skip 3d vis")
        return

    canvas = None

    bbox3d_vis_configs = deepcopy(vis_configs)
    bbox3d_vis_configs.update(vis_configs.get("box3d", {}))
    vis_image = vis_det_boxes_3d(
        vis_image,
        calib,
        distCoeffs,
        det_2d_3d_boxes,
        bbox3d_vis_configs,
    )

    # bev
    if bbox3d_vis_configs.get("draw_bev", True):
        canvas = vis_det_boxes_3d_bev(det_2d_3d_boxes, bbox3d_vis_configs)

    if canvas is not None:
        vis_image = blend_top_right(vis_image, canvas)

    return vis_image


def vis_det_objects(
    vis_image: np.ndarray,
    det_objects: DetObjects,
    vis_configs: Dict,
    calib: Optional[np.ndarray] = None,
    distCoeffs: Optional[np.ndarray] = None,
):

    canvas = None

    for f in fields(det_objects):
        field_val = getattr(det_objects, f.name)
        if f.type == DetBoxes2D:
            bbox2d_vis_configs = deepcopy(vis_configs)
            bbox2d_vis_configs.update(vis_configs.get("box2d", {}))
            vis_image = vis_det_boxes_2d(
                vis_image, field_val, bbox2d_vis_configs
            )

        elif f.type == MultipleBoxes2D:
            bbox2d_vis_configs = deepcopy(vis_configs)
            bbox2d_vis_configs.update(vis_configs.get("box2d", {}))
            vis_image = vis_multiple_boxes_2d(
                vis_image, field_val, bbox2d_vis_configs
            )

        elif issubclass(f.type, DetBoxes3D):
            if calib is None or distCoeffs is None:
                logger.info("calib or distCoeffs not provided, skip 3d vis")
                continue
            bbox3d_vis_configs = deepcopy(vis_configs)
            bbox3d_vis_configs.update(vis_configs.get("box3d", {}))
            vis_image = vis_det_boxes_3d(
                vis_image,
                calib,
                distCoeffs,
                field_val,
                bbox3d_vis_configs,
            )

            # bev
            if bbox3d_vis_configs.get("draw_bev", True):
                canvas = vis_det_boxes_3d_bev(field_val, bbox3d_vis_configs)

        elif f.type == Points2D_2:
            kps_2_vis_configs = deepcopy(vis_configs)
            kps_2_vis_configs.update(vis_configs.get("points2", {}))
            vis_image = vis_n_points_2d(
                vis_image,
                field_val,
                kps_2_vis_configs,
            )

        elif f.type == Lines2D:
            lines_vis_configs = deepcopy(vis_configs)
            lines_vis_configs.update(vis_configs.get("lines", {}))
            vis_image = vis_lines_2d(vis_image, field_val, lines_vis_configs)

    for det_object in iter(det_objects):
        text = ""
        box = None
        for f in fields(det_object):
            field_val = getattr(det_object, f.name)

            if type(field_val) == DetBox2D:
                assert box is None
                box = [int(x) for x in field_val.box.numpy()]

            if hasattr(field_val, "to_text"):
                add_text = field_val.to_text()
                text = "|".join([text, add_text]) if text else add_text

        assert box is not None

        draw_text(
            vis_image,
            text,
            (box[0], box[1]),
            vis_configs["color"],
            1.0,
        )

    if canvas is not None:
        vis_image = blend_top_right(vis_image, canvas)

    return vis_image


def vis_mask(vis_image, data, vis_configs):
    mask = data.mask.numpy().astype(np.uint8)
    seg_map = np.zeros_like(vis_image)
    for label in np.unique(mask):
        mask_l_mask = mask == label
        seg_map[mask_l_mask] = vis_configs["colormap"][label]
    seg_map = seg_map[:, :, ::-1]

    vis_image = vis_image.astype(float) * vis_configs[
        "alpha"
    ] + seg_map.astype(float) * (1 - vis_configs["alpha"])

    return vis_image.astype(np.uint8)
