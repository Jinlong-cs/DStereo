# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import warnings
from dataclasses import fields
from typing import Union

import cv2
import numpy as np
import torch

from hat.core.data_struct.app_struct import DetObjects
from hat.core.data_struct.base_struct import ClsLabel, DetBox2D, Mask
from hat.core.proj_spec.detection import classname2id
from hat.utils.apply_func import convert_numpy
from hat.visualize.bbox2d import draw_bbox, draw_text

__all__ = ["ObjectRender"]

logger = logging.getLogger(__name__)


class ObjectRender:
    """Render model outputs for multitask model results in CloudModel."""

    @staticmethod
    def draw_seg(
        img: Union[torch.Tensor, np.ndarray],
        pred_seg: Union[torch.Tensor, np.ndarray],
        *,
        colormap: np.ndarray,
        void_index: int = -1,
        alpha: float = 0.5,
    ):
        """Apply masks on image."""
        if isinstance(pred_seg, Mask):
            pred_seg = pred_seg.mask
        pred_seg = convert_numpy(pred_seg)
        img = convert_numpy(img)

        pred_seg = pred_seg % len(colormap)
        ret = colormap[pred_seg]

        valid_flag = pred_seg != void_index
        img[valid_flag] = (
            alpha * img[valid_flag] + (1 - alpha) * ret[valid_flag]
        )

        return img.astype(np.uint8)

    @staticmethod
    def draw_mask2d_objects(
        img: Union[torch.Tensor, np.ndarray],
        det_objects: DetObjects,
        *,
        colormap: np.ndarray,
    ):
        """Apply masks on instance."""
        img_h, img_w = img.shape[:2]
        img = convert_numpy(img)
        for _idx, det_object in enumerate(det_objects):
            mask = None
            ins_info = []

            for a in fields(det_object):
                if a.type == Mask:
                    mask = getattr(det_object, a.name).mask
                elif a.type == ClsLabel:
                    ins_info.append(f"{getattr(det_object, a.name).cls_name}")
                else:
                    warnings.warn(f"not implemented {a.type}")
            assert mask is not None, "not found Mask field"

            color = colormap[_idx % len(colormap)]
            # draw mask
            ObjectRender.draw_seg(
                img,
                mask.int(),
                colormap=np.vstack((np.zeros(3), color)),
                void_index=0,
            )
            # draw attributes
            if len(ins_info) > 0:
                y, x = torch.nonzero(mask).float().mean(0)
                y = max(min(img_h - 100, int(y)), 15)
                x = max(min(int(x), img_w - 100), 2)
            for ins_info_i in ins_info:
                draw_text(img, ins_info_i, (x, y), color.tolist(), 2)
                y += 25
        return img

    @staticmethod
    def draw_box2d_objects(
        img: Union[torch.Tensor, np.ndarray],
        det_objects: DetObjects,
        *,
        colormap: np.ndarray,
        score_thresh: float = 0.3,
    ):
        """Apply rect bbox on instance."""
        img = convert_numpy(img)
        for det_object in det_objects:
            bbox = None
            ins_info = []

            for a in fields(det_object):
                if a.type == DetBox2D:
                    bbox = getattr(det_object, a.name)
                elif a.type == ClsLabel:
                    ins_info.append(f"{getattr(det_object, a.name).cls_name}")
                else:
                    warnings.warn(f"not implemented {a.type}")
            assert bbox is not None, "not found DetBox2D field"

            # filter with score threshold
            score = bbox.score
            if score < score_thresh:
                continue

            xmin, ymin, xmax, ymax = [int(x) for x in bbox.box]
            classname = bbox.cls_name

            color_idx = classname2id.get(classname, int(bbox.cls_idx))
            bcolor = colormap[color_idx].tolist()

            # draw 2d bbox
            img = draw_bbox(img, [xmin, ymin, xmax, ymax], bcolor, 2)
            # draw classname & score
            y = ymin - 10 if ymin - 10 > 10 else ymin + 10
            draw_text(img, f"{classname} {score:.2f}", (xmin, y), bcolor, 2)
            # draw other attributes
            for ins_info_i in ins_info:
                y -= 10
                draw_text(img, ins_info_i, (xmin, y), bcolor, 1.0)

        return img

    @staticmethod
    def draw_polygon2d_objects(
        img: Union[torch.Tensor, np.ndarray],
        det_objects: dict,
        *,
        colormap: np.ndarray,
        score_thresh: float = 0.3,
    ):
        """Apply polygon on instance."""
        img = convert_numpy(img)
        assert "polygons" in det_objects
        polygons = det_objects["polygons"].tolist()
        scores = det_objects["scores"].tolist()
        cls_idxs = det_objects["cls_idxs"].tolist()
        classname = det_objects["cls_name"]
        for polygon, score, cls_idx in zip(polygons, scores, cls_idxs):

            # filter with score threshold
            if score < score_thresh:
                continue

            xmin, ymin = int(polygon[0]), int(polygon[1])

            color_idx = classname2id.get(classname, int(cls_idx))
            bcolor = colormap[color_idx].tolist()

            # draw 2d polygon
            pts = np.array(
                [list(polygon[i : i + 2]) for i in range(0, len(polygon), 2)],
                dtype=np.int32,
            )
            pts = pts.reshape((-1, 1, 2))
            img = cv2.polylines(img, [pts], 1, bcolor, 2)

            # draw classname & score
            y = ymin - 10 if ymin - 10 > 10 else ymin + 10
            draw_text(img, f"{classname} {score:.2f}", (xmin, y), bcolor, 2)

        return img
