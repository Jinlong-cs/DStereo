# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import warnings

import numpy as np
from easydict import EasyDict

from hat.registry import OBJECT_REGISTRY
from .det_seg_2d_anno_dataset import DetSeg2DAnnoDatasetToDetFormat

logger = logging.getLogger(__name__)


__all__ = ["TrafficLens2DToDetFormat"]


@OBJECT_REGISTRY.register
class TrafficLens2DToDetFormat(DetSeg2DAnnoDatasetToDetFormat):
    def __call__(self, data):

        assert "img" in data.keys()
        assert "anno" in data.keys()

        img = data["img"]
        anno = data["anno"]

        def _is_selected_class_id(class_id):
            if self.classidmap is None:
                return True
            return class_id in self.classidmap.keys()

        def _remap_class_id(class_id):
            if self.classidmap is None:
                return class_id
            return self.classidmap[class_id]

        def _get_gt_crop(inst):
            lt = inst.points_data[self.lt_point_id]
            rb = inst.points_data[self.rb_point_id]
            class_id = inst.class_id[0]
            if _is_selected_class_id(class_id):
                class_id = _remap_class_id(class_id)
            else:
                return None
            hard_flag = inst.is_hard[0]
            if hard_flag in [1, True, "1", "True"]:
                class_id *= -1
            gt_cropes_i = [lt[0], lt[1], rb[0], rb[1], class_id]
            gt_cropes_i_wh = self._get_bbox_wh(gt_cropes_i)
            if (
                gt_cropes_i_wh[0] < self.min_edge_size
                or gt_cropes_i_wh[1] < self.min_edge_size
            ):
                msg = (
                    "Ignore gt_cropes %s since its min edge size is invalid..."
                    % gt_cropes_i
                )  # noqa
                warnings.warn(msg)
                return None
            return gt_cropes_i

        def _get_gt_box(inst):
            lt = inst.points_data[self.lt_point_id + 10]
            rb = inst.points_data[self.rb_point_id + 10]
            attribute = inst.attribute
            if len(attribute) != 2:
                return None
            else:
                category, color = attribute
                # The background class is 0, label+=1.
                gt_lens_category = -1 if category == -1 else category + 1
                gt_lens_color = -1 if color == -1 else color + 1
            hard_flag = inst.is_hard[0]
            class_id = inst_i.class_id[0]
            if _is_selected_class_id(class_id):
                class_id = _remap_class_id(class_id)
            if hard_flag in [1, True, "1", "True"]:
                class_id *= -1
            gt_box = [
                lt[0],
                lt[1],
                rb[0],
                rb[1],
                class_id,
                gt_lens_category,
                gt_lens_color,
            ]
            gt_bbox_wh = self._get_bbox_wh(gt_box)
            if (
                gt_bbox_wh[0] < self.min_edge_size
                or gt_bbox_wh[1] < self.min_edge_size
            ):
                msg = (
                    "Ignore gt_cropes %s since its min edge size is invalid..."
                    % gt_bbox_wh
                )  # noqa
                warnings.warn(msg)
                return None
            return gt_box

        gt_boxes = []
        gt_cropes = []
        for inst_i in anno["instances"]:
            if isinstance(inst_i, dict):
                inst_i = EasyDict(inst_i)
            # TODO: There are duplicate boxes.
            if _get_gt_crop(inst_i) is not None:
                gt_cropes.append(_get_gt_crop(inst_i))
            if _get_gt_box(inst_i) is not None:
                gt_boxes.append(_get_gt_box(inst_i))

        gt_cropes = (
            np.array(gt_cropes, dtype=np.float32)
            if len(gt_cropes) > 0
            else np.zeros((0, 5), dtype=np.float32)
        )
        gt_boxes = (
            np.array(gt_boxes, dtype=np.float32)
            if len(gt_boxes) > 0
            else np.zeros((0, 7), dtype=np.float32)
        )

        ig_regions = []
        for inst_i in anno["ignore_regions"]:
            if isinstance(inst_i, dict):
                inst_i = EasyDict(inst_i)

            if "left_top" in inst_i:
                lt = inst_i["left_top"]
                rb = inst_i["right_bottom"]
            else:
                lt = inst_i["contour"][0]
                rb = inst_i["contour"][1]
            class_id = inst_i.class_id[0]
            if _is_selected_class_id(class_id):
                class_id = _remap_class_id(class_id)
            else:
                continue
            ig_regions_i = [lt[0], lt[1], rb[0], rb[1], class_id]
            ig_regions_i_wh = self._get_bbox_wh(ig_regions_i)
            if ig_regions_i_wh[0] <= 0 or ig_regions_i_wh[1] <= 0:
                msg = "Ignore invalid ig_regions %s" % ig_regions_i
                warnings.warn(msg)
                continue
            ig_regions.append(ig_regions_i)

        ig_regions = (
            np.array(ig_regions, dtype=np.float32)
            if len(ig_regions) > 0
            else np.zeros((0, 5), dtype=np.float32)
        )

        return {
            "img": img,
            "gt_boxes": gt_boxes,
            "gt_cropes": gt_cropes,
            "ig_regions": ig_regions,
            "roi_list": data["roi_list"],
        }
