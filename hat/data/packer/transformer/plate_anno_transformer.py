import json
import os
import re
import warnings
from typing import Any, Dict, List

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY
from .anno_ts_utils import (
    _get_box_10_points,
    check_image_completeness,
    check_obj,
)


@OBJECT_REGISTRY.register
class RearPlateKps4ToBBox(object):
    def __init__(self, classname: str) -> None:
        self.classname = classname
        self._bbox_occlusion = [
            "full_visible",
            "occluded",
            "heavily_occluded",
            "invisible",
        ]

    def __call__(self, item: tuple) -> Any:
        if len(item) == 1 and item[0] is not None:
            image_dir, anno = item[0]
        elif len(item) == 2:
            image_dir, anno = item
        else:
            return None

        for obj in anno.get(self.classname, []):
            points = np.array(obj["data"]).reshape(4, 2)
            xmin = points[:, 0].min()
            ymin = points[:, 1].min()
            xmax = points[:, 0].max()
            ymax = points[:, 1].max()

            num_occllusion_points = 0
            bbox_ignore = "no"
            for point_attr in obj["point_attrs"]:
                if point_attr["point_label"]["occlusion"] != "full_visible":
                    num_occllusion_points += 1
                if point_attr["point_label"]["ignore"] == "yes":
                    bbox_ignore = "yes"
            num_occllusion_points = min(
                len(self._bbox_occlusion) - 1, num_occllusion_points
            )
            bbox_occlusion = self._bbox_occlusion[num_occllusion_points]

            obj["data"] = [xmin, ymin, xmax, ymax]
            obj["struct_type"] = "rect"
            obj["label_type"] = "boxes"
            bbox_attrs = {
                "ignore": bbox_ignore,
                "occlusion": bbox_occlusion,
            }
            obj["attrs"] = bbox_attrs

        return ((image_dir, anno),)


@OBJECT_REGISTRY.register
class DenseBoxSubboxDetAnnoTs(object):
    """
    Default annotation transformer.

    for subbox detection that packed in the
    densebox image record format.

    Args:
        config:
            Configure
        root_dir:
            Image root
    """

    def __init__(
        self,
        anno_config: Dict,
        root_dir: str,
        verbose: bool = True,
        skip_invalid: bool = True,
    ):
        self.verbose = verbose
        self.root_dir = root_dir
        self.config = anno_config
        self.skip_invalid = skip_invalid

    def _get_20_points(self, bbox_1, bbox_2):
        points_data = _get_box_10_points(bbox_1) + _get_box_10_points(
            bbox_2
        )  # noqa
        return points_data

    def __call__(self, item):
        if len(item) == 1 and item[0] is not None:
            image_dir, anno = item[0]
        elif len(item) == 2:
            image_dir, anno = item
        else:
            return None

        instances = []
        ignore_regions = []
        image_url = os.path.abspath(image_dir)
        if not os.path.exists(image_url):
            if self.skip_invalid:
                if self.verbose:
                    warnings.warn(
                        "WARNING: skip invalid image: %s" % (image_url)
                    )
                return None
            else:
                raise RuntimeError("No such image: %s" % (image_url))

        if not check_image_completeness(image_url):
            if self.verbose:
                warnings.warn(
                    "WARNING: skip premature end image: %s" % (image_url)
                )  # noqa
            return None

        parent_boxes = []
        for obj in anno.get(self.config["parent_box_classname"], []):
            x1, y1, x2, y2 = map(float, obj["data"])
            height = y2 - y1
            width = x2 - x1
            if height <= 0 or width <= 0:
                continue
            if check_obj(obj, self.config.get("parent_box_remove_condiction")):
                continue
            parent_boxes.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "matched": False,
                    "id": obj.get("id", ""),
                }
            )

        children_boxes = []
        for obj in anno.get(self.config["children_box_classname"], []):
            x1, y1, x2, y2 = map(float, obj["data"])
            height = y2 - y1
            width = x2 - x1
            if height <= 0 or width <= 0:
                continue
            if check_obj(
                obj, self.config.get("children_box_remove_condiction")
            ):  # noqa
                continue
            if check_obj(
                obj, self.config.get("children_box_ignore_condiction")
            ):  # noqa
                ignore_region = {
                    "left_top": [x1, y1],
                    "right_bottom": [x2, y2],
                    "class_id": [self.config["current_class_id"]],
                }
                ignore_regions.append(ignore_region)
            elif check_obj(
                obj, self.config.get("children_box_hard_condiction")
            ):  # noqa
                children_boxes.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "hard": True,
                        "matched": False,
                        "id": obj.get("id", ""),
                    }
                )
            elif check_obj(
                obj, self.config.get("children_box_positive_condiction")
            ):
                children_boxes.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "hard": False,
                        "matched": False,
                        "id": obj.get("id", ""),
                    }
                )
            else:
                warnings.warn(
                    "WARNING, not matched obj: %s" % (json.dumps(obj))
                )  # noqa

        if self.config["match_mode"] == "matching_with_overlaps":
            if len(parent_boxes) and len(children_boxes):
                matched_results = matching_with_overlaps(
                    list(map(lambda x: x["bbox"], parent_boxes)),
                    list(map(lambda x: x["bbox"], children_boxes)),
                )
                for parent_id, children_id, overlap in matched_results:
                    if overlap > self.config["match_overlap_threshold"]:
                        parent_box = parent_boxes[parent_id]
                        parent_box["matched"] = True
                        children_box = children_boxes[children_id]
                        children_box["matched"] = True
                        points_data = self._get_20_points(
                            parent_box["bbox"], children_box["bbox"]
                        )
                        instance = {
                            "points_data": points_data,
                            "class_id": [self.config["current_class_id"]],
                            "attribute": [],
                            "is_hard": [int(children_box["hard"])],
                        }
                        instances.append(instance)
            for parent_box in parent_boxes:
                if not parent_box["matched"]:
                    points_data = self._get_20_points(
                        parent_box["bbox"], [-10000, -10000, -10000, -10000]
                    )
                    instance = {
                        "points_data": points_data,
                        "class_id": [self.config["current_class_id"]],
                        "attribute": [],
                        "is_hard": [False],
                    }
                    instances.append(instance)
            for children_box in children_boxes:
                if not children_box["matched"]:
                    points_data = self._get_20_points(
                        children_box["bbox"], children_box["bbox"]
                    )
                    instance = {
                        "points_data": points_data,
                        "class_id": [self.config["current_class_id"]],
                        "attribute": [],
                        "is_hard": [int(children_box["hard"])],
                    }
                    instances.append(instance)
        elif (
            self.config["match_mode"]
            == "pack_children_then_parent_and_ignore_unmatched_parent"
        ):  # noqa
            if len(parent_boxes) and len(children_boxes):
                matched_results = matching_with_overlaps(
                    list(map(lambda x: x["bbox"], parent_boxes)),
                    list(map(lambda x: x["bbox"], children_boxes)),
                )
                for parent_id, children_id, overlap in matched_results:
                    if overlap > self.config["match_overlap_threshold"]:
                        parent_box = parent_boxes[parent_id]
                        parent_box["matched"] = True
                        children_box = children_boxes[children_id]
                        children_box["matched"] = True
                        points_data = self._get_20_points(
                            children_box["bbox"], parent_box["bbox"]
                        )
                        instance = {
                            "points_data": points_data,
                            "class_id": [self.config["current_class_id"]],
                            "attribute": [],
                            "is_hard": [int(children_box["hard"])],
                        }
                        instances.append(instance)
                for parent_box in parent_boxes:
                    if not parent_box["matched"]:
                        x1, y1, x2, y2 = parent_box["bbox"]
                        ignore_region = {
                            "left_top": [x1, y1],
                            "right_bottom": [x2, y2],
                            "class_id": [self.config["parent_class_id"]],
                        }
                        ignore_regions.append(ignore_region)
        elif (
            self.config["match_mode"]
            == "pack_parent_then_children_and_ignore_unmatched_children"
        ):  # noqa
            if len(parent_boxes) and len(children_boxes):
                matched_results = matching_with_overlaps(
                    list(map(lambda x: x["bbox"], parent_boxes)),
                    list(map(lambda x: x["bbox"], children_boxes)),
                )
                for parent_id, children_id, overlap in matched_results:
                    if overlap > self.config["match_overlap_threshold"]:
                        parent_box = parent_boxes[parent_id]
                        parent_box["matched"] = True
                        children_box = children_boxes[children_id]
                        children_box["matched"] = True
                        points_data = self._get_20_points(
                            parent_box["bbox"], children_box["bbox"]
                        )
                        instance = {
                            "points_data": points_data,
                            "class_id": [self.config["current_class_id"]],
                            "attribute": [],
                            "is_hard": [int(children_box["hard"])],
                        }
                        instances.append(instance)
            for parent_box in parent_boxes:
                if not parent_box["matched"]:
                    points_data = self._get_20_points(
                        parent_box["bbox"], [-10000, -10000, -10000, -10000]
                    )
                    instance = {
                        "points_data": points_data,
                        "class_id": [self.config["current_class_id"]],
                        "attribute": [],
                        "is_hard": [False],
                    }
                    instances.append(instance)
            for children_box in children_boxes:
                if not children_box["matched"]:
                    x1, y1, x2, y2 = map(float, children_box["bbox"])
                    ignore_region = {
                        "left_top": [x1, y1],
                        "right_bottom": [x2, y2],
                        "class_id": [self.config["current_class_id"]],
                    }
                    ignore_regions.append(ignore_region)
        elif self.config["match_mode"] == "matching_with_belongto_attr":
            belong_to_list = anno.get("belong_to", [])
            if len(belong_to_list) != 0:
                matched_results = matching_with_belongto_attr(
                    parent_boxes, children_boxes, belong_to_list, self.config
                )  # noqa
                for parent_id, children_id in matched_results:
                    parent_box = parent_boxes[parent_id]
                    parent_box["matched"] = True
                    children_box = children_boxes[children_id]
                    children_box["matched"] = True
                    points_data = self._get_20_points(
                        parent_box["bbox"], children_box["bbox"]
                    )  # noqa
                    instance = {
                        "points_data": points_data,
                        "class_id": [self.config["current_class_id"]],
                        "attribute": [],
                        "is_hard": [int(children_box["hard"])],
                    }
                    instances.append(instance)
            for parent_box in parent_boxes:
                if not parent_box["matched"]:
                    points_data = self._get_20_points(
                        parent_box["bbox"], [-10000, -10000, -10000, -10000]
                    )  # noqa
                    instance = {
                        "points_data": points_data,
                        "class_id": [self.config["current_class_id"]],
                        "attribute": [],
                        "is_hard": [False],
                    }
                    instances.append(instance)
        elif (
            self.config["match_mode"]
            == "matching_with_belongto_attr_and_remove_unmatched_parent"
        ):  # noqa
            belong_to_list = anno.get("belong_to", [])
            if len(belong_to_list) != 0:
                matched_results = matching_with_belongto_attr(
                    parent_boxes, children_boxes, belong_to_list, self.config
                )  # noqa
                for parent_id, children_id in matched_results:
                    parent_box = parent_boxes[parent_id]
                    parent_box["matched"] = True
                    children_box = children_boxes[children_id]
                    children_box["matched"] = True
                    points_data = self._get_20_points(
                        parent_box["bbox"], children_box["bbox"]
                    )  # noqa
                    instance = {
                        "points_data": points_data,
                        "class_id": [self.config["current_class_id"]],
                        "attribute": [],
                        "is_hard": [int(children_box["hard"])],
                    }
                    instances.append(instance)
        else:
            raise Exception(
                "Invalid match mode: %s" % self.config["match_mode"]
            )

        if self.config.get("remove_empty_images", False) and not len(
            instances
        ):  # noqa
            return None
        np.random.shuffle(instances)
        img = cv2.imread(image_url, cv2.IMREAD_UNCHANGED)
        if img is None:
            if self.verbose:
                warnings.warn("WARNING: skip invalid image: %s" % (image_url))
            return None
        img_url = os.path.relpath(image_url, self.root_dir)
        if self.config.get("remove_zh_image_path", False) and re.findall(
            "[\u4e00-\u9fa5]", img_url
        ):  # noqa
            if self.verbose:
                warnings.warn(
                    "WARNING, deprecated params: skip zh image path: %s"
                    % (image_url)
                )  # noqa
            return None
        img_h = img.shape[0]
        img_w = img.shape[1]
        img_c = img.shape[2] if len(img.shape) == 3 else 1

        if self.config.get("default_ignore_full_image", False):
            for class_id in range(1, self.config["num_classes"] + 1):
                if class_id != self.config["current_class_id"]:
                    ignore_region = {
                        "left_top": (0, 0),
                        "right_bottom": (img_w, img_h),
                        "class_id": [class_id],
                    }
                    ignore_regions.append(ignore_region)

        img_dict = {
            "img_url": img_url,
            "img_h": img_h,
            "img_w": img_w,
            "img_c": img_c,
            "instances": instances,
            "ignore_regions": ignore_regions,
        }

        return img_dict


def matching_with_overlaps(
    parent_bboxes: List[List], children_bboxes: List[List]
):
    """
    Bipartite graph matching.

    with negative IoU between parent bounding bboxes
    and children bounding boxes

    Args:
        parent_bboxes: of list
            List of parent bounding bbox
        children_bboxes: of list
            List of children bounding bbox
    """

    from sklearn.utils.linear_assignment_ import linear_assignment

    parent_bbox_areas = cal_bbox_areas(parent_bboxes)
    children_bbox_areas = cal_bbox_areas(children_bboxes)
    intersection_bbox_areas_matrix = cal_intersection_areas(
        parent_bboxes, children_bboxes
    )
    union_areas_matrix = (
        parent_bbox_areas.reshape(-1, 1)
        + children_bbox_areas.reshape(1, -1)
        - intersection_bbox_areas_matrix
    )
    contains_matrix = (
        intersection_bbox_areas_matrix / children_bbox_areas.reshape(1, -1)
    )
    ious_matrix = intersection_bbox_areas_matrix / union_areas_matrix
    matched_pairs = linear_assignment(-ious_matrix)
    results = []
    for parent_id, children_id in matched_pairs:
        overlap = contains_matrix[parent_id, children_id]
        results.append((parent_id, children_id, overlap))
    return results


def cal_bbox_areas(bboxes: List[List]):
    """
    Calculate areas of a group of bounding boxes.

    Args:
        bboxes: of list
            List of bounding bbox
    """
    bboxes = np.asarray(bboxes)
    ws = np.maximum(bboxes[:, 2] - bboxes[:, 0] + 1, 0)
    hs = np.maximum(bboxes[:, 3] - bboxes[:, 1] + 1, 0)
    return ws * hs


def cal_intersection_areas(lhs_bboxes: List[List], rhs_bboxes: List[List]):
    """
    Calculate areas of intersection boxes between two group of bounding boxes.

    Args:
        lhs_bboxes: of list
            List of bounding bbox
        rhs_bboxes: of list
            List of bounding bbox
    """
    if not len(lhs_bboxes) or not len(rhs_bboxes):
        iou_matrix = np.array([]).reshape(len(lhs_bboxes), len(rhs_bboxes))
        return iou_matrix

    lhs_bboxes = np.asarray(lhs_bboxes)
    rhs_bboxes = np.asarray(rhs_bboxes)

    lhs_x1 = lhs_bboxes[:, 0].reshape(-1, 1)
    rhs_x1 = rhs_bboxes[:, 0].reshape(1, -1)

    lhs_y1 = lhs_bboxes[:, 1].reshape(-1, 1)
    rhs_y1 = rhs_bboxes[:, 1].reshape(1, -1)

    lhs_x2 = lhs_bboxes[:, 2].reshape(-1, 1)
    rhs_x2 = rhs_bboxes[:, 2].reshape(1, -1)

    lhs_y2 = lhs_bboxes[:, 3].reshape(-1, 1)
    rhs_y2 = rhs_bboxes[:, 3].reshape(1, -1)

    i_x1 = np.maximum(lhs_x1, rhs_x1)
    i_y1 = np.maximum(lhs_y1, rhs_y1)
    i_x2 = np.minimum(lhs_x2, rhs_x2)
    i_y2 = np.minimum(lhs_y2, rhs_y2)

    i_ws = np.maximum(i_x2 - i_x1 + 1.0, 0.0)
    i_hs = np.maximum(i_y2 - i_y1 + 1.0, 0.0)
    i_areas = i_ws * i_hs
    return i_areas


def matching_with_belongto_attr(
    parent_bboxes: List[List],
    children_bboxes: List[List],
    belong_to_list: List,
    config: Dict,
):  # noqa
    """
    Matching with "belongto" attribute between pareng bounding boxes and
    children bounding boxes

    Args:
        parent_bboxes:
            List of parent bounding bbox
        children_bboxes:
            List of children bounding bbox
    """
    results = []
    parent_mapping_dict = {}
    children_mapping_dict = {}

    for parent_idx, parent_bbox in enumerate(parent_bboxes):
        parent_mapping_dict[parent_bbox["id"]] = parent_idx
    for children_idx, children_bbox in enumerate(children_bboxes):
        children_mapping_dict[children_bbox["id"]] = children_idx

    for belong_to in belong_to_list:
        parent_str, children_str = belong_to.split(":")
        parent_cls_name, parent_box_id = parent_str.split("|")
        children_cls_name, children_box_id = children_str.split("|")

        if (
            parent_cls_name != config["parent_box_classname"]
            or children_cls_name != config["children_box_classname"]
        ):
            continue

        parent_id = parent_mapping_dict.get(int(parent_box_id), None)
        children_id = children_mapping_dict.get(int(children_box_id), None)

        if parent_id is not None and children_id is not None:
            results.append((parent_id, children_id))

    return results
