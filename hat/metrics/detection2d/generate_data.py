# Copyright (c) Horizon Robotics. All rights reserved.

"""
Make modifications to the box(GT and DET) according to setting.

The value returned contains the modified box.
"""
import copy
import logging

from hat.core import cvt_pack_info as packcvt
from hat.core.camera import Camera
from .box_property import (
    get_area,
    get_bottom,
    get_euclidean_distance,
    get_height,
    get_iou,
    get_left,
    get_longside,
    get_region_distance,
    get_right,
    get_shortside,
    get_top,
    get_whr,
    get_width,
)

logger = logging.getLogger(__name__)


def gen_roi_data(
    data,
    cover=False,
    bbox_min_left=None,
    bbox_max_left=None,
    bbox_min_right=None,
    bbox_max_right=None,
    bbox_min_bottom=None,
    bbox_max_bottom=None,
    bbox_min_top=None,
    bbox_max_top=None,
    bbox_min_shortside=None,
    bbox_max_shortside=None,
    bbox_min_longside=None,
    bbox_max_longside=None,
    bbox_min_height=None,
    bbox_max_height=None,
    bbox_min_width=None,
    bbox_max_width=None,
    bbox_min_aspect_ratio=None,
    bbox_max_aspect_ratio=None,
    bbox_min_area=None,
    bbox_max_area=None,
    min_distance=None,
    max_distance=None,
    max_distance_lateral=None,
    min_distance_forward=None,
    max_distance_forward=None,
    dynamic_crop=None,
    dynamic_resize=None,
):
    if not cover:
        images = copy.deepcopy(list(data["images"]))
        image_tags_dict = copy.deepcopy(data["image_tags_dict"])
        gts_dict = copy.deepcopy(data["gts_dict"])
        dets_dict = copy.deepcopy(data["dets_dict"])
    else:
        images = list(data["images"])
        image_tags_dict = data["image_tags_dict"]
        gts_dict = data["gts_dict"]
        dets_dict = data["dets_dict"]
    for image_key in images:
        try:
            gts_image = gts_dict[image_key]
            dets_image = dets_dict[image_key]
            dynamic_region = image_tags_dict[image_key].get(
                "dynamic_region", None
            )
            normal_gt_ids = []
            for gt in gts_image:
                if gt["gt_type"] == "normal":
                    out_roi = False
                    if bbox_min_left is not None:
                        gt_left = get_left(gt["bbox"], gt["bbox_type"])
                        if gt_left < bbox_min_left:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_left_min:{}".format(gt_left),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_max_left is not None:
                        gt_left = get_left(gt["bbox"], gt["bbox_type"])
                        if gt_left > bbox_max_left:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_left_max:{}".format(gt_left),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_min_right is not None:
                        gt_right = get_right(gt["bbox"], gt["bbox_type"])
                        if gt_right < bbox_min_right:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_right_min:{}".format(gt_right),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_max_right is not None:
                        gt_right = get_right(gt["bbox"], gt["bbox_type"])
                        if gt_right > bbox_max_right:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_right_max:{}".format(gt_right),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_min_top is not None:
                        gt_top = get_top(gt["bbox"], gt["bbox_type"])
                        if gt_top < bbox_min_top:
                            gt["info"]["hard"].append(
                                ["bbox_top_min:{}".format(gt_top), "from_rois"]
                            )
                            out_roi = True
                    if bbox_max_top is not None:
                        gt_top = get_top(gt["bbox"], gt["bbox_type"])
                        if gt_top > bbox_max_top:
                            gt["info"]["hard"].append(
                                ["bbox_top_max:{}".format(gt_top), "from_rois"]
                            )
                            out_roi = True
                    if bbox_min_bottom is not None:
                        gt_bottom = get_bottom(gt["bbox"], gt["bbox_type"])
                        if gt_bottom < bbox_min_bottom:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_bottom_min:{}".format(gt_bottom),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_max_bottom is not None:
                        gt_bottom = get_bottom(gt["bbox"], gt["bbox_type"])
                        if gt_bottom > bbox_max_bottom:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_bottom_max:{}".format(gt_bottom),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_min_shortside is not None:
                        gt_shortside = get_shortside(
                            gt["bbox"], gt["bbox_type"]
                        )
                        if gt_shortside < bbox_min_shortside:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_min_shortside:{}".format(
                                        gt_shortside
                                    ),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_max_shortside is not None:
                        gt_shortside = get_shortside(
                            gt["bbox"], gt["bbox_type"]
                        )
                        if gt_shortside > bbox_max_shortside:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_max_shortside:{}".format(
                                        gt_shortside
                                    ),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_min_longside is not None:
                        gt_longside = get_longside(gt["bbox"], gt["bbox_type"])
                        if gt_longside < bbox_min_longside:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_min_longside:{}".format(gt_longside),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_max_longside is not None:
                        gt_longside = get_longside(gt["bbox"], gt["bbox_type"])
                        if gt_longside > bbox_max_longside:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_max_longside:{}".format(gt_longside),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_min_height is not None:
                        gt_height = get_height(gt["bbox"], gt["bbox_type"])
                        if gt_height < bbox_min_height:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_min_height:{}".format(gt_height),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_max_height is not None:
                        gt_height = get_height(gt["bbox"], gt["bbox_type"])
                        if gt_height > bbox_max_height:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_max_height:{}".format(gt_height),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_min_width is not None:
                        gt_width = get_width(gt["bbox"], gt["bbox_type"])
                        if gt_width < bbox_min_width:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_min_width:{}".format(gt_width),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_max_width is not None:
                        gt_width = get_width(gt["bbox"], gt["bbox_type"])
                        if gt_width > bbox_max_width:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_max_width:{}".format(gt_width),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_min_aspect_ratio is not None:
                        gt_whr = get_whr(gt["bbox"], gt["bbox_type"])
                        if gt_whr < bbox_min_aspect_ratio:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_min_aspect_ratio:{}".format(gt_whr),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_max_aspect_ratio is not None:
                        gt_whr = get_whr(gt["bbox"], gt["bbox_type"])
                        if gt_whr > bbox_max_aspect_ratio:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_max_aspect_ratio:{}".format(gt_whr),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_min_area is not None:
                        gt_area = get_area(gt["bbox"], gt["bbox_type"])
                        if gt_area < bbox_min_area:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_min_area:{}".format(gt_area),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if bbox_max_area is not None:
                        gt_area = get_area(gt["bbox"], gt["bbox_type"])
                        if gt_area > bbox_max_area:
                            gt["info"]["hard"].append(
                                [
                                    "bbox_max_area:{}".format(gt_area),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if min_distance is not None:
                        gt_distance = get_euclidean_distance(
                            gt["distance_lateral"], gt["distance_forward"]
                        )
                        if (
                            gt_distance < min_distance
                            or gt["distance_forward"] < 0
                        ):
                            gt["info"]["hard"].append(
                                [
                                    "min_distance:{}".format(gt_distance),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if max_distance is not None:
                        gt_distance = get_euclidean_distance(
                            gt["distance_lateral"], gt["distance_forward"]
                        )
                        if (
                            gt_distance > max_distance
                            or gt["distance_forward"] < 0
                        ):
                            gt["info"]["hard"].append(
                                [
                                    "max_distance:{}".format(gt_distance),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if max_distance_lateral is not None:
                        if (
                            abs(gt["distance_lateral"]) > max_distance_lateral
                            or gt["distance_forward"] < 0
                        ):
                            gt["info"]["hard"].append(
                                [
                                    "max_distance_lateral:{}".format(
                                        abs(gt["distance_lateral"])
                                    ),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if min_distance_forward is not None:
                        if (
                            gt["distance_forward"] < min_distance_forward
                            or gt["distance_forward"] < 0
                        ):
                            gt["info"]["hard"].append(
                                [
                                    "min_distance_forward:{}".format(
                                        gt["distance_forward"]
                                    ),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if max_distance_forward is not None:
                        if (
                            gt["distance_forward"] > max_distance_forward
                            or gt["distance_forward"] < 0
                        ):
                            gt["info"]["hard"].append(
                                [
                                    "max_distance_forward:{}".format(
                                        gt["distance_forward"]
                                    ),
                                    "from_rois",
                                ]
                            )
                            out_roi = True
                    if dynamic_crop is not None:
                        iog = get_iou(
                            [
                                gt["bbox"]["x1"],
                                gt["bbox"]["y1"],
                                gt["bbox"]["x2"],
                                gt["bbox"]["y2"],
                            ],
                            dynamic_region,
                            iou_type="iog",
                        )
                        if iog < dynamic_crop:
                            gt["info"]["hard"].append(
                                ["dynamic_crop", "from_rois"]
                            )
                            out_roi = True
                    if dynamic_resize is not None:
                        iog = get_iou(
                            [
                                gt["bbox"]["x1"],
                                gt["bbox"]["y1"],
                                gt["bbox"]["x2"],
                                gt["bbox"]["y2"],
                            ],
                            dynamic_region,
                            iou_type="iog",
                        )
                        if iog > dynamic_resize:
                            gt["info"]["hard"].append(
                                ["dynamic_resize", "from_rois"]
                            )
                            out_roi = True
                    if out_roi:
                        gt["gt_type"] = "hard"
                        if gt["eval_type"] == "TP":
                            gt["eval_type"] = "IGNORE"
                if gt["gt_type"] == "normal" and gt["eval_type"] == "TP":
                    normal_gt_ids.append(gt["id"])

            for det in dets_image:
                if det["eval_type"] == "TP":
                    if det["matched_gt_id"] not in normal_gt_ids:
                        det["eval_type"] = "IGNORE"
                elif det["eval_type"] == "FP":
                    out_roi = False
                    if bbox_min_left is not None:
                        det_left = get_left(det["bbox"], det["bbox_type"])
                        if det_left < bbox_min_left:
                            out_roi = True
                    if bbox_max_left is not None:
                        det_left = get_left(det["bbox"], det["bbox_type"])
                        if det_left > bbox_max_left:
                            out_roi = True
                    if bbox_min_right is not None:
                        det_right = get_right(det["bbox"], det["bbox_type"])
                        if det_right < bbox_min_right:
                            out_roi = True
                    if bbox_max_right is not None:
                        det_right = get_right(det["bbox"], det["bbox_type"])
                        if det_right > bbox_max_right:
                            out_roi = True
                    if bbox_min_top is not None:
                        det_top = get_top(det["bbox"], det["bbox_type"])
                        if det_top < bbox_min_top:
                            out_roi = True
                    if bbox_max_top is not None:
                        det_top = get_top(det["bbox"], det["bbox_type"])
                        if det_top > bbox_max_top:
                            out_roi = True
                    if bbox_min_bottom is not None:
                        det_bottom = get_bottom(det["bbox"], det["bbox_type"])
                        if det_bottom < bbox_min_bottom:
                            out_roi = True
                    if bbox_max_bottom is not None:
                        det_bottom = get_bottom(det["bbox"], det["bbox_type"])
                        if det_bottom > bbox_max_bottom:
                            out_roi = True
                    if bbox_min_shortside is not None:
                        det_shortside = get_shortside(
                            det["bbox"], det["bbox_type"]
                        )
                        if det_shortside < bbox_min_shortside:
                            out_roi = True
                    if bbox_max_shortside is not None:
                        det_shortside = get_shortside(
                            det["bbox"], det["bbox_type"]
                        )
                        if det_shortside > bbox_max_shortside:
                            out_roi = True
                    if bbox_min_longside is not None:
                        det_longside = get_longside(
                            det["bbox"], det["bbox_type"]
                        )
                        if det_longside < bbox_min_longside:
                            out_roi = True
                    if bbox_max_longside is not None:
                        det_longside = get_longside(
                            det["bbox"], det["bbox_type"]
                        )
                        if det_longside > bbox_max_longside:
                            out_roi = True
                    if bbox_min_height is not None:
                        det_height = get_height(det["bbox"], det["bbox_type"])
                        if det_height < bbox_min_height:
                            out_roi = True
                    if bbox_max_height is not None:
                        det_height = get_height(det["bbox"], det["bbox_type"])
                        if det_height > bbox_max_height:
                            out_roi = True
                    if bbox_min_width is not None:
                        det_width = get_width(det["bbox"], det["bbox_type"])
                        if det_width < bbox_min_width:
                            out_roi = True
                    if bbox_max_width is not None:
                        det_width = get_width(det["bbox"], det["bbox_type"])
                        if det_width > bbox_max_width:
                            out_roi = True
                    if bbox_min_aspect_ratio is not None:
                        det_whr = get_whr(det["bbox"], det["bbox_type"])
                        if det_whr < bbox_min_aspect_ratio:
                            out_roi = True
                    if bbox_max_aspect_ratio is not None:
                        det_whr = get_whr(det["bbox"], det["bbox_type"])
                        if det_whr > bbox_max_aspect_ratio:
                            out_roi = True
                    if bbox_min_area is not None:
                        det_area = get_area(det["bbox"], det["bbox_type"])
                        if det_area < bbox_min_area:
                            out_roi = True
                    if bbox_max_area is not None:
                        det_area = get_area(det["bbox"], det["bbox_type"])
                        if det_area > bbox_max_area:
                            out_roi = True
                    if min_distance is not None:
                        det_distance = get_euclidean_distance(
                            det["distance_lateral"], det["distance_forward"]
                        )
                        if (
                            det_distance < min_distance
                            or det["distance_forward"] < 0
                        ):
                            out_roi = True
                    if max_distance is not None:
                        det_distance = get_euclidean_distance(
                            det["distance_lateral"], det["distance_forward"]
                        )
                        if (
                            det_distance > max_distance
                            or det["distance_forward"] < 0
                        ):
                            out_roi = True
                    if max_distance_lateral is not None:
                        if (
                            abs(det["distance_lateral"]) > max_distance_lateral
                            or det["distance_forward"] < 0
                        ):
                            out_roi = True
                    if min_distance_forward is not None:
                        if (
                            det["distance_forward"] < min_distance_forward
                            or det["distance_forward"] < 0
                        ):
                            out_roi = True
                    if max_distance_forward is not None:
                        if (
                            det["distance_forward"] > max_distance_forward
                            or det["distance_forward"] < 0
                        ):
                            out_roi = True
                    if dynamic_crop is not None:
                        iou = get_iou(
                            [
                                det["bbox"]["x1"],
                                det["bbox"]["y1"],
                                det["bbox"]["x2"],
                                det["bbox"]["y2"],
                            ],
                            dynamic_region,
                            iou_type="iog",
                        )
                        if iou < dynamic_crop:
                            out_roi = True
                        elif (
                            get_region_distance(
                                det["bbox"], det["bbox_type"], dynamic_region
                            )
                            <= 20
                        ):
                            det["eval_type"] = "IGNORE"
                    if dynamic_resize is True:
                        iog = get_iou(
                            [
                                det["bbox"]["x1"],
                                det["bbox"]["y1"],
                                det["bbox"]["x2"],
                                det["bbox"]["y2"],
                            ],
                            dynamic_region,
                            iou_type="iog",
                        )
                        if iog > dynamic_resize:
                            out_roi = True
                    if out_roi:
                        det["eval_type"] = "IGNORE"
                else:
                    assert det["eval_type"] == "IGNORE"
        except BaseException as e:
            logger.error("gen roi data %s error." % image_key)
            logger.error(e)
            raise e
    data_new = {
        "images": images,
        "image_tags_dict": image_tags_dict,
        "gts_dict": gts_dict,
        "dets_dict": dets_dict,
    }
    return data_new


def gen_image_tags_data(data, cover=False, **kwargs):
    if not cover:
        images = copy.deepcopy(list(data["images"]))
        image_tags_dict = copy.deepcopy(data["image_tags_dict"])
        gts_dict = copy.deepcopy(data["gts_dict"])
        dets_dict = copy.deepcopy(data["dets_dict"])
    else:
        images = list(data["images"])
        image_tags_dict = data["image_tags_dict"]
        gts_dict = data["gts_dict"]
        dets_dict = data["dets_dict"]

    images_new = []
    image_tags_dict_new = {}
    gts_dict_new = {}
    dets_dict_new = {}
    for image_key in images:
        try:
            image_tag = image_tags_dict[image_key]
            keep = True
            for key, value in kwargs.items():
                value = value if isinstance(value, list) else [value]
                if image_tag[key] not in value:
                    keep = False
            if keep:
                images_new.append(image_key)
                image_tags_dict_new[image_key] = image_tags_dict[image_key]
                gts_dict_new[image_key] = gts_dict[image_key]
                dets_dict_new[image_key] = dets_dict[image_key]
        except BaseException as e:
            logger.error("gen image tags data %s error." % image_key)
            logger.error(e)
            raise e
    data_new = {
        "images": images_new,
        "image_tags_dict": image_tags_dict_new,
        "gts_dict": gts_dict_new,
        "dets_dict": dets_dict_new,
    }
    return data_new


def gen_det_cls_all_data(data, cover=False):
    if not cover:
        images = copy.deepcopy(list(data["images"]))
        image_tags_dict = copy.deepcopy(data["image_tags_dict"])
        gts_dict = copy.deepcopy(data["gts_dict"])
        dets_dict = copy.deepcopy(data["dets_dict"])
    else:
        images = list(data["images"])
        image_tags_dict = data["image_tags_dict"]
        gts_dict = data["gts_dict"]
        dets_dict = data["dets_dict"]

    for image_key in images:
        try:
            gts = gts_dict[image_key]
            dets = dets_dict[image_key]
            for det in dets:
                if (
                    det["eval_type"] == "FP"
                    and det["det_cls_type"] == "REMOVE"
                ):
                    det["eval_type"] = "IGNORE"
                if det["eval_type"] == "TP":
                    for gt in gts:
                        if gt["id"] == det["matched_gt_id"]:
                            matched_gt = gt
                    assert matched_gt["eval_type"] == "TP"
                    if det["det_cls_type"] == "REMOVE":
                        det["eval_type"] = "IGNORE"
                        matched_gt["eval_type"] = "FN"
                    elif det["det_cls_type"] == "NO_CLS_OUTPUT":
                        det["eval_type"] = "IGNORE"
                        matched_gt["eval_type"] = "IGNORE"
                    elif matched_gt["gt_cls_type"] == "IGNORE_DET":
                        det["eval_type"] = "IGNORE"
                        matched_gt["eval_type"] = "IGNORE"
                    elif (
                        matched_gt["gt_cls_type"] != "IGNORE"
                        and det["det_cls_type"] != "IGNORE"
                        and det["det_cls_type"] != matched_gt["gt_cls_type"]
                    ):
                        det["eval_type"] = "FP"
                        matched_gt["eval_type"] = "FN"
                    else:
                        pass
        except BaseException as e:
            logger.error("get det cls all data %s error." % image_key)
            logger.error(e)
            raise e
    data_new = {
        "images": images,
        "image_tags_dict": image_tags_dict,
        "gts_dict": gts_dict,
        "dets_dict": dets_dict,
    }
    return data_new


def gen_det_cls_one_data(data, det_cls, cover=False, keep_ignore_data=True):
    if not cover:
        images = copy.deepcopy(list(data["images"]))
        image_tags_dict = copy.deepcopy(data["image_tags_dict"])
        gts_dict = copy.deepcopy(data["gts_dict"])
        dets_dict = copy.deepcopy(data["dets_dict"])
    else:
        images = list(data["images"])
        image_tags_dict = data["image_tags_dict"]
        gts_dict = data["gts_dict"]
        dets_dict = data["dets_dict"]

    valid_images = []
    for image_key in images:
        try:
            gts = gts_dict[image_key]
            dets = dets_dict[image_key]
            match_gt = False
            match_det = False
            for gt in gts:
                if det_cls in gt["gt_cls_type"].split("|"):
                    match_gt = True
                if gt["gt_type"] == "normal" and det_cls not in gt[
                    "gt_cls_type"
                ].split("|"):
                    gt["gt_type"] = "hard"
                    if gt["eval_type"] == "TP":
                        gt["eval_type"] = "IGNORE"
            for det in dets:
                if det_cls in det["det_cls_type"].split("|"):
                    match_det = True
                if det["eval_type"] == "FP" and det_cls not in det[
                    "det_cls_type"
                ].split("|"):
                    det["eval_type"] = "IGNORE"
                if det["eval_type"] == "TP":
                    for gt in gts:
                        if gt["id"] == det["matched_gt_id"]:
                            matched_gt = gt
                    if matched_gt["gt_type"] != "normal":
                        det["eval_type"] = "IGNORE"
                    else:
                        assert matched_gt[
                            "eval_type"
                        ] == "TP" and det_cls in matched_gt[
                            "gt_cls_type"
                        ].split(
                            "|"
                        )
                        if det["det_cls_type"] == "NO_CLS_OUTPUT":
                            det["eval_type"] = "IGNORE"
                            matched_gt["eval_type"] = "IGNORE"
                        elif (
                            det["det_cls_type"] != "IGNORE"
                            and det["det_cls_type"]
                            != matched_gt["gt_cls_type"]
                        ):
                            det["eval_type"] = "IGNORE"
                            matched_gt["eval_type"] = "FN"
                        else:
                            pass
            if not keep_ignore_data and not match_gt and not match_det:
                gts_dict.pop(image_key)
                dets_dict.pop(image_key)
                image_tags_dict.pop(image_key)
            if keep_ignore_data:
                valid_images.append(image_key)
            elif match_gt or match_det:
                valid_images.append(image_key)
        except BaseException as e:
            logger.error("gen det cls one data: %s error." % image_key)
            logger.error(e)
            raise e
    data_new = {
        "images": valid_images,
        "image_tags_dict": image_tags_dict,
        "gts_dict": gts_dict,
        "dets_dict": dets_dict,
    }
    return data_new


def gen_seq_data(data, cover=False, det_min_age=5):
    if not cover:
        images = data["images"]
        image_tags_dict = copy.deepcopy(list(data["image_tags_dict"]))
        gts_dict = copy.deepcopy(data["gts_dict"])
        dets_dict = copy.deepcopy(data["dets_dict"])
    else:
        images = data["images"]
        image_tags_dict = list(data["image_tags_dict"])
        gts_dict = data["gts_dict"]
        dets_dict = data["dets_dict"]

    for image_key in images:
        try:
            # gts = gts_dict[image_key]
            dets = dets_dict[image_key]
            for det in dets:
                if det["eval_type"] == "FP" and det["age"] <= det_min_age:
                    det["eval_type"] = "IGNORE"
                # if det['eval_type'] == 'TP' and det['age'] <= det_min_age:
                #     for gt in gts:
                #         if gt['id'] == det['matched_gt_id']:
                #             matched_gt = gt
                #     assert matched_gt['eval_type'] == 'TP'
                #     det['eval_type'] = 'IGNORE'
                #     matched_gt['eval_type'] = 'FN'
        except BaseException as e:
            logger.error("gen seq data %s error." % image_key)
            logger.error(e)
            raise e
    data_new = {
        "images": images,
        "image_tags_dict": image_tags_dict,
        "gts_dict": gts_dict,
        "dets_dict": dets_dict,
    }
    return data_new


# def gen_cipv_data(data):
#     images = data['images']
#     gts_dict = copy.deepcopy(data['gts_dict'])
#     dets_dict = copy.deepcopy(data['dets_dict'])
#
#     gts_dict_new = {}
#     dets_dict_new = {}
#     for image_key in images:
#         gts = gts_dict[image_key]
#         dets = dets_dict[image_key]
#         gts_new = []
#         dets_new = []
#         for gt in gts:
#             if gt['cipv']:
#                 gts_new.append(gt)
#                 for det in dets:
#                     if det['matched_gt_id'] == gt['id']:
#                         dets_new.append(det)
#         for det in dets:
#             if det['cipv']:
#                 dets_new.append(det)
#                 for gt in gts:
#                     if gt['id'] == det['matched_gt_id']:
#                         gts_new.append(gt)
#
#         gts_dict_new[image_key] = gts_new
#         dets_dict_new[image_key] = dets_new
#
#     data_new = {'images': images, 'gts_dict': gts_dict_new,
# 'dets_dict': dets_dict_new}
#     return data_new


def get_distance(
    gt_boxes_dict, det_boxes_dict, cam_info_dict, bbox_undistort=False
):
    lut_dic = {}
    for image_key in det_boxes_dict:
        try:
            cam_info = cam_info_dict[image_key]
            if cam_info is None:
                continue

            version = cam_info.get("version", 0)
            if version == 0:
                cam = Camera(
                    cam_info["pitch"],
                    cam_info["yaw"],
                    cam_info["roll"],
                    cam_info["cameraX"],
                    cam_info["cameraY"],
                    cam_info["cameraZ"],
                    cam_info["focalU"],
                    cam_info["focalV"],
                    cam_info["centerU"],
                    cam_info["centerV"],
                    version=version,
                )
            elif version == 1:
                cam = Camera(
                    cam_info["roll"],
                    cam_info["pitch"],
                    cam_info["yaw"],
                    cam_info["cameraX"],
                    cam_info["cameraY"],
                    cam_info["cameraZ"],
                    cam_info["focalU"],
                    cam_info["focalV"],
                    cam_info["centerU"],
                    cam_info["centerV"],
                    version=version,
                )
            else:
                NotImplementedError(
                    f"version should be 0 or 1, but get {version}"
                )

            (
                camera_type,
                camera_fov,
                mat_distort,
                mat_camera,
                cam_hash,
                img_height,
                img_width,
            ) = packcvt.get_camera_info(cam_info)
            if bbox_undistort is False:
                if cam_hash not in lut_dic:
                    logger.debug("{} has new lut".format(image_key))
                    lut_dic[cam_hash] = packcvt.get_undistort_lut(
                        camera_type,
                        camera_fov,
                        mat_distort,
                        mat_camera,
                        img_height,
                        img_width,
                    )
                undisort_lut = lut_dic[cam_hash]

            for gt in gt_boxes_dict[image_key]:
                bbox = gt.bbox
                if bbox_undistort is True:
                    bbox_undistorted = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
                else:
                    bbox_undistorted = packcvt.undistort_bbox(
                        [bbox.x1, bbox.y1, bbox.x2, bbox.y2], undisort_lut
                    )
                distance_forward, distance_lateral = cam.get_bbox_distance(
                    bbox_undistorted
                )
                gt.distance_forward = distance_forward
                gt.distance_lateral = distance_lateral

            for det in det_boxes_dict[image_key]:
                bbox = det.bbox
                if bbox_undistort is True:
                    bbox_undistorted = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
                else:
                    bbox_undistorted = packcvt.undistort_bbox(
                        [bbox.x1, bbox.y1, bbox.x2, bbox.y2], undisort_lut
                    )
                distance_forward, distance_lateral = cam.get_bbox_distance(
                    bbox_undistorted
                )
                det.distance_forward = distance_forward
                det.distance_lateral = distance_lateral
        except BaseException as e:
            logger.error("get distance: %s error." % image_key)
            logger.error(e)
            raise e
    return gt_boxes_dict, det_boxes_dict


def get_cls(data, gt2eval, det2eval):
    det2eval = {str(k): v for k, v in det2eval.items()}
    images = data["images"]
    gts_dict = data["gts_dict"]
    dets_dict = data["dets_dict"]
    for image_key in images:
        try:
            gts = gts_dict[image_key]
            for gt in gts:
                gt_cls_type_raw = gt["gt_cls_type_raw"]
                gt_cls_type_raw_list = gt_cls_type_raw.split("|")
                try:
                    gt_cls_type_list = sorted(
                        [
                            gt2eval[gt_cls_type_raw]
                            for gt_cls_type_raw in gt_cls_type_raw_list
                        ]
                    )
                except Exception as e:
                    logger.error(
                        "gt_cls_type_raw_list is : %s\n"
                        "gt2eval.keys: %s \n"
                        "There some keys in gt_cls_type_raw_list "
                        "but not in profile `gt2eval` attr."
                        % (
                            ",".join(gt_cls_type_raw_list),
                            ",".join(list(gt2eval.keys())),
                        )
                    )
                    raise e
                gt_cls_type = "|".join(gt_cls_type_list)
                gt["gt_cls_type"] = gt_cls_type
            dets = dets_dict[image_key]
            for det in dets:
                if det["det_cls_type_raw"] == "":
                    det_cls_type_raw = "NO_CLS_OUTPUT"
                elif isinstance(det["det_cls_type_raw"], (int, float)):
                    det_cls_type_raw = str(det["det_cls_type_raw"])
                else:
                    det_cls_type_raw = det["det_cls_type_raw"]
                det_cls_type_raw_list = det_cls_type_raw.split("|")
                try:
                    det_cls_type_list = sorted(
                        [
                            det2eval[det_cls_type_raw]
                            for det_cls_type_raw in det_cls_type_raw_list
                        ]
                    )
                except Exception as e:
                    logger.error(
                        "det_cls_type_raw_list is : %s\n"
                        "det2eval.keys: %s \n"
                        "There some keys in det_cls_type_raw_list "
                        "but not in profile `det2eval` attr."
                        % (
                            ",".join(det_cls_type_raw_list),
                            ",".join(list(det2eval.keys())),
                        )
                    )
                    raise e
                det_cls_type = "|".join(det_cls_type_list)
                det["det_cls_type"] = det_cls_type
        except BaseException as e:
            logger.error("get cls: %s error." % image_key)
            logger.error(e)
            raise e
    return data


def get_data(images, image_tags_dict, eval_results):
    det_boxes, gt_boxes = eval_results
    dets_dict = {}
    gts_dict = {}
    for image, dets, gts in zip(images, det_boxes, gt_boxes):
        dets_to_dict = []
        for det in dets:
            det.bbox = {
                "x1": det.bbox.x1,
                "y1": det.bbox.y1,
                "x2": det.bbox.x2,
                "y2": det.bbox.y2,
            }
            if getattr(det, "undist_bbox", None):
                det.undist_bbox = {
                    "x1": det.undist_bbox.x1,
                    "y1": det.undist_bbox.y1,
                    "x2": det.undist_bbox.x2,
                    "y2": det.undist_bbox.y2,
                }
            dets_to_dict.append(det.__dict__)
        dets_dict[image] = dets_to_dict
        gts_to_dict = []
        for gt in gts:
            delattr(gt, "crowd_group")
            gt.bbox = {
                "x1": gt.bbox.x1,
                "y1": gt.bbox.y1,
                "x2": gt.bbox.x2,
                "y2": gt.bbox.y2,
            }
            if getattr(gt, "undist_bbox", None):
                gt.undist_bbox = {
                    "x1": gt.undist_bbox.x1,
                    "y1": gt.undist_bbox.y1,
                    "x2": gt.undist_bbox.x2,
                    "y2": gt.undist_bbox.y2,
                }
            if getattr(gt, "fit_dist_bbox", None):
                gt.fit_dist_bbox = {
                    "x1": gt.fit_dist_bbox.x1,
                    "y1": gt.fit_dist_bbox.y1,
                    "x2": gt.fit_dist_bbox.x2,
                    "y2": gt.fit_dist_bbox.y2,
                }
            gts_to_dict.append(gt.__dict__)
        gts_dict[image] = gts_to_dict

    data = {
        "images": images,
        "image_tags_dict": image_tags_dict,
        "gts_dict": gts_dict,
        "dets_dict": dets_dict,
    }
    return data
