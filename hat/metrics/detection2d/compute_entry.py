# Copyright (c) Horizon Robotics. All rights reserved.

import logging

import numpy as np
from tqdm import tqdm

from hat.core.crowd import get_crowd_groups
from .box_property import bbox_critical, bbox_errors, lower_overlap_for_small
from .generate_data import get_data
from .utils import CrowdGroup, CrowdGTGroup

logger = logging.getLogger(__name__)


def compute(
    images, det_boxes_dict, gt_boxes_dict, critical_label_dict, config
):
    gt_boxes = [gt_boxes_dict[k] for k in images]
    det_boxes = [det_boxes_dict[k] for k in images]
    crowd_mode = config["crowd_mode"]
    default_overlap = config["default_overlap"]
    # error_types = config["error_types"]
    focus_gt = config["focus_gt"]
    focus_gt_iou = config["focus_gt_iou"]
    # images_length = len(images)
    for image_idx, image in enumerate(tqdm(images)):
        gts = gt_boxes_dict[image]
        try:
            if crowd_mode:
                crowd_iou_thresh = config["crowd_iou_thresh"]
                min_crowd_group_size = config["min_crowd_group_size"]
                gt_instances = list(
                    filter(lambda gt: gt.gt_type != "ignore", gts)
                )
                gt_instances_bboxes = list(
                    map(
                        lambda gt: [
                            gt.bbox.x1,
                            gt.bbox.y1,
                            gt.bbox.x2,
                            gt.bbox.y2,
                        ],
                        gt_instances,
                    )
                )
                crowd_group_indexes_list = get_crowd_groups(
                    gt_instances_bboxes, crowd_iou_thresh
                )
                crowd_group_indexes_list = list(
                    filter(
                        lambda g: len(g) >= min_crowd_group_size,
                        crowd_group_indexes_list,
                    )
                )
                crowd_groups = list(
                    map(
                        lambda crowd_group_indexes: CrowdGTGroup(
                            list(
                                map(
                                    lambda idx: gt_instances[idx],
                                    crowd_group_indexes,
                                )
                            )
                        ),
                        crowd_group_indexes_list,
                    )
                )
            dets = det_boxes_dict[image]
            num_obj = len(dets)
            if num_obj > 0:
                idx = np.argsort([-det.score for det in dets])
            else:
                idx = []
            for j in idx:
                det = dets[j]
                kmax = -1  # matched to normal
                hmax = -1  # matched to hard
                imax = -1  # matched to ignore
                gt_matched = False  # matched to gt or not
                ov_max = -1000000
                hv_max = -1000000
                iv_max = -1000000  # max(intersection/detection_area)
                for gt in gts:
                    if gt.eval_type == "TP" and not focus_gt:
                        continue

                    if config["lower_overlap_for_small"]:
                        thr = min(gt.overlap, lower_overlap_for_small(gt.bbox))
                    else:
                        thr = gt.overlap

                    ov = gt.bbox.iou(det.bbox, config["normal_iou_type"])
                    hv = gt.bbox.iou(det.bbox, config["hard_iou_type"])
                    iv = gt.bbox.iou(det.bbox, config["ignore_iou_type"])

                    if focus_gt:
                        if gt.gt_type == "normal" and ov > focus_gt_iou:
                            gt_matched = True
                        if gt.eval_type == "TP":
                            continue

                    if gt.gt_type == "normal":
                        if ov > ov_max and ov > thr:
                            kmax = gt.id
                            ov_max = ov
                    elif gt.gt_type == "ignore":
                        if iv > iv_max and iv >= gt.overlap:
                            imax = gt.id
                            iv_max = iv
                    elif gt.gt_type == "hard":
                        if hv > hv_max and hv > gt.overlap:
                            hmax = gt.id
                            hv_max = hv
                # matched to normal instance
                if kmax >= 0:
                    det.matched_gt_id = kmax
                    det.eval_type = "TP"
                    for gt in gts:
                        if gt.id == kmax:
                            gt.eval_type = "TP"
                            det.error = bbox_errors(gt.bbox, det.bbox)
                # matched to hard instance
                elif hmax >= 0:
                    det.matched_gt_id = hmax
                    det.eval_type = "IGNORE"
                    for gt in gts:
                        if gt.id == hmax:
                            gt.eval_type = "IGNORE"
                # matched to ignore region
                elif imax >= 0:
                    det.eval_type = "IGNORE"
                    for gt in gts:
                        if gt.id == imax:
                            gt.eval_type = "IGNORE"
                # if det not in critical area, ignore det
                elif critical_label_dict is not None and not bbox_critical(
                    det.bbox, critical_label_dict[image]
                ):
                    det.eval_type = "IGNORE"
                if not gt_matched and focus_gt:
                    det.eval_type = "IGNORE"

                if crowd_mode and len(crowd_groups) > 0:
                    overlaps_with_crowds = list(
                        map(
                            lambda crowd_group: crowd_group.intersection(
                                det
                            ).area
                            / det.bbox.area,
                            crowd_groups,
                        )
                    )
                    max_idx = np.argmax(overlaps_with_crowds)
                    if overlaps_with_crowds[max_idx] > default_overlap:
                        if det.eval_type == "FP":
                            det.eval_type = "IGNORE"
                        crowd_groups[max_idx].matched_dets.append((j, det))
            if crowd_mode:
                for gt in gts:
                    if (
                        gt.gt_type == "normal"
                        and gt.eval_type == "FN"
                        and gt.crowd_group is not None
                    ):
                        idxs_dets = gt.crowd_group.matched_dets
                        crowd_dets = CrowdGroup(
                            list(map(lambda idx_det: idx_det[1], idxs_dets))
                        )
                        if (
                            crowd_dets.intersection(gt).area / gt.bbox.area
                            > default_overlap
                        ):
                            gt.eval_type = "IGNORE"
        except BaseException as e:
            logger.error("eval %s, image_key: %s error." % (image_idx, image))
            logger.error(e)
            raise e
    return det_boxes, gt_boxes


def compute_detection_metric(
    gt_boxes_dict,
    det_boxes_dict,
    critical_label_dict,
    config,
    image_tags_dict=None,
):
    if image_tags_dict is None:
        image_tags_dict = {}
    gt_images = list(gt_boxes_dict.keys())
    det_images = list(det_boxes_dict.keys())
    if set(gt_images) != set(det_images):
        logger.warning(
            "Warnning! gt images and det images not match: {} vs {}, \
            Evaluating common images".format(
                len(gt_images), len(det_images)
            )
        )
        images = list(set(gt_images).intersection(set(det_images)))
        gt_boxes_dict = {image: gt_boxes_dict[image] for image in images}
        det_boxes_dict = {image: det_boxes_dict[image] for image in images}
    else:
        images = gt_images

    # images = sorted(images)
    eval_results = compute(
        images, det_boxes_dict, gt_boxes_dict, critical_label_dict, config
    )
    data = get_data(images, image_tags_dict, eval_results)
    return data
