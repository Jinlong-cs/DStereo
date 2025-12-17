# Copyright (c) Horizon Robotics. All rights reserved.
import collections
import json
import logging
import os

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

from hat.metrics.detection2d.generate_result import NpEncoder
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.visualize.semantic_segmentation.visualize import save_eval_images
from .freespace import eval_freespace, get_freespace_res
from .get_files import (
    get_image_dict,
    get_undistort_image,
    get_undistort_label,
    writeout_json_all,
    writeout_labels_dict,
)
from .get_roi import draw_vp, roi_valid

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SegMetric(EvalMetric):
    """Semantic segmentation metric.

    Args:
        outfile_dir: path to output directory.
        labels_dict: define the labels, the value means
            labels, the keys describe the label.
        freespace_dict: define the freespcae, keys including
            freespace_id, margin, thredshold and so on.
        roi_dict: dict define the roi.
        min_gt_num: minimum gt numbers for image which can contribute to the metric.
        image_tags: define the scene/time/weather tags.
    """  # noqa: E501

    def __init__(
        self,
        outfile_dir: str,
        labels_dict: dict,
        freespace_dict: dict,
        roi_dict: dict,
        image_tags: dict,
        min_gt_num: int = 0,
    ):
        self.outfile_dir = outfile_dir
        self.labels_dict = labels_dict
        self.freespace_dict = freespace_dict
        self.roi_dict = roi_dict
        self.min_gt_num = min_gt_num
        self.config_tags = image_tags
        self.reslist = []
        self.reset()

    def reset(self):
        self.reslist = []
        self.TPs = 0
        self.FNs = 0
        self.FPs = 0
        self.valid_labels = sorted(self.labels_dict)
        num_classes = len(self.labels_dict)

        if num_classes == 0:
            logger.warning(
                "len labels dict is 0. Please check profile `class` attr."
            )
        self.label_num_list = np.array([0] * num_classes)
        self.conf_all = np.zeros((num_classes, num_classes))
        self.images_num = 0

        if self.freespace_dict is not None:
            self.TPs_f = 0
            self.FNs_f = 0
            self.FPs_f = 0
            self.conf_all_f = np.zeros((2, 2))
            num_no_freespace = num_classes - len(
                self.freespace_dict["freespace_id"]
            )
            self.conf_all_f_categorys = np.zeros(
                (num_no_freespace, num_no_freespace)
            )
            self.dist_vertical_all = []
            self.dist_around_all = []
            if "margin" in self.freespace_dict.keys():
                self.vertical_margins_acc_all = []
                self.around_margins_acc_all = []
                self.has_margin = 1
                if len(self.freespace_dict["margin"]) == 0:
                    logger.error(
                        "Please check profile `freespace.margin.item`."
                        " It need len > 0"
                    )
                assert len(self.freespace_dict["margin"])
                self.margin_list = self.freespace_dict["margin"]
                self.conf_all_f_categorys_with_margin = np.zeros(
                    (
                        len(self.margin_list),
                        num_no_freespace,
                        num_no_freespace,
                    )
                )
                if "threshold" in self.freespace_dict.keys():
                    if len(self.freespace_dict["threshold"]) == 0:
                        logger.error(
                            "Please check profile `freespace.threshold.item`. "
                            "It need len > 0"
                        )
                    assert len(self.freespace_dict["threshold"])
                    self.threshold_list = self.freespace_dict["threshold"]
                    self.has_threshold = 1
                elif "thredshold" in self.freespace_dict.keys():
                    if len(self.freespace_dict["thredshold"]) == 0:
                        logger.error(
                            "Please check profile `freespace.threshold.item`. "
                            "It need len > 0"
                        )
                    assert len(self.freespace_dict["thredshold"])
                    self.threshold_list = self.freespace_dict["thredshold"]
                    self.has_threshold = 1
                else:
                    self.has_threshold = 0
            else:
                self.has_margin = 0
        else:
            self.has_margin = 0

    def update(self, image_info):
        try:
            image_name = image_info.pop("img_name")
            image = image_info.pop("image")
            gt_label = image_info.pop("gt_label")
            pred_label = image_info.pop("pred_label")
            image_tag = image_info.pop("image_tag")
            keep = False

            if self.config_tags is None or image_tag is None:
                keep = True

            if self.config_tags is not None:  # config image tags
                config_tags_dict = {}
                image_tag_dict = {}

                config_tags_dict["time"] = self.config_tags.get("time", None)
                config_tags_dict["scene"] = self.config_tags.get("scene", None)
                config_tags_dict["weather"] = self.config_tags.get(
                    "weather", None
                )

                if image_tag is not None:
                    image_tag_dict["time"] = image_tag.get("time", None)
                    image_tag_dict["scene"] = image_tag.get("scene", None)
                    image_tag_dict["weather"] = image_tag.get("weather", None)

                if len(image_tag_dict) > 0:
                    for key in image_tag_dict:
                        if config_tags_dict[key] is None:
                            continue
                        else:
                            if image_tag_dict[key] is not None:
                                if (
                                    image_tag_dict[key]
                                    in config_tags_dict[key]
                                ):
                                    keep = True
                                else:
                                    keep = False
                                    break
                            else:
                                continue

            if keep:
                if self.roi_dict is not None and len(image_info.keys()) > 0:
                    mask_array, label_index, image_camera = roi_valid(
                        image_info, self.roi_dict
                    )
                    if not self.roi_dict["distort"]:
                        if image_camera is not None:
                            image = get_undistort_image(image, image_camera)
                            (
                                image_vp_x,
                                image_vp_y,
                            ) = image_camera.get_vanishing_point_undistort()
                            draw_vp(image, image_vp_x, image_vp_y)
                            gt_label = get_undistort_label(
                                gt_label, image_camera
                            )
                            pred_label = get_undistort_label(
                                pred_label, image_camera
                            )
                    else:
                        if image_camera is not None:
                            (
                                image_vp_x,
                                image_vp_y,
                            ) = image_camera.get_vanishing_point_distort()
                            draw_vp(image, image_vp_x, image_vp_y)

                    if mask_array is not None:
                        if mask_array.shape[:2] != gt_label.shape[:2]:
                            mask_array = cv2.resize(
                                mask_array,
                                (gt_label.shape[1], gt_label.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        gt_label[mask_array == 0] = 255
                        pred_label[mask_array == 0] = 255

                # avoid error if all gt labels are 255 for confusion_matrix()
                if (
                    np.sum(gt_label == 255)
                    == gt_label.shape[0] * gt_label.shape[1]
                ):
                    conf = np.zeros(
                        (len(self.valid_labels), len(self.valid_labels)),
                        dtype=np.int64,
                    )
                else:
                    conf = confusion_matrix(
                        gt_label.reshape(-1),
                        pred_label.reshape(-1),
                        labels=self.valid_labels,
                    )
                class_pixels = conf.sum(1)
                class_pixels += class_pixels == 0  # avoid zero in denominator
                TP = np.diag(conf)
                FN = class_pixels - TP
                FP = conf.sum(0) - TP
                cls_ious = TP / (FN + TP + FP + 0.0)
                cls_accs = TP / (TP + FP + 0.0)
                cls_recs = TP / (FN + TP + 0.0)
                diff = np.logical_and(
                    (pred_label != gt_label), (gt_label != 255)
                ).astype(np.uint8)[:, :, np.newaxis]

                if self.freespace_dict is not None:
                    clsnum = len(self.valid_labels)
                    no_freespace_id = []
                    for i in range(clsnum):
                        if i not in self.freespace_dict["freespace_id"]:
                            no_freespace_id.append(i)
                    freespace_res = eval_freespace(
                        gt_label,
                        pred_label,
                        clsnum,
                        conf,
                        self.freespace_dict,
                        self.has_margin,
                        no_freespace_id,
                    )
                else:
                    freespace_res = None

                score_dict = get_image_dict(
                    image_name,
                    cls_ious,
                    cls_accs,
                    cls_recs,
                    self.labels_dict,
                    freespace_res,
                    TP,
                    FN,
                    FP,
                )
                image_dict = dict()  # noqa: C408
                image_dict["image_name"] = image_name
                image_dict["image_tag"] = image_tag
                save_eval_images(
                    self.outfile_dir,
                    image_name,
                    diff,
                    gt_label,
                    pred_label,
                    image,
                )
                if self.has_margin == 1:
                    image_dict["gt_freespace_line"] = (
                        freespace_res["gt_line"].astype(np.int32).tolist()
                    )
                    image_dict["pred_freespace_line"] = (
                        freespace_res["pred_line"].astype(np.int32).tolist()
                    )
                else:
                    image_dict["gt_freespace_line"] = None
                    image_dict["pred_freespace_line"] = None
        except BaseException as e:
            logger.error("eval one %s error" % image_name)
            logger.error(e)
            raise e
        eval_dict = collections.OrderedDict()
        eval_dict["TP"] = TP if keep else None
        eval_dict["FN"] = FN if keep else None
        eval_dict["FP"] = FP if keep else None
        eval_dict["cls_ious"] = cls_ious if keep else None
        eval_dict["cls_accs"] = cls_accs if keep else None
        eval_dict["cls_recs"] = cls_recs if keep else None
        eval_dict["conf"] = conf if keep else None
        eval_dict["score_dict"] = score_dict if keep else None
        eval_dict["image_dict"] = image_dict if keep else None
        eval_dict["freespace_res"] = freespace_res if keep else None
        self.reslist.append(eval_dict)
        return eval_dict

    def get(self):
        image_json_path = os.path.join(self.outfile_dir, "images.json")
        scores_json_path = os.path.join(self.outfile_dir, "scores.json")
        self.images_json_fin = open(image_json_path, "a")
        self.scores_json_fin = open(scores_json_path, "a")

        for i in range(len(self.reslist)):
            res_detail = self.reslist[i]
            (TP, FN, FP, conf, score_dict, image_dict) = (
                res_detail["TP"],
                res_detail["FN"],
                res_detail["FP"],
                res_detail["conf"],
                res_detail["score_dict"],
                res_detail["image_dict"],
            )
            if image_dict is not None:
                self.label_num_list[conf.sum(1) != 0] += 1
                self.TPs += TP
                self.FNs += FN
                self.FPs += FP
                self.conf_all += conf
                if self.freespace_dict is not None:
                    freespace_res = res_detail["freespace_res"]
                    conf_f = freespace_res["conf_f"]
                    conf_f_categorys = freespace_res["conf_f_categorys"]
                    TP_f = freespace_res["TP_f"]
                    FN_f = freespace_res["FN_f"]
                    FP_f = freespace_res["FP_f"]
                    margin_flag = freespace_res["margin_flag"]

                    self.TPs_f += TP_f
                    self.FNs_f += FN_f
                    self.FPs_f += FP_f
                    self.conf_all_f += conf_f
                    self.conf_all_f_categorys += conf_f_categorys
                    self.dist_vertical_all = np.append(
                        self.dist_vertical_all, freespace_res["dist_vertical"]
                    )
                    self.dist_around_all = np.append(
                        self.dist_around_all, freespace_res["dist_around"]
                    )
                    if margin_flag == 1:
                        self.conf_all_f_categorys_with_margin += np.array(
                            freespace_res["conf_f_categorys_with_margin"]
                        )
                        vertical_margins_acc = freespace_res[
                            "vertical_margins_acc"
                        ]
                        around_margins_acc = freespace_res[
                            "around_margins_acc"
                        ]
                        self.vertical_margins_acc_all.append(
                            vertical_margins_acc
                        )
                        self.around_margins_acc_all.append(around_margins_acc)
                self.images_num += 1
                self.images_json_fin.write(
                    "%s\n" % json.dumps(image_dict, cls=NpEncoder)
                )
                self.scores_json_fin.write(
                    "%s\n" % json.dumps(score_dict, cls=NpEncoder)
                )
        self.images_json_fin.close()
        self.scores_json_fin.close()

        try:
            if self.images_num > 0:
                cls_ious_all = self.TPs / (
                    self.FNs + self.TPs + self.FPs + 0.0
                ).astype(np.float64)
                cls_accs_all = self.TPs / (self.FPs + self.TPs).astype(
                    np.float64
                )
                cls_recs_all = self.TPs / (self.FNs + self.TPs).astype(
                    np.float64
                )
            else:
                cls_ious_all, cls_accs_all, cls_recs_all = 0.0, 0.0, 0.0
                self.freespace_dict = None
        except Exception as e:
            logger.error("Please check dataset images num from Above log.")
            raise e

        if self.freespace_dict is not None:
            freespace_res_dict = get_freespace_res(
                TPs_f=self.TPs_f,
                FNs_f=self.FNs_f,
                FPs_f=self.FPs_f,
                conf_all_f=self.conf_all_f,
                dist_vertical_all=self.dist_vertical_all,
                dist_around_all=self.dist_around_all,
                has_margin=self.has_margin,
                has_threshold=self.has_threshold,
                margin_list=self.margin_list,
                vertical_margins_acc_all=self.vertical_margins_acc_all,
                around_margins_acc_all=self.around_margins_acc_all,
                conf_all_f_categorys_with_margin=self.conf_all_f_categorys_with_margin,  # noqa: E501
                labels_dict=self.labels_dict,
                freespace_dict=self.freespace_dict,
                threshold_list=self.threshold_list,
                reslist=self.reslist,
            )
        else:
            freespace_res_dict = None

        writeout_json_all(
            os.path.join(self.outfile_dir, "all.json"),
            cls_ious_all,
            cls_accs_all,
            cls_recs_all,
            self.conf_all,
            self.labels_dict,
            freespace_res_dict,
            self.label_num_list,
            self.images_num,
            self.min_gt_num,
        )
        writeout_labels_dict(
            os.path.join(self.outfile_dir, "labels.json"), self.labels_dict
        )
        result = json.load(open(os.path.join(self.outfile_dir, "all.json")))
        mean_iou = result["mean_iou"]
        if "freespace" in result.keys():
            freespace_iou = result["freespace"]["freespace_iou"]
            content = [
                {"name": "mIOU", "value": mean_iou},
                {"name": "IOU Freespace", "value": freespace_iou},
            ]
        else:
            content = [{"name": "mIOU", "value": mean_iou}]
        return content
