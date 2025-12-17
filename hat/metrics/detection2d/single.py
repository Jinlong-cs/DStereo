# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import logging
import os
import shutil

import cv2
import numpy as np

try:
    from hatbc.message.structure import BBox2D
except ImportError:
    BBox2D = None

from hat.core.eval_setting.parameter_def import DetectionSettingConfig
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages
from .box_property import get_pixel_distance
from .compute_entry import compute_detection_metric
from .generate_data import (
    gen_det_cls_all_data,
    gen_image_tags_data,
    gen_roi_data,
    gen_seq_data,
    get_cls,
    get_distance,
)
from .generate_result import write_eval_stats, write_eval_stats_sep
from .utils import DET, GT

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Detection2dSingle(EvalMetric):
    """Single-frame image detection metrics.

    Args:
        config: config setting.
        save_dir: path used to save json file.
    """

    @require_packages("hatbc")
    def __init__(
        self,
        config: DetectionSettingConfig,
        save_dir: str,
    ):
        self.config = config
        self.save_dir = save_dir

        self.classname = config["classname"]
        self.gt_cls_attr_name = config["gt_cls_attr_name"]
        self.default_type = config["default_type"]
        self.default_overlap = config["default_overlap"]
        self.det_cls_attr_name = config["det_cls_attr_name"]
        self.det_cls_list = config["det_cls"]
        self.image_tags = config["image_tags"]
        self.rois = config["rois"]
        self.det_min_ages = config["det_min_ages"]
        self.target_recalls = config["target_recalls"]
        self.target_precisions = config["target_precisions"]
        self.target_thresholds = config["target_thresholds"]

        self.cam_info_dict = {}
        self.image_tags_dict = {}
        self.image_size_dict = {}
        self.gt_bboxes = {}
        self.det_bboxes = {}

        self.gt_cnt = 0
        self.det_cnt = 0

        if len(self.config["criticalRegions"]):
            self.critical_regions = config["criticalRegions"]
            self.critical_label_dict = {}
        else:
            self.critical_label_dict = None

    def parse_gt(self, anno):
        image_key = anno["image_key"]
        try:
            self.cam_info_dict[image_key] = anno.get("attrs", {}).get(
                "camera_default", None
            )
            self.image_tags_dict[image_key] = anno.get("attrs", {}).get(
                "tags", {}
            )
            img_w, img_h = int(anno["width"]), int(anno["height"])
            self.image_size_dict[image_key] = (img_w, img_h)
            dynamic_region = anno.get("attrs", {}).get("dynamic_region", None)
            self.image_tags_dict[image_key]["dynamic_region"] = dynamic_region
            self.image_tags_dict[image_key]["image_width"] = img_w
            self.image_tags_dict[image_key]["image_height"] = img_h

            # get clip_region
            clip_region = None
            if self.config["clip_to_image"]:
                clip_region = [0, 0, img_w, img_h]
            if len(self.config["clip_region"]) == 4:
                if clip_region is None:
                    clip_region = self.config["clip_region"]
                else:
                    clip_region = [
                        max(clip_region[0], self.config["clip_region"][0]),
                        max(clip_region[1], self.config["clip_region"][1]),
                        min(clip_region[2], self.config["clip_region"][2]),
                        min(clip_region[3], self.config["clip_region"][3]),
                    ]
            if self.config["dynamic_clip"] is True:
                if clip_region is None:
                    clip_region = dynamic_region
                else:
                    clip_region = [
                        max(clip_region[0], dynamic_region[0]),
                        max(clip_region[1], dynamic_region[1]),
                        min(clip_region[2], dynamic_region[2]),
                        min(clip_region[3], dynamic_region[3]),
                    ]
            self.image_tags_dict[image_key]["clip_region"] = clip_region

            if len(self.config["criticalRegions"]):
                assert "parsing" in anno.keys(), (
                    "parsing anno not exist "
                    "while critical regions defined in config"
                )
                cts = anno["parsing"] if anno["parsing"] else []
                critical_label = np.zeros((img_h, img_w), dtype=np.uint8)
                for c in cts:
                    is_critical = any(
                        list(
                            map(
                                lambda critical_region: all(
                                    key in c["attrs"]
                                    and c["attrs"][key] == critical_region[key]
                                    for key in critical_region
                                ),
                                self.critical_regions,
                            )
                        )
                    )
                    if is_critical:
                        pts = c["data"]
                        if None in pts:
                            continue
                        ct_pts = []
                        for pt in pts:
                            ct_pts.append(
                                (int(float(pt[0])), int(float(pt[1])))
                            )
                        ct_pts = np.array(ct_pts)
                        cv2.drawContours(critical_label, [ct_pts], -1, (1), -1)
                self.critical_label_dict[image_key] = critical_label

            self.gt_bboxes[image_key] = []
            GT.count = 0
            if self.classname not in anno:
                return
            for anno_idx, obj in enumerate(anno[self.classname]):
                bbox = BBox2D(list(map(float, obj["data"])))

                cutin_config = self.config["cutin"] != dict()  # noqa: C408
                if cutin_config:
                    is_cutin = (
                        get_pixel_distance(bbox, 0, img_w)
                        <= self.config["cutin"]["distance"]
                    )

                # if outside_img then continue
                outside_image = bbox.outside(img_w, img_h)
                if outside_image:
                    continue

                if (
                    self.gt_cls_attr_name
                    and self.gt_cls_attr_name not in list(obj["attrs"].keys())
                ):
                    msg = (
                        "There is a parameter `gt_cls_attr_name` corresponding to the value %s in your profile. \n"  # noqa: E501
                        "But it is detected that there is no such value in the gt attr attribute of this line. \n"  # noqa: E501
                        "The corresponding image_key is %s, in the %s anno. (All start from 0)"  # noqa: E501
                        % (self.gt_cls_attr_name, image_key, anno_idx)
                    )
                    logger.error(msg)
                    raise Exception(msg)
                gt_cls_type_raw = (
                    obj["attrs"][self.gt_cls_attr_name]
                    if self.gt_cls_attr_name
                    else None
                )
                gt = GT(
                    bbox,
                    self.default_type,
                    self.default_overlap,
                    gt_cls_type_raw=gt_cls_type_raw,
                )

                # add all the attr to gt
                setattr(gt, "gt_attrs", obj["attrs"])  # noqa: B010

                # handle attr_hard
                attrs = copy.deepcopy(self.config["attrs"])
                if (
                    cutin_config is True
                    and is_cutin is True
                    and self.config["cutin"]["attrs"] != dict()  # noqa: C408
                ):
                    for attr_name, profiles in self.config["cutin"][
                        "attrs"
                    ].items():
                        if attr_name not in attrs:
                            attrs[attr_name] = profiles
                        else:
                            for profile in profiles:
                                value = profile["value"]
                                for profiles_ori in attrs[attr_name]:
                                    value_ori = profiles_ori["value"]
                                    if value == value_ori:
                                        attrs[attr_name].remove(profiles_ori)
                                        attrs[attr_name].append(profile)

                for attr_name, profiles in attrs.items():
                    if attr_name in obj["attrs"]:
                        value = obj["attrs"][attr_name]
                        for profile in profiles:
                            if value == profile["value"]:
                                gt.info[profile["type"]].append(
                                    (
                                        attr_name + ":" + value,
                                        profile["overlap"],
                                    )
                                )
                    else:
                        logger.info(
                            "attribute {} not exist in annotation".format(
                                attr_name
                            )
                        )

                # handle keyPartOcclusion hard
                if len(self.config["keyPartOcclusion"]):
                    keyPartOcclusion_score = 0
                    for attr_name, weight_value_score in self.config[
                        "keyPartOcclusion"
                    ]["components"].items():
                        assert (
                            attr_name in obj["attrs"]
                        ), "attribute {} not exist in annotation".format(
                            attr_name
                        )
                        value = obj["attrs"][attr_name]
                        weight = weight_value_score["weight"]
                        for profile in weight_value_score["value_score"]:
                            if value == profile["value"]:
                                score = profile["score"]
                        keyPartOcclusion_score += weight * score

                    for profile in self.config["keyPartOcclusion"]["types"]:
                        low, high = profile["value"].split(",")
                        if float(low) <= keyPartOcclusion_score < float(high):
                            gt.info[profile["type"]].append(
                                (
                                    "keyPartOcclusion"
                                    + ":"
                                    + str(keyPartOcclusion_score),
                                    profile["overlap"],
                                )
                            )

                # only_cutin
                if (
                    cutin_config is True
                    and self.config["cutin"]["only_cutin"] is True
                    and is_cutin is False
                ):
                    gt.info["hard"].append(
                        ("only_cutin: true", self.default_overlap)
                    )

                # clip gt bbox
                if clip_region is not None:
                    old_area = gt.bbox.area
                    gt.bbox.clip_to_region(clip_region)
                    new_area = gt.bbox.area
                    clip_area_ratio = 1 - float(new_area) / old_area
                    if clip_area_ratio > self.config["clip_hard_thresh"]:
                        gt.info["hard"].append(
                            (
                                "clip_hard_thresh: {}".format(
                                    self.config["clip_hard_thresh"]
                                ),
                                self.config["clip_hard_overlap"],
                            )
                        )

                # if remove then continue
                remove = gt.set_type()
                if remove:
                    continue

                if gt.bbox.valid:
                    self.gt_bboxes[image_key].append(gt)
                    self.gt_cnt += 1

        except BaseException as e:
            logger.error("load gts image_key: %s error." % (image_key))
            logger.error(e)
            raise e

    def parse_pred(self, anno):
        image_key = anno["image_key"]
        try:
            # get clip_region
            clip_region = None
            if self.config["clip_to_image"]:
                img_w, img_h = self.image_size_dict[
                    image_key.encode("utf8").decode("utf8")
                ]
                clip_region = [0, 0, img_w, img_h]
            if len(self.config["clip_region"]) == 4:
                if clip_region is None:
                    clip_region = self.config["clip_region"]
                else:
                    clip_region = [
                        max(clip_region[0], self.config["clip_region"][0]),
                        max(clip_region[1], self.config["clip_region"][1]),
                        min(clip_region[2], self.config["clip_region"][2]),
                        min(clip_region[3], self.config["clip_region"][3]),
                    ]
            if self.config["dynamic_clip"] is True:
                dynamic_region = self.image_tags_dict[image_key][
                    "dynamic_region"
                ]
                if clip_region is None:
                    clip_region = dynamic_region
                else:
                    clip_region = [
                        max(clip_region[0], dynamic_region[0]),
                        max(clip_region[1], dynamic_region[1]),
                        min(clip_region[2], dynamic_region[2]),
                        min(clip_region[3], dynamic_region[3]),
                    ]

            self.det_bboxes[image_key] = []
            DET.count = 0
            if self.classname not in anno:
                return
            for anno_idx, obj in enumerate(anno[self.classname]):
                bbox = list(map(float, obj["bbox"]))

                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    logger.warning(
                        "detection bbox {} in image {} is not valid, skip.".format(  # noqa: E501
                            obj["bbox"], image_key
                        )
                    )
                    logger.info(
                        "detection bbox {} in image {} is not valid, skip.".format(  # noqa: E501
                            obj["bbox"], image_key
                        )
                    )
                    continue

                bbox = BBox2D(bbox)

                image_x1 = 0 if clip_region is None else clip_region[0]
                image_x2 = (
                    self.image_size_dict[
                        image_key.encode("utf8").decode("utf8")
                    ][0]
                    if clip_region is None
                    else clip_region[2]
                )
                if (
                    self.config["cutin"] != dict()  # noqa: C408
                    and self.config["cutin"]["only_cutin"] is True
                ):
                    is_cutin = get_pixel_distance(
                        bbox, image_x1, image_x2
                    ) <= (self.config["cutin"]["distance"] + 20)
                    if is_cutin is False:
                        continue

                outside_img = bbox.outside(
                    self.image_size_dict[
                        image_key.encode("utf8").decode("utf8")
                    ][0],
                    self.image_size_dict[
                        image_key.encode("utf8").decode("utf8")
                    ][1],
                )
                if outside_img:
                    continue

                score = obj["bbox_score"]
                if score < self.config["score_thresh"]:
                    continue
                if (
                    self.det_cls_attr_name
                    and self.det_cls_attr_name not in list(obj["attrs"].keys())
                ):
                    msg = (
                        "There is a parameter `det_cls_attr_name` corresponding to the value %s in your profile. \n"  # noqa: E501
                        "But it is detected that there is no such value in the det attr attribute of this line. \n"  # noqa: E501
                        "The corresponding image_key is %s, in the %s anno. (All start from 0)"  # noqa: E501
                        % (self.det_cls_attr_name, image_key, anno_idx)
                    )
                    logger.error(msg)
                    raise Exception(msg)
                det_cls_type_raw = (
                    obj["attrs"][self.det_cls_attr_name]
                    if self.det_cls_attr_name
                    else None
                )
                det = DET(bbox, score=score, det_cls_type_raw=det_cls_type_raw)

                age = (
                    obj["attrs"]["age"]
                    if "attrs" in obj and "age" in obj["attrs"]
                    else None
                )
                setattr(det, "age", age)  # noqa: B010

                setattr(det, "det_attrs", obj.get("attrs", {}))  # noqa: B010

                # clip det bbox
                if clip_region is not None:
                    # old_area = det.bbox.area
                    det.bbox.clip_to_region(clip_region)
                    # new_area = det.bbox.area
                    # clip_area_ratio = 1 - float(new_area) / old_area
                    # if clip_area_ratio > config['clip_hard_thresh']:
                    #     continue

                if det.bbox.valid:
                    self.det_bboxes[image_key].append(det)
                    self.det_cnt += 1
        except BaseException as e:
            logger.error("load dets image_key: %s error." % (image_key))
            logger.error(e)
            raise e

    def update(self, gt, pred):
        assert (
            gt["image_key"] == pred["image_key"]
        ), "The pred and gt do not belong to the same image."
        self.parse_gt(gt)
        self.parse_pred(pred)

    def get(self):
        logger.info("load gts done. Gts number: %s" % self.gt_cnt)
        logger.info("load dets done. Dets number: %s" % self.det_cnt)
        if self.classname not in ["Traffic_light_shell", "traffic_sign"]:
            bbox_undistort = self.config["bbox_undistort"]
            self.gt_bboxes, self.det_bboxes = get_distance(
                self.gt_bboxes,
                self.det_bboxes,
                self.cam_info_dict,
                bbox_undistort=bbox_undistort,
            )
        data = compute_detection_metric(
            self.gt_bboxes,
            self.det_bboxes,
            self.critical_label_dict,
            self.config,
            self.image_tags_dict,
        )
        all_eval_data = {
            "DET": {
                "data": copy.deepcopy(data),
                "eval_types": [],
            }
        }
        all_summry = {}
        eval_types_single_mode = []
        if len(self.det_cls_list):
            data = get_cls(
                data, self.config["gt2eval"], self.config["det2eval"]
            )
            for det_cls in self.det_cls_list:
                if det_cls == "SEP":
                    all_eval_data["DET_CLS_SEP"] = {
                        "data": copy.deepcopy(data),
                        "eval_types": self.config["eval_types"],
                    }
                elif det_cls == "ALL":
                    data_all = gen_det_cls_all_data(data, cover=False)
                    all_eval_data["DET_CLS_ALL"] = {
                        "data": data_all,
                        "eval_types": self.config["eval_types"],
                    }
                elif (
                    "SEP" not in self.det_cls_list
                    and det_cls in self.config["eval_types"]
                ):
                    eval_types_single_mode.append(det_cls)

            if len(eval_types_single_mode) == 1:
                all_eval_data["DET_CLS_SINGLE"] = {
                    "data": copy.deepcopy(data),
                    "eval_types": eval_types_single_mode,
                }
            elif len(eval_types_single_mode) > 1:
                all_eval_data["DET_CLS_SEP"] = {
                    "data": copy.deepcopy(data),
                    "eval_types": eval_types_single_mode,
                }

        for mode, eval_data in all_eval_data.items():
            data = eval_data["data"]
            eval_types = eval_data["eval_types"]
            save_dir = os.path.join(self.save_dir, mode)

            if len(self.image_tags):
                if len(self.image_tags) > 1:
                    raise Exception("Required num image_tags <= 1")
                else:
                    data = gen_image_tags_data(
                        data,
                        True,
                        **self.image_tags[0],
                    )

            if len(self.rois):
                if len(self.rois) > 1:
                    raise Exception("Required num rois <= 1")
                else:
                    data = gen_roi_data(data, True, **self.rois[0])

            if len(self.det_min_ages):
                if len(self.det_min_ages) > 1:
                    raise Exception("Required num det min ges <= 1")
                else:
                    det_min_age = self.det_min_ages[0]
                    data = gen_seq_data(data, True, det_min_age=det_min_age)

            if mode in ["DET_CLS_SEP", "DET_CLS_SINGLE"]:
                summary = write_eval_stats_sep(
                    eval_types,
                    data,
                    save_dir,
                    target_recalls=self.target_recalls,
                    target_precisions=self.target_precisions,
                    target_thresholds=self.target_thresholds,
                    num_workers=(
                        self.config["write_eval_stats_sep_num_workers"]
                        if mode == "DET_CLS_SEP"
                        else 0
                    ),
                    mode=mode,
                    keep_ignore_data=self.config["keep_ignore_data"],
                )
            else:
                summary = write_eval_stats(
                    data,
                    save_dir,
                    target_recalls=self.target_recalls,
                    target_precisions=self.target_precisions,
                    target_thresholds=self.target_thresholds,
                    mode=mode,
                )
            all_summry[mode] = summary

            eval_data["save_dir"] = save_dir
            eval_data["summary"] = summary

            eval_data["output_json"] = os.path.join(
                save_dir, f"{mode}_all.json"
            )
            eval_data["result_json"] = os.path.join(
                save_dir, f"{mode}_result.json"
            )
            eval_data["tables_json"] = os.path.join(
                save_dir, f"{mode}_tables.json"
            )
            eval_data["errors_json"] = os.path.join(
                save_dir, f"{mode}_errors.json"
            )
            eval_data["result_png"] = os.path.join(
                save_dir, f"{mode}_result.png"
            )

        # Generate total all.json, result.json, tables.json, errors.json.
        # Compatible with previous usage.
        output_json = os.path.join(self.save_dir, "all.json")
        result_json = os.path.join(self.save_dir, "result.json")
        tables_json = os.path.join(self.save_dir, "tables.json")
        errors_json = os.path.join(self.save_dir, "errors.json")
        result_png = os.path.join(self.save_dir, "result.png")

        if "DET_CLS_SEP" in all_eval_data:
            target_mode = "DET_CLS_SEP"
        elif "DET_CLS_SINGLE" in all_eval_data:
            target_mode = "DET_CLS_SINGLE"
        elif "DET_CLS_ALL" in all_eval_data:
            target_mode = "DET_CLS_ALL"
        else:
            target_mode = "DET"

        if target_mode != "DET_CLS_SEP":
            shutil.copy(all_eval_data[target_mode]["output_json"], output_json)
            shutil.copy(all_eval_data[target_mode]["result_json"], result_json)
            shutil.copy(all_eval_data[target_mode]["errors_json"], errors_json)
            shutil.copy(all_eval_data[target_mode]["tables_json"], tables_json)
            shutil.copy(all_eval_data[target_mode]["result_png"], result_png)
        else:
            for fname in os.listdir(all_eval_data[target_mode]["save_dir"]):
                src_path = os.path.join(
                    all_eval_data[target_mode]["save_dir"], fname
                )
                dst_path = os.path.join(
                    self.save_dir, fname.split(f"{target_mode}_")[-1]
                )
                shutil.copy(src_path, dst_path)

        logger.info("eval done")
        return all_summry[target_mode]
