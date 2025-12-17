# Copyright (c) Horizon Robotics. All rights reserved.
import collections
import json
import logging
import os

import cv2
import numpy as np
import yaml

from hat.metrics.detection2d.generate_result import NpEncoder

logger = logging.getLogger(__name__)


def get_list(directory):
    thelist = []
    images_names = os.listdir(directory)
    for image_name in images_names:
        thelist.append(os.path.splitext(image_name)[0])
    logger.info("find %s file" % len(thelist))
    return thelist


def get_list_json(json_path):
    thelist = []
    image_source = {}
    if not os.path.isfile(json_path):
        logger.error("invalid images.json path, please check.")
    json_file = open(json_path, "r")
    for line in json_file.readlines():
        line = json.loads(line)
        if "image_name" in line:
            image_key = line["image_name"]
        if "image_key" in line:
            image_key = line["image_key"]
        thelist.append(os.path.splitext(image_key)[0])
        image_source[image_key] = line["image_source"]
    logger.info("find %s file" % len(thelist))
    return thelist, image_source


def get_gt_list(directory):
    images_names = os.listdir(directory)
    thelist = [i.split("_label")[0] for i in images_names if "label" in i]
    logger.info("find %s file" % len(thelist))
    return thelist


def get_json_list(json_file):
    thelist = []
    with open(json_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            image_info = json.loads(line.strip())
            thelist.append(os.path.splitext(image_info["image_name"])[0])
    return thelist


def get_image_info_dict(json_file):
    image_info_dict = collections.OrderedDict()
    idx = 0
    try:
        with open(json_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                image_info = json.loads(line.strip())
                if "image_name" in image_info:
                    image_info_dict[image_info["image_name"]] = image_info
                elif "image_key" in image_info:
                    image_info_dict[image_info["image_key"]] = image_info
                idx += 1
    except Exception as e:
        logger.error(
            "Please check whether the images.json file is reasonable. \n"
            "You need to ensure that each row is data in json format.\n "
            "And contains the `image_name` attribute. \n"
            "Please check line %s. (line from 0)" % idx
        )
        raise e
    return image_info_dict


def check_validation(images_dir, gt_dir, pred_dir, images_json=None):
    if os.path.isdir(images_dir):
        images_list = get_list(images_dir)
    else:
        images_list, _ = get_list_json(images_dir)
    gt_list = get_gt_list(gt_dir)
    pred_list = get_list(pred_dir)
    problem_images = set(images_list) - (set(gt_list) & set(pred_list))
    if len(problem_images) > 0:
        for image in problem_images:
            logger.error(
                "%s not in ground truth directory or prediction directory"
                % image
            )
        return False
    else:
        if images_json is not None:
            images_info_list = get_json_list(images_json)
            problem_images = set(images_list) - set(images_info_list)
            if len(problem_images) > 0:
                for image in problem_images:
                    logger.error(
                        "%s not match between images dir and image json"
                        % image
                    )
                return False
            else:
                return True
        else:
            return True


def readin_config(config_file):
    try:
        handle = open(config_file)
        thedict = yaml.safe_load(handle)
        handle.close()
    except Exception as e:
        logger.error("Please check config yaml is legal")
        raise e
    labels_dict = {}
    try:
        if "class" in thedict and "classes" not in thedict:
            thedict["classes"] = thedict["class"]
        for element in thedict["classes"]:
            labels_dict[element["label"]] = {
                "color": element["color"],
                "color_mode": thedict.get("color_mode", "bgr"),
                "name": element["name"],
                "ignore": element["ignore"],
            }
        up_scale = thedict.get("up_scale", 1)
        if "freespace" in thedict.keys():
            freespace_dict = thedict["freespace"][0]
        else:
            freespace_dict = None
        if "roi" in thedict.keys():
            roi_dict = thedict["roi"][0]
        else:
            roi_dict = None
        min_gt_num = thedict.get("min_gt_num", 0)
        if "label_remap" in thedict:
            label_remap = np.arange(256, dtype=np.int64)
            for i in thedict["label_remap"]:
                for src in i["src_label"]:
                    label_remap[int(src)] = int(i["dst_label"])
        else:
            label_remap = None
        if "image_tags" in thedict.keys():
            image_tags_dict = thedict["image_tags"][0]
        else:
            image_tags_dict = None
    except Exception as e:
        logger.error(
            "Please check whether the `classes` attribute of "
            "the profile file is legal.\n"
            "This attribute should contain a list. \n"
            "Each element contains at least four attributes: "
            "`label`, `color`, `name`, and `ignore`. \n"
            "And if there is a `freespace` or `roi` attribute, "
            "the attribute must have a value."
        )
        raise e
    return (
        labels_dict,
        up_scale,
        freespace_dict,
        roi_dict,
        min_gt_num,
        label_remap,
        image_tags_dict,
    )


def get_image_dict(
    image_name,
    cls_ious,
    cls_accs,
    cls_recs,
    labels_dict,
    freespace_res,
    TP,
    FN,
    FP,
):
    image_dict = collections.OrderedDict()
    image_dict["image_name"] = image_name
    image_dict["classes"] = []

    valid_labels = sorted(labels_dict.keys())

    sum_iou = 0
    sum_acc = 0
    sum_rec = 0
    count = 0
    for index in range(len(valid_labels)):
        label = valid_labels[index]
        class_name = labels_dict[label]["name"]
        class_dict = collections.OrderedDict()
        class_dict["name"] = class_name
        class_dict["iou"] = round(cls_ious[index].tolist(), 4)
        class_dict["acc"] = round(cls_accs[index].tolist(), 4)
        class_dict["rec"] = round(cls_recs[index].tolist(), 4)
        class_dict["TP"] = TP[index].item()
        class_dict["FN"] = FN[index].item()
        class_dict["FP"] = FP[index].item()
        image_dict["classes"].append(class_dict)
        if labels_dict[label]["ignore"] is False:
            sum_iou += cls_ious[index]
            sum_acc += cls_accs[index]
            sum_rec += cls_recs[index]
            count += 1
    image_dict["mean_iou"] = round(sum_iou / count, 4)
    image_dict["mean_acc"] = round(sum_acc / count, 4)
    image_dict["mean_rec"] = round(sum_rec / count, 4)

    if freespace_res is not None:
        cls_ious_f = freespace_res["cls_ious_f"]
        cls_accs_f = freespace_res["cls_accs_f"]
        margin_flag = freespace_res["margin_flag"]
        image_dict["freespace_iou"] = round(cls_ious_f[0], 4)
        image_dict["freespace_acc"] = round(cls_accs_f[0], 4)
        if margin_flag == 1:
            vertical_margins_acc = freespace_res["vertical_margins_acc"]
            around_margins_acc = freespace_res["around_margins_acc"]
            margin_list = freespace_res["margin_list"]
            for index in range(len(margin_list)):
                image_dict[
                    "vertical_margin_{}_acc".format(str(margin_list[index]))
                ] = round(vertical_margins_acc[index], 4)
                image_dict[
                    "around_margin_{}_acc".format(str(margin_list[index]))
                ] = round(around_margins_acc[index], 4)
    return image_dict


def writeout_json_images(jsonfile_path, images_dict):
    handle = open(jsonfile_path, "w")
    json.dump(images_dict, handle, indent=4, cls=NpEncoder)
    handle.close()
    return


def writeout_json_all(
    jsonfile_path,
    cls_ious_all,
    cls_accs_all,
    cls_recs_all,
    conf,
    labels_dict,
    freespace_res_all,
    label_num_list,
    images_num,
    min_gt_num=0,
):
    handle = open(jsonfile_path, "w")
    all_dict = collections.OrderedDict()
    all_dict["confusion_matrix"] = conf.tolist()
    all_dict["classes"] = []

    valid_labels = sorted(labels_dict.keys())
    sum_iou = 0
    sum_acc = 0
    sum_rec = 0
    count = 0
    if images_num == 0:
        all_dict["mean_iou"] = None
        all_dict["mean_acc"] = None
        all_dict["mean_rec"] = None
    else:
        for index in range(len(valid_labels)):
            label = valid_labels[index]
            class_name = labels_dict[label]["name"]
            class_dict = collections.OrderedDict()
            class_dict["name"] = class_name
            class_dict["iou"] = round(cls_ious_all[index].tolist(), 4)
            class_dict["acc"] = round(cls_accs_all[index].tolist(), 4)
            class_dict["rec"] = round(cls_recs_all[index].tolist(), 4)
            class_dict["number"] = label_num_list[index]
            all_dict["classes"].append(class_dict)
            if (
                labels_dict[label]["ignore"] is False
                and class_dict["number"] >= min_gt_num
            ):
                sum_iou += cls_ious_all[index]
                sum_acc += cls_accs_all[index]
                sum_rec += cls_recs_all[index]
                count += 1
        if count > 0:
            all_dict["mean_iou"] = round(sum_iou / count, 4)
            all_dict["mean_acc"] = round(sum_acc / count, 4)
            all_dict["mean_rec"] = round(sum_rec / count, 4)
        else:
            all_dict["mean_iou"] = 0
            all_dict["mean_acc"] = 0
            all_dict["mean_rec"] = 0
    all_dict["number"] = images_num

    if freespace_res_all is not None:
        # write freespace general result
        freespace_res_dict = collections.OrderedDict()
        freespace_general_dict = freespace_res_all["freespace_general_res"]
        freespace_res_dict["freespace_iou"] = round(
            freespace_general_dict["cls_ious_all_f"][0], 4
        )
        freespace_res_dict["freespace_acc"] = round(
            freespace_general_dict["cls_accs_all_f"][0], 4
        )
        freespace_res_dict["confusion_matrix"] = freespace_general_dict[
            "conf_all_f"
        ].tolist()

        # freespace distance statistic
        freespace_distance_dict = freespace_res_all["freespace_distance_res"]
        freespace_res_dict["freespace_distance_statistic"] = []
        for key, v in freespace_distance_dict.items():
            freespace_res_dict["freespace_distance_statistic"].append({key: v})

        # write freespace margin result
        freespace_margin_res_dict = freespace_res_all["freespace_margin_res"]
        if "margin_list" in freespace_margin_res_dict.keys():
            margin_list = freespace_margin_res_dict["margin_list"]
            freespace_vertical_margin_acc = freespace_margin_res_dict[
                "freespace_vertical_margin_acc"
            ]
            freespace_around_margin_acc = freespace_margin_res_dict[
                "freespace_around_margin_acc"
            ]
            freespace_res_dict["margin_result"] = []
            for margin_num in range(len(margin_list)):

                # 1. acc(point level) and precision(image level)
                vertical_margin_detail = collections.OrderedDict()
                vertical_margin_detail["acc_point"] = round(
                    freespace_vertical_margin_acc[margin_num], 4
                )
                around_margin_detail = collections.OrderedDict()
                around_margin_detail["acc_point"] = round(
                    freespace_around_margin_acc[margin_num], 4
                )
                if "threshold_list" in freespace_margin_res_dict.keys():
                    threshold_list = freespace_margin_res_dict[
                        "threshold_list"
                    ]
                    freespace_vertical_precision = freespace_margin_res_dict[
                        "freespace_vertical_precision"
                    ]
                    freespace_around_precision = freespace_margin_res_dict[
                        "freespace_around_precision"
                    ]
                    for thred_num in range(len(threshold_list)):
                        vertical_margin_detail[
                            "precison_{}_image".format(
                                threshold_list[thred_num]
                            )
                        ] = round(
                            freespace_vertical_precision[
                                margin_num, thred_num
                            ],
                            4,
                        )
                        around_margin_detail[
                            "precison_{}_image".format(
                                threshold_list[thred_num]
                            )
                        ] = round(
                            freespace_around_precision[margin_num, thred_num],
                            4,
                        )
                freespace_res_dict["margin_result"].append(
                    {
                        "vertical_margin_{}".format(
                            str(margin_list[margin_num])
                        ): vertical_margin_detail
                    }
                )
                freespace_res_dict["margin_result"].append(
                    {
                        "around_margin_{}".format(
                            str(margin_list[margin_num])
                        ): around_margin_detail
                    }
                )

                # 2. categorys metrics
                freespace_res_dict[
                    "categories_result_vertical_margin_{}".format(
                        str(margin_list[margin_num])
                    )
                ] = []
                freespace_category_res_dict = freespace_res_all[
                    "freespace_margin_res"
                ]["category_metrics_dict"][margin_num]
                for label_name, metric in freespace_category_res_dict.items():
                    freespace_res_dict[
                        "categories_result_vertical_margin_{}".format(
                            str(margin_list[margin_num])
                        )
                    ].append({label_name: metric})
        all_dict["freespace"] = freespace_res_dict

    json.dump(all_dict, handle, indent=4, cls=NpEncoder)
    handle.close()
    return


def writeout_labels_dict(jsonfile_path, labels_dict):
    handle = open(jsonfile_path, "w")
    labels_info = {"labels": []}
    for label in labels_dict:
        label_info = labels_dict[label].copy()
        label_info["label"] = label
        labels_info["labels"].append(label_info)
    json.dump(labels_info, handle, indent=4, cls=NpEncoder)
    handle.close()


def write_config(config_file):
    handle = open(config_file, "w")
    config_dict = {
        "classes": [
            {
                "label": 0,
                "name": "background",
                "color": [0, 0, 0],
                "ignore": True,
            },
            {
                "label": 1,
                "name": "lane",
                "color": [255, 0, 0],
                "ingore": False,
            },
            {
                "label": 2,
                "name": "curb",
                "color": [0, 255, 0],
                "ignore": False,
            },
        ],
        "up_scale": 4,
    }
    yaml.dump(config_dict, handle)
    handle.close()
    return


def get_line(label, anno_map, ignore_idx=-1):
    h, w = label.shape
    line = np.zeros(w)
    line_categorys = np.zeros(w)
    freespace_label = anno_map[label]
    for x in range(w):
        for y in range(freespace_label.shape[0])[::-1]:
            if freespace_label[y, x] == 1:
                line[x] = y + 1
                line_categorys[x] = label[y, x]
                break
            elif freespace_label[y, x] == 255:
                line[x] = ignore_idx  # ignore idx
                line_categorys[x] = 255
                continue
    return line, line_categorys


def get_distance_transform(line, h, w):
    label = np.zeros((h, w), dtype=np.uint8) + 1
    for i in range(w):
        if line[i] == -1:
            continue
        label[int(line[i]) - 1, i] = 0
    gt_distance_map = cv2.distanceTransform(label, cv2.DIST_L1, 5)
    return gt_distance_map


def get_undistort_image(image, image_camera):
    undisort_image = image_camera.get_undistorted_image(
        image, image.shape[0], image.shape[1]
    )
    return undisort_image


def get_undistort_label(label, image_camera):
    undisort_label = image_camera.get_undistorted_label_map(
        label, label.shape[0], label.shape[1]
    )
    return undisort_label


def dmp_path_to_local(path, is_on_cluster=True):
    if "dmpv2" in path:
        if is_on_cluster:
            path = path.replace("dmpv2://", "/bucket/input/")
        else:
            path = path.replace("dmpv2://", "/horizon-bucket/")
    else:
        logger.warning("image source is not dmp path, please check")
    return path
