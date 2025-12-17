# Copyright (c) Horizon Robotics. All rights reserved.


import json
import os

EVAL_TYPES_LIST = ["PersonRideMotorcycle", "PersonRideBicycle", "other"]
BBOX_OCC_ATTR = ["full_visible", "occluded"]
KP_STATUS = {
    "two_occ": {
        "full_visible": 1,
        "occluded": 1,
        "outside": 0,
    }
}
FAIL_OFFSET = {
    "full_visible": {
        "high": 0,
        "middle": 0.001,
        "low": 0.002,
        "super_low": 0.004,
    },
    "occluded": {
        "high": 0.003,
        "middle": 0.006,
        "low": 0.008,
        "super_low": 0.01,
    },
    "outside": {
        "high": 0,
        "middle": 0.0,
        "low": 0.0,
        "super_low": 0.0,
    },
}


def base_check_eval(label):
    # check object type
    if label["bbox_type"] not in EVAL_TYPES_LIST:
        return False

    # check occlusion
    if label["bbox_occ"] not in BBOX_OCC_ATTR:
        return False

    # check ignore
    if label["bbox_ignore"].lower() == "yes":
        return False

    return True


def parse_gt_anno(anno):
    image_key = anno["image_key"]  # imgpath

    rois = {}
    for roi in anno["person"]:
        roi_id = roi["id"]
        rois[roi_id] = roi

    belong_map = {}
    for belong in anno.get("belong_to", []):
        roi, points = belong.split(":")
        roi_id = int(roi.split("|")[1])
        points_id = int(points.split("|")[1])
        belong_map[points_id] = roi_id

    tmp_anno = {}
    for points in anno.get("p_WheelKeyPoints_2", []):
        point_id = points["id"]
        points_data = points["data"]
        point_attrs = points["point_attrs"]
        if point_id in belong_map.keys():
            roi_id = belong_map[point_id]
            this_roi = rois[roi_id]
            bbox_type = this_roi["attrs"]["type"]
            bbox_blur = (
                this_roi["attrs"]["blur"]
                if "blur" in this_roi["attrs"]
                else "Normal"
            )
            bbox_occ = this_roi["attrs"]["occlusion"]
            bbox_ignore = this_roi["attrs"]["ignore"]
            tmp_info = {
                "point_id": point_id,
                "bbox_type": bbox_type,
                "bbox_blur": bbox_blur,
                "bbox_occ": bbox_occ,
                "bbox_ignore": bbox_ignore,
                "roi_bbox": this_roi["data"],
                "wheel_kps_0": points_data[0],
                "wheel_kps_1": points_data[1],
                "wheel_kps_occ_0": point_attrs[0]["point_label"]["occlusion"],
                "wheel_kps_occ_1": point_attrs[1]["point_label"]["occlusion"],
                "wheel_kps_conf_0": point_attrs[0]["point_label"][
                    "Corner_confidence"
                ],
                "wheel_kps_conf_1": point_attrs[1]["point_label"][
                    "Corner_confidence"
                ],
            }
            need_eval = base_check_eval(tmp_info)
            if need_eval:
                eval_kps_two_occ = [
                    KP_STATUS["two_occ"][tmp_info["wheel_kps_occ_0"]],
                    KP_STATUS["two_occ"][tmp_info["wheel_kps_occ_1"]],
                ]
                fail_offset = [
                    FAIL_OFFSET[tmp_info["wheel_kps_occ_0"]][
                        tmp_info["wheel_kps_conf_0"]
                    ],
                    FAIL_OFFSET[tmp_info["wheel_kps_occ_1"]][
                        tmp_info["wheel_kps_conf_1"]
                    ],
                ]
                tmp_info["eval_kps_two_occ"] = eval_kps_two_occ
                tmp_info["fail_offset"] = fail_offset
                tmp_anno.update({roi_id: tmp_info})
    return image_key, tmp_anno


def loadGtContentsfromJson(gt_json, img_prefix):
    """Load and parse gt contents from json."""
    base_gt_dict = {}
    for line in open(gt_json, "r").readlines():
        anno = json.loads(line)
        image_key, label = parse_gt_anno(anno)
        assert base_gt_dict.get(image_key) is None
        if not os.path.exists(os.path.join(img_prefix, image_key)):
            print(image_key)
            continue
        base_gt_dict[image_key] = label
    return base_gt_dict
