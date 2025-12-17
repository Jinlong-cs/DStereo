import json

import cv2
import numpy as np

INF = 100000000

gt_type_map = {
    "normal": {
        "name": "GT:normal",
        "color": (0, 255, 255),
        "thickness": 2,
        "display": True,
    },
    "ignore": {
        "name": "GT:ignore",
        "color": (240, 250, 240),
        "thickness": 1,
        "display": True,
    },
    "hard": {
        "name": "GT:hard",
        "color": (4, 104, 4),
        "thickness": 1,
        "display": True,
    },
    "remove": {"display": False},
}
gt_eval_type_map = {
    "FN": {
        "name": "GT:FN",
        "display": True,
        "thickness": 2,
    },
    "TP": {
        "name": "GT:TP",
        "display": True,
        "thickness": 1,
    },
    "IGNORE": {
        "name": "GT:ignore",
        "display": True,
        "thickness": 1,
    },
}
det_type_map = {
    "FP": {
        "name": "DET:FP",
        "color": (0, 0, 255),
        "thickness": 2,
        "display": True,
    },
    "TP": {
        "name": "DET:TP",
        "color": (0, 255, 0),
        "thickness": 1,
        "display": True,
    },
    "IGNORE": {
        "name": "DET:ignore",
        "color": (255, 255, 0),
        "thickness": 1,
        "display": True,
    },
}

comp_model_type_map = {
    "FP": {
        "name": "DIFF FP",
        "color": (0, 0, 255),
        "thickness": 2,
        "display": True,
    },
    "FN": {
        "name": "DIFF FN",
        "color": (0, 255, 255),
        "thickness": 2,
        "display": True,
    },
}

other_map = {
    "dynamic_region": {
        "name": "ROI",
        "color": (128, 0, 128),
        "thickness": 2,
        "display": True,
    }
}

stability_map = {
    "det": {
        "name": "DET: TP",
        "color": (0, 0, 255),
        "thickness": 2,
        "display": True,
    },
    "fit": {
        "name": "GT: FIT",
        "color": (255, 0, 0),
        "thickness": 2,
        "display": True,
    },
}

STABILITY_SHOW_IMG_HEIGHT = 128
STABILITY_SHOW_IMG_WIDTH = 128


def _list_samples(result_file):
    all_result = json.load(result_file)
    samples = list(
        map(
            lambda image_key: {
                "image_key": image_key,
                "det_bboxes": all_result["dets_dict"][image_key],
                "gt_bboxes": all_result["gts_dict"][image_key],
                "image_tags_dict": all_result["image_tags_dict"][image_key]
                if "image_tags_dict" in all_result
                and image_key in all_result["image_tags_dict"]
                else None,
            },
            all_result["images"],
        )
    )
    return samples


def list_samples_order_by_fp(result_file):
    samples = _list_samples(result_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda det: det["score"],
                    list(
                        filter(
                            lambda det: det["eval_type"] == "FP",
                            sample["det_bboxes"],
                        )
                    ),
                )
            )
            + [-1]
        ),
        reverse=True,
    )
    return samples


def list_samples_order_by_tp(result_file):
    samples = _list_samples(result_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda det: det["score"],
                    list(
                        filter(
                            lambda det: det["eval_type"] == "TP",
                            sample["det_bboxes"],
                        )
                    ),
                )
            )
            + [-1]
        ),
        reverse=True,
    )
    return samples


def list_samples_order_by_compare_fp(
    samples1, samples2, score_thresh=0.3, iou=0.8
):
    samples = {}
    for idx in range(len(samples1)):
        det_bboxes1 = list(
            filter(
                lambda bbox: bbox["eval_type"] == "FP"
                and bbox["score"] >= score_thresh,
                samples1[idx]["det_bboxes"],
            )
        )
        det_bboxes2 = list(
            filter(
                lambda bbox: bbox["eval_type"] == "FP"
                and bbox["score"] >= score_thresh,
                samples2[idx]["det_bboxes"],
            )
        )
        det_bboxes1_lst = []
        for idx1, det_bbox1 in enumerate(
            sorted(det_bboxes1, key=lambda b: b["bbox"]["y2"])
        ):
            for _, det_bbox2 in enumerate(
                sorted(det_bboxes2, key=lambda b: b["bbox"]["y2"])
            ):
                bxmin = np.max(
                    [det_bbox1["bbox"]["x1"], det_bbox2["bbox"]["x1"]]
                )
                bymin = np.max(
                    [det_bbox1["bbox"]["y1"], det_bbox2["bbox"]["y1"]]
                )
                bxmax = np.min(
                    [det_bbox1["bbox"]["x2"], det_bbox2["bbox"]["x2"]]
                )
                bymax = np.min(
                    [det_bbox1["bbox"]["y2"], det_bbox2["bbox"]["y2"]]
                )
                bwidth = bxmax - bxmin
                bhight = bymax - bymin
                inner = bwidth * bhight
                union = (
                    (det_bbox1["bbox"]["x2"] - det_bbox1["bbox"]["x1"])
                    * (det_bbox1["bbox"]["y2"] - det_bbox1["bbox"]["y1"])
                    + (det_bbox2["bbox"]["x2"] - det_bbox2["bbox"]["x1"])
                    * (det_bbox1["bbox"]["y2"] - det_bbox1["bbox"]["y1"])
                    - inner
                )
                if inner / union > iou:
                    det_bboxes1_lst.append(idx1)
                    break
        score = 0
        for idx1, det_bbox in enumerate(
            sorted(det_bboxes1, key=lambda b: b["bbox"]["y2"])
        ):
            if idx1 in det_bboxes1_lst:
                continue
            score = np.max([score, det_bbox["score"]])
        samples[samples1[idx]["image_key"]] = score
    rsts = sorted(samples.items(), key=lambda d: d[1], reverse=True)
    return rsts


def list_samples_order_by_compare_fn(
    samples1, samples2, score_thresh=0.3, iou=0.8
):
    samples = {}
    for idx in range(len(samples1)):
        sample1_score_list = [
            -(
                lambda tp_dets: min(
                    list(map(lambda det: det["score"], tp_dets))
                )
                if len(tp_dets)
                else 100
            )(
                list(
                    filter(
                        lambda det: det["eval_type"] == "TP",
                        samples1[idx]["det_bboxes"],
                    )
                )
            )
        ]
        sample2_score_list = [
            -(
                lambda tp_dets: min(
                    list(map(lambda det: det["score"], tp_dets))
                )
                if len(tp_dets)
                else 100
            )(
                list(
                    filter(
                        lambda det: det["eval_type"] == "TP",
                        samples2[idx]["det_bboxes"],
                    )
                )
            )
        ]
        gt_bboxes1 = list(
            filter(
                lambda bbox: bbox["eval_type"] == "FN"
                and bbox["gt_type"] == "normal"
                and max([get_area(bbox)] + sample1_score_list) >= score_thresh,
                samples1[idx]["gt_bboxes"],
            )
        )
        gt_bboxes2 = list(
            filter(
                lambda bbox: bbox["eval_type"] == "FN"
                and bbox["gt_type"] == "normal"
                and max([get_area(bbox)] + sample2_score_list) >= score_thresh,
                samples2[idx]["gt_bboxes"],
            )
        )
        gt_bboxes1_lst = []
        for idx1, gt_bbox1 in enumerate(
            sorted(gt_bboxes1, key=lambda b: b["bbox"]["y2"])
        ):
            for _, gt_bbox2 in enumerate(
                sorted(gt_bboxes2, key=lambda b: b["bbox"]["y2"])
            ):
                bxmin = np.max(
                    [gt_bbox1["bbox"]["x1"], gt_bbox2["bbox"]["x1"]]
                )
                bymin = np.max(
                    [gt_bbox1["bbox"]["y1"], gt_bbox2["bbox"]["y1"]]
                )
                bxmax = np.min(
                    [gt_bbox1["bbox"]["x2"], gt_bbox2["bbox"]["x2"]]
                )
                bymax = np.min(
                    [gt_bbox1["bbox"]["y2"], gt_bbox2["bbox"]["y2"]]
                )
                bwidth = bxmax - bxmin
                bhight = bymax - bymin
                inner = bwidth * bhight
                union = (
                    (gt_bbox1["bbox"]["x2"] - gt_bbox1["bbox"]["x1"])
                    * (gt_bbox1["bbox"]["y2"] - gt_bbox1["bbox"]["y1"])
                    + (gt_bbox2["bbox"]["x2"] - gt_bbox2["bbox"]["x1"])
                    * (gt_bbox1["bbox"]["y2"] - gt_bbox1["bbox"]["y1"])
                    - inner
                )
                if inner / union > iou:
                    gt_bboxes1_lst.append(idx1)
                    break
        score = (
            samples[samples1[idx]["image_key"]]
            if samples1[idx]["image_key"] in samples.keys()
            else 0
        )
        for idx1, gt_bbox in enumerate(
            sorted(gt_bboxes1, key=lambda b: b["bbox"]["y2"])
        ):
            if idx1 in gt_bboxes1_lst:
                continue
            score = np.max(
                [score, max([get_area(gt_bbox)] + sample1_score_list)]
            )
        samples[samples1[idx]["image_key"]] = score
    rsts = sorted(samples.items(), key=lambda d: d[1], reverse=True)
    return rsts


def get_area(gt):
    if gt["bbox_type"] == "BBox2D" or gt["bbox_type"] == "BBOX_2D":
        area = (gt["bbox"]["x2"] - gt["bbox"]["x1"]) * (
            gt["bbox"]["y2"] - gt["bbox"]["y1"]
        )
    elif gt["bbox_type"] == "BBOX_rotation":
        cnt = np.array(
            [
                [gt["bbox"]["x1"], gt["bbox"]["y1"]],
                [gt["bbox"]["x2"], gt["bbox"]["y2"]],
                [gt["bbox"]["x3"], gt["bbox"]["y3"]],
                [gt["bbox"]["x4"], gt["bbox"]["y4"]],
            ]
        ).astype(int)
        box = cv2.minAreaRect(cnt)
        area = box[1][0] * box[1][1]
    else:
        raise NotImplementedError(
            "Bbox Type not Implemented yet: {}".format(gt["bbox_type"])
        )
    return area


def list_samples_order_by_fn(result_file):
    samples = _list_samples(result_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda gt: get_area(gt),
                    list(
                        filter(
                            lambda gt: gt["gt_type"] == "normal"
                            and gt["eval_type"] == "FN",
                            sample["gt_bboxes"],
                        )
                    ),
                )
            )
            + [
                -(
                    lambda tp_dets: min(
                        list(map(lambda det: det["score"], tp_dets))
                    )
                    if len(tp_dets)
                    else 100
                )(
                    list(
                        filter(
                            lambda det: det["eval_type"] == "TP",
                            sample["det_bboxes"],
                        )
                    )
                )
            ]
        ),
        reverse=True,
    )
    return samples


def list_samples_order_by_name(result_file):
    samples = _list_samples(result_file)
    samples.sort(key=lambda sample: sample["image_key"])
    return samples


def list_bad_cases_order_by_slice_name_then_image_key(result_file):
    samples = json.load(result_file)["bad_case"]
    samples.sort(
        key=lambda sample: (
            sample["slice_name"],
            sample["track_id"],
            sample["image_key"],
        )
    )
    return samples


def _draw_legend(image, startx=10, starty=10, text_scale=1.5):
    offset = 17 * text_scale
    board_startx = startx - 10
    board_starty = starty - 10
    board_endx = startx + int(round(150 * text_scale))
    board_endy = (
        int(
            round(
                offset
                * (len(gt_type_map) + len(det_type_map) + len(other_map) + 1)
            )
        )
        + 10
    )
    alpha = 0.5
    image[board_starty:board_endy, board_startx:board_endx] = (
        alpha * image[board_starty:board_endy, board_startx:board_endx]
    )
    shadow_color = (0, 0, 0)

    gt_types = list(map(lambda k: gt_type_map[k], gt_type_map))
    det_types = list(map(lambda k: det_type_map[k], det_type_map))
    other_types = list(map(lambda k: other_map[k], other_map))
    for _, type_info in enumerate(gt_types + det_types + other_types):
        if not type_info["display"]:
            continue
        name = type_info["name"]
        color = type_info["color"]
        thickness = 2
        x1, x2, y1, y2 = (
            startx,
            startx + int(round(0.75 * offset)),
            starty,
            starty + int(round(0.75 * offset)),
        )
        # pts = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]).reshape(
        #    (-1, 2)
        # )
        cv2.rectangle(
            image, (x1 + 2, y1 + 2), (x2 + 2, y2 + 2), shadow_color, thickness
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            name,
            (
                startx + 35 + thickness,
                starty + int(round(0.72 * offset)) + thickness,
            ),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (0, 0, 0),
            thickness,
        )
        cv2.putText(
            image,
            name,
            (startx + 35, starty + int(round(0.72 * offset))),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            color,
            thickness,
        )
        starty += int(round(offset))


def _draw_comp_legend(
    image, startx=10, starty=10, text_scale=1.1, type_name=None
):
    offset = 17 * text_scale
    board_startx = startx - 10
    board_starty = starty - 10
    board_endx = startx + int(round(150 * text_scale))
    board_endy = int(round(offset * (len(comp_model_type_map)))) + 10
    alpha = 0.5
    image[board_starty:board_endy, board_startx:board_endx] = (
        alpha * image[board_starty:board_endy, board_startx:board_endx]
    )
    shadow_color = (0, 0, 0)

    comp_model_types = list(
        map(lambda k: comp_model_type_map[k], comp_model_type_map)
    )
    for _, type_info in enumerate(comp_model_types):
        if not type_info["display"]:
            continue
        name = type_info["name"]
        color = type_info["color"]
        thickness = 2
        if type_name is not None and name.find(type_name) == -1:
            continue
        x1, x2, y1, y2 = (
            startx,
            startx + int(round(0.75 * offset)),
            starty,
            starty + int(round(0.75 * offset)),
        )
        # pts = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]).reshape(
        #    (-1, 2)
        # )
        cv2.rectangle(
            image, (x1 + 2, y1 + 2), (x2 + 2, y2 + 2), shadow_color, thickness
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            name,
            (
                startx + 25 + thickness,
                starty + int(round(0.72 * offset)) + thickness,
            ),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (0, 0, 0),
            thickness,
        )
        cv2.putText(
            image,
            name,
            (startx + 25, starty + int(round(0.72 * offset))),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            color,
            thickness,
        )
        starty += int(round(offset))


def draw_sample(
    image,
    sample,
    score_thresh=0.3,
    max_score_thresh=INF,
    box_shadow=False,
    text_shadow=True,
):
    h, w, c = image.shape
    gt_bboxes = sample["gt_bboxes"]
    det_bboxes = sample["det_bboxes"]
    image_tags_dict = (
        sample["image_tags_dict"] if "image_tags_dict" in sample else None
    )
    text_scale = w / 1280.0
    shadow_color = (0, 0, 0)
    text_offset = (0, -5)
    for gt_bbox in sorted(gt_bboxes, key=lambda b: b["bbox"]["y2"]):
        cls_type = gt_bbox.get("gt_cls_type", None)
        gt_type = gt_bbox["gt_type"]
        gt_eval_type = gt_bbox["eval_type"]
        gt_bbox_type = gt_bbox["bbox_type"]
        if not gt_type_map[gt_type]["display"]:
            continue
        if gt_eval_type_map[gt_eval_type]["display"]:
            thickness = min(
                gt_eval_type_map[gt_eval_type]["thickness"],
                gt_type_map[gt_type]["thickness"],
            )
        else:
            continue
        color = gt_type_map[gt_type]["color"]
        if gt_bbox_type == "BBox2D" or gt_bbox_type == "BBOX_2D":
            if box_shadow:
                cv2.rectangle(
                    image,
                    (
                        int(gt_bbox["bbox"]["x1"] + thickness),
                        int(gt_bbox["bbox"]["y1"] + thickness),
                    ),
                    (
                        int(gt_bbox["bbox"]["x2"] + thickness),
                        int(gt_bbox["bbox"]["y2"] + thickness),
                    ),
                    shadow_color,
                    thickness,
                )
            cv2.rectangle(
                image,
                (int(gt_bbox["bbox"]["x1"]), int(gt_bbox["bbox"]["y1"])),
                (int(gt_bbox["bbox"]["x2"]), int(gt_bbox["bbox"]["y2"])),
                color,
                thickness,
            )
        elif gt_bbox_type == "BBOX_rotation":
            cv2.line(
                image,
                (int(gt_bbox["bbox"]["x1"]), int(gt_bbox["bbox"]["y1"])),
                (int(gt_bbox["bbox"]["x2"]), int(gt_bbox["bbox"]["y2"])),
                color,
                thickness,
            )
            cv2.line(
                image,
                (int(gt_bbox["bbox"]["x2"]), int(gt_bbox["bbox"]["y2"])),
                (int(gt_bbox["bbox"]["x3"]), int(gt_bbox["bbox"]["y3"])),
                color,
                thickness,
            )
            cv2.line(
                image,
                (int(gt_bbox["bbox"]["x3"]), int(gt_bbox["bbox"]["y3"])),
                (int(gt_bbox["bbox"]["x4"]), int(gt_bbox["bbox"]["y4"])),
                color,
                thickness,
            )
            cv2.line(
                image,
                (int(gt_bbox["bbox"]["x4"]), int(gt_bbox["bbox"]["y4"])),
                (int(gt_bbox["bbox"]["x1"]), int(gt_bbox["bbox"]["y1"])),
                color,
                thickness,
            )
        else:
            raise NotImplementedError(
                "Bbox Type not Implemented yet: {}".format(gt_bbox_type)
            )
        if text_shadow:
            cv2.putText(
                image,
                cls_type,
                (int(gt_bbox["bbox"]["x1"]), int(gt_bbox["bbox"]["y2"] + 10)),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                shadow_color,
                thickness,
            )
        cv2.putText(
            image,
            cls_type,
            (int(gt_bbox["bbox"]["x1"]), int(gt_bbox["bbox"]["y2"] + 10)),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            color,
            thickness,
        )
    for det_bbox in sorted(det_bboxes, key=lambda b: b["bbox"]["y2"]):
        cls_type = det_bbox.get("det_cls_type", None)
        det_type = det_bbox["eval_type"]
        det_bbox_type = det_bbox["bbox_type"]
        if det_type != "TP" and not (
            score_thresh <= det_bbox["score"] <= max_score_thresh
        ):
            continue
        if not det_type_map[det_type]["display"]:
            continue
        color = det_type_map[det_type]["color"]
        thickness = det_type_map[det_type]["thickness"]
        if det_bbox_type == "BBox2D" or det_bbox_type == "BBOX_2D":
            if box_shadow:
                cv2.rectangle(
                    image,
                    (
                        int(det_bbox["bbox"]["x1"] + thickness),
                        int(det_bbox["bbox"]["y1"] + thickness),
                    ),
                    (
                        int(det_bbox["bbox"]["x2"] + thickness),
                        int(det_bbox["bbox"]["y2"] + thickness),
                    ),
                    shadow_color,
                    thickness,
                )
            cv2.rectangle(
                image,
                (int(det_bbox["bbox"]["x1"]), int(det_bbox["bbox"]["y1"])),
                (int(det_bbox["bbox"]["x2"]), int(det_bbox["bbox"]["y2"])),
                color,
                thickness,
            )
        elif det_bbox_type == "BBOX_rotation":
            cv2.line(
                image,
                (int(det_bbox["bbox"]["x1"]), int(det_bbox["bbox"]["y1"])),
                (int(det_bbox["bbox"]["x2"]), int(det_bbox["bbox"]["y2"])),
                color,
                thickness,
            )
            cv2.line(
                image,
                (int(det_bbox["bbox"]["x2"]), int(det_bbox["bbox"]["y2"])),
                (int(det_bbox["bbox"]["x3"]), int(det_bbox["bbox"]["y3"])),
                color,
                thickness,
            )
            cv2.line(
                image,
                (int(det_bbox["bbox"]["x3"]), int(det_bbox["bbox"]["y3"])),
                (int(det_bbox["bbox"]["x4"]), int(det_bbox["bbox"]["y4"])),
                color,
                thickness,
            )
            cv2.line(
                image,
                (int(det_bbox["bbox"]["x4"]), int(det_bbox["bbox"]["y4"])),
                (int(det_bbox["bbox"]["x1"]), int(det_bbox["bbox"]["y1"])),
                color,
                thickness,
            )
        else:
            raise NotImplementedError(
                "Bbox Type not Implemented yet: {}".format(det_bbox_type)
            )
        if text_shadow:
            cv2.putText(
                image,
                "%.2f" % det_bbox["score"],
                (
                    int(det_bbox["bbox"]["x1"] + thickness + text_offset[0]),
                    int(det_bbox["bbox"]["y1"] + thickness + text_offset[1]),
                ),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                shadow_color,
                thickness,
            )
            cv2.putText(
                image,
                cls_type,
                (
                    int(det_bbox["bbox"]["x1"]),
                    int(det_bbox["bbox"]["y2"] + 10),
                ),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                shadow_color,
                thickness,
            )
        cv2.putText(
            image,
            "%.2f" % det_bbox["score"],
            (
                int(det_bbox["bbox"]["x1"] + text_offset[0]),
                int(det_bbox["bbox"]["y1"] + text_offset[1]),
            ),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            color,
            thickness,
        )
        cv2.putText(
            image,
            cls_type,
            (int(det_bbox["bbox"]["x1"]), int(det_bbox["bbox"]["y2"] + 10)),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            color,
            thickness,
        )

    if (
        image_tags_dict
        and "dynamic_region" in image_tags_dict
        and image_tags_dict["dynamic_region"] is not None
    ):
        cv2.rectangle(
            image,
            (
                int(image_tags_dict["dynamic_region"][0]),
                int(image_tags_dict["dynamic_region"][1]),
            ),
            (
                int(image_tags_dict["dynamic_region"][2]),
                int(image_tags_dict["dynamic_region"][3]),
            ),
            other_map["dynamic_region"]["color"],
            other_map["dynamic_region"]["thickness"],
        )
        cv2.putText(
            image,
            ",".join(
                list(map(lambda a: str(a), image_tags_dict["dynamic_region"]))
            ),
            (
                int(image_tags_dict["dynamic_region"][0]),
                int(image_tags_dict["dynamic_region"][1] * 0.95),
            ),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            other_map["dynamic_region"]["color"],
            other_map["dynamic_region"]["thickness"],
        )

    startx = 20
    starty = 20
    _draw_legend(image, startx, starty, text_scale * 1.3)


def draw_compare_sample_fp(image, sample1, sample2, score_thresh=0.3, iou=0.8):
    h, w, c = image.shape
    det_bboxes1 = list(
        filter(
            lambda bbox: bbox["eval_type"] == "FP"
            and bbox["score"] >= score_thresh,
            sample1["det_bboxes"],
        )
    )
    det_bboxes2 = list(
        filter(
            lambda bbox: bbox["eval_type"] == "FP"
            and bbox["score"] >= score_thresh,
            sample2["det_bboxes"],
        )
    )
    text_scale = w / 1280.0
    shadow_color = (0, 0, 255)
    text_offset = (0, -5)
    det_bboxes1_lst = []
    for idx1, det_bbox1 in enumerate(
        sorted(det_bboxes1, key=lambda b: b["bbox"]["y2"])
    ):
        for _, det_bbox2 in enumerate(
            sorted(det_bboxes2, key=lambda b: b["bbox"]["y2"])
        ):
            bxmin = np.max([det_bbox1["bbox"]["x1"], det_bbox2["bbox"]["x1"]])
            bymin = np.max([det_bbox1["bbox"]["y1"], det_bbox2["bbox"]["y1"]])
            bxmax = np.min([det_bbox1["bbox"]["x2"], det_bbox2["bbox"]["x2"]])
            bymax = np.min([det_bbox1["bbox"]["y2"], det_bbox2["bbox"]["y2"]])
            bwidth = bxmax - bxmin
            bhight = bymax - bymin
            inner = bwidth * bhight
            union = (
                (det_bbox1["bbox"]["x2"] - det_bbox1["bbox"]["x1"])
                * (det_bbox1["bbox"]["y2"] - det_bbox1["bbox"]["y1"])
                + (det_bbox2["bbox"]["x2"] - det_bbox2["bbox"]["x1"])
                * (det_bbox1["bbox"]["y2"] - det_bbox1["bbox"]["y1"])
                - inner
            )
            if inner / union > iou:
                det_bboxes1_lst.append(idx1)
                break
    for idx, det_bbox in enumerate(
        sorted(det_bboxes1, key=lambda b: b["bbox"]["y2"])
    ):
        det_type = det_bbox["eval_type"]
        if idx in det_bboxes1_lst:
            continue
        color = comp_model_type_map[det_type]["color"]
        thickness = comp_model_type_map[det_type]["thickness"]
        cv2.rectangle(
            image,
            (int(det_bbox["bbox"]["x1"]), int(det_bbox["bbox"]["y1"])),
            (int(det_bbox["bbox"]["x2"]), int(det_bbox["bbox"]["y2"])),
            color,
            thickness,
        )
        cv2.putText(
            image,
            "%.2f" % det_bbox["score"],
            (
                int(det_bbox["bbox"]["x1"] + thickness + text_offset[0]),
                int(det_bbox["bbox"]["y1"] + thickness + text_offset[1]),
            ),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            shadow_color,
            thickness,
        )
    startx = 20
    starty = 20
    _draw_comp_legend(image, startx, starty, text_scale * 1.3, "FP")


def draw_compare_sample_fn(image, sample1, sample2, score_thresh=0.3, iou=0.8):
    h, w, c = image.shape
    sample1_score_list = [
        -(
            lambda tp_dets: min(list(map(lambda det: det["score"], tp_dets)))
            if len(tp_dets)
            else 100
        )(
            list(
                filter(
                    lambda det: det["eval_type"] == "TP", sample1["det_bboxes"]
                )
            )
        )
    ]
    sample2_score_list = [
        -(
            lambda tp_dets: min(list(map(lambda det: det["score"], tp_dets)))
            if len(tp_dets)
            else 100
        )(
            list(
                filter(
                    lambda det: det["eval_type"] == "TP", sample2["det_bboxes"]
                )
            )
        )
    ]
    gt_bboxes1 = list(
        filter(
            lambda bbox: bbox["eval_type"] == "FN"
            and bbox["gt_type"] == "normal"
            and max([get_area(bbox)] + sample1_score_list) >= score_thresh,
            sample1["gt_bboxes"],
        )
    )
    gt_bboxes2 = list(
        filter(
            lambda bbox: bbox["eval_type"] == "FN"
            and bbox["gt_type"] == "normal"
            and max([get_area(bbox)] + sample2_score_list) >= score_thresh,
            sample2["gt_bboxes"],
        )
    )
    text_scale = w / 1280.0
    gt_bboxes1_lst = []
    for idx1, gt_bbox1 in enumerate(
        sorted(gt_bboxes1, key=lambda b: b["bbox"]["y2"])
    ):
        for _, gt_bbox2 in enumerate(
            sorted(gt_bboxes2, key=lambda b: b["bbox"]["y2"])
        ):
            bxmin = np.max([gt_bbox1["bbox"]["x1"], gt_bbox2["bbox"]["x1"]])
            bymin = np.max([gt_bbox1["bbox"]["y1"], gt_bbox2["bbox"]["y1"]])
            bxmax = np.min([gt_bbox1["bbox"]["x2"], gt_bbox2["bbox"]["x2"]])
            bymax = np.min([gt_bbox1["bbox"]["y2"], gt_bbox2["bbox"]["y2"]])
            bwidth = bxmax - bxmin
            bhight = bymax - bymin
            inner = bwidth * bhight
            union = (
                (gt_bbox1["bbox"]["x2"] - gt_bbox1["bbox"]["x1"])
                * (gt_bbox1["bbox"]["y2"] - gt_bbox1["bbox"]["y1"])
                + (gt_bbox2["bbox"]["x2"] - gt_bbox2["bbox"]["x1"])
                * (gt_bbox1["bbox"]["y2"] - gt_bbox1["bbox"]["y1"])
                - inner
            )
            if inner / union > iou:
                gt_bboxes1_lst.append(idx1)
                break
    for idx, gt_bbox in enumerate(
        sorted(gt_bboxes1, key=lambda b: b["bbox"]["y2"])
    ):
        gt_type = gt_bbox["eval_type"]
        if idx in gt_bboxes1_lst:
            continue
        color = comp_model_type_map[gt_type]["color"]
        thickness = comp_model_type_map[gt_type]["thickness"]
        cv2.rectangle(
            image,
            (int(gt_bbox["bbox"]["x1"]), int(gt_bbox["bbox"]["y1"])),
            (int(gt_bbox["bbox"]["x2"]), int(gt_bbox["bbox"]["y2"])),
            color,
            thickness,
        )

    startx = 20
    starty = 20
    _draw_comp_legend(image, startx, starty, text_scale * 1.3, "FN")


def _create_draw_concat_compare_sample_func(draw_compare_sample_func):
    def draw_concat_compare_sample_func(
        image, sample1, sample2, score_thresh=0.3, iou=0.8
    ):
        sample1_image = image.copy()
        draw_sample(sample1_image, sample1, score_thresh=score_thresh)
        sample2_image = image.copy()
        draw_sample(sample2_image, sample2, score_thresh=score_thresh)
        compare_image = image.copy()
        draw_compare_sample_func(
            compare_image, sample1, sample2, score_thresh, iou
        )
        concat_compare = np.vstack(
            (
                np.hstack((image, compare_image)),
                np.hstack((sample1_image, sample2_image)),
            )
        )
        return concat_compare

    return draw_concat_compare_sample_func


draw_concat_compare_sample_fp = _create_draw_concat_compare_sample_func(
    draw_compare_sample_fp
)


draw_concat_compare_sample_fn = _create_draw_concat_compare_sample_func(
    draw_compare_sample_fn
)


def draw_stability_bad_case(image, sample):
    det_bbox = [
        sample["det_bbox"]["x1"],
        sample["det_bbox"]["y1"],
        sample["det_bbox"]["x2"],
        sample["det_bbox"]["y2"],
    ]

    fit_dist_bbox = [
        sample["fit_dist_bbox"]["x1"],
        sample["fit_dist_bbox"]["y1"],
        sample["fit_dist_bbox"]["x2"],
        sample["fit_dist_bbox"]["y2"],
    ]

    max_w_h = max(det_bbox[2] - det_bbox[0], det_bbox[3] - det_bbox[1])
    cx = 0.5 * (det_bbox[0] + det_bbox[2])
    cy = 0.5 * (det_bbox[1] + det_bbox[3])

    src = np.array(
        [
            [cx - 0.5 * max_w_h, cy - 0.5 * max_w_h],
            [cx - 0.5 * max_w_h, cy + 0.5 * max_w_h],
            [cx + 0.5 * max_w_h, cy + 0.5 * max_w_h],
        ],
        dtype=np.float32,
    )

    dst = np.array(
        [
            [
                0.25 * STABILITY_SHOW_IMG_WIDTH,
                0.25 * STABILITY_SHOW_IMG_HEIGHT,
            ],
            [
                0.25 * STABILITY_SHOW_IMG_WIDTH,
                0.75 * STABILITY_SHOW_IMG_HEIGHT,
            ],
            [
                0.75 * STABILITY_SHOW_IMG_WIDTH,
                0.75 * STABILITY_SHOW_IMG_HEIGHT,
            ],
        ],
        dtype=np.float32,
    )

    M = cv2.getAffineTransform(src, dst)
    image = cv2.warpAffine(
        image, M, (STABILITY_SHOW_IMG_HEIGHT, STABILITY_SHOW_IMG_WIDTH)
    )

    if stability_map["det"]["display"]:
        thickness = stability_map["det"]["thickness"]
        color = stability_map["det"]["color"]
        # draw det rect
        pt1 = (det_bbox[0], det_bbox[1], 1)
        pt2 = (det_bbox[2], det_bbox[3], 1)
        pt1 = tuple(map(int, np.dot(M, pt1)))[:2]
        pt2 = tuple(map(int, np.dot(M, pt2)))[:2]
        cv2.rectangle(image, pt1, pt2, color, thickness)
        cv2.putText(
            image,
            "det",
            (pt1[0], pt1[1] - thickness),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            color,
        )

    if stability_map["fit"]["display"]:
        thickness = stability_map["fit"]["thickness"]
        color = stability_map["fit"]["color"]
        # draw fit rect
        pt1 = (fit_dist_bbox[0], fit_dist_bbox[1], 1)
        pt2 = (fit_dist_bbox[2], fit_dist_bbox[3], 1)
        pt1 = tuple(map(int, np.dot(M, pt1)))[:2]
        pt2 = tuple(map(int, np.dot(M, pt2)))[:2]
        cv2.rectangle(image, pt1, pt2, color, thickness)
        cv2.putText(
            image,
            "fit",
            (pt2[0] - 20, pt1[1] - thickness),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            color,
        )

    return image
