import cv2

gt_eval_type_map = {
    "UNMATCHED": {
        "name": "GT:unmatched",
        "color": (0, 255, 255),
        "display": True,
        "thickness": 2,
    },
    "MATCHED": {
        "name": "GT:matched",
        "color": (0, 128, 128),
        "display": True,
        "thickness": 1,
    },
    "HARD": {
        "name": "GT:hard",
        "color": (4, 104, 4),
        "display": True,
        "thickness": 1,
    },
    "IGNORE": {
        "name": "GT:ignore",
        "color": (240, 250, 240),
        "display": True,
        "thickness": 1,
    },
    "REMOVE": {"display": False},
}
det_type_map = {
    "FP": {
        "name": "DET:FP",
        "color": (0, 0, 255),
        "thickness": 2,
        "display": True,
    },
    "UNKNOWN": {
        "name": "DET:unknown",
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


def sort_samples_by_fp(result, fp_type=None):
    result.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda fp_det: fp_det["score"],
                    list(
                        filter(
                            lambda det: det["eval_type"] == "FP"
                            and (fp_type is None or det["fp_type"] == fp_type),
                            sample["dets"],
                        )
                    ),
                )
            )
            + [-1]
        ),
        reverse=True,
    )
    return result


def _draw_legend(image, startx=10, starty=10, text_scale=1.5):
    offset = 17 * text_scale
    board_startx = startx - 10
    board_starty = starty - 10
    board_endx = startx + int(round(150 * text_scale))
    num_gt_eval_type = len(
        list(
            filter(lambda t: gt_eval_type_map[t]["display"], gt_eval_type_map)
        )
    )
    num_det_eval_type = len(det_type_map)
    board_endy = (
        int(round(offset * (num_gt_eval_type + num_det_eval_type + 1))) + 10
    )
    alpha = 0.5
    image[board_starty:board_endy, board_startx:board_endx] = (
        alpha * image[board_starty:board_endy, board_startx:board_endx]
    )
    shadow_color = (0, 0, 0)

    gt_types = list(map(lambda k: gt_eval_type_map[k], gt_eval_type_map))
    det_types = list(map(lambda k: det_type_map[k], det_type_map))
    for _, type_info in enumerate(gt_types + det_types):
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
        #     (-1, 2)
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


def draw_sample(image, sample, box_shadow=False, text_shadow=True):
    h, w, c = image.shape
    gt_bboxes = sample["gts"]
    det_bboxes = sample["dets"]
    text_scale = w / 1280.0
    shadow_color = (0, 0, 0)
    text_offset = (0, -5)
    for gt_bbox in sorted(gt_bboxes, key=lambda b: b["bbox"]["y2"]):
        gt_eval_type = gt_bbox["eval_type"]
        gt_bbox_type = gt_bbox["bbox_type"]
        if gt_eval_type_map[gt_eval_type]["display"]:
            thickness = min(
                gt_eval_type_map[gt_eval_type]["thickness"],
                gt_eval_type_map[gt_eval_type]["thickness"],
            )
        else:
            continue
        color = gt_eval_type_map[gt_eval_type]["color"]
        if gt_bbox_type == "BBox2D":
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
    for det_bbox in sorted(det_bboxes, key=lambda b: b["bbox"]["y2"]):
        det_type = det_bbox["eval_type"]
        det_bbox_type = det_bbox["bbox_type"]
        if not det_type_map[det_type]["display"]:
            continue
        color = det_type_map[det_type]["color"]
        thickness = det_type_map[det_type]["thickness"]
        if det_bbox_type == "BBox2D":
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

    startx = 20
    starty = 20
    _draw_legend(image, startx, starty, text_scale * 1.3)
