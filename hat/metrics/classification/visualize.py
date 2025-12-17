import json
import logging
import os
import traceback

import cv2
import numpy as np

THUMBNAIL_WIDTH = 128
THUMBNAIL_HEIGHT = 128
MAX_NUM_ROW = 10
MAX_NUM_COLUMN = 10
BACKGROUND_COLOR = (255, 255, 255)
TEXT_COLOR = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0)
# padding_ratio = 0.05


def draw_confusion_matrix_samples(
    eval_result: dict,
    image_dir: str,
    output_dir: str,
    output_json: bool = False,
):
    confusion_matrix_samples = {}
    for obj in eval_result["objects"]:
        cm_key = (obj["gt_id"], obj["pred_id"])
        confusion_matrix_samples.setdefault(cm_key, []).append(obj)

    for cm_key in confusion_matrix_samples:
        gt_id, pred_id = cm_key
        gt_category = (
            eval_result["categorys"][gt_id] if gt_id >= 0 else "IGNORE"
        )
        pred_category = (
            eval_result["categorys"][pred_id] if pred_id >= 0 else "IGNORE"
        )
        objs = confusion_matrix_samples[cm_key]
        if not len(objs):
            continue
        objs.sort(key=lambda obj: obj["score"], reverse=True)

        num_column = min(MAX_NUM_COLUMN, len(objs))
        num_row = min(MAX_NUM_ROW, int(np.ceil(len(objs) / float(num_column))))
        canvas = (
            np.ones(
                (num_row * THUMBNAIL_HEIGHT, num_column * THUMBNAIL_WIDTH, 3),
                dtype=np.uint8,
            )
            * np.array(BACKGROUND_COLOR)
        )

        for row in range(num_row):
            for col in range(num_column):
                obj = {"image_key": ""}
                try:
                    obj_idx = row * num_column + col
                    if obj_idx >= len(objs):
                        break
                    obj = objs[obj_idx]
                    if not os.path.exists(
                        os.path.join(image_dir, obj["image_key"])
                    ):
                        continue
                    img = cv2.imread(os.path.join(image_dir, obj["image_key"]))
                    x1, y1, x2, y2 = map(
                        lambda x: max(0, int(round(x))), obj["bbox"]
                    )
                    obj_img = img[y1:y2, x1:x2, :]
                    obj_img = cv2.resize(
                        obj_img, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
                    )
                    cv2.putText(
                        obj_img,
                        text="%.2f" % obj["score"],
                        org=(1, 14),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=SHADOW_COLOR,
                    )
                    cv2.putText(
                        obj_img,
                        text="%.2f" % obj["score"],
                        org=(0, 13),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=TEXT_COLOR,
                    )
                    canvas[
                        row * THUMBNAIL_HEIGHT : (row + 1) * THUMBNAIL_HEIGHT,
                        col * THUMBNAIL_WIDTH : (col + 1) * THUMBNAIL_WIDTH,
                        :,
                    ] = obj_img
                except Exception:
                    logging.warning(
                        "skip visual image, because got error: {}. \
                        ImageKey: {}".format(
                            traceback.format_exc(), obj["image_key"]
                        )
                    )

        cv2.imwrite(
            os.path.join(
                output_dir,
                "GT_%s_PRED_%s_NUM_%d.png"
                % (gt_category, pred_category, len(objs)),
            ),
            canvas,
        )

        if gt_category != pred_category and output_json:
            json_path = os.path.join(
                output_dir,
                "GT_%s_PRED_%s_NUM_%d.json"
                % (gt_category, pred_category, len(objs)),
            )  # noqa
            file = open(json_path, "w+", encoding="utf-8")
            images = [obj["image_key"] for obj in objs]
            json.dump(
                {"GT_%s_PRED_%s" % (gt_category, pred_category): images},
                file,
                indent=2,
            )  # noqa
            file.close()
