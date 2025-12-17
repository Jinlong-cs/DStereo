import json
from typing import List

import numpy as np

from hat.core.virtual_camera import CameraModelType
from hat.visualize.detection3d.visualize import (
    draw_bbox3d_with_filter,
    draw_bird_eye_view_with_filter,
    draw_mask,
)

failure_vis_cfg = {
    "score_threshold": 0.0,
    "dep_threshold": 300,
    "out_size": 384,
    "world_size": 100,
    "class_names": ["det", "gt"],
    "colors_map": {
        "TP": (0, 255, 0),
        "FP": (0, 0, 255),
        "ignore": (240, 250, 240),
        "gt": (0, 255, 255),
    },
    "layout": "HWC",
    "to_rgb": False,
    "thickness": 2,
    "truncation_thresh": 0.5,
    "min_depth_dist": 0.5,
    "max_depth_dist": 300,
    "draw_truncation_2d": True,
}


def list_failure_samples(result_file: str) -> List[dict]:
    all_result = json.load(result_file)
    samples = list(
        map(
            lambda image_key: {
                "image_key": image_key,
                "det_bboxes": all_result["dets_dict"][image_key],
                "gt_bboxes": all_result["gts_dict"][image_key],
                "img_meta": all_result["img_meta"][image_key],
            },
            all_result["images"],
        )
    )
    return samples


def _list_samples(result_file):
    all_result = json.load(result_file)
    samples = list(
        map(
            lambda image_key: {
                "image_key": image_key,
                "det_bboxes": all_result["dets_dict"][image_key],
                "gt_bboxes": all_result["gts_dict"][image_key],
                "img_meta": all_result["img_meta"][image_key],
            },
            all_result["images"],
        )
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
                        filter(  # noqa
                            lambda det: det["eval_type"] == "TP",
                            sample["det_bboxes"],
                        )
                    ),
                )
            )
            + [-1]
        ),  # noqa
        reverse=True,
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
                        filter(  # noqa
                            lambda det: det["eval_type"] == "FP"
                            or det["eval_type"] == "ignore",
                            sample["det_bboxes"],
                        )
                    ),
                )
            )
            + [-1]
        ),  # noqa
        reverse=True,
    )
    return samples


def list_samples_order_by_drot(result_file):
    samples = _list_samples(result_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda det: det["metrics"]["drot"],
                    list(
                        filter(  # noqa
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


def list_samples_order_by_dxy(result_file):
    samples = _list_samples(result_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda det: det["metrics"]["dxy"],
                    list(
                        filter(  # noqa
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


def draw_sample(image, sample: dict):
    initial_image = image.copy()
    initial_img_h, initial_img_w = initial_image.shape[:2]
    gt_bboxes = sample["gt_bboxes"]
    det_bboxes = sample["det_bboxes"]
    meta = sample["img_meta"]

    camera = None
    camera_model = meta.get("camera_model", None)
    if camera_model is not None:
        thickness = int(np.ceil(min(initial_img_h, initial_img_w) / 1000.0))
        failure_vis_cfg["thickness"] = thickness
        camera_cls = CameraModelType[camera_model].value
        camera = camera_cls(
            camera_matrix=np.array(meta["calib"]),
            distcoeffs=np.array(meta["distCoeffs"]),
            image_size=[meta["image_width"], meta["image_height"]],
            is_virtual=False,
        )

    failure_vis_cfg["draw_truncation_2d"] = True
    draw_bbox3d_with_filter(
        image,
        det_bboxes,
        failure_vis_cfg,
        meta,
        apply_dist_coeff=True,
        camera=camera,
    )
    failure_vis_cfg["draw_truncation_2d"] = False
    draw_bbox3d_with_filter(
        image,
        gt_bboxes,
        failure_vis_cfg,
        meta,
        apply_dist_coeff=True,
        camera=camera,
    )
    # draw mask
    draw_mask(image, meta["ignore_mask"])
    image = draw_bird_eye_view_with_filter(
        image, gt_bboxes + det_bboxes, failure_vis_cfg, meta
    )
    if camera_model is not None:
        drawed_img_h, drawed_img_w = image.shape[:2]
        display_image = np.zeros(
            (
                drawed_img_h + initial_img_h,
                max(drawed_img_w, initial_img_w),
                3,
            ),
            dtype=np.uint8,
        )
        display_image[:initial_img_h, :initial_img_w, :] = initial_image
        display_image[initial_img_h:, :drawed_img_w, :] = image
        return display_image
    return image
