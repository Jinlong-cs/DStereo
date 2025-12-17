import json
from io import BytesIO
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from hat.core.data_struct.base_struct import DetBoxes3D
from hat.visualize.detection_bev.visualize import (
    draw_multi_camera_imgs,
    get_virtual_params,
)

vis_cfg = {
    "score_threshold": 0.2,
    "dep_threshold": 300,
    "colors_map": {
        "TP": (0, 255, 0),
        "FP": (255, 0, 0),
        "FN": (255, 255, 0),
        "gt_ignore": (240, 250, 240),
        "det_ignore": (255, 255, 255),
        "GT": (0, 255, 255),
    },
    "thickness": 2,
}


def list_samples(gt_results_file: str, pred_results_file: str) -> List[Dict]:
    with open(gt_results_file, "r") as w:
        all_gts_result = json.load(w)
    with open(pred_results_file, "r") as w:
        all_preds_result = json.load(w)
    samples = []
    for gt_scence in all_gts_result:
        scence_key = gt_scence["scence_key"]
        img_meta = {}
        for key, img_info in gt_scence["image_objects"].items():
            img_meta[key] = dict(  # noqa
                width=img_info["width"],
                height=img_info["height"],
                calib=img_info["calib"],
                camera_view=img_info["meta"]["camera"],
                image_source=img_info["meta"]["image_source"],
                camera_model=img_info["meta"].get(
                    "camera_model", "PinholeCamera"
                ),
            )
        gt_bboxes = list(gt_scence["objects_3d"].values())
        pred_scences = list(
            filter(
                lambda data: data["scence_key"] == scence_key,
                all_preds_result,
            )
        )
        det_bboxes = (
            list(pred_scences[0]["objects_3d"].values())
            if len(pred_scences) > 0
            else []
        )
        samples.append(
            dict(  # noqa
                scence_key=scence_key,
                det_bboxes=det_bboxes,
                gt_bboxes=gt_bboxes,
                img_meta=img_meta,
                vis_score_threshold=gt_scence["meta"]["eval_vis_cfg"][
                    "vis_score_threshold"
                ],
                vis_image_layout=gt_scence["meta"]["eval_vis_cfg"][
                    "vis_image_layout"
                ],
                vis_camera=gt_scence["meta"]["eval_vis_cfg"]["vis_camera"],
                bird_eye_size=gt_scence["meta"]["eval_vis_cfg"][
                    "bird_eye_size"
                ],
                vcs_range=gt_scence["meta"]["eval_vis_cfg"]["vcs_range"],
                is_multi_category_eval=gt_scence["meta"]["eval_vis_cfg"][
                    "is_multi_category_eval"
                ],
            )
        )

    return samples


def list_samples_order_by_tp(
    gt_results_file: str, pred_results_file: str
) -> List[Dict]:
    samples = list_samples(gt_results_file, pred_results_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda det: det["score"],
                    list(
                        filter(  # noqa
                            lambda det: det["is_tp"] == 1,
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


def list_samples_order_by_fp(
    gt_results_file: str, pred_results_file: str
) -> List[Dict]:
    samples = list_samples(gt_results_file, pred_results_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda det: det["score"],
                    list(
                        filter(  # noqa
                            lambda det: det["is_tp"] == 0 or det["ignore"],
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


def list_samples_order_by_fn(
    gt_results_file: str, pred_results_file: str
) -> List[Dict]:
    samples = list_samples(gt_results_file, pred_results_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda gt: -np.linalg.norm(gt["location"]),
                    list(
                        filter(  # noqa
                            lambda gt: gt["is_fn"] == 1 and not gt["ignore"],
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
                            lambda det: det["is_tp"] == 1,
                            sample["det_bboxes"],
                        )
                    )
                )
            ]
        ),  # noqa
        reverse=True,
    )
    return samples


def list_samples_order_by_drot(
    gt_results_file: str, pred_results_file: str
) -> List[Dict]:
    samples = list_samples(gt_results_file, pred_results_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda det: det["drot"],
                    list(
                        filter(  # noqa
                            lambda det: det["is_tp"] == 1,
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


def list_samples_order_by_dxyp(
    gt_results_file: str, pred_results_file: str
) -> List[Dict]:
    samples = list_samples(gt_results_file, pred_results_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda det: det["dxyp"],
                    list(
                        filter(  # noqa
                            lambda det: det["is_tp"] == 1,
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


def draw_sample(imgs_dict: Dict[str, np.ndarray], sample: Dict) -> np.ndarray:
    img_meta = sample["img_meta"]
    vis_camera = sample["vis_camera"]
    vis_image_layout = sample["vis_image_layout"]
    vis_score_threshold = sample.get("vis_score_threshold", None)
    if vis_score_threshold is not None:
        vis_cfg.update({"score_threshold": vis_score_threshold})
    is_multi_category_eval = sample["is_multi_category_eval"]
    cameras_img_key = {
        meta["camera_view"]: key for key, meta in img_meta.items()
    }
    img_list = []
    img_meta_list = []
    for camera in vis_camera:
        if camera in cameras_img_key:
            image = imgs_dict[cameras_img_key[camera]]
            image_meta = img_meta[cameras_img_key[camera]]
            if image is not None:
                img_list.append(image)
                img_meta_list.append(image_meta)
    gt_cls = []
    for data in sample["gt_bboxes"]:
        if data["ignore"]:
            gt_cls.append(-1)
        else:
            gt_cls.append(data["is_fn"])
    gt_bboxes = DetBoxes3D(
        cls_idxs=torch.Tensor(gt_cls),
        cls_name_mapping={-1: "gt_ignore", 1: "FN", 0: "GT"},
        x=torch.Tensor([data["location"][0] for data in sample["gt_bboxes"]]),
        y=torch.Tensor([data["location"][1] for data in sample["gt_bboxes"]]),
        z=torch.Tensor([data["location"][2] for data in sample["gt_bboxes"]]),
        l=torch.Tensor(
            [data["dimensions"][2] for data in sample["gt_bboxes"]]
        ),
        w=torch.Tensor(
            [data["dimensions"][1] for data in sample["gt_bboxes"]]
        ),
        h=torch.Tensor(
            [data["dimensions"][0] for data in sample["gt_bboxes"]]
        ),
        yaw=torch.Tensor([data["rotation_y"] for data in sample["gt_bboxes"]]),
    )
    gt_addition_info = None
    if (
        len(sample["gt_bboxes"]) > 0
        and sample["gt_bboxes"][0]["category"] is not None
    ):
        gt_addition_info = {
            "camera_model": [
                v.get("camera_model", "PinholeCamera") for v in img_meta_list
            ],
        }
        if is_multi_category_eval:
            gt_addition_info.update(
                {
                    "category": [
                        bbox["category"] for bbox in sample["gt_bboxes"]
                    ]
                }
            )
    det_bboxes = DetBoxes3D(  # noqa
        cls_idxs=torch.Tensor(
            [data["is_tp"] for data in sample["det_bboxes"]]
        ),
        cls_name_mapping={-1: "det_ignore", 0: "FP", 1: "TP"},
        x=torch.Tensor([data["location"][0] for data in sample["det_bboxes"]]),
        y=torch.Tensor([data["location"][1] for data in sample["det_bboxes"]]),
        z=torch.Tensor([data["location"][2] for data in sample["det_bboxes"]]),
        l=torch.Tensor(
            [data["dimensions"][2] for data in sample["det_bboxes"]]
        ),
        w=torch.Tensor(
            [data["dimensions"][1] for data in sample["det_bboxes"]]
        ),
        h=torch.Tensor(
            [data["dimensions"][0] for data in sample["det_bboxes"]]
        ),
        yaw=torch.Tensor(
            [data["rotation_y"] for data in sample["det_bboxes"]]
        ),
        scores=torch.Tensor([data["score"] for data in sample["det_bboxes"]]),
    )
    det_addition_info = None
    if (
        len(sample["gt_bboxes"]) > 0
        and sample["gt_bboxes"][0]["category"] is not None
    ):
        det_addition_info = {
            "camera_model": [
                v.get("camera_model", "PinholeCamera") for v in img_meta_list
            ],
        }
        if is_multi_category_eval:
            det_addition_info.update(
                {
                    "category": [
                        bbox["category"] for bbox in sample["det_bboxes"]
                    ]
                }
            )
    # draw bird eye view map
    bird_eye_size = sample["bird_eye_size"]
    vcs_range = sample["vcs_range"]
    bev_map = np.zeros((bird_eye_size[0], bird_eye_size[1], 3), dtype=np.uint8)
    cu, cv, vf = get_virtual_params(bird_eye_size, vcs_range)
    cv2.circle(
        bev_map,
        (int(cu + 0.5), int(cv + 0.5)),
        3,
        (0, 255, 0),
        thickness=2,
    )
    for r in range(10, int(abs(vcs_range[0])) + 5, 10):
        r = int(vf * r)
        cv2.circle(bev_map, (cu, cv), r, (0, 255, 0), 1)
    # draw gt
    calib_list = [meta["calib"] for meta in img_meta_list]
    img_list, bev_map = draw_multi_camera_imgs(
        img_list,
        bev_map,
        calib_list,
        gt_bboxes,
        vis_cfg,
        bird_eye_size=bird_eye_size,
        vcs_range=vcs_range,
        additional_info=gt_addition_info,
    )
    # draw pred
    img_list, bev_map = draw_multi_camera_imgs(
        img_list,
        bev_map,
        calib_list,
        det_bboxes,
        vis_cfg,
        bird_eye_size=bird_eye_size,
        vcs_range=vcs_range,
        additional_info=det_addition_info,
    )
    # concat
    row = max([layout[0] for layout in vis_image_layout.values()]) + 1
    col = max([layout[1] for layout in vis_image_layout.values()]) + 1
    fig = plt.figure(figsize=(col * 18, row * 12))
    # vis bird eye view map
    layout = vis_image_layout["bird_eye_view"]
    ax = fig.add_subplot(row, col, layout[0] * col + layout[1] + 1)
    ax.imshow(bev_map)
    ax.axis("off")
    ax.set_title("bird eye map", fontsize=18)
    # vis each camera image
    for camera in vis_camera:
        if (
            camera in cameras_img_key
            and imgs_dict[cameras_img_key[camera]] is not None
        ):
            vis_img = img_list.pop(0)
            layout = vis_image_layout[camera]
            ax = fig.add_subplot(row, col, layout[0] * col + layout[1] + 1)
            ax.imshow(vis_img)
            ax.axis("off")
            ax.set_title(f"{camera}:{cameras_img_key[camera]}", fontsize=18)
    plt.tight_layout()
    buff = BytesIO()
    fig.canvas.print_jpg(buff)
    data = buff.getvalue()
    return data


def get_multi_img(sample: Dict) -> Dict:
    img_dict = {}
    img_meta = sample["img_meta"]
    for key, meta in img_meta.items():
        img_path = meta["image_source"].replace("dmpv2://", "/horizon-bucket/")
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_dict[key] = image
    return img_dict
