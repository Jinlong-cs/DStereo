import json
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np
import torch

from hat.core.data_struct.base_struct import DetRotBoxes2D
from hat.visualize.detection_bev_rotbox.visualize import draw_multi_camera_imgs

vis_cfg = {
    "score_threshold": 0.2,
    "dep_threshold": 300,
    "colors_map": {
        "TP": (0, 255, 0),  # green
        "FP": (255, 0, 0),  # red
        "FN": (255, 255, 0),  # yellow
        "gt_ignore": (240, 250, 240),
        "det_ignore": (255, 255, 255),
        "GT": (0, 255, 255),
    },
    "thickness": 2,
}


def list_samples(result_file: str) -> List[Dict]:
    with open(result_file, "r") as w:
        all_result = json.load(w)
    samples = []
    for inst_info in all_result["all_info_list"]:
        for ts, info in inst_info.items():
            img_meta = {}
            for camera_name in info["meta"]["imgs_meta"]:
                img_info = info["meta"]["imgs_meta"][camera_name]
                image_key = img_info["image_key"]
                img_meta[image_key] = dict(  # noqa
                    width=img_info["shape"][1],
                    height=img_info["shape"][0],
                    calib=img_info["calib"],
                    camera_view=camera_name,
                    camera_model=img_info.get("camera_model", "PinholeCamera"),
                )
            all_bboxes = info["object_info"]
            det_bboxes = []
            gt_bboxes = []
            for box in all_bboxes:
                box_attr = box["attr"]
                if box_attr == "FN":
                    gt_bboxes.append(box)
                elif box_attr == "FP":
                    det_bboxes.append(box)
                else:
                    gt_bboxes.append(box["gt"])
                    det_bboxes.append(box)
            samples.append(
                dict(  # noqa
                    timestamp=ts,
                    all_bboxes=all_bboxes,
                    det_bboxes=det_bboxes,
                    gt_bboxes=gt_bboxes,
                    img_meta=img_meta,
                    vis_score_threshold=info["meta"]["eval_vis_cfg"][
                        "vis_score_threshold"
                    ],
                    vis_image_layout=info["meta"]["eval_vis_cfg"][
                        "vis_image_layout"
                    ],
                    vis_bird_eye_size=info["meta"]["eval_vis_cfg"][
                        "vis_bird_eye_size"
                    ],
                    bev_size=info["meta"]["eval_vis_cfg"]["bev_size"],
                    vcs_range=info["meta"]["eval_vis_cfg"]["vcs_range"],
                    is_multi_category_eval=info["meta"]["eval_vis_cfg"][
                        "is_multi_category_eval"
                    ],
                )
            )

    return samples


def list_samples_order_by_tp(result_file: str) -> List[Dict]:
    """Sort by the maximum predicted tp socre in descending order per frame."""
    samples = list_samples(result_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda det: det["pred_score"],
                    list(
                        filter(  # noqa
                            lambda det: det["attr"] == "TP",
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


def list_samples_order_by_fp(result_file: str) -> List[Dict]:
    """Sort by the maximum predicted fp socre in descending order per frame."""
    samples = list_samples(result_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda det: det["pred_score"],
                    list(
                        filter(  # noqa
                            lambda det: det["attr"] == "FP",
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


def list_samples_order_by_fn(result_file: str) -> List[Dict]:
    """Sort by the maximum fn dictance and tp pred socre in descending order per frame."""  # noqa
    samples = list_samples(result_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda gt: -np.linalg.norm(gt["vcs_discobj_loc"]),
                    list(
                        filter(  # noqa
                            lambda gt: gt["attr"] == "FN"
                            and not gt["vcs_discobj_ignore"],
                            sample["gt_bboxes"],
                        )
                    ),
                )
            )
            + [
                -(
                    lambda tp_dets: min(
                        list(map(lambda det: det["pred_score"], tp_dets))
                    )
                    if len(tp_dets)
                    else 100
                )(
                    list(
                        filter(
                            lambda det: det["attr"] == "TP",
                            sample["det_bboxes"],
                        )
                    )
                )
            ]
        ),  # noqa
        reverse=True,
    )
    return samples


def list_samples_order_by_drot(result_file: str) -> List[Dict]:
    """Sort by the maximum predicted tp drot in descending order per frame."""
    samples = list_samples(result_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda det: det["drot"],
                    list(
                        filter(  # noqa
                            lambda det: det["attr"] == "TP",
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


def list_samples_order_by_dxy(result_file: str) -> List[Dict]:
    """Sort by the maximum predicted tp dxy in descending order per frame."""
    samples = list_samples(result_file)
    samples.sort(
        key=lambda sample: max(
            list(
                map(
                    lambda det: det["dxy"],
                    list(
                        filter(  # noqa
                            lambda det: det["attr"] == "TP",
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


def bev_6v(
    multiview_imgs: list,
    per_extra_img_size: tuple,
    cam2row_idx: dict,
    bev_img: np.array,
):
    # generate 6v
    img_row1 = np.zeros(
        (
            per_extra_img_size[1],
            per_extra_img_size[0] * 3,
            3,
        )
    )

    img_row1[
        :, per_extra_img_size[0] : per_extra_img_size[0] * 2, :
    ] = cv2.resize(
        multiview_imgs[cam2row_idx["row1_6v_idx"][0]],
        per_extra_img_size,
    )

    img_row2 = np.zeros(
        (
            per_extra_img_size[1],
            per_extra_img_size[0] * 3,
            3,
        )
    )
    img_row2[:, 0 : per_extra_img_size[0], :] = cv2.resize(
        multiview_imgs[cam2row_idx["row2_6v_idx"][0]],
        per_extra_img_size,
    )

    img_row2[
        :,
        per_extra_img_size[0] * 2 : per_extra_img_size[0] * 3,
        :,
    ] = cv2.resize(
        multiview_imgs[cam2row_idx["row2_6v_idx"][1]],
        per_extra_img_size,
    )

    img_row3 = np.concatenate(
        [
            cv2.resize(multiview_imgs[idx], per_extra_img_size)
            for idx in cam2row_idx["row3_6v_idx"]
        ],
        axis=1,
    )

    ret_img = np.concatenate((img_row1, img_row2, img_row3), axis=0)

    return ret_img


def bev_4v(
    multiview_imgs: list,
    per_extra_img_size: tuple,
    cam2row_idx: dict,
    bev_img: np.array,
):
    raise NotImplementedError


def bev_7v(
    multiview_imgs: list,
    per_extra_img_size: tuple,
    cam2row_idx: dict,
    bev_img: np.array,
):
    img_6v = bev_6v(multiview_imgs, bev_img)
    img_6v[
        0 : per_extra_img_size[1],
        per_extra_img_size[0] * 2 : per_extra_img_size[0] * 3,
        :,
    ] = cv2.resize(
        multiview_imgs[cam2row_idx["row1_narrow_idx"][0]],
        per_extra_img_size,
    )
    ret_img = np.concatenate((img_6v, bev_img), axis=1)

    return ret_img


def bev_10v(
    multiview_imgs: list,
    per_extra_img_size: tuple,
    cam2row_idx: dict,
    bev_img: np.array = None,
):
    # 6v
    img_row1 = np.zeros(
        (
            per_extra_img_size[1],
            per_extra_img_size[0] * 3,
            3,
        )
    )
    img_row1[
        :, per_extra_img_size[0] : per_extra_img_size[0] * 2, :
    ] = cv2.resize(
        multiview_imgs[cam2row_idx["row1_6v_idx"][0]],
        per_extra_img_size,
    )

    img_row2 = np.zeros(
        (
            per_extra_img_size[1],
            per_extra_img_size[0] * 3,
            3,
        )
    )
    img_row2[:, 0 : per_extra_img_size[0], :] = cv2.resize(
        multiview_imgs[cam2row_idx["row2_6v_idx"][0]],
        per_extra_img_size,
    )

    img_row2[
        :, per_extra_img_size[0] : per_extra_img_size[0] * 2, :
    ] = cv2.resize(
        multiview_imgs[cam2row_idx["row1_4v_idx"][0]],
        per_extra_img_size,
    )

    img_row2[
        :,
        per_extra_img_size[0] * 2 : per_extra_img_size[0] * 3,
        :,
    ] = cv2.resize(
        multiview_imgs[cam2row_idx["row2_6v_idx"][1]],
        per_extra_img_size,
    )

    img_row3 = np.concatenate(
        [
            cv2.resize(multiview_imgs[idx], per_extra_img_size)
            for idx in cam2row_idx["row2_4v_idx"]
        ],
        axis=1,
    )

    img_row4 = np.concatenate(
        [
            cv2.resize(multiview_imgs[idx], per_extra_img_size)
            for idx in cam2row_idx["row3_6v_idx"]
        ],
        axis=1,
    )
    ret_img = np.concatenate((img_row1, img_row2, img_row3, img_row4), axis=0)
    if bev_img is not None:
        ret_img = np.concatenate((ret_img, bev_img), axis=1)
    return ret_img


def bev_11v(
    multiview_imgs: list,
    per_extra_img_size: tuple,
    cam2row_idx: dict,
    bev_img: np.array,
):
    img_10v = bev_10v(multiview_imgs, per_extra_img_size, cam2row_idx, bev_img)
    img_10v[
        0 : per_extra_img_size[1],
        per_extra_img_size[0] * 2 : per_extra_img_size[0] * 3,
        :,
    ] = cv2.resize(
        multiview_imgs[cam2row_idx["row1_narrow_idx"][0]],
        per_extra_img_size,
    )

    return img_10v


def draw_sample(imgs_dict: Dict[str, np.ndarray], sample: Dict) -> np.ndarray:
    img_meta = sample["img_meta"]
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
    for camera in cameras_img_key:
        image = imgs_dict[cameras_img_key[camera]]
        image_meta = img_meta[cameras_img_key[camera]]
        if image is not None:
            img_list.append(image)
            img_meta_list.append(image_meta)
    gt_cls = []
    for data in sample["gt_bboxes"]:
        if data["vcs_discobj_ignore"]:
            gt_cls.append(-1)
        else:
            gt_cls.append(1)
    gt_bboxes = DetRotBoxes2D(
        cls_idxs=torch.Tensor(gt_cls),
        cls_name_mapping={-1: "gt_ignore", 1: "FN", 0: "GT"},
        x=torch.Tensor(
            [data["vcs_discobj_loc"][0] for data in sample["gt_bboxes"]]
        ),
        y=torch.Tensor(
            [data["vcs_discobj_loc"][1] for data in sample["gt_bboxes"]]
        ),
        w=torch.Tensor(
            [data["vcs_discobj_wh"][0] for data in sample["gt_bboxes"]]
        ),
        h=torch.Tensor(
            [data["vcs_discobj_wh"][1] for data in sample["gt_bboxes"]]
        ),
        yaw=torch.Tensor(
            [data["vcs_discobj_yaw"] for data in sample["gt_bboxes"]]
        ),
        vcs_points=torch.Tensor(
            [data["vcs_points"] for data in sample["gt_bboxes"]]
        ),
    )
    gt_addition_info = None
    if (
        len(sample["gt_bboxes"]) > 0
        and sample["gt_bboxes"][0]["vcs_discobj_cls"] is not None
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
                        bbox["vcs_discobj_cls"] for bbox in sample["gt_bboxes"]
                    ]
                }
            )
    det_bboxes = DetRotBoxes2D(  # noqa
        cls_idxs=torch.Tensor(
            [data["attr"] == "TP" for data in sample["det_bboxes"]]
        ),
        cls_name_mapping={-1: "det_ignore", 0: "FP", 1: "TP"},
        x=torch.Tensor([data["pred_loc"][0] for data in sample["det_bboxes"]]),
        y=torch.Tensor([data["pred_loc"][1] for data in sample["det_bboxes"]]),
        w=torch.Tensor([data["pred_wh"][0] for data in sample["det_bboxes"]]),
        h=torch.Tensor([data["pred_wh"][1] for data in sample["det_bboxes"]]),
        yaw=torch.Tensor([data["pred_yaw"] for data in sample["det_bboxes"]]),
        scores=torch.Tensor(
            [data["pred_score"] for data in sample["det_bboxes"]]
        ),
        vcs_points=torch.Tensor(
            [data["pred_vcs_points"] for data in sample["det_bboxes"]]
        ),
    )
    det_addition_info = None
    if (
        len(sample["det_bboxes"]) > 0
        and sample["det_bboxes"][0]["pred_bev_discobj_cls_id"] is not None
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
                        bbox["pred_bev_discobj_cls_id"]
                        for bbox in sample["det_bboxes"]
                    ]
                }
            )
    # draw bird eye view map
    vis_bird_eye_size = sample["vis_bird_eye_size"]
    bev_size = sample["bev_size"]
    vis_ipm_scale = vis_bird_eye_size[0] / bev_size[0]
    vcs_range = sample["vcs_range"]
    if "bird_eye_view" in cameras_img_key.keys():
        bev_map = img_list.pop(-1)
    else:
        bev_map = np.zeros(
            (vis_bird_eye_size[0], vis_bird_eye_size[1], 3), dtype=np.uint8
        )
    if tuple(vis_bird_eye_size) != bev_map.shape[:2]:
        resize_size = [
            int(bev_map.shape[1] * vis_ipm_scale),
            int(bev_map.shape[0] * vis_ipm_scale),
        ]
        bev_map = cv2.resize(bev_map, resize_size)
    # draw gt
    calib_list = [meta["calib"] for meta in img_meta_list]
    img_list, bev_map = draw_multi_camera_imgs(
        img_list,
        bev_map,
        calib_list,
        gt_bboxes,
        vis_cfg,
        bev_size=bev_size,
        vis_ipm_scale=vis_ipm_scale,
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
        bev_size=bev_size,
        vis_ipm_scale=vis_ipm_scale,
        vcs_range=vcs_range,
        additional_info=det_addition_info,
    )
    # vis each camera image and bird view
    cam2idx = {}
    for idx, cam in enumerate(cameras_img_key):
        cam2idx[cam] = idx
    cam2row_idx = defaultdict(list)
    multi_views_type = "bev_" + str(len(cameras_img_key) - 1) + "v"
    assert multi_views_type in [
        "bev_6v",
        "bev_4v",
        "bev_7v",
        "bev_10v",
        "bev_11v",
    ]

    if multi_views_type in ["bev_6v", "bev_7v", "bev_10v", "bev_11v"]:
        cam2row_idx["row1_6v_idx"] = [
            cam2idx[cam]
            for cam in [
                "camera_front",
            ]
        ]
        cam2row_idx["row2_6v_idx"] = [
            cam2idx[cam]
            for cam in [
                "camera_front_left",
                "camera_front_right",
            ]
        ]
        cam2row_idx["row3_6v_idx"] = [
            cam2idx[cam]
            for cam in [
                "camera_rear_left",
                "camera_rear",
                "camera_rear_right",
            ]
        ]
    if multi_views_type in ["bev_4v", "bev_10v", "bev_11v"]:
        cam2row_idx["row1_4v_idx"] = [
            cam2idx[cam]
            for cam in [
                "fisheye_front",
            ]
        ]
        cam2row_idx["row2_4v_idx"] = [
            cam2idx[cam]
            for cam in ["fisheye_left", "fisheye_rear", "fisheye_right"]
        ]

    if multi_views_type in ["bev_7v", "bev_11v"]:
        cam2row_idx["row1_narrow_idx"] = [
            cam2idx[cam]
            for cam in [
                "camera_front_30fov",
            ]
        ]
    vis_img_scale = vis_bird_eye_size[0] / bev_size[0]
    vis_image_layout = sample["vis_image_layout"]
    per_extra_img_size = (
        int(bev_size[1] * vis_img_scale / vis_image_layout[1]),
        int(bev_size[0] * vis_img_scale / vis_image_layout[0]),
    )  # (w,h)
    func = eval(f"{multi_views_type}")
    img = func(img_list, per_extra_img_size, cam2row_idx, bev_map).astype(
        np.uint8
    )
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.putText(
        img,
        sample["timestamp"],
        (10, 30),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 255, 0),
        2,
    )
    # save imgs to local path, for debug
    # timestamp = sample["timestamp"]
    # cv2.imwrite(f"./{timestamp}.jpg", img)
    return img
