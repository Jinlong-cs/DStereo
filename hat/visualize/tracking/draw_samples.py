import os
import pickle
from io import BytesIO
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from hat.visualize.tracking.visualize import (
    draw_multi_camera_imgs,
    get_virtual_params,
)

vis_cfg = {
    "score_threshold": 0.2,
    "dep_threshold": 300,
    "colors_map": {
        "TP": (0, 255, 0),
        "FP": (255, 0, 0),
        "FN": (255, 255, 255),
        "SWITCH": (255, 255, 0),
        "GT": (0, 255, 255),
    },
    "thickness": 2,
}


def list_samples(result_file: str, sort_key: str = "tp") -> List[Dict]:
    with open(result_file, "rb") as f:
        all_result = pickle.load(f)
    samples = {}
    eval_vis_cfg = all_result["eval_vis_cfg"]
    vis_camera = eval_vis_cfg["vis_camera"]
    for sample_id, gt in all_result["gts"].items():
        pred = all_result["preds"][sample_id]
        scene_id = gt.topic
        if scene_id not in samples:
            samples[scene_id] = []
        img_meta = {}
        for camera_frame in gt.get_messages("camera"):
            channel_id = camera_frame.meta.channel
            if channel_id not in vis_camera:
                continue
            img_meta[channel_id] = camera_frame

        samples[scene_id].append(
            dict(  # noqa
                scene_id=scene_id,
                timestamp=gt.meta.timestamp,
                gt=gt.get_messages("overall")[0],
                pred=pred.get_messages("overall")[0],
                img_meta=img_meta,
                vis_score_threshold=eval_vis_cfg["vis_score_threshold"],
                vis_image_layout=eval_vis_cfg["vis_image_layout"],
                vis_camera=vis_camera,
                bird_eye_size=eval_vis_cfg["bird_eye_size"],
                vcs_range=eval_vis_cfg["vcs_range"],
            )
        )
    for scene_id, sample in samples.items():
        samples[scene_id] = sorted(sample, key=lambda x: x["timestamp"])

    sort_index = all_result["sort_index"].get(sort_key)
    sorted_samples = []
    if sort_index is None:
        for _, sample in samples.items():
            sorted_samples.extend(sample)
    else:
        sort_index = sorted(
            sort_index.items(), key=lambda x: x[1], reverse=True
        )
        for scene_id, _ in sort_index:
            sorted_samples.extend(samples[scene_id])
    return sorted_samples


def list_samples_order_by_tp(result_file: str) -> List[Dict]:
    return list_samples(result_file, "tp")


def list_samples_order_by_fp(result_file: str) -> List[Dict]:
    return list_samples(result_file, "fp")


def list_samples_order_by_fn(result_file: str) -> List[Dict]:
    return list_samples(result_file, "fn")


def list_samples_order_by_switch(result_file: str) -> List[Dict]:
    return list_samples(result_file, "switch")


def list_samples_order_by_dist(result_file: str) -> List[Dict]:
    return list_samples(result_file, "dist")


def draw_sample(
    imgs_dict: Dict[str, np.ndarray], sample: Dict, save_local_path: str = None
) -> np.ndarray:
    img_meta = sample["img_meta"]
    vis_camera = sample["vis_camera"]
    vis_image_layout = sample["vis_image_layout"]
    vis_score_threshold = sample.get("vis_score_threshold", None)
    if vis_score_threshold is None:
        vis_score_threshold = vis_cfg["score_threshold"]
    cameras_img_key = {key: meta.image.url for key, meta in img_meta.items()}
    img_list = []
    img_meta_list = []
    for camera in vis_camera:
        if camera in cameras_img_key:
            image = imgs_dict.get(camera)
            image_meta = img_meta.get(camera)
            if image is not None:
                img_list.append(image)
                img_meta_list.append(image_meta)
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
    vis_cfg.update({"score_threshold": -99.0})
    img_list, bev_map = draw_multi_camera_imgs(
        img_list,
        bev_map,
        img_meta_list,
        sample["gt"],
        vis_cfg,
        bird_eye_size=bird_eye_size,
        vcs_range=vcs_range,
    )
    # draw pred
    vis_cfg.update({"score_threshold": vis_score_threshold})
    img_list, bev_map = draw_multi_camera_imgs(
        img_list,
        bev_map,
        img_meta_list,
        sample["pred"],
        vis_cfg,
        bird_eye_size=bird_eye_size,
        vcs_range=vcs_range,
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
        if camera in cameras_img_key and imgs_dict[camera] is not None:
            vis_img = img_list.pop(0)
            layout = vis_image_layout[camera]
            ax = fig.add_subplot(row, col, layout[0] * col + layout[1] + 1)
            ax.imshow(vis_img)
            ax.axis("off")
            ax.set_title(f"{camera}:{cameras_img_key[camera]}", fontsize=18)
    plt.tight_layout()
    if save_local_path:
        save_name = "%s/%s_%s.jpg" % (
            save_local_path,
            sample["scene_id"],
            sample["timestamp"],
        )
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        plt.savefig(save_name, dpi=100)
        plt.clf()
    buff = BytesIO()
    fig.canvas.print_jpg(buff)
    data = buff.getvalue()
    return data
