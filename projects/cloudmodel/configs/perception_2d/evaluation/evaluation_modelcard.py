from typing import List

import numpy as np
import torch
from evaluation import save_predict_result
from hatbc.message.frame import CameraFrame, Image


def dict_to_camera_frame(sample):
    cf_list = []
    for idx, img in enumerate(sample["img"]):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        cf = CameraFrame(
            image=Image(
                data=img,
                name=sample["img_name"][idx],
                layout=sample["layout"][idx],
                color_space=sample["color_space"][idx],
                shape=sample["img_shape"][idx],
            )
        )
        cf_list.append(cf)
    return cf_list


def camera_frame_list_to_dict(msgs: List[CameraFrame]):
    """Convert camera frame to hat dict.

    Args:
        msgs: list of camera frame.
    """
    img_meta = dict(
        img=[],
        img_name=[],
        layout=[],
        img_shape=[],
        color_space=[],
        img_height=[],
        img_width=[],
    )

    for msg in msgs:
        img = msg.image.data
        img_meta["img"].append(img)
        img_meta["img_shape"].append(img.shape)
        img_meta["layout"].append(msg.image.layout.lower())
        img_meta["img_name"].append(msg.image.name)
        img_meta["color_space"].append(msg.image.color_space)
        img_meta["img_height"].append(img.shape[0])
        img_meta["img_width"].append(img.shape[1])
    img_meta["img"] = np.stack(img_meta["img"])

    return img_meta


def reformat_mc_prediction_fn(batch, model_outs, **kwargs):
    task_name = kwargs.get("task_name", None)
    name_dict = kwargs.get("name_dict", None)
    assert task_name, "`task_name` can not be None."

    task_results = model_outs[task_name]

    if isinstance(batch, tuple) and isinstance(batch[0][0], CameraFrame):
        batch = camera_frame_list_to_dict(batch[0])

    all_results = save_predict_result(
        batch, task_results, task_name, name_dict=name_dict
    )
    return all_results
