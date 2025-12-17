import os
from typing import List

import numpy as np
from hatbc.message import CameraFrame
from hatbc.message.structure.image import ColorSpace

# env variables
num_machines = int(os.getenv("HAT_NUM_MACHINES", "1"))
pipeline_test = os.getenv("HAT_PIPELINE_TEST", "0") == "1"
training_step = os.getenv("HAT_TRAINING_STEP", "int_infer")
enable_model_tracking = os.getenv("HAT_ENABLE_MODEL_TRACKING") == "1"

model_checkpoint = os.getenv("HAT_PILOT_MODEL_CHECKPOINT")
model_setting = os.getenv("HAT_PILOT_MODEL_SETTING")
model_name_postfix = os.getenv("HAT_PILOT_MODEL_NAME_POSTFIX")
model_version = os.getenv("HAT_PILOT_MODEL_VERSION", "v0.0.1")
model_thresh = os.getenv("HAT_PILOT_MODEL_THRESH")
tasks = os.getenv("HAT_PILOT_TASKS")
resume_training = os.getenv("HAT_PILOT_RESUME_TRAINING", None)
split_flag = os.getenv("HAT_PILOT_SPLIT_FLAG", None)
eval_data_setting = os.getenv("HAT_PILOT_EVAL_DATA_SETTING", None)

pack_infer_vis = os.getenv("HAT_PILOT_PACK_INFER_VIS", "1") == "1"
pack_infer_consist = os.getenv("HAT_PILOT_PACK_INFER_CONSIST", "0") == "1"


def cam_frame_to_dict(data: List[CameraFrame]):
    """Convert camera frame to hat dict.

    Args:
        msg: camera frame.
        keep_attr: whether to keep all attributes. Defaults to False.
    """

    results = []
    for msg in data:
        if msg.image.data is None:
            msg.image.read()

        assert msg.image.color_space == ColorSpace.YUV444P
        assert msg.image.layout.lower() == "hwc"

        img_meta = dict(
            meta=msg.meta,
            img=msg.image.data,
            img_shape=msg.image.data.shape,
            layout=msg.image.layout.lower(),
            img_url=msg.image.url,
            color_space=msg.image.color_space,
            img_name=msg.image.name,
            img_height=msg.image.shape[0],
            img_width=msg.image.shape[1],
        )

        param = msg.camera_param
        if param is not None:
            img_meta["calib"] = np.array(
                [
                    [param.focal_u, 0.0, param.center_u, 0.0],
                    [0.0, param.focal_v, param.center_v, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
            img_meta["distCoeffs"] = np.array(param.distort)

            calib_all = {}
            calib_all["center_u"] = param.center_u
            calib_all["center_v"] = param.center_v
            calib_all["camera_z"] = param.camera_z
            calib_all["focal_u"] = param.focal_u
            calib_all["focal_v"] = param.focal_v
            calib_all["roll"] = param.roll
            calib_all["pitch"] = param.pitch
            calib_all["distort"] = param.distort
            calib_all["image_height"] = msg.image.shape[1]
            calib_all["image_width"] = msg.image.shape[2]

            img_meta["calib_all"] = calib_all

        results.append(img_meta)

    return results
