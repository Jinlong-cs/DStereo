import os
import sys

import torch
from image_fail_segmentation import (  # noqa
    desc_id,
    enable_model_tracking,
    input_hw,
    march,
    model_checkpoint,
    model_name,
    model_setting,
    model_version,
    val_model,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from hatbc.message import CameraFrame
from hatbc.message.structure.image import ColorSpace
from hdflow.v2.format.tat_msg.camera_frame import (
    CameraFrameConverter as TatCameraFrameConverter,
)
from project_common import cam_frame_to_dict
from tat._core import PixFormat
from tat.matrix.pack_sdk import TopicChannel

from hat.core.data_struct.app_struct import reformat_to_hatbc_msg
from hat.data.collates.collates import default_collate_v2
from hat.data.transforms.classification import ConvertLayout
from hat.data.transforms.common import BgrToYuv444V2
from hat.data.transforms.frame import CV2AdptiveResolutionInput
from hat.data.transforms.inverse_transforms import FrcnnInvTransforms
from hat.utils.apply_func import multi_apply_wrapper

sys.modules.pop("project_common")

task_names = "image_fail_parsing"


def img_converter(img):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img.copy())
    img = img.permute((2, 0, 1))
    transform = BgrToYuv444V2(rgb_input=True)
    img = transform(img)
    img = img.permute((1, 2, 0))
    return CameraFrame.from_dict(
        {
            "image": {
                "data": img,
                "layout": "hwc",
                "color_space": ColorSpace.YUV444P,
            }
        }
    )


transforms = dict(
    image=img_converter,
    pack=TatCameraFrameConverter(
        image_topic_channel=TopicChannel("image", 0),
        camera_param_topic_channel=TopicChannel("camera_runtime", 0),
        image_layout="HWC",
        image_pix_format=PixFormat.YUV444P,
    ),
)

img_transforms = [CV2AdptiveResolutionInput(input_hw)]

pre_processor = [
    cam_frame_to_dict,
    multi_apply_wrapper(img_transforms[0]),
    multi_apply_wrapper(ConvertLayout()),
    default_collate_v2,
    dict(
        type="ConvertDataType",
        convert_map={"img": torch.float32},
    ),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

post_processor = [
    FrcnnInvTransforms(img_transforms=img_transforms, return_meta=True),
    reformat_to_hatbc_msg,
]
inference = dict(
    type="Inference",
    model=val_model,
    device=None,
    march=march,
    pre_processors=pre_processor,
    post_processors=post_processor,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=model_checkpoint
                if model_checkpoint
                else f"aidi://{model_name}/{model_version}/qat",
                enable_tracking=enable_model_tracking,
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
        ],
    ),
)
