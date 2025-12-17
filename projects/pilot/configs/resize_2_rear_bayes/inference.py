import os

os.environ["NO_HDFLOW"] = "1"

import torch
from common import (
    cam_frame_to_dict,
    crop_roi_3d,
    input_hw,
    is_with_pe,
    model_setting,
    pe_config,
    training_step,
)
from models import march, val_model
from schedule import get_fuse_patterns_by_stage

from hat.core.data_struct.app_struct import reformat_to_hatbc_msg
from hat.data.collates.collates import default_collate_v2
from hat.data.transforms.classification import ConvertLayout
from hat.data.transforms.frame import CropImgPatch, CV2AdptiveResolutionInput
from hat.data.transforms.real3d import PEGenerator
from hat.utils.apply_func import multi_apply_wrapper

pre, cur = get_fuse_patterns_by_stage(training_step)

img_transforms = [CV2AdptiveResolutionInput(input_hw)]

# TODO: Get rid of inefficient CPU transforms
preprocessors = [
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

if is_with_pe:
    preprocessors.insert(3, multi_apply_wrapper(PEGenerator(pe_config)))

if "galaxy" in model_setting.lower():
    crop = CropImgPatch(crop_roi_3d, is_buf=False)
    preprocessors.insert(1, multi_apply_wrapper(crop))
    img_transforms.insert(0, crop)

inference = dict(
    type="Inference",
    device=None,
    march=march,
    pre_processors=preprocessors,
    post_processors=[
        dict(
            type="FrcnnInvTransforms",
            img_transforms=img_transforms,
            return_meta=True,
        ),
        reformat_to_hatbc_msg,
    ],
    model=val_model,
    model_convert_pipeline=dict(
        type="QATFuseBNConvertPipeline",
        qat_mode="with_bn_reverse_fold",
        pre_stage_fuse_patterns=pre,
        cur_stage_fuse_patterns=cur,
        fuse_part_configs=dict(
            fuse_method="fuse_norm",
            regex=True,
            strict=False,
        ),
    ),
)
