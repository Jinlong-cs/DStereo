import os

os.environ["NO_HDFLOW"] = "1"

import torch
from common import (
    cam_frame_to_dict,
    input_hw,
    is_with_pe,
    pe_config,
    training_step,
)
from models import march, val_model
from schedule import get_fuse_patterns_by_stage

from hat.core.data_struct.app_struct import reformat_to_hatbc_msg
from hat.data.collates.collates import default_collate_v2
from hat.data.transforms.classification import ConvertLayout
from hat.data.transforms.frame import CV2InverseTransform
from hat.data.transforms.real3d import PEGenerator
from hat.utils.apply_func import multi_apply_wrapper

original_wh = (1920, 1280)
pre, cur = get_fuse_patterns_by_stage(training_step)

preprocessors = [
    cam_frame_to_dict,
    multi_apply_wrapper(ConvertLayout()),
    default_collate_v2,
    dict(
        type="TorchVisionAdapter",
        interface="Resize",
        size=input_hw,
    ),
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
    preprocessors.insert(2, multi_apply_wrapper(PEGenerator(pe_config)))

inference = dict(
    type="Inference",
    device=None,
    march=march,
    pre_processors=preprocessors,
    post_processors=[
        CV2InverseTransform(
            scale_wh=[0.5, 0.5],
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
