import os
from copy import deepcopy

import torch
from common import (
    input_hw,
    is_with_pe,
    model_checkpoint,
    model_setting,
    pe_config,
)
from models import march, val_model  # noqa

from hat.data.collates.collates import default_collate_v2

HAT_BUCKET_PATH = (
    "/home/cidata"
    if os.path.exists("/home/cidata")
    else "/horizon-bucket/HDLTAlgorithm"
)

model_name = os.getenv("HAT_PILOT_MODEL_NAME")
assert model_name is not None
device_ids = [0]

# reset postprocess args in seg task
# pred results for consistency check of seg task dont need postproc
for seg_task in ["default_segmentation", "lane_segmentation"]:
    if seg_task in val_model["task_modules"]:
        postprocess = val_model["task_modules"][seg_task]["postprocess"]
        postprocess["out_strides"] = [1]
        postprocess["input_padding"] = [0, 0, 0, 0]

callbacks = [
    dict(
        type="SaveDetConsistencyResult",
        output_dir=f"./dump_res/{model_name}",
        dump_extra_key=model_name,
        task_type_mapping=dict(
            vehicle="FullCar",
            rear="Car",
            person="Ped",
            cyclist="Cyclist",
        ),
        task_name_mapping=dict(
            vehicle_ground_line="ground_line",
            vehicle_flank="ground_line",
            vehicle_wheel_kps="wheel_kps",
            vehicle_wheel_detection="wheel",
            rear_plate_detection="plate",
            person_face_detection="face",
        ),
    ),
    dict(
        type="SaveSegConsistencyResult",
        output_dir=f"./dump_res/{model_name}",
        dump_extra_key=model_name,
        task_list=["default_segmentation", "lane_segmentation"],
    ),
]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="YUVFrames",
        img_dir=f"{HAT_BUCKET_PATH}/users/tian.li/consistency/{model_setting}/resize_2",
        calib_path=f"{HAT_BUCKET_PATH}/users/tian.li/consistency/{model_setting}/resize_2.json",
        im_hw=input_hw,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    collate_fn=default_collate_v2,
)
if is_with_pe:
    data_loader["dataset"]["enable_calib_all"] = True
    data_loader["dataset"]["transforms"] = [
        dict(type="PEGenerator", pe_config=deepcopy(pe_config))
    ]

int_infer_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=model_checkpoint,
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=data_loader,
    batch_processor=dict(
        type="BasicBatchProcessor",
        need_grad_update=False,
        batch_transforms=dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
        inverse_transforms=dict(
            type="RescaleTransforms",
            factor=2,
            exclude_task=["default_segmentation", "lane_segmentation"],
        ),
    ),
    callbacks=callbacks,
    num_epochs=1,
    share_callbacks=False,
)
