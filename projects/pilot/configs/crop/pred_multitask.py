import os

import torch
from common import model_checkpoint, model_setting, origin_img_hw, roi_region
from models import march, val_model  # noqa

HAT_BUCKET_PATH = (
    "/home/cidata"
    if os.path.exists("/home/cidata")
    else "/horizon-bucket/HDLTAlgorithm"
)

model_name = os.getenv("HAT_PILOT_MODEL_NAME")
assert model_name is not None
device_ids = [0]


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
]

val_transforms = [
    dict(
        type="CropImgPatch",
        static_roi=roi_region,
        is_buf=False,
    )
]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="YUVFrames",
        img_dir=f"{HAT_BUCKET_PATH}/users/tian.li/consistency/{model_setting}/crop",
        im_hw=origin_img_hw,
        transforms=val_transforms,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
)

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
            type="FrcnnInvTransforms",
            img_transforms=val_transforms,
        ),
    ),
    callbacks=callbacks,
    num_epochs=1,
    share_callbacks=False,
)
