import os
from copy import deepcopy

import torch
from common import (
    decoder3d_input_hw,
    input_hw,
    is_with_pe,
    model_checkpoint,
    model_setting,
    pe_config,
    standardized_calib_all,
    with_cam_standiardization,
)
from models import val_model
from multitask import update_embedding_state_dict

from hat.data.collates.collates import default_collate_v2

HAT_BUCKET_PATH = (
    "/home/cidata"
    if os.path.exists("/home/cidata")
    else "/horizon-bucket/HDLTAlgorithm"
)

model_name = os.getenv("HAT_PILOT_MODEL_NAME")
assert model_name is not None
device_ids = [0]
march = "bayes"

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

if "lmdb" in model_setting:
    model_setting = model_setting.replace("_lmdb", "")

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="YUVFrames",
        img_dir=f"{HAT_BUCKET_PATH}/users/xinjie.wang/consistency/v2/{model_setting.lower()}/resize_2_rear/yuv_img",
        calib_path=f"{HAT_BUCKET_PATH}/users/xinjie.wang/consistency/v2/{model_setting.lower()}/resize_2_rear/camera_0.json",
        im_hw=input_hw,
        enable_calib_all=True,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    collate_fn=default_collate_v2,
)

inverse_transforms = dict(
    type="RescaleTransforms",
    factor=2,
    exclude_task=["default_segmentation", "lane_segmentation"],
)

if is_with_pe and not with_cam_standiardization:
    pe_crop_config = deepcopy(pe_config)
    if "galaxy" in model_setting.lower():
        pe_crop_config.update(
            input_hw=decoder3d_input_hw,
            crop_roi_3d=(0, 0, 1920, 1024),
        )
    data_loader["dataset"]["transforms"] = [
        dict(type="PEGenerator", pe_config=deepcopy(pe_crop_config))
    ]
elif with_cam_standiardization:
    uvmap_software = None
    if "niofy" in model_setting.lower():
        from hat.data.transforms.camera_standardization import load_uvmap

        uvmap_software = load_uvmap(
            f"{HAT_BUCKET_PATH}/users/xinjie.wang/consistency/v2/{model_setting.lower()}/resize_2_rear/uvmap/map_x_float_2.000000.dat",
            f"{HAT_BUCKET_PATH}/users/xinjie.wang/consistency/v2/{model_setting.lower()}/resize_2_rear/uvmap/map_y_float_2.000000.dat",
            decoder3d_input_hw[0],
            decoder3d_input_hw[1],
        )
        uvmap_software_invert = load_uvmap(
            f"{HAT_BUCKET_PATH}/users/xinjie.wang/consistency/v2/{model_setting.lower()}/resize_2_rear/uvmap_invert/map_x_float_h320_w480.dat",
            f"{HAT_BUCKET_PATH}/users/xinjie.wang/consistency/v2/{model_setting.lower()}/resize_2_rear/uvmap_invert/map_y_float_h320_w480.dat",
            input_hw[0] // 2,
            input_hw[1] // 2,
        )
    crop_standardized_calib_all = deepcopy(standardized_calib_all)
    if "galaxy" in model_setting.lower():
        if "center_v" in crop_standardized_calib_all:
            crop_standardized_calib_all["center_v"] -= 100
        crop_standardized_calib_all["image_height"] = 1080
        crop_standardized_calib_all["crop_region"] = (0, 0, 1920, 1024)
    transforms = [
        dict(
            type="CameraStandardization",
            meta_key=(),
            **crop_standardized_calib_all,
            uvmap=uvmap_software,
            uvmap_invert=uvmap_software_invert,
            project_3d_to_vcs=True,
            use_LUT=True,
        )
    ]
    if is_with_pe:
        pe_crop_config = deepcopy(pe_config)
        if "galaxy" in model_setting.lower():
            pe_crop_config.update(
                input_hw=decoder3d_input_hw,
                crop_roi_3d=(0, 0, 1920, 1024),
            )
        pe_generator = dict(
            type="PEGenerator",
            pe_config=pe_crop_config,
        )
        transforms.append(pe_generator)
    data_loader["dataset"]["transforms"] = transforms
    inverse_transforms = dict(
        type="ComposeInverseTransforms",
        transforms=[
            dict(
                type="FrcnnInvTransforms",
                img_transforms=transforms,
            ),
            inverse_transforms,
        ],
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
                state_dict_update_func=update_embedding_state_dict,
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
        inverse_transforms=inverse_transforms,
    ),
    callbacks=callbacks,
    num_epochs=1,
    share_callbacks=False,
)
