import os

import torch
from bev_common import grid_quant_scale, per_view_shape, warp_sizes
from common import input_hw, model_checkpoint, model_setting
from models import test_model, val_model

HAT_BUCKET_PATH = (
    "/home/cidata"
    if os.path.exists("/home/cidata")
    else "/horizon-bucket/HDLTAlgorithm"
)

model_name = os.getenv("HAT_PILOT_MODEL_NAME")
assert model_name is not None
device_ids = [0]
march = "bayes"


# use val_model with decoder, but modify model cfg
for task_key, task_cfg in val_model["task_modules"].items():
    if "bev" in task_key:
        # modify compile model
        bev_fusion = task_cfg["model"]["stage2_module"]["bevfusion"]
        bev_fusion["compile_model"] = True
        # replace input layout
        val_model["opt_inputs"] = test_model["opt_inputs"]
        val_model["inputs"] = test_model["inputs"]
        val_model["task_inputs"] = test_model["task_inputs"]

        task_cfg["model"]["stage2_module"]["head"]["postprocess"][
            "center_offset"
        ] = False

# reset postprocess args in seg task
# pred results for consistency check of seg task dont need postproc
for seg_task in ["default_segmentation", "lane_segmentation"]:
    if seg_task in val_model["task_modules"]:
        postprocess = val_model["task_modules"][seg_task]["model"][
            "postprocess"
        ]
        postprocess["out_strides"] = [1]
        postprocess["input_padding"] = [0, 0, 0, 0]


callbacks = [
    dict(
        type="SaveBevDetConsistencyResult",
        output_dir=f"./dump_res/{model_name}",
        dump_extra_key=model_name,
        task_type_mapping=dict(
            bev_3d_vehicle="FullCar",
            bev_3d_vehicle_cls="FullCar",
            bev_3d_cyclist="Cyclist",
            bev_3d_cyclist_cls="Cyclist",
            bev_3d_pedestrian="Ped",
        ),
    ),
    dict(
        type="SaveDetConsistencyResult",
        output_dir=f"./dump_res/{model_name}",
        dump_extra_key=model_name,
        task_type_mapping=dict(
            vehicle_plate="plate",
            face="face",
        ),
        task_name_mapping=dict(
            vehicle_plate="plate",
            face="face",
        ),
        bev_mode=True,
    ),
    dict(
        type="SaveSegConsistencyResult",
        output_dir=f"./dump_res/{model_name}",
        dump_extra_key=model_name,
        task_list=["default_segmentation", "lane_segmentation"],
        bev_mode=True,
    ),
]
view_img_hw = {
    view.replace("camera_", ""): input_hw for view, _ in per_view_shape.items()
}
view_homo_hw = {
    view: size for view, size in zip(view_img_hw.keys(), warp_sizes)
}

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="YUVMultiCamera",
        data_dir=f"{HAT_BUCKET_PATH}/users/yiding.liu/consistency/{model_setting.lower()}/",
        json_path=f"{HAT_BUCKET_PATH}/users/yiding.liu/consistency/{model_setting.lower()}/img_info.json",
        view_img_hw=view_img_hw,
        view_homo_hw=view_homo_hw,
        grid_quant_scale=grid_quant_scale,
        transforms=[
            dict(type="ToTensor", to_yuv=False),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
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
                allow_miss=True,
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
        inverse_transforms=dict(
            type="RescaleTransforms",
            factor=2,
            exclude_task=[
                "default_segmentation",
                "lane_segmentation",
                "bev_3d_vehicle_cls",
                "bev_3d_pedestrian",
                "bev_3d_cyclist_cls",
            ],
        ),
    ),
    callbacks=callbacks,
    num_epochs=1,
    share_callbacks=False,
)
