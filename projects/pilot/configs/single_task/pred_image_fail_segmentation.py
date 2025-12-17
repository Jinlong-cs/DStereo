import os

import torch
from image_fail_segmentation import (  # noqa
    input_hw,
    march,
    model_checkpoint,
    model_setting,
    val_model,
)

model_name = os.getenv("HAT_PILOT_MODEL_NAME")
assert model_name is not None
device_ids = [0]

# reset postprocess args in seg task
# pred results for consistency check of seg task dont need postprocess
postprocess = val_model["task_modules"]["image_fail_parsing"]["postprocess"]
postprocess["out_strides"] = [1]
postprocess["input_padding"] = [0, 0, 0, 0]

callbacks = [
    dict(
        type="SaveSegConsistencyResult",
        output_dir=f"./dump_res/{model_name}",
        dump_extra_key=model_name,
        task_list=["image_fail_parsing"],
        task_name_map={"image_fail_parsing": "iqa_parsing"},
    ),
]

HAT_BUCKET_PATH = "/pilot_data_raw"

if input_hw == [256, 480]:
    pred_img_dir = f"{HAT_BUCKET_PATH}/consistency/480x256_j5/"  # noqa
elif input_hw == [320, 480]:
    if "galaxy" in model_setting:
        pred_img_dir = f"{HAT_BUCKET_PATH}/consistency/480x320_j5/"  # noqa
    else:
        pred_img_dir = f"{HAT_BUCKET_PATH}/consistency/480x320/"  # noqa
else:
    pred_img_dir = f"{HAT_BUCKET_PATH}/consistency/512x320/"

# clipping: begin
HAT_BUCKET_PATH = (
    "/home/cidata"
    if os.path.exists("/home/cidata")
    else "/horizon-bucket/HDLTAlgorithm"
)
if "lmdb" in model_setting:
    model_setting = model_setting.replace("_lmdb", "")

if input_hw == [256, 480]:
    pred_img_dir = f"{HAT_BUCKET_PATH}/users/meng01.wang/consistency_v2/480x256_j5/"  # noqa
elif input_hw == [320, 480]:
    if march == "bayes":
        if "galaxy" in model_setting:
            pred_img_dir = f"{HAT_BUCKET_PATH}/users/meng01.wang/consistency_v2/480x320_j5/"  # noqa
        else:
            pred_img_dir = f"{HAT_BUCKET_PATH}/users/meng01.wang/consistency/{model_setting.lower()}"  # noqa
    else:
        pred_img_dir = (
            f"{HAT_BUCKET_PATH}/users/meng01.wang/consistency/480x320/"  # noqa
        )
else:
    pred_img_dir = f"{HAT_BUCKET_PATH}/users/meng01.wang/consistency/512x320/"
# clipping: end


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="YUVFrames",
        img_dir=pred_img_dir,
        im_hw=input_hw,
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
            type="RescaleTransforms",
            factor=2,
            exclude_task=["image_fail_parsing"],
        ),
    ),
    callbacks=callbacks,
    num_epochs=1,
    share_callbacks=False,
)
