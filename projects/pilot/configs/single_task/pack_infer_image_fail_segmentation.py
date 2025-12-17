import os

import torch
from eval_image_fail_segmentation import val_transforms
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

from hat.core.proj_spec.lane_parsing import get_colormap

suffix = os.getenv("HAT_PILOT_PREDICTION_NAME_SUFFIX", "")
prediction_name = f"{model_name}_{suffix}" if suffix else model_name

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"

camera_view_names = [
    "camera_front_left",
    "camera_front_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
]

pred_batch_size = 64

all_callbacks = []
all_data_loaders = []

tasks = (dict(name="image_fail_parsing", important=True),)
task_names = [task["name"] for task in tasks]

roi_region = (0, 0, input_hw[1], input_hw[0])

# data
pack_path = "/horizon-bucket/plat-qa-data/issue-data/artificial-collection/multicam/x8b/C385/CA_YU_C385128/20230115/ADAS_20230115-144540_003_0.pack"
pack_name = os.path.basename(pack_path)
view_index = int(pack_name[-6])
pack_root = os.path.dirname(pack_path)
pred_path = f"./tmp_viz_imgs/{model_name}/{pack_name[:-5]}"

for _ in task_names:
    obj_key = task_names[0]
    visualize_callback = dict(
        type="ComposeVisualize",
        callbacks=[
            dict(
                type="DetMultitaskVisualize",
                out_keys=[obj_key],
                output_dir=pred_path,
                vis_configs=dict(
                    image_fail_parsing=dict(
                        colormap=get_colormap(task_names[0], desc_id),
                        alpha=0.7,
                    ),
                ),
                save_viz_imgs=True,
            ),
        ],
    )
    callbacks = [
        visualize_callback,
        dict(
            type="StatsMonitor",
            log_freq=5,
        ),
    ]

    all_callbacks.append(callbacks)

    data_loader = dict(
        type=torch.utils.data.DataLoader,
        sampler=dict(
            type=torch.utils.data.DistributedSampler,
            shuffle=False,
        ),
        dataset=dict(
            type="PackDataset",
            pack_path=os.path.join(pack_root, pack_name[:-6] + "$Index.pack"),
            pix_format="rgb",
            camera_view_names=[
                camera_view_names[view_index],
            ],
            camera_calib=True,
            transforms=val_transforms,
            to_buf=True,
        ),
        batch_size=pred_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )

    all_data_loaders.append(data_loader)

qat_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=all_data_loaders,
    batch_processor=dict(
        type="BasicBatchProcessor",
        need_grad_update=False,
        inverse_transforms=dict(
            type="FrcnnInvTransforms",
            img_transforms=val_transforms,
        ),
    ),
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
    callbacks=all_callbacks,
    num_epochs=1,
    share_callbacks=False,
)
int_infer_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=all_data_loaders,
    batch_processor=dict(
        type="BasicBatchProcessor",
        need_grad_update=False,
        inverse_transforms=dict(
            type="FrcnnInvTransforms",
            img_transforms=val_transforms,
        ),
    ),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=model_checkpoint
                if model_checkpoint
                else f"aidi://{model_name}/{model_version}/int_infer",
                enable_tracking=enable_model_tracking,
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    callbacks=all_callbacks,
    num_epochs=1,
    share_callbacks=False,
)
