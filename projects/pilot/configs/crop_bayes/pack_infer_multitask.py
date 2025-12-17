import os
from copy import deepcopy
from importlib import import_module

import torch
from common import (  # is_with_pe,; pe_config,
    enable_model_tracking,
    model_checkpoint,
    model_name,
    model_version,
    pred_batch_size,
    roi_region,
    tasks,
    training_step,
    val_transforms,
    vanishing_point,
    vis_tasks,
)
from models import val_model  # noqa
from schedule import get_fuse_patterns_by_stage, train_stages

from hat.core.proj_spec.parsing import colormap

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"

suffix = os.getenv("HAT_PILOT_PREDICTION_NAME_SUFFIX", "")
prediction_name = f"{model_name}_{suffix}" if suffix else model_name
eval_part_graph_model_only = False

camera_view_names = [
    "camera_front_left",
    "camera_front_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
]


all_callbacks = []
all_data_loaders = []

colormap = {i: color for i, color in enumerate(colormap)}
colormap[255] = [0, 0, 0]

# data
pack_path = "/horizon-bucket/plat-qa-data/issue-data/artificial-collection/pilot5/0820/Galaxy/LX客户搬运/ADPRO-7840/ADAS_20221111-170851_57_4.pack"
pack_name = os.path.basename(pack_path)
view_index = int(pack_name[-6])
pack_root = os.path.dirname(pack_path)
pred_path = f"./tmp_viz_imgs/{model_name}/{pack_name[:-5]}"

roi_from_img_meta = True
if roi_from_img_meta:
    val_transforms[1]["type"] = "DynamicCropImgPatch"
    val_transforms[1]["roi_source"] = "img_meta"
else:
    val_transforms[1]["type"] = "DynamicCropImgPatch"
    val_transforms[1]["static_roi"] = (
        vanishing_point[0] - roi_region[0],
        vanishing_point[1] - roi_region[1],
        roi_region[2] - roi_region[0],
        roi_region[3] - roi_region[1],
    )

task_names = [task["name"] for task in tasks]

for task_key in task_names:

    if "segmentation" in task_key:
        obj_key = import_module(task_key).task_name
        task_val_transforms = deepcopy(val_transforms)
        task_val_transforms[2].update(static_roi=(0, 0, 1920, 1024))
    else:
        obj_key = import_module(task_key).object_type
        task_val_transforms = deepcopy(val_transforms)

    visualize_callback = dict(
        type="ComposeVisualize",
        callbacks=[
            dict(
                type="DetMultitaskVisualize",
                out_keys=vis_tasks,
                output_dir=os.path.join(pred_path, "images"),
                vis_configs=dict(
                    vehicle=dict(
                        color=(0, 255, 0),
                        thickness=2,
                        points2=dict(),
                    ),
                    vehicle_heatmap_3d_detection=dict(
                        color=(0, 255, 0),
                        thickness=2,
                    ),
                    rear=dict(
                        color=(0, 255, 255),
                        thickness=2,
                    ),
                    person=dict(
                        color=(255, 0, 0),
                        thickness=2,
                    ),
                    cyclist=dict(
                        color=(255, 255, 0),
                        thickness=2,
                    ),
                    default_segmentation=dict(
                        colormap=colormap,
                        alpha=0.7,
                    ),
                    lane_segmentation=dict(
                        colormap=colormap,
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
            transforms=task_val_transforms,
            to_buf=True,
            camera_calib_key="calib_all",
            crop_roi=True,
            with_cam=False,
        ),
        batch_size=pred_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
    )

    # TODO
    # if is_with_pe:
    #     pe_config["verbose"] = 0
    #     if eval_type == "detection_3d" or extra_task_key in [
    #         "vehicle_roi_3d",
    #         "person_roi_3d",
    #         "cyclist_roi_3d",
    #     ]:
    #         data_loader["dataset"].update(
    #             use_dataset_extrinsic=True,
    #         )
    #         pe_config["verbose"] = 1

    #     data_loader["dataset"].update(
    #         pe_config=deepcopy(pe_config),
    #     )

    all_data_loaders.append(data_loader)


base_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=all_data_loaders,
    batch_processor=dict(
        type="MultiBatchProcessor"
        if eval_part_graph_model_only
        else "BasicBatchProcessor",
        need_grad_update=False,
        inverse_transforms=dict(
            type="FrcnnInvTransforms",
            img_transforms=val_transforms,
        ),
    ),
    callbacks=all_callbacks,
    num_epochs=1,
    share_callbacks=False,
)

for stage in train_stages:
    pre, cur = get_fuse_patterns_by_stage(stage)

    predictor = deepcopy(base_predictor)
    predictor.update(
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
            checkpoint_mode="resume",
            checkpoint_configs=dict(
                checkpoint_path=(
                    model_checkpoint
                    if model_checkpoint
                    else f"aidi://{model_name}/{model_version}/{training_step}"
                ),
                enable_tracking=enable_model_tracking,
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            qconfig_params=None,
        )
    )
    globals()[f"{stage}_predictor"] = predictor


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
                checkpoint_path=(
                    model_checkpoint
                    if model_checkpoint
                    else f"aidi://{model_name}/{model_version}/{training_step}"
                ),
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
        inverse_transforms=dict(
            type="FrcnnInvTransforms",
            img_transforms=val_transforms,
        ),
    ),
    callbacks=callbacks,
    num_epochs=1,
    share_callbacks=False,
)
