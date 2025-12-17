import os
from collections import defaultdict
from copy import deepcopy
from importlib import import_module

import torch

from hat.core.data_struct.app_struct import reformat_bev_det_to_aidi_eval
from hat.data.collates.collates import collate_3d
from hat.utils import Config
from projects.pilot.configs.bev_7v.base import get_homo_transforms
from projects.pilot.configs.bev_7v.bev_3d_base import (
    bev_common_transforms,
    use_distorted_offset,
)
from projects.pilot.configs.bev_7v.common import (
    H_persp_view_scale,
    camera_view_names,
    crop_roi,
    enable_model_tracking,
    front_camera_view_names,
    img_ori_size,
    img_resize_scale,
    model_checkpoint,
    model_name,
    model_setting,
    model_version,
    narrow_camera_view_names,
    side_camera_view_names,
    spatial_resolution,
    tasks,
    val_batch_size_per_gpu,
    vcs_plane_heights,
    vcs_range,
)
from projects.pilot.configs.bev_7v.models import val_model
from projects.pilot.configs.bev_7v.multitask import freeze_bn_callback
from projects.pilot.configs.bev_7v.schedule import train_stages

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"

suffix = os.getenv("HAT_PILOT_PREDICTION_NAME_SUFFIX", "")
prediction_name = f"{model_name}_{suffix}" if suffix else model_name

device_ids = [0]
log_rank_zero_only = True

eval_tags = ["all"]

# task args
task_args = dict()

task_adapter_func = dict()
all_callbacks = []
all_data_loaders = []

# data
eval_data_setting = model_setting.lower().replace("_lmdb", "")
ds_path = os.path.join(
    os.path.dirname(__file__),
    f"../datasets/{eval_data_setting}_eval_datasets.py",
)
dataset_ids = Config.fromfile(ds_path).dataset_ids

local_datapath = os.path.join(
    bucket_root, "auto_eval/adas_eval/eval_platform/fs/"
)

anno_camera_view_names = {
    view_name: view_name.replace("camera_", "")
    for view_name in camera_view_names
}

multi_view_transforms = {
    view: [
        dict(type="YUVTurboJPEGDecoder", to_string=True),
        dict(
            type="BPUPyramidResizer",
            scale_wh=(img_resize_scale[view], img_resize_scale[view]),
            pyramid_type="ips",
        ),
        dict(
            type="CropImgPatch",
            static_roi=crop_roi[view],
            is_buf=False,
        ),
        dict(type="ToTensor", to_yuv=False),
        dict(type="Normalize", mean=128.0, std=128.0),
    ]
    for view in camera_view_names
}

multi_view_collect = dict(
    type="MultiViewImgCollect",
    target_keys=["img", "side_img", "narrow_img"],
    collect_keys=[
        front_camera_view_names,
        side_camera_view_names,
        narrow_camera_view_names,
    ],
)

# bev relate
bev_3d_transforms = [
    bev_common_transforms["ANCResize3DV"],
    bev_common_transforms["ANCCrop3DV"],
    bev_common_transforms["ANCPad3DV"],
]

homo_transforms = get_homo_transforms(
    transforms_list=bev_3d_transforms, camera_view_names=camera_view_names
)

homo_cfg = dict(
    calib_path=None,
    homo_path=None,
    spatial_resolution=spatial_resolution,
    vcs_range=vcs_range,
    camera_view_names=camera_view_names,
    task_camera_view_names=camera_view_names,
    per_view_shape=img_ori_size,
    vcs_plane_heights=vcs_plane_heights,
    use_distorted_offset=use_distorted_offset,
    homo_transforms=homo_transforms,
    H_persp_view_scale=H_persp_view_scale,
)

task_names = [task["name"] for task in tasks]

for eval_type, all_datasets in dataset_ids.items():
    for task_cfg, datasets in all_datasets.items():

        if isinstance(task_cfg, tuple):
            extra_task_key, task_key = task_cfg
        else:
            extra_task_key = None
            task_key = task_cfg

        # skip datasets of tasks not included in current model
        if extra_task_key is not None:
            if extra_task_key not in task_names:
                continue
        elif task_key not in task_names:
            continue

        if eval_type == "bev_3d":
            task_name = import_module(task_key).task_name
            obj_key = import_module(task_key).object_type
            task_output_name = task_key
            reformat_output_fn = reformat_bev_det_to_aidi_eval
            reformat_out_fn_kwargs = dict(
                obj_key=task_name, dump_obj_key=obj_key
            )
        else:
            raise TypeError(f"Unsupport eval_type for {eval_type}")

        for ds in datasets:
            ds_key = "datasets"
            ds_gt = "gt.json"
            if isinstance(ds, tuple):
                if len(ds) == 2:
                    ds, ds_key = ds
                elif len(ds) == 3:
                    ds, ds_key, ds_gt = ds
                else:
                    raise ValueError
            dataset_path = os.path.join(
                local_datapath, str(ds), "datasets", ds_key
            )
            data_path = os.path.join(dataset_path, "data")

            if os.path.exists(
                os.path.join(
                    local_datapath,
                    str(ds),
                    "datasets",
                    ds_key,
                    ds_gt,
                )
            ):
                gt_path = os.path.join(
                    local_datapath,
                    str(ds),
                    "datasets",
                    ds_key,
                    ds_gt,
                )
            else:
                gt_path = None

            eval_callback = dict(
                type="AIDIEval",
                aidi_eval_dataset_id=ds,
                output_root=f"./eval_res/{ds}",
                prediction_name=prediction_name,
                prediction_tags=eval_tags,
                project_id=os.getenv("PROJECT_ID"),
                reformat_output_fn=reformat_output_fn,
                reformat_out_fn_kwargs=reformat_out_fn_kwargs,
            )
            visualize_callback = dict(
                type="ComposeVisualize",
                callbacks=[
                    dict(
                        type="BEVMultitaskVisualize",
                        out_keys=[task_output_name],
                        output_dir=f"./tmp_viz_imgs/{model_name}/{ds}",
                        bird_eye_size=(
                            512,
                            960,
                        ),
                        vcs_range=vcs_range,
                        vis_image_layout=dict(
                            front_30fov=[0, 0],
                            front=[0, 1],
                            front_left=[1, 0],
                            bird_eye_view=[1, 1],
                            front_right=[1, 2],
                            rear_left=[2, 0],
                            rear=[2, 1],
                            rear_right=[2, 2],
                        ),
                        vis_configs=dict(
                            bev_3d_vehicle=dict(
                                colors_map=defaultdict(lambda: (0, 255, 0)),
                                score_threshold=0.3,
                            ),
                            bev_3d_vehicle_cls=dict(
                                colors_map=defaultdict(lambda: (0, 255, 0)),
                                score_threshold=0.3,
                            ),
                            bev_3d_pedestrian=dict(
                                colors_map=defaultdict(lambda: (0, 255, 0)),
                                score_threshold=0.3,
                            ),
                            bev_3d_cyclist=dict(
                                colors_map=defaultdict(lambda: (0, 255, 0)),
                                score_threshold=0.3,
                            ),
                            bev_3d_cyclist_cls=dict(
                                colors_map=defaultdict(lambda: (0, 255, 0)),
                                score_threshold=0.3,
                            ),
                        ),
                        ori_img_keys=[
                            "img_ori",
                            "side_img_ori",
                            "narrow_img_ori",
                        ],
                        save_viz_imgs=True,
                    ),
                ],
            )
            callbacks = [
                # visualize_callback,
                eval_callback,
                dict(
                    type="StatsMonitor",
                    log_freq=5,
                ),
            ]
            all_callbacks.append(callbacks)

            if eval_type == "bev_3d":
                transforms = deepcopy(multi_view_transforms)
                data_loader = dict(
                    type=torch.utils.data.DataLoader,
                    collate_fn=collate_3d,
                    sampler=dict(
                        type=torch.utils.data.DistributedSampler,
                        shuffle=False,
                    ),
                    dataset=dict(
                        type="MultiViewImgDataset",
                        img_dir=data_path,
                        camera_list=camera_view_names,
                        view_shapes=img_ori_size,
                        anno_json_file=gt_path,
                        homo_cfg=homo_cfg,
                        to_rgb=True,
                        transforms=transforms,
                        anno_view_key_map=anno_camera_view_names,
                        multi_view_collect=multi_view_collect,
                    ),
                    batch_size=val_batch_size_per_gpu,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    drop_last=False,
                )

            else:
                pass

            all_data_loaders.append(data_loader)

base_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=all_data_loaders,
    batch_processor=dict(
        type="BasicBatchProcessor",
        need_grad_update=False,
    ),
    callbacks=all_callbacks,
    num_epochs=1,
    share_callbacks=False,
)

for stage in train_stages:
    predictor = deepcopy(base_predictor)
    if stage == "float_freeze_bn":
        predictor["callbacks"] += freeze_bn_callback
    predictor.update(
        model_convert_pipeline=dict(
            type="FloatQatConvertPipeline",
            qat_mode="fuse_bn",
            checkpoint_mode="resume",
            enable_qat=stage == "qat",
            checkpoint_configs=dict(
                checkpoint_path=model_checkpoint
                if model_checkpoint
                else f"aidi_artifact://{model_name}/{stage}/{model_version}/{stage}-checkpoint-last.pth.tar",
                enable_tracking=enable_model_tracking,
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            qconfig_params=None,
        )
    )
    globals()[f"{stage}_predictor"] = predictor
