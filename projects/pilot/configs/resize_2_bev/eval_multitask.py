import os
from collections import defaultdict
from copy import deepcopy
from importlib import import_module

import torch
from bev_common import (
    cam_standardized_transform,
    get_homo_transforms,
    standardized_cam_calibs,
)
from bev_dynamic_base import (
    bev_common_transforms,
    camera_view_names,
    per_view_shape,
    spatial_resolution,
    use_distorted_offset,
    vcs_range,
)
from common import (
    enable_model_tracking,
    model_checkpoint,
    model_name,
    model_setting,
    model_version,
    pred_batch_size,
    pred_batch_size_bev,
    raw_image_hw,
    tasks,
    training_step,
    vis_tasks_2d,
    vis_tasks_bev,
    with_cam_standiardization,
)
from multitask import val_model
from schedule import train_stages

from hat.core.data_struct.app_struct import (
    reformat_bev_det_to_aidi_eval,
    reformat_det_to_aidi_eval,
    reformat_seg_to_aidi_eval,
)
from hat.core.proj_spec.parsing import colormap
from hat.data.collates.collates import collate_3d
from hat.utils import Config

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"

suffix = os.getenv("HAT_PILOT_PREDICTION_NAME_SUFFIX", "")
prediction_name = f"{model_name}_{suffix}" if suffix else model_name

device_ids = [0]
log_rank_zero_only = True
# If eval_part_graph_model_only == True, forward only part graph model
# for diff task datasets, it usually speeds eval up
# and it should be True when evaluting both bev and 2d tasks.
eval_part_graph_model_only = True

eval_tags = ["resize", "all_region"]


# task args
task_args = dict(
    vehicle_plate_detection=dict(
        dump_obj_key="plate",
    ),
    face_detection=dict(
        dump_obj_key="face",
    ),
)

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

val_transforms = [
    dict(type="YUVTurboJPEGDecoder", to_string=True),
    dict(
        type="BPUPyramidResizer",
        scale_wh=(1 / 2, 1 / 2),
        pyramid_type="ips",
    ),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=128.0, std=128.0),
]

camera_view_names = [v.replace("camera_", "") for v in camera_view_names]
per_view_shape = {
    view.replace("camera_", ""): shape
    for view, shape in per_view_shape.items()
}
if with_cam_standiardization:
    standardized_cam_calibs = {
        view.replace("camera_", ""): calib_all
        for view, calib_all in standardized_cam_calibs.items()
    }

# bev relate
bev_3d_transforms = [
    bev_common_transforms["Resize3DV"],
    bev_common_transforms["Crop3DV"],
    bev_common_transforms["ResizeHomo"],
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
    per_view_shape=per_view_shape,
    norm_homo=True,
    use_distorted_offset=use_distorted_offset,
    homo_transforms=homo_transforms,
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

        if eval_type == "detection":
            obj_key = import_module(task_key).object_type
            task_output_name = obj_key
            reformat_output_fn = reformat_det_to_aidi_eval

            _kwargs = dict(
                obj_key=obj_key,
                det_task_key=task_key,
                extra_task_key=extra_task_key,
            )
            if extra_task_key in task_adapter_func:
                _kwargs.update(
                    dict(custom_func=task_adapter_func[extra_task_key])
                )
            elif task_key in task_adapter_func:
                _kwargs.update(dict(custom_func=task_adapter_func[task_key]))

            if extra_task_key in task_args:
                _kwargs.update(task_args[extra_task_key])
            elif task_key in task_args:
                _kwargs.update(task_args[task_key])
            reformat_out_fn_kwargs = dict(
                **_kwargs,
            )
        elif eval_type == "bev_detection":
            task_name = import_module(task_key).task_name
            obj_key = import_module(task_key).object_type
            task_output_name = task_key
            reformat_output_fn = reformat_bev_det_to_aidi_eval
            reformat_out_fn_kwargs = dict(
                obj_key=task_name, dump_obj_key=obj_key
            )
        elif eval_type == "semantic_segmentation":
            obj_key = import_module(task_key).task_name
            task_output_name = obj_key
            reformat_output_fn = reformat_seg_to_aidi_eval
            reformat_out_fn_kwargs = dict(
                obj_key=obj_key,
            )

        for ds in datasets:
            if eval_type == "bev_detection":
                data_path = os.path.join(
                    local_datapath, str(ds), "datasets", "datasets", "data"
                )
                if os.path.exists(
                    os.path.join(
                        local_datapath,
                        str(ds),
                        "datasets",
                        "datasets",
                        "gt.json",
                    )
                ):
                    gt_path = os.path.join(
                        local_datapath,
                        str(ds),
                        "datasets",
                        "datasets",
                        "gt.json",
                    )
                else:
                    gt_path = None
            else:
                data_path = os.path.join(local_datapath, str(ds), "datasets")

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
                        out_keys=[task_output_name]
                        if eval_part_graph_model_only
                        else vis_tasks_bev,
                        output_dir=f"./tmp_viz_imgs/{model_name}/{ds}",
                        bird_eye_size=(
                            raw_image_hw[0] // 2,
                            raw_image_hw[1] // 2,
                        ),
                        vcs_range=vcs_range,
                        vis_image_layout=dict(
                            front_left=[0, 0],
                            bird_eye_view=[0, 1],
                            front_right=[0, 2],
                            rear_left=[1, 0],
                            rear=[1, 1],
                            rear_right=[1, 2],
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
                        save_viz_imgs=False,
                    )
                    if task_key in vis_tasks_bev
                    else dict(
                        type="DetMultitaskVisualize",
                        out_keys=[obj_key]
                        if eval_part_graph_model_only
                        else vis_tasks_2d,
                        output_dir=f"./tmp_viz_imgs/{model_name}/{ds}",
                        vis_configs=dict(
                            vehicle_plate=dict(
                                color=(255, 128, 0),
                                thickness=2,
                            ),
                            face=dict(
                                color=(255, 128, 0),
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
                        save_viz_imgs=False,
                    ),
                ],
            )
            callbacks = [
                visualize_callback,
                eval_callback,
                dict(
                    type="StatsMonitor",
                    log_freq=5,
                ),
            ]
            all_callbacks.append(callbacks)

            if eval_type == "bev_detection":
                transforms = deepcopy(val_transforms)
                if with_cam_standiardization:
                    transforms.append(cam_standardized_transform)
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
                        view_shapes=per_view_shape,
                        anno_json_file=gt_path,
                        homo_cfg=homo_cfg,
                        to_rgb=True,
                        transforms=transforms,
                        standardized_cam_calibs=standardized_cam_calibs,
                    ),
                    batch_size=pred_batch_size_bev,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False,
                    drop_last=False,
                )
            else:
                data_loader = dict(
                    type=torch.utils.data.DataLoader,
                    sampler=dict(
                        type=torch.utils.data.DistributedSampler,
                        shuffle=False,
                    ),
                    dataset=dict(
                        type="ModelEvalRawDataset",
                        data_path=data_path,
                        to_rgb=True,
                        buf_only=True,
                        return_orig_img=True,
                        transforms=val_transforms,
                    ),
                    batch_size=pred_batch_size,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False,
                    drop_last=False,
                )

            if eval_part_graph_model_only:
                all_data_loaders.append(
                    dict(
                        type="MultitaskLoader",
                        return_task=True,
                        loaders={task_output_name: data_loader},
                    )
                )
            else:
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
    predictor = deepcopy(base_predictor)
    predictor.update(
        model_convert_pipeline=dict(
            type="FloatQatConvertPipeline",
            qat_mode="fuse_bn",
            checkpoint_mode="resume",
            enable_qat=stage == "qat",
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
