import os
from copy import deepcopy

import torch
from hatbc.utils import _as_list

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.utils.config import Config
from hat.utils.filesystem import join_path
from projects.pilot.configs.bev_7v_temporal.base import (
    convert_to_split_dataloader,
    get_data_dict,
    get_dataloader,
    get_dataloader_list,
    get_dataset_list,
    get_datasets,
    get_homo_transforms,
    get_template_dataset,
    get_update_bev_dataset_func,
    get_update_metric_func,
    remove_none,
)
from projects.pilot.configs.bev_7v_temporal.bev_elevation_base import (
    roi_resize,
    roi_vcs_range,
    stage2_stride2channels,
    task_in_stride,
    task_out_size,
)
from projects.pilot.configs.bev_7v_temporal.common import (
    H_persp_view_scale,
    backbone,
    bev_backbone,
    bev_fusion,
    bev_fusion_upsampling,
    bev_neck,
    bn_kwargs,
    bucket_root,
    cal_homo_offset_on_gpu,
    camera_view_names,
    common_transforms,
    deploy_mode,
    head,
    img_ori_size,
    img_resize_wh_size,
    length_of_clip,
    log_freq,
    narrow_backbone,
    narrow_head,
    narrow_pafpn_neck,
    offset_save_path,
    pafpn_neck,
    save_prefix,
    side_backbone,
    side_head,
    side_pafpn_neck,
    spatial_resolution,
    temporal_fusion,
    train_batch_size_per_gpu,
    train_num_frames_per_iter,
    train_num_workers,
    training_step,
    use_split_dataloader,
    val_batch_size_per_gpu,
    val_num_frames_per_iter,
    val_num_workers,
    vcs_plane_heights,
    vcs_range,
)

cfg_dir = os.path.dirname(__file__)

# -------------------------- TASK ---------------------------
task_name = "bev_seg"
bev_seg_type = "HDMap_6"
use_resize_head_parser = False
task_loss_weight = 0.3
loss_weight = 1.0

enable_tensorboard = False

if not deploy_mode:
    raise NotImplementedError(
        f"【{task_name}】just for temporal deploy; not temporal training"
    )

# seg maps
STAGE2_SEG_CLASS_DICT = {
    "HDMap_6": 6,
    "HDMap_6_3": 3,
}
STAGE2_SEG_NAMES_DICT = {
    "HDMap_6": [
        "others",
        "roadedge",
        "roadarrow",
        "solid_lanes",
        "stopline",
        "crosswalk",
    ],
    "HDMap_6_3": [
        "others",
        "roadedge",
        "solid_lanes",
    ],
}


# ------------------ DATASET SETTING -------------------
train_data_version = "pipeline_test"
val_data_version = "pipeline_test"

# train data
dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_seg"
)

bev_common_transforms = deepcopy(common_transforms)
collect_3dv = bev_common_transforms["ANCCollect3DV"]
collect_3dv["gt_bev_seg_idxs"] = [0]
load_data_types = [
    "timestamp",
    "img_name",
    "pose",
    "gt_bev_seg_anno",
    "gt_bev_discrete_obj",
]
load_data_types = remove_none(load_data_types)
collect_3dv["load_data_types"] = load_data_types
bev_transforms = [
    collect_3dv,
    dict(
        type="ANCBevSegTargetGenerator",
        vcs_range=roi_vcs_range,
        bev_size=task_out_size,
        gt_name=f"gt_{task_name}",
    ),
    bev_common_transforms["ANCResize3DV"],
    bev_common_transforms["ANCCrop3DV"],
    bev_common_transforms["ANCClassRemap"],
    common_transforms["ANCTemporalHomo"],
    bev_common_transforms["ANCToTensor3DV"],
    bev_common_transforms["ANCPrepareTempoDataBEV"],
    dict(type="ANCSetTemporalClearFlag", clr_mode="clip"),
    dict(type="AddKeys", kv={"return_latest_flag": True}),
    dict(type="AddKeys", kv={"task_name": task_name}),
]
bev_transforms = remove_none(bev_transforms)

use_distorted_offset = True
template_dataset = get_template_dataset(
    img_load_size=list(img_resize_wh_size.values()),
    camera_view_names=camera_view_names,
    per_view_shape=img_ori_size,
    transforms=bev_transforms,
    homo_transforms=get_homo_transforms(
        transforms_list=bev_transforms,
        camera_view_names=camera_view_names,
    ),
    spatial_resolution=spatial_resolution,
    H_persp_view_scale=H_persp_view_scale,
    vcs_range=vcs_range,
    use_distorted_offset=use_distorted_offset,
    vcs_plane_heights=vcs_plane_heights,
    cal_homo_offset_on_gpu=cal_homo_offset_on_gpu,
    offset_save_path=offset_save_path,
    temporal_bev=True,
    length_of_clip=length_of_clip,
    train_num_frames_per_iter=train_num_frames_per_iter,
)

update_bev_dataset = get_update_bev_dataset_func(
    added_info=[
        "homo_path",
        "calib_path",
        "bev_static_lmdb_path",
        "bev_occlusion_data_path",
        "camera_module_type",
        "homo_noise",
        "bev_discrete_obj_lmdb_path",
    ],
    update_trans_info={
        "ANCCollect3DV": "fill_fake_temporal_data",
    },
)


# for val, not dilate small obj gt
val_template_dataset = deepcopy(template_dataset)
val_template_dataset["num_frames_per_iter"] = val_num_frames_per_iter
val_template_dataset["transforms"][0]["img_idxs"] = list(
    range(val_num_frames_per_iter)
)
val_template_dataset["transforms"][0]["pose_idxs"] = list(
    range(val_num_frames_per_iter + 1)
)


# ----------------------- DATALODER---------------------------
data_yaml = os.path.join(dataset_dir, f"{task_name}_dataset.yaml")


def get_train_dataloader():
    url = os.path.join(dataset_dir, f"{task_name}_train_version.py")
    train_dataset_info = Config.fromfile(url)
    train_dataset_dict = train_dataset_info[train_data_version]

    data_dict = get_data_dict(data_yaml)
    data_dict = join_path(
        bucket_root, data_dict, ["camera_module_type", "camera_view_names"]
    )
    train_datasets = get_datasets(
        train_dataset_dict,
        None,
        template_dataset,
        data_dict,
        update_bev_dataset,
        homo_noise=None,
    )
    data_loader = get_dataloader(
        train_datasets,
        train_num_workers,
        train_batch_size_per_gpu,
        shuffle=True,
        persistent_workers=train_num_workers > 0,
    )

    if use_split_dataloader:
        data_loader = convert_to_split_dataloader(data_loader)
    return data_loader


def get_val_dataloader():
    val_url = os.path.join(dataset_dir, f"{task_name}_val_version.yaml")
    val_data_version_dict = get_data_dict(val_url)
    val_dataset_dict = {
        val_version: val_data_version_dict[val_version]
        for val_version in _as_list(val_data_version)
    }

    data_dict = get_data_dict(data_yaml)
    data_dict = join_path(
        bucket_root, data_dict, ["camera_module_type", "camera_view_names"]
    )

    val_dataset_list = get_dataset_list(
        list(val_dataset_dict.values()),
        None,
        val_template_dataset,
        data_dict,
        update_bev_dataset,
        homo_noise=None,
        repeat_dataset_times=1,
    )

    val_data_loader_list = get_dataloader_list(
        val_dataset_list, val_num_workers, val_batch_size_per_gpu, False, False
    )
    return val_data_loader_list


# -------------------------- MODEL --------------------------
train_inputs = {
    f"gt_{task_name}": torch.zeros(
        (1, task_out_size[0], task_out_size[1]), dtype=torch.long
    ),
}

inputs = dict(
    train=train_inputs,
    val={},
    deploy={},
)


def get_model(mode):

    bev_head = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEVStaticHead",
            in_strides=[task_in_stride],
            out_strides=[task_in_stride],
            merge_branches=use_resize_head_parser,
            stride2channels=stage2_stride2channels,
            out_nums=[6],
            output_name=[f"pred_{task_name}"],
            quanti_last_conv=True,
            dequant_out=False
            if use_resize_head_parser or training_step == "int_infer"
            else True,
            bn_kwargs=bn_kwargs,
        ),
        head_parser=None,
        loss=dict(
            type="ANCBEVSegLoss",
            pred_names=[f"pred_{task_name}_frame0"],
            loss_names=[f"loss_{task_name}"],
            loss_seg_cfg=dict(
                type="CrossEntropyLoss",
                reduction="mean",
                loss_weight=6.0 * loss_weight * task_loss_weight,
                class_weight=(
                    [0.8] + [2] * (STAGE2_SEG_CLASS_DICT[bev_seg_type] - 1)
                ),
                ignore_index=255,
            ),
            conf_loss_weight=None,
            gt_name=f"gt_{task_name}",
            conf_gt_name=f"pred_{task_name}_frame0",
        ),
        postprocess=[
            dict(
                type="ArgmaxPostprocess",
                data_name=f"pred_{task_name}_frame0",
                dim=1,
                keepdim=True if training_step == "int_infer" else False,
            ),
        ],
        target=dict(
            type="ANCBEVTarget",
            gt_name=f"gt_{task_name}",
            bev_height=task_out_size[0],
            bev_width=task_out_size[1],
            gt_shape=(-1, *task_out_size),
        ),
        node_name="bev_stage2_seg_head",
    )

    model = dict(
        type="MultiViewTwoStageBEVModule",
        # view_img_key to module
        multi_view_module=dict(
            img=dict(
                type="BEVStageOneModule",
                backbone=backbone,
                neck=pafpn_neck,
                head=head,
            ),
            side_img=dict(
                type="BEVStageOneModule",
                backbone=side_backbone,
                neck=side_pafpn_neck,
                head=side_head,
            ),
            narrow_img=dict(
                type="BEVStageOneModule",
                backbone=narrow_backbone,
                neck=narrow_pafpn_neck,
                head=narrow_head,
            ),
        ),
        bev_fusion_module=dict(
            type="BEVStageTwoModule",
            bevfusion=bev_fusion[mode],
            temporal_fusion=temporal_fusion[mode],
            bev_fusion_upsample=bev_fusion_upsampling,
            backbone=bev_backbone,
            neck=bev_neck,
            roi_resizes=roi_resize,
            head=bev_head,
        ),
    )

    return model


# -------------------------- TRAIN METRIC --------------------------
metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(
            type="LossShow",
            name=f"loss_{task_name}",
        ),
    ],
    metric_update_func=update_metric_using_regex(
        [
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}.*loss_{task_name}$",
            ),
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


# ------------------------- VALIDATION SETTING-----------------------
val_metric_updater_list = []
pred_name = f"pred_{task_name}"
for dataset_key in _as_list(val_data_version):
    save_dir = os.path.join(save_prefix, "eval", task_name, dataset_key)
    os.makedirs(save_dir, exist_ok=True)
    save_metric_path = os.path.join(
        save_dir, f"{task_name}_{training_step}_eval_metric.json"
    )

    val_metric_updater = dict(
        type="MetricUpdater",
        metrics=[
            dict(
                type="ANCMeanIOUV2",
                name="BEVMIOU",
                task_name=task_name,
                gt_name=f"gt_{task_name}",
                pred_name=pred_name,
                seg_class=STAGE2_SEG_NAMES_DICT[bev_seg_type],
                ignore_index=255,
                verbose=True,
                save_path=save_metric_path,
                global_ignore_index=[0, 2],
                result_prefix="wide",
            ),
        ],
        metric_update_func=get_update_metric_func(
            mode="val",
            task_name=task_name,
            pred_patterns=[
                {
                    pred_name: f"^.*pred_{task_name}_frame0",
                },
            ],
        ),
        step_log_freq=-1,
        epoch_log_freq=1,
        log_prefix=f"Validation_{task_name}_{dataset_key}",
    )
    val_metric_updater_list.append(val_metric_updater)


# -------------------------- TENSORBOAED --------------------------
if enable_tensorboard:
    # tb_update_func
    raise NotImplementedError
