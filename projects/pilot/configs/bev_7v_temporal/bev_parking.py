import copy
import json
import os
import re

import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.utils import Config
from hat.utils.apply_func import _as_list, is_list_of_type
from hat.utils.filesystem import join_path
from projects.pilot.configs.bev_7v_temporal.base import (
    convert_to_split_dataloader,
    get_data_dict,
    get_dataloader,
    get_dataloader_list,
    get_dataset_list,
    get_datasets,
    get_homo_transforms,
    get_roi_resize_cfg,
    get_template_dataset,
    get_update_bev_dataset_func,
    reformat_compile_vcs_range,
    remove_none,
    repeat_metric_updater_by_name,
)
from projects.pilot.configs.bev_7v_temporal.bev_discobj_base import (
    bev_common_transforms,
    get_bev_tb_update_func,
    get_inputs,
    get_train_metric_updater,
    val_common_transforms,
    vismask_vcsrange_cfg,
)
from projects.pilot.configs.bev_7v_temporal.common import (
    H_persp_view_scale,
    backbone,
    bev_backbone,
    bev_fusion,
    bev_fusion_upsampling,
    bev_neck,
    bevfusion_output_size,
    bucket_root,
    camera_view_names,
    deploy_head,
    deploy_narrow_head,
    deploy_side_head,
    deploy_stage2_inputs_key,
    do_val_visualize,
    front_camera_view_names,
    head,
    img_ori_size,
    img_resize_wh_size,
    length_of_clip,
    log_freq,
    multi_view_collect,
    narrow_backbone,
    narrow_camera_view_names,
    narrow_head,
    narrow_pafpn_neck,
    num_views,
    offset_save_path,
    pafpn_neck,
    pipeline_test,
    save_prefix,
    side_backbone,
    side_camera_view_names,
    side_head,
    side_pafpn_neck,
    spatial_resolution,
    temporal_fusion,
    train_batch_size_per_gpu,
    train_global_sample_interval,
    train_num_frames_per_iter,
    train_num_workers,
    training_step,
    use_split_dataloader,
    val_batch_size_per_gpu,
    val_num_frames_per_iter,
    val_num_workers,
    vcs_origin_coord,
    vcs_plane_heights,
    vcs_range,
)

# -------------------------- task --------------------------
task_name = "bev_parking"
base_task_name = "bev_disc"
enable_tensorboard = True

# -------------------------- data --------------------------
train_data_version = "v2_8_1_temporal"
val_data_version = "psd_v1_6_0_temporal"

if pipeline_test:
    train_data_version = "pipeline_test"
    val_data_version = "pipeline_test_temporal"

dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_discrete_obj"
)

task_name_list = [
    task_name + "_" + val_version for val_version in _as_list(val_data_version)
]
# ------------------ BEV parking transformation setting -------------------
roi_vcs_range = (-12.8, -12.8, 25.6, 12.8)
task_in_stride = (2, 8)
psd_task_out_sizes = ((192, 128), (48, 32))

task_out_size = psd_task_out_sizes[0]

bev_discobj_psd_stage2_output_resolution = (
    abs(roi_vcs_range[2] - roi_vcs_range[0]) / task_out_size[0],
    abs(roi_vcs_range[3] - roi_vcs_range[1]) / task_out_size[1],
)  # (height, witdh)

psd_resolution_m_per_grid_h = (
    -roi_vcs_range[0] + roi_vcs_range[2]
) / task_out_size[0]
psd_resolution_m_per_grid_w = (
    -roi_vcs_range[0] + roi_vcs_range[2]
) / task_out_size[0]
assert (psd_resolution_m_per_grid_h - psd_resolution_m_per_grid_w) < 1e-5
psd_resolution_m_per_grid = psd_resolution_m_per_grid_h

global_downsample_factor = 4
global_resolution_m_per_grid = (
    global_downsample_factor * psd_resolution_m_per_grid
)
global_kernel_size = round(2.4 / global_resolution_m_per_grid)
assert global_kernel_size % 2 == 1

local_downsample_factor = 1
local_resolution_m_per_grid = (
    local_downsample_factor * psd_resolution_m_per_grid
)
local_kernel_size = round(1.0 / local_resolution_m_per_grid)
assert local_kernel_size % 2 == 1
psd_eval_bev_range_list = []

eval_psd_vcs_range_list = [
    (0, 0, 0, 0),
    (-3, -3, 6, 3),
    (-7, -7, 10, 7),
    (-10, -10, 14, 10),
    (-10, -10, 20, 10),
]

for eval_vcs_range in eval_psd_vcs_range_list:
    eval_bev_range = (
        (
            (roi_vcs_range[3] - eval_vcs_range[3])
            / psd_resolution_m_per_grid,  # u_left
            (roi_vcs_range[2] - eval_vcs_range[2])
            / psd_resolution_m_per_grid,  # v_down
            (roi_vcs_range[3] - eval_vcs_range[1])
            / psd_resolution_m_per_grid,  # u_right
            (roi_vcs_range[2] - eval_vcs_range[0])
            / psd_resolution_m_per_grid,  # v_up
        )
        if eval_vcs_range is not None
        else None
    )
    psd_eval_bev_range_list.append(eval_bev_range)

psd_metric_param = dict(
    metric_global_threshold=0.18,
    metric_global_topk=50,
    metric_global_downsample_factor=global_downsample_factor,
    metric_global_max_distance=0.4 / psd_resolution_m_per_grid,
    metric_global_kernel_size=global_kernel_size,
    metric_local_threshold=0.25,
    metric_local_topk=50,
    metric_local_downsample_factor=local_downsample_factor,
    metric_local_max_distance=0.2 / psd_resolution_m_per_grid,
    metric_local_kernel_size=local_kernel_size,
    metric_threshold_occupancy_feature=0.5,
    metric_threshold_junction_type_feature=0.5,
    metric_threshold_fuse_distance=1.1 / psd_resolution_m_per_grid,
    metric_iou_threshold=0.5,
    metric_nms_distance_threshold=1.4 / psd_resolution_m_per_grid,
    metric_resolution_m_per_grid=psd_resolution_m_per_grid,
)

global_feature_size = (
    task_out_size[0] // psd_metric_param["metric_global_downsample_factor"],
    task_out_size[1] // psd_metric_param["metric_global_downsample_factor"],
)
local_feature_size = (
    task_out_size[0] // psd_metric_param["metric_local_downsample_factor"],
    task_out_size[1] // psd_metric_param["metric_local_downsample_factor"],
)
global_slot_weight_dict = {  # noqa
    0: 1.0,
    1: 1.0,
    2: 1.0,
    -1: 0.0,
}
local_slot_weight_dict = {  # noqa
    0: 1.0,
    1: 2.0,
    2: 1.0,
    -1: 0.0,
}
local_radius_m = 0.7
local_radius_grid = int(local_radius_m / psd_resolution_m_per_grid)
local_point01_cls_weight = 4.0
local_point01_offset_weight = 4.0
local_far_weight = 1.0


use_distorted_offset = True
cal_homo_offset_on_gpu = True

bev_psd_target = dict(
    type="ANCBEVPSDTargetGenerator",
    global_target=dict(
        type="ANCBEVPSDGlobalTargetGenerator",
        input_size=task_out_size,
        out_size=global_feature_size,
        slot_weight_dict=global_slot_weight_dict,
        res_key="bev_psd_obj",
    ),
    local_target=dict(
        type="ANCBEVPSDLocalTargetGenerator",
        input_size=task_out_size,
        out_size=local_feature_size,
        radius=local_radius_grid,
        slot_weight_dict=local_slot_weight_dict,
        cls_weight=local_point01_cls_weight,
        offset_weight=local_point01_offset_weight,
        far_weight=local_far_weight,
        res_key="bev_psd_obj",
    ),
    max_objs=100,
    input_size=task_out_size,
    vismask_vcsrange_cfg=vismask_vcsrange_cfg,
    use_vis_mask=True,
    visable_condition={
        "visable_threshold": 2,
        "extend_line_length": 5,  # m
        # "ignore_threshold" : 2,
    },
    with_rod=False,
    vcs_range=roi_vcs_range,
    resolution_m_per_grid=[
        psd_resolution_m_per_grid_w,
        psd_resolution_m_per_grid_h,
    ],
    res_key="bev_psd_obj",
)

bev_discobj_target_list = []
bev_discobj_target_list.append(bev_psd_target)

load_data_types = [
    "timestamp",
    "img_name",
    "gt_bev_parking_obj",
    # "pack_dir",
    # "img_paths",
    "pose",
]

use_vis_mask = True
if use_vis_mask:
    load_data_types.append("bev_occlusion_mask")

load_data_types = remove_none(load_data_types)

bev_common_transforms = copy.deepcopy(bev_common_transforms)
val_common_transforms = copy.deepcopy(val_common_transforms)
collect_3dv = copy.deepcopy(bev_common_transforms["ANCCollect3DV"])
collect_3dv["gt_bev_discobj_idx"] = 0
collect_3dv["load_data_types"] = load_data_types

bev_discobj_transforms = [
    collect_3dv,
    *bev_discobj_target_list,
    bev_common_transforms["ANCResize3DV"],
    bev_common_transforms["ANCCrop3DV"],
    bev_common_transforms["ANCTemporalHomo"],
    val_common_transforms if do_val_visualize else None,
    bev_common_transforms["ANCToTensor3DV"],
    bev_common_transforms["ANCPrepareTempoDataBEV"],
    dict(type="ANCSetTemporalClearFlag", clr_mode="clip"),
    dict(type="AddKeys", kv={"return_latest_flag": True}),
    dict(type="AddKeys", kv={"task_name": task_name}),
]
bev_discobj_transforms = remove_none(bev_discobj_transforms)

homo_transforms = get_homo_transforms(
    transforms_list=bev_discobj_transforms, camera_view_names=camera_view_names
)

template_dataset = get_template_dataset(
    img_load_size=list(img_resize_wh_size.values()),
    camera_view_names=camera_view_names,
    per_view_shape=img_ori_size,
    transforms=bev_discobj_transforms,
    homo_transforms=homo_transforms,
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
        "bev_parking_obj_lmdb_path",
        "calib_path",
        "camera_module_type",
        "homo_noise",
        "bev_occlusion_data_path",
    ],
)

val_template_dataset = copy.deepcopy(template_dataset)
val_template_dataset["num_frames_per_iter"] = val_num_frames_per_iter
val_template_dataset["transforms"][0]["img_idxs"] = list(
    range(val_num_frames_per_iter)
)
val_template_dataset["transforms"][0]["pose_idxs"] = list(
    range(val_num_frames_per_iter + 1)
)


select_strides = _as_list(task_in_stride)

# task roi resize config
roi_resize_cfg_stride2 = get_roi_resize_cfg(
    input_size=bevfusion_output_size,
    in_stride=select_strides[0],
    output_size=psd_task_out_sizes[0],
    ori_vcs_range=vcs_range,
    roi_vcs_range=roi_vcs_range,
)
roi_resize_cfg_stride8 = get_roi_resize_cfg(
    input_size=bevfusion_output_size,
    in_stride=select_strides[1],
    output_size=psd_task_out_sizes[1],
    ori_vcs_range=vcs_range,
    roi_vcs_range=roi_vcs_range,
)

psd_roi_resize_stride2 = dict(
    type="RoiCropResize",
    in_strides=[2, 4, 8, 16, 32],
    target_stride=roi_resize_cfg_stride2["in_stride"],
    output_size=roi_resize_cfg_stride2["output_size"],
    roi_box=roi_resize_cfg_stride2["roi_box"],
    node_name="bev_psd_roi_resize_stride2",
)

psd_roi_resize_stride8 = dict(
    type="RoiCropResize",
    in_strides=[2, 4, 8, 16, 32],
    target_stride=roi_resize_cfg_stride8["in_stride"],
    output_size=roi_resize_cfg_stride8["output_size"],
    roi_box=roi_resize_cfg_stride8["roi_box"],
    node_name="bev_psd_roi_resize_stride8",
)

psd_roi_resize = [psd_roi_resize_stride2, psd_roi_resize_stride8]

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
        global_sample_interval=train_global_sample_interval,
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


# -------------------------- model --------------------------
train_inputs, val_inputs, deploy_inputs = get_inputs(
    num_views,
    vcs_plane_heights,
    task_out_size,
)
pop_keys = [
    "gt_bev_discrete_obj",
    "annos_bev_discrete_obj",
]
for pop_key in pop_keys:
    train_inputs.pop(pop_key, None)
    val_inputs.pop(pop_key, None)

train_inputs["gt_bev_psd_obj"] = {
    "global_classification_obj": torch.zeros(
        [1, 1, global_feature_size[0], global_feature_size[1]],
        dtype=torch.float32,
    ),
    "global_offset_obj": torch.zeros(
        [1, 8, global_feature_size[0], global_feature_size[1]],
        dtype=torch.float32,
    ),
    "global_occupancy_obj": torch.zeros(
        [1, 1, global_feature_size[0], global_feature_size[1]],
        dtype=torch.float32,
    ),
    "global_slot_type_obj": torch.zeros(
        [1, 3, global_feature_size[0], global_feature_size[1]],
        dtype=torch.float32,
    ),
    "global_direction_obj": torch.zeros(
        [1, 2, global_feature_size[0], global_feature_size[1]],
        dtype=torch.float32,
    ),
    "global_classification_grad": torch.zeros(
        [1, 1, global_feature_size[0], global_feature_size[1]],
        dtype=torch.float32,
    ),
    "global_offset_grad": torch.zeros(
        [1, 8, global_feature_size[0], global_feature_size[1]],
        dtype=torch.float32,
    ),
    "global_occupancy_grad": torch.zeros(
        [1, 1, global_feature_size[0], global_feature_size[1]],
        dtype=torch.float32,
    ),
    "global_slot_type_grad": torch.zeros(
        [1, 3, global_feature_size[0], global_feature_size[1]],
        dtype=torch.float32,
    ),
    "global_direction_grad": torch.zeros(
        [1, 2, global_feature_size[0], global_feature_size[1]],
        dtype=torch.float32,
    ),
    "local_classification_obj": torch.zeros(
        [1, 4, local_feature_size[0], local_feature_size[1]],
        dtype=torch.float32,
    ),
    "local_offset_obj": torch.zeros(
        [1, 8, local_feature_size[0], local_feature_size[1]],
        dtype=torch.float32,
    ),
    "local_sline_angle_obj": torch.zeros(
        [1, 8, local_feature_size[0], local_feature_size[1]],
        dtype=torch.float32,
    ),
    "local_point_type_obj": torch.zeros(
        [1, 4, local_feature_size[0], local_feature_size[1]],
        dtype=torch.float32,
    ),
    "local_classification_grad": torch.zeros(
        [1, 4, local_feature_size[0], local_feature_size[1]],
        dtype=torch.float32,
    ),
    "local_offset_grad": torch.zeros(
        [1, 8, local_feature_size[0], local_feature_size[1]],
        dtype=torch.float32,
    ),
    "local_sline_angle_grad": torch.zeros(
        [1, 8, local_feature_size[0], local_feature_size[1]],
        dtype=torch.float32,
    ),
    "local_point_type_grad": torch.zeros(
        [1, 4, local_feature_size[0], local_feature_size[1]],
        dtype=torch.float32,
    ),
}
train_inputs["annos_bev_psd_obj"] = {
    "global": torch.zeros([1, 100, 16], dtype=torch.float32),
    "local": torch.zeros([1, 100, 22], dtype=torch.float32),
}

val_inputs = copy.deepcopy(train_inputs)
inputs = dict(
    train=train_inputs,
    val=val_inputs,
    deploy=deploy_inputs,
)


# ----------------------------- DESC ---------------------------
bev_common_desc = dict(
    vcs_origin_coord=vcs_origin_coord,
    bev_stage2_input_resolution=spatial_resolution,
    bev_stage2_output_resolution=bev_discobj_psd_stage2_output_resolution,
    visible_range=reformat_compile_vcs_range(roi_vcs_range),
    warp_offset_range=reformat_compile_vcs_range(vcs_range),
    vcs_plane_heights=vcs_plane_heights,
)


def get_bev_psd_desc():
    per_tensor_desc = [
        dict(
            task="bevpsd_j5_stage2",
            name="global_cls",
            metric=psd_metric_param,
            **bev_common_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="global_offset",
            **bev_common_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="global_occupancy",
            **bev_common_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="global_slot_type",
            **bev_common_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="global_direction",
            **bev_common_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="local_cls",
            **bev_common_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="local_offset",
            **bev_common_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="local_sline_angle",
            **bev_common_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="local_point_type",
            **bev_common_desc,
        ),
    ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


num_slot_type = 3
psd_task_loss_weight = 0.1
local_loss_weights = [2, 2, 1, 1]
local_loss_weights = [i * psd_task_loss_weight for i in local_loss_weights]
global_loss_weights = [1, 1, 1, 1, 1]
global_loss_weights = [i * psd_task_loss_weight for i in global_loss_weights]


def get_model(mode):

    bev_psd_postprocess = dict(
        type="ANCBevPSDDecoder",
        input_size=task_out_size,
        downsample_factor_globalslot_feature=psd_metric_param[
            "metric_global_downsample_factor"
        ],
        downsample_factor_localjunction_feature=psd_metric_param[
            "metric_local_downsample_factor"
        ],
        threshold_globalslot_feature=psd_metric_param[
            "metric_global_threshold"
        ],
        threshold_localjunction_feature=psd_metric_param[
            "metric_local_threshold"
        ],
        topk_globalslot_feature=psd_metric_param["metric_global_topk"],
        topk_localjunction_feature=psd_metric_param["metric_local_topk"],
        threshold_occupancy_feature=psd_metric_param[
            "metric_threshold_occupancy_feature"
        ],
        threshold_junction_type_feature=psd_metric_param[
            "metric_threshold_junction_type_feature"
        ],
        threshold_fuse_distance=psd_metric_param[
            "metric_threshold_fuse_distance"
        ],
        global_kernel_size=psd_metric_param["metric_global_kernel_size"],
        local_kernel_size=psd_metric_param["metric_local_kernel_size"],
        nms_distance_threshold=psd_metric_param[
            "metric_nms_distance_threshold"
        ],
    )
    bev_psd_head = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEVPSDHead",
            global_head=dict(
                type="ANCBEVPSDGlobalHead",
                num_slot_type=num_slot_type,
                in_channels=96,
                out_channels=32,
                stack=1,
                group_base=8,
                in_strides=task_in_stride,
                out_stride=global_downsample_factor * 2,
            ),
            local_head=dict(
                type="ANCBEVPSDLocalHead",
                in_channels=48,
                out_channels=32,
                stack=1,
                group_base=8,
                in_strides=task_in_stride,
                out_stride=local_downsample_factor * 2,
            ),
        ),
        loss=None,
        postprocess=None,
        prefix="bev_stage2_psd_head",
        node_name="bev_stage2_psd_head",
    )

    if mode == "train":
        bev_psd_head["loss"] = dict(
            type="ANCBEVPSDLoss",
            local_loss=dict(
                type="ANCBEVPSDLocalLoss",
                classification_loss=dict(
                    type="GaussianFocalLoss", alpha=2.0, gamma=4.0
                ),
                offset_loss=dict(
                    type="L1Loss",
                    reduction="none",
                ),
                offset_weight=[4.0, 1.0],
                sline_angle_loss=dict(type=torch.nn.MSELoss, reduction="none"),
                point_type_loss=dict(
                    type=torch.nn.BCEWithLogitsLoss, reduction="none"
                ),
                loss_weights=local_loss_weights,
            ),
            global_loss=dict(
                type="ANCBEVPSDGlobalLoss",
                classification_loss=dict(type="FocalLossV2", reduction="none"),
                offset_loss=dict(type="L1Loss", reduction="none"),
                occupancy_loss=dict(
                    type=torch.nn.BCEWithLogitsLoss, reduction="none"
                ),
                slot_type_loss=dict(
                    type=torch.nn.BCEWithLogitsLoss, reduction="none"
                ),
                direction_loss=dict(type=torch.nn.MSELoss, reduction="none"),
                loss_weights=global_loss_weights,
            ),
            gt_name="gt_bev_psd_obj",
        )

    elif mode == "val":
        bev_psd_head["postprocess"] = bev_psd_postprocess

    elif mode == "deploy":
        bev_psd_head["convert_to_dict"] = False
        bev_psd_head["head_parser"] = None
        bev_psd_head["target"] = None
        bev_psd_head["postprocess"] = dict(
            type="MultiInputSequential",
            modules=[
                # AddDesc should be the last module
                dict(
                    type="AddDesc",
                    per_tensor_desc=get_bev_psd_desc(),
                ),
            ],
        )

    multi_view_module = dict(
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
    )
    bevfusion_pick_keys = None
    if mode == "deploy":
        multi_view_module = dict()
        for key in camera_view_names:
            if key in front_camera_view_names:
                multi_view_module[key] = dict(
                    type="BEVStageOneModule",
                    backbone=backbone,
                    neck=pafpn_neck,
                    head=deploy_head,
                )
            elif key in side_camera_view_names:
                multi_view_module[key] = dict(
                    type="BEVStageOneModule",
                    backbone=side_backbone,
                    neck=side_pafpn_neck,
                    head=deploy_side_head,
                )
            elif key in narrow_camera_view_names:
                multi_view_module[key] = dict(
                    type="BEVStageOneModule",
                    backbone=narrow_backbone,
                    neck=narrow_pafpn_neck,
                    head=deploy_narrow_head,
                )
            else:
                raise TypeError
        bevfusion_pick_keys = deploy_stage2_inputs_key

    model = dict(
        type="MultiViewTwoStageBEVModule",
        # view_img_key to module
        multi_view_module=multi_view_module,
        multi_view_collect=multi_view_collect,
        bev_fusion_module=dict(
            type="BEVStageTwoModule",
            bevfusion=bev_fusion[mode],
            temporal_fusion=temporal_fusion[mode],
            bev_fusion_upsample=bev_fusion_upsampling,
            backbone=bev_backbone,
            neck=bev_neck,
            roi_resizes=psd_roi_resize,
            head=bev_psd_head,
        ),
        bevfusion_pick_keys=bevfusion_pick_keys,
    )

    return model


# -------------------------- TRAIN METRIC --------------------------
save_dir = os.path.join(save_prefix, task_name, training_step)
os.makedirs(save_dir, exist_ok=True)
psd_eval_result_path = os.path.join(save_dir, "psd_eval_metric.json")


def get_psd_train_metrics_patterns():
    metrics = [
        # dict(type="LossShow", name="loss_psd"),
        dict(type="LossShow", name="loss_psd_global_classification"),
        dict(type="LossShow", name="loss_psd_global_offset"),
        dict(type="LossShow", name="loss_psd_global_occupancy"),
        dict(type="LossShow", name="loss_psd_global_slot"),
        dict(type="LossShow", name="loss_psd_global_sideness"),
        dict(type="LossShow", name="loss_psd_global_dir"),
        dict(type="LossShow", name="loss_psd_local_cls"),
        dict(type="LossShow", name="loss_psd_local_offset"),
        dict(type="LossShow", name="loss_psd_local_angle"),
        dict(type="LossShow", name="loss_psd_loss_point_type"),
    ]

    per_metric_patterns = [  # corresponding to metrics
        dict(
            label_pattern=None,
            pred_pattern="^.*bev_stage2_psd_head.*global_classification_loss$",  # noqa
        ),
        dict(
            label_pattern=None,
            pred_pattern="^.*bev_stage2_psd_head.*global_offset_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern="^.*bev_stage2_psd_head.*global_occupancy_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern="^.*bev_stage2_psd_head.*global_slot_type_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern="^.*bev_stage2_psd_head.*global_sideness_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern="^.*bev_stage2_psd_head.*global_direction_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern="^.*bev_stage2_psd_head.*local_classification_loss$",  # noqa
        ),
        dict(
            label_pattern=None,
            pred_pattern="^.*bev_stage2_psd_head.*local_offset_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern="^.*bev_stage2_psd_head.*local_sline_angle_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern="^.*bev_stage2_psd_head.*local_point_type_loss$",
        ),
    ]

    return metrics, per_metric_patterns


def get_psd_val_metrics_patterns():
    val_metrics = [
        dict(
            type="ANCBEVPSDMetric",
            name="BEVPSDMetric",
            iou_threshold=psd_metric_param["metric_iou_threshold"],
            global_max_distance=psd_metric_param["metric_global_max_distance"],
            local_max_distance=psd_metric_param["metric_local_max_distance"],
            resolution_m_per_grid=psd_metric_param[
                "metric_resolution_m_per_grid"
            ],
            validation_bev_range_list=psd_eval_bev_range_list,
            eval_result_path=psd_eval_result_path,
            result_prefix="small",
        )
    ]
    val_per_metric_patterns = [
        dict(
            label_pattern=f"^.*{task_name}.*annos_bev_psd_obj",
            pred_pattern="^.*bev_stage2_psd_head.*decode_label*",
        ),
    ]
    return val_metrics, val_per_metric_patterns


psd_metrics, psd_per_metric_patterns = get_psd_train_metrics_patterns()

psd_val_metrics, psd_val_per_metric_patterns = get_psd_val_metrics_patterns()


metric_updater = get_train_metric_updater(
    task_name=task_name,
    metrics=psd_metrics,
    per_metric_patterns=psd_per_metric_patterns,
    log_freq=log_freq,
)

# -------------------------- VALIDATION SETTING --------------------------


def val_flat_condition(key, values):
    """
    Validation need annotations and flating it will result in a bug with more
    than 255 parameters. So this function will not flat annotations
    """
    flag = None
    if "decode_label" in key or "annos_bev_psd_obj" in key:
        flag = False
    elif isinstance(values, dict) or not is_list_of_type(values, torch.Tensor):
        flag = True
    else:
        flag = False
    return flag


def get_val_metric_updater(
    task_name,
    metrics,
    per_metric_patterns,
    log_freq,
):
    val_metric_updater = dict(
        type="MetricUpdater",
        metrics=metrics,
        metric_update_func=update_metric_using_regex(
            per_metric_patterns=per_metric_patterns,
            flat_condition=val_flat_condition,
        ),
        step_log_freq=-1,
        epoch_log_freq=1,
        log_prefix="Validation " + task_name,
    )

    return val_metric_updater


_val_metric_updater = get_val_metric_updater(
    task_name=task_name,
    metrics=psd_val_metrics,
    per_metric_patterns=psd_val_per_metric_patterns,
    log_freq=log_freq,
)


replace_key_list = ["eval_result_path", "name"]
val_metric_updater_list = repeat_metric_updater_by_name(
    task_name_list, _val_metric_updater, replace_key_list
)
# -------------------------- tensorboard --------------------------

gt_hm_key_regex = re.compile(
    f"^.*{task_name}.*gt_bev_discrete_obj_bev_discobj_weight_hm"
)
pred_hm_key_regex = re.compile(f"^.*{task_name}.*pred_bev_discobj_hm")

if enable_tensorboard:
    tb_update_func = get_bev_tb_update_func(
        task_name, pred_hm_key_regex, gt_hm_key_regex
    )
