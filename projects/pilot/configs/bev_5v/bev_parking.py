import copy
import json
import os
import re

import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.utils import Config
from hat.utils.apply_func import _as_list, is_list_of_type
from hat.utils.filesystem import join_path
from projects.pilot.configs.bev_5v.base import (
    convert_to_split_dataloader,
    get_data_dict,
    get_dataloader,
    get_dataloader_list,
    get_dataset_list,
    get_datasets,
    get_homo_transforms,
    get_template_dataset,
    get_update_bev_dataset_func,
    reformat_compile_vcs_range,
    remove_none,
    repeat_metric_updater_by_name,
    update_content,
)
from projects.pilot.configs.bev_5v.bev_discobj_base import (
    bev_common_transforms,
    get_bev_tb_update_func,
    get_inputs,
    get_train_metric_updater,
    high_sp_resolution,
    high_sp_vcs_range,
    img_scale_before_ipm,
    val_common_transforms,
    vis_common,
    vis_common_transforms,
    vismask_vcsrange_cfg,
)
from projects.pilot.configs.bev_5v.common import (
    H_persp_view_scale,
    backbone,
    bev_backbone,
    bev_fusion,
    bev_fusion_upsampling,
    bev_neck,
    block_warp_padding,
    bucket_root,
    cal_homo_offset_on_gpu,
    camera_view_names,
    deploy_head,
    deploy_homo_offset_key,
    deploy_mode,
    deploy_round_head,
    do_val_visualize,
    fisheye_camera_view_names,
    fisheye_warp_range,
    front_camera_view_names,
    head,
    img_ori_size,
    img_resize_wh_size,
    ipm_output_size,
    log_freq,
    multi_view_collect,
    num_views,
    offset_save_path,
    pafpn_neck,
    pipeline_test,
    round_backbone,
    round_head,
    round_pafpn_neck,
    save_prefix,
    spatial_resolution,
    train_batch_size_per_gpu,
    train_global_sample_interval,
    train_num_workers,
    training_step,
    use_split_dataloader,
    val_batch_size_per_gpu,
    val_num_workers,
    vcs_origin_coord,
    vcs_plane_heights,
    vcs_range,
)

# -------------------------- task --------------------------
if deploy_mode:
    raise NotImplementedError(f'"{__file__}": not for deploy or pack_infer.')

task_name = "bev_parking"
base_task_name = "bev_disc"
with_psd = True
with_rod = True
aux_near_training = True

psd_task_loss_weight = 3.0
rod_task_loss_weight = 0.8
psd_res_key = "bev_psd_obj"
rod_res_key = "bev_parkingrod_obj"

bev_batch_size = train_batch_size_per_gpu
enable_tensorboard = True
use_random_rotation = False
use_rec = True

save_dir = os.path.join(save_prefix, task_name, training_step)
os.makedirs(save_dir, exist_ok=True)
psd_eval_result_path = os.path.join(save_dir, "psd_eval_metric.json")
rod_eval_result_path = os.path.join(save_dir, "rod_eval_metric.json")

# -------------------------- data --------------------------
train_data_version = "v3_1_0"
val_data_version = [
    "psd_ek_v1_0_0_outdoor",
    "psd_ek_v1_0_0_underground",
    "psd_ek_v1_0_0_parking_outdoor",
    "psd_ek_v1_0_0_parking_underground",
    "psd_v1_6_0",
    "psd_v1_7_0_parking",  # only underground
    "psd_v1_8_0_parking_outdoor",
    "rod_v1.4.0",
    "rod_v1.5.0_underground",
    "rod_v1.5.0_outdoor",
    # EK testsite & demo parkinglot
    # TODO: only keep underfitting set after V3.2.0
    "psd_ek_demo_v1_0_0_outdoor",
    "psd_ek_demo_v1_0_0_underground",
    "psd_ek_demo_v1_0_0_underground_hotel",
    "psd_ek_demo_v1_0_0_parking_outdoor",
    "psd_ek_demo_v1_0_0_parking_underground",
    "psd_ek_demo_v1_0_0_parking_underground_hotel",
    "psd_ek_demo_v1_0_0_parkingrod_outdoor",
    "psd_ek_demo_v1_0_0_parkingrod_underground",
    "psd_ek_demo_v1_0_0_parkingrod_underground_hotel",
]

if pipeline_test:
    train_data_version = "pipeline_test"
    val_data_version = "pipeline_test"

dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_discrete_obj"
)
url = os.path.join(dataset_dir, f"{task_name}_train_version.py")
train_dataset_info = Config.fromfile(url)

train_dataset_dict = train_dataset_info[train_data_version]

val_url = os.path.join(dataset_dir, f"{task_name}_val_version.yaml")
val_data_version_dict = get_data_dict(val_url)
val_dataset_dict = {
    val_version: val_data_version_dict[val_version]
    for val_version in _as_list(val_data_version)
}

task_name_list = [
    task_name + "_" + val_version for val_version in _as_list(val_data_version)
]

load_data_types = [
    "timestamp",
    "img_name",
    "gt_bev_parking_obj",
    # "pack_dir",
    # "img_paths",
    # load_pose_type if temporal_bev else None,
]

use_vis_mask = True
if use_vis_mask:
    load_data_types.append("bev_occlusion_mask")

load_data_types = remove_none(load_data_types)

collect_3dv = copy.deepcopy(bev_common_transforms["ANCCollect3DV"])
collect_3dv["gt_bev_discobj_idx"] = 0
collect_3dv["load_data_types"] = load_data_types

# ------------- psd config -------------
bev_psd_gt_size = (192, 128)
psd_roi_vcs_range = (-12.8, -12.8, 25.6, 12.8)
psd_in_stride = 2
bev_psd_stage2_output_resolution = (
    abs(psd_roi_vcs_range[2] - psd_roi_vcs_range[0]) / bev_psd_gt_size[0],
    abs(psd_roi_vcs_range[3] - psd_roi_vcs_range[1]) / bev_psd_gt_size[1],
)  # (height, width)

num_slot_type = 3
local_loss_weights = [2, 2, 1, 1]
local_loss_weights = [i * psd_task_loss_weight for i in local_loss_weights]
global_loss_weights = [1, 1, 1, 1, 1]
global_loss_weights = [i * psd_task_loss_weight for i in global_loss_weights]

psd_resolution_m_per_grid_h = (
    -psd_roi_vcs_range[0] + psd_roi_vcs_range[2]
) / bev_psd_gt_size[0]
psd_resolution_m_per_grid_w = (
    -psd_roi_vcs_range[0] + psd_roi_vcs_range[2]
) / bev_psd_gt_size[0]
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

# roi in raw gt size
replace_roi = (98, 49, 143, 79)
# reweight range
roi_padding = (5, 5, -5, -5)
reweight_roi = [0.75, 4] + [r - p for r, p in zip(replace_roi, roi_padding)]
reweight_roi_near = [0.75, 2] + [
    (r - p) * 2 for r, p in zip(replace_roi, roi_padding)
]
# roi in upsampling bev feature map for auxiliary near range training
near_roi = tuple([r * 2 for r in replace_roi])
near_roi_padding = (10, 10, -10, -10)
local_near_downsample_factor = 0.5
local_near_resolution_m_per_grid = (
    local_near_downsample_factor * psd_resolution_m_per_grid
)


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
            (vcs_range[3] - eval_vcs_range[3])
            / psd_resolution_m_per_grid,  # u_left
            (vcs_range[2] - eval_vcs_range[2])
            / psd_resolution_m_per_grid,  # v_down
            (vcs_range[3] - eval_vcs_range[1])
            / psd_resolution_m_per_grid,  # u_right
            (vcs_range[2] - eval_vcs_range[0])
            / psd_resolution_m_per_grid,  # v_up
        )
        if eval_vcs_range is not None
        else None
    )
    psd_eval_bev_range_list.append(eval_bev_range)

# metric_global_threshold: classification score threhold for global head.
# metric_global_topk: center point number kept for global head.
# metric_global_downsample_factor: downsample factor of global head.
# metric_global_max_distance: coordinate distance threshold to judge
#     whether global corner point prediction is right.
# metric_global_kernel_size: kernel size used to pool max score
#     for global head.
# metric_local_threshold: classification score threshold for
#     local head.
# metric_local_topk: corner point number kept for local head.
# metric_local_downsample_factor: downsample factor of local head.
# metric_local_max_distance: coordinate distance threshold to judge
#     whether local corner point prediction is right.
# metric_local_kernel_size: kernel size used to pool max score
#     for local head.
# metric_threshold_fuse_distance: coordinate distance threshold to
#     fuse global and local head corner point prediction.
# metric_iou_threshold: iou threshold to judge whether detect slot or not.
# metric_nms_distance_threshold: distance for nms.
# metric_start_bev_range: start eval bev range(bottom, right, top, left)
# metric_end_bev_range: end eval vcs_range (bottom, right, top, left)
# metric_resolution_m_per_grid: meter per grid
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


# ------------- parkingrod config -------------
bev_parkingrod_gt_size = (96, 64)
rod_roi_vcs_range = (-12.8, -12.8, 25.6, 12.8)
rod_out_stride = 8
bev_parkingrod_stage2_output_resolution = (
    abs(rod_roi_vcs_range[2] - rod_roi_vcs_range[0])
    / bev_parkingrod_gt_size[0],
    abs(rod_roi_vcs_range[3] - rod_roi_vcs_range[1])
    / bev_parkingrod_gt_size[1],
)  # (height, width)

# parkingrod roi in raw gt size
parkingrod_replace_roi = (49, 25, 72, 40)
# reweight range
parkingrod_roi_padding = (2, 2, -2, -2)
parkingrod_reweight_roi = None
# parkingrod_reweight_roi = [1.0, 4.0] + [
#     r - p for r, p in zip(parkingrod_replace_roi, parkingrod_roi_padding)
# ]

rod_resolution_m_per_grid_h = (
    rod_roi_vcs_range[2] - rod_roi_vcs_range[0]
) / bev_parkingrod_gt_size[0]
rod_resolution_m_per_grid_w = (
    rod_roi_vcs_range[3] - rod_roi_vcs_range[1]
) / bev_parkingrod_gt_size[1]
assert (rod_resolution_m_per_grid_h - rod_resolution_m_per_grid_w) < 1e-5
rod_resolution_m_per_grid = rod_resolution_m_per_grid_h

num_parkingrod_class = 3
rod_loss_weight = {
    "cls_loss": 1.0,
    "endpoint_offset_loss": 1.0,
    "aux_loss": 0.1,
}
for key in rod_loss_weight.keys():
    rod_loss_weight[key] *= rod_task_loss_weight
rod_dilate_rate = 1

half_park_space_length = 1.2
parkingrod_kernel_size = round(
    half_park_space_length / rod_resolution_m_per_grid
)
assert parkingrod_kernel_size % 2 == 1
ego_vcs_range = (-1, -1, 4, 1)
eval_vcs_range = (-10, -10, 20, 10)
dep_intervals = (2, 6, 9)

parkingrod_metric_param = dict(
    downsampling_factor=1,  # stay decoder unchange
    threshold_parkingrod_feature=0.20,
    object_pool_kernel_size=parkingrod_kernel_size,
    nms_distance_threshold=1.0 / rod_resolution_m_per_grid,
    topk_rod_feature=200,
    strip_dictance_threshold=0.8 / rod_resolution_m_per_grid,
    rod_distance_threshold=1.0 / rod_resolution_m_per_grid,
    angle_threshold=30,
    resolution_m_per_grid=rod_resolution_m_per_grid,
    ego_vcs_range=ego_vcs_range,
    eval_vcs_range=eval_vcs_range,
    dep_intervals=dep_intervals,
)


# ------------------ BEV psd and parkingrod transformation setting -------------------
# psd target generator
global_feature_size = (
    bev_psd_gt_size[0] // psd_metric_param["metric_global_downsample_factor"],
    bev_psd_gt_size[1] // psd_metric_param["metric_global_downsample_factor"],
)
local_feature_size = (
    bev_psd_gt_size[0] // psd_metric_param["metric_local_downsample_factor"],
    bev_psd_gt_size[1] // psd_metric_param["metric_local_downsample_factor"],
)
local_near_feature_size = (
    int(bev_psd_gt_size[0] // local_near_downsample_factor),
    int(bev_psd_gt_size[1] // local_near_downsample_factor),
)
global_slot_weight_dict = {  # noqa
    0: 1.0,
    1: 1.0,
    2: 1.0,
    -1: 0.0,
}
local_slot_weight_dict = {  # noqa
    0: 1.0,
    1: 1.0,
    2: 1.0,
    -1: 0.0,
}
local_radius_m = 0.7
local_radius_grid = int(local_radius_m / psd_resolution_m_per_grid)
local_far_weight = 1.0

# global_target: Global target module for bev psd task.
# local_target: Local target module for bev psd task.
# max_objs: size of global and local align to max_objs
# input_size(sequence of int): The size of input feature size.
# visable_condition(dict): Store some threshold labels are visible,
#     "extend_line_length": A line extending from the center of parking-slot
#         to the entrance line, default: 5(m),
#     "visable_threshold": All the points(grids) along this line,
#         if more than 2 points are in the visible area, the label is retained
# vcs_range(sequence of float): The groundtruth vcs range,
#     (bottom, right, top, left)m
# resolution_m_per_grid(List):
#     [spatial resolution width, spatial resolution height]
# vismask_vcsrange_cfg (dict): Raw offline vismask image size
#     as key, and corresponding vcs range as value.
bev_psd_target = dict(
    type="ANCBEVPSDTargetGenerator",
    global_target=dict(
        type="ANCBEVPSDGlobalTargetGenerator",
        input_size=bev_psd_gt_size,
        out_size=global_feature_size,
        slot_weight_dict=global_slot_weight_dict,
        res_key=psd_res_key,
    ),
    local_target=dict(
        type="ANCBEVPSDLocalTargetGenerator",
        input_size=bev_psd_gt_size,
        out_size=local_feature_size,
        radius=local_radius_grid,
        slot_weight_dict=local_slot_weight_dict,
        cls_weight=4.0,
        offset_weight=4.0,
        far_weight=local_far_weight,
        res_key=psd_res_key,
        reweight=reweight_roi,
    ),
    local_near_target=dict(
        type="ANCBEVPSDLocalTargetGenerator",
        input_size=bev_psd_gt_size,
        out_size=local_near_feature_size,
        radius=local_radius_grid,
        slot_weight_dict=local_slot_weight_dict,
        cls_weight=4.0,
        offset_weight=4.0,
        far_weight=local_far_weight,
        res_key=psd_res_key,
        gt_postfix="_near",
        reweight=reweight_roi_near,
    )
    if aux_near_training
    else None,
    max_objs=100,
    input_size=bev_psd_gt_size,
    vismask_vcsrange_cfg=vismask_vcsrange_cfg,
    use_vis_mask=use_vis_mask,
    with_rod=True,
    visable_condition={
        "visable_threshold": 2,
        "extend_line_length": 5,  # m
        # "ignore_threshold" : 2,
    },
    vcs_range=vcs_range,
    resolution_m_per_grid=[
        psd_resolution_m_per_grid_w,
        psd_resolution_m_per_grid_h,
    ],
    res_key=psd_res_key,
)

# parkingrod target generator
rod_weight_dict = {
    0: 1.0,  # limited_strip
    1: 1.0,  # limited_rod
    2: 1.0,  # other
}
bev_parkingrod_target = dict(
    type="ANCBEVParkingRodTargetGenerator",
    max_objs=200,
    len_gt_info=17,
    target_size=bev_parkingrod_gt_size,
    roi_vcs_range=rod_roi_vcs_range,
    rod_weight_dict=rod_weight_dict,
    num_parkingrod_class=num_parkingrod_class,
    resolution_m_per_grid=rod_resolution_m_per_grid,
    dilate_rate=rod_dilate_rate,
    use_vis_mask=use_vis_mask,
    visable_condition={
        "visable_threshold": 1,
    },
    vismask_vcsrange_cfg=vismask_vcsrange_cfg,
    with_psd=True,
    res_key=rod_res_key,
    reweight=None,  # parkingrod_reweight_roi,
)
val_transforms = update_content(
    val_common_transforms,
    bev_size=ipm_output_size,
    block_warp_padding=block_warp_padding,
    meta_name="meta_info",
    return_name="ipm",
)

bev_discobj_transforms = [
    collect_3dv,
    bev_psd_target if with_psd else None,
    bev_parkingrod_target if with_rod else None,
    bev_common_transforms["ANCResize3DV"],
    bev_common_transforms["ANCCrop3DV"],
    None,  # temporal_transform if temporal_bev else None,
    vis_common_transforms if do_val_visualize else None,
    bev_common_transforms["ANCToTensor3DV"],
]
bev_discobj_transforms.append(bev_common_transforms["ANCPrepareDataBEV"])

bev_discobj_transforms = remove_none(bev_discobj_transforms)

homo_transforms = get_homo_transforms(
    transforms_list=bev_discobj_transforms, camera_view_names=camera_view_names
)
use_distorted_offset = True

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
    temporal_bev=False,
    length_of_clip=None,
    train_num_frames_per_iter=None,
)

url = os.path.join(dataset_dir, f"{task_name}_dataset.yaml")
data_dict = get_data_dict(url)
data_dict = join_path(bucket_root, data_dict, ["camera_module_type"])

update_bev_dataset = get_update_bev_dataset_func(
    added_info=[
        "bev_parking_obj_lmdb_path",
        "calib_path",
        "camera_module_type",
        "homo_noise",
        "bev_occlusion_data_path",
    ],
    do_val_visualize=do_val_visualize,
)

train_homo_noise = None
set_homo_noise = False
if set_homo_noise:
    train_homo_noise = {
        "noise_value": (0.2, 0.2, 0.2, 0.0, 0.0, 0.04),
        # It is the exact value of generated noise when the noise type is
        # 'specific_cam'. When the noise type is 'random_cam' or 'random_vcs',
        # it is the upper bound value of generated random noises. The format is
        # (roll, pitch, yaw, x, y, z), the units are degrees and meters.
        "noise_type": "random_cam",
        "noise_view_names": camera_view_names,
    }
val_with_homo_noise = False
val_homo_noise = None

train_datasets = get_datasets(
    train_dataset_dict,
    None,
    template_dataset,
    data_dict,
    update_bev_dataset,
    homo_noise=train_homo_noise if set_homo_noise else None,
    global_sample_interval=train_global_sample_interval,
)


val_template_dataset = copy.deepcopy(template_dataset)
if do_val_visualize:
    val_template_dataset["homo_gen"]["return_offset_in_meta_info"] = True
    homo_gen_high_sp = copy.deepcopy(val_template_dataset["homo_gen"])
    homo_transforms = copy.deepcopy(homo_gen_high_sp["homo_transforms"])
    homo_gen_high_sp.update(
        vcs_range=high_sp_vcs_range,
        spatial_resolution=(high_sp_resolution, high_sp_resolution),
        H_persp_view_scale=img_scale_before_ipm,
        homo_transforms=homo_transforms,
    )
    val_template_dataset["homo_gen_high_sp"] = homo_gen_high_sp

val_dataset_list = get_dataset_list(
    list(val_dataset_dict.values()),
    None,
    val_template_dataset,
    data_dict,
    update_bev_dataset,
    homo_noise=val_homo_noise if val_with_homo_noise else None,
    repeat_dataset_times=3 if val_with_homo_noise else 1,
)

# ----------------------- DATALODER---------------------------
data_loader = get_dataloader(
    train_datasets,
    train_num_workers,
    train_batch_size_per_gpu,
    shuffle=True,
    persistent_workers=train_num_workers > 0,
)

if use_split_dataloader:
    data_loader = convert_to_split_dataloader(data_loader)

val_data_loader_list = get_dataloader_list(
    val_dataset_list, val_num_workers, val_batch_size_per_gpu, False, False
)

# -------------------------- model --------------------------
inputs, val_inputs, deploy_inputs = get_inputs(
    num_views, vcs_plane_heights, bev_psd_gt_size, 1
)
inputs.pop("gt_bev_discrete_obj", None)
inputs.pop("annos_bev_discrete_obj", None)
if with_psd:
    inputs[f"gt_{psd_res_key}"] = {
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
    if aux_near_training:
        inputs[f"gt_{psd_res_key}"].update(
            {
                "local_classification_obj_near": torch.zeros(
                    [
                        1,
                        4,
                        local_near_feature_size[0],
                        local_near_feature_size[1],
                    ],
                    dtype=torch.float32,
                ),
                "local_offset_obj_near": torch.zeros(
                    [
                        1,
                        8,
                        local_near_feature_size[0],
                        local_near_feature_size[1],
                    ],
                    dtype=torch.float32,
                ),
                "local_sline_angle_obj_near": torch.zeros(
                    [
                        1,
                        8,
                        local_near_feature_size[0],
                        local_near_feature_size[1],
                    ],
                    dtype=torch.float32,
                ),
                "local_point_type_obj_near": torch.zeros(
                    [
                        1,
                        4,
                        local_near_feature_size[0],
                        local_near_feature_size[1],
                    ],
                    dtype=torch.float32,
                ),
                "local_classification_grad_near": torch.zeros(
                    [
                        1,
                        4,
                        local_near_feature_size[0],
                        local_near_feature_size[1],
                    ],
                    dtype=torch.float32,
                ),
                "local_offset_grad_near": torch.zeros(
                    [
                        1,
                        8,
                        local_near_feature_size[0],
                        local_near_feature_size[1],
                    ],
                    dtype=torch.float32,
                ),
                "local_sline_angle_grad_near": torch.zeros(
                    [
                        1,
                        8,
                        local_near_feature_size[0],
                        local_near_feature_size[1],
                    ],
                    dtype=torch.float32,
                ),
                "local_point_type_grad_near": torch.zeros(
                    [
                        1,
                        4,
                        local_near_feature_size[0],
                        local_near_feature_size[1],
                    ],
                    dtype=torch.float32,
                ),
            }
        )
    inputs[f"annos_{psd_res_key}"] = {
        "global": torch.zeros([1, 100, 16], dtype=torch.float32),
        "local": torch.zeros([1, 100, 22], dtype=torch.float32),
    }
if with_rod:
    inputs[f"gt_{rod_res_key}"] = {
        "classification_obj": torch.zeros(
            [1, 1, bev_parkingrod_gt_size[0], bev_parkingrod_gt_size[1]],
            dtype=torch.float32,
        ),
        "endpoint_offset_obj": torch.zeros(
            [1, 4, bev_parkingrod_gt_size[0], bev_parkingrod_gt_size[1]],
            dtype=torch.float32,
        ),
        "slot_01offset_obj": torch.zeros(
            [1, 4, bev_parkingrod_gt_size[0], bev_parkingrod_gt_size[1]],
            dtype=torch.float32,
        ),
        "rod2slot_offset_obj": torch.zeros(
            [1, 2, bev_parkingrod_gt_size[0], bev_parkingrod_gt_size[1]],
            dtype=torch.float32,
        ),
        "classification_weight_mask": torch.zeros(
            [1, 1, bev_parkingrod_gt_size[0], bev_parkingrod_gt_size[1]],
            dtype=torch.float32,
        ),
        "endpoint_offset_weight_mask": torch.zeros(
            [1, 4, bev_parkingrod_gt_size[0], bev_parkingrod_gt_size[1]],
            dtype=torch.float32,
        ),
        "slot_01offset_weight_mask": torch.zeros(
            [1, 4, bev_parkingrod_gt_size[0], bev_parkingrod_gt_size[1]],
            dtype=torch.float32,
        ),
        "rod2slot_offset_weight_mask": torch.zeros(
            [1, 2, bev_parkingrod_gt_size[0], bev_parkingrod_gt_size[1]],
            dtype=torch.float32,
        ),
    }
    inputs[f"annos_{rod_res_key}"] = {
        "parking_rod": torch.zeros([1, 200, 13], dtype=torch.float32),
    }

val_inputs = copy.deepcopy(inputs)
if do_val_visualize:
    val_inputs["meta_info_high_sp"] = None
    val_inputs["high_sp_ipm"] = None

inputs = dict(
    train=inputs,
    val=val_inputs,
    deploy=deploy_inputs,
)

# training set
bev_parking_out_name = [
    "bev_stage2_psd_head" if with_psd else None,
    "bev_stage2_parkingrod_head" if with_rod else None,
]
bev_parking_out_name = remove_none(bev_parking_out_name)

bev_parking_gt_name = [
    "gt_bev_psd_obj" if with_psd else None,
    "gt_bev_parkingrod_obj" if with_rod else None,
]
bev_parking_gt_name = remove_none(bev_parking_gt_name)

psd_head_prefix_name = "bev_stage2_psd_small_head"
rod_head_prefix_name = "bev_stage2_parkingrod_small_head"


# ----------------------------- DESC ---------------------------
bev_psd_desc = dict(
    vcs_origin_coord=vcs_origin_coord,
    bev_stage2_input_resolution=spatial_resolution,
    bev_stage2_output_resolution=bev_psd_stage2_output_resolution,
    visible_range=reformat_compile_vcs_range(psd_roi_vcs_range),
    warp_offset_range=reformat_compile_vcs_range(vcs_range),
    vcs_plane_heights=vcs_plane_heights,
    fisheye_warp_offset_range=reformat_compile_vcs_range(fisheye_warp_range),
)


def get_bev_psd_desc():
    per_tensor_desc = [
        dict(
            task="bevpsd_j5_stage2",
            name="global_cls",
            metric=psd_metric_param,
            **bev_psd_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="global_offset",
            **bev_psd_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="global_occupancy",
            **bev_psd_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="global_slot_type",
            **bev_psd_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="global_direction",
            **bev_psd_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="local_cls",
            **bev_psd_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="local_offset",
            **bev_psd_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="local_sline_angle",
            **bev_psd_desc,
        ),
        dict(
            task="bevpsd_j5_stage2",
            name="local_point_type",
            **bev_psd_desc,
        ),
    ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


bev_parkingrod_desc = dict(
    vcs_origin_coord=vcs_origin_coord,
    bev_stage2_input_resolution=spatial_resolution,
    bev_stage2_output_resolution=bev_parkingrod_stage2_output_resolution,
    visible_range=reformat_compile_vcs_range(rod_roi_vcs_range),
    warp_offset_range=reformat_compile_vcs_range(vcs_range),
    vcs_plane_heights=vcs_plane_heights,
    fisheye_warp_offset_range=reformat_compile_vcs_range(fisheye_warp_range),
)


def get_bev_parkingrod_desc():
    per_tensor_desc = [
        dict(
            task="bev_parkingrod_j5_stage2",
            name="parkingrod_cls",
            metric=parkingrod_metric_param,
            **bev_parkingrod_desc,
        ),
        dict(
            task="bev_parkingrod_j5_stage2",
            name="parkingrod_endpoint_offset",
            **bev_parkingrod_desc,
        ),
    ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


def get_model(mode):
    # psd stage node
    bev_psd_postprocess = dict(
        type="ANCBevPSDDecoder",
        input_size=bev_psd_gt_size,
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
    bev_psd_stage2_out_module = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEVPSDHead",
            global_head=dict(
                type="ANCBEVPSDGlobalHead",
                num_slot_type=num_slot_type,
                in_channels=192,
                out_channels=32,
                stack=1,
                group_base=8,
                in_strides=[2, 4, 8, 16, 32],
                out_stride=global_downsample_factor * 4,
            ),
            local_head=dict(
                type="ANCBEVPSDLocalHead",
                in_channels=48,
                out_channels=32,
                stack=1,
                group_base=8,
                in_strides=[2, 4, 8, 16, 32],
                out_stride=local_downsample_factor * 4,
            ),
        ),
        loss=None,
        postprocess=None,
        prefix=psd_head_prefix_name,
        node_name=psd_head_prefix_name,
    )

    # parkingrod stage node
    bev_parkingrod_postprocess = dict(
        type="ANCBEVParkingrodDecoder",
        input_size=bev_parkingrod_gt_size,
        downsample_factor_feature=parkingrod_metric_param[
            "downsampling_factor"
        ],
        score_threshold=parkingrod_metric_param[
            "threshold_parkingrod_feature"
        ],
        nms_distance_threshold=parkingrod_metric_param[
            "nms_distance_threshold"
        ],
        topk=parkingrod_metric_param["topk_rod_feature"],
        kernel_size=parkingrod_metric_param["object_pool_kernel_size"],
    )
    bev_parkingrod_stage2_out_module = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEVParkingRodHead",
            num_parkingrod_class=num_parkingrod_class,
            in_strides=[2, 4, 8, 16, 32],
            out_strides=rod_out_stride,
            in_channels=96,
            mid_channels=48,
            sep_conv=False,
            stack=1,
            use_bias=True,
            aux_branch=True,
        ),
        head_parser=None,
        target=None,
        loss=None,
        postprocess=None,
        prefix=rod_head_prefix_name,
        node_name=rod_head_prefix_name,
    )
    bev_head = dict(
        type="MultiHeadOutputModule",
        heads=[bev_psd_stage2_out_module, bev_parkingrod_stage2_out_module],
    )

    if mode == "train":
        bev_psd_stage2_out_module["loss"] = dict(
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
            local_loss_near=dict(
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
                crop_roi=tuple(
                    [r - p for r, p in zip(near_roi, near_roi_padding)]
                ),
                gt_postfix="_near",
            )
            if aux_near_training
            else None,
            gt_name=f"gt_{psd_res_key}",
        )

        if aux_near_training:
            bev_psd_stage2_out_module["head"]["local_head_near"] = dict(
                type="ANCBEVPSDLocalHead",
                in_channels=48,
                out_channels=32,
                stack=1,
                group_base=8,
                in_strides=[2, 4, 8, 16, 32],
                out_stride=local_downsample_factor * 2,
                crop_roi=tuple(
                    [r - p for r, p in zip(near_roi, near_roi_padding)]
                ),
                crop_roi_output=None,
            )

        bev_parkingrod_stage2_out_module["loss"] = dict(
            type="ANCBEVParkingRodLoss",
            classification_loss=dict(type="FocalLossV2", reduction="none"),
            endpoint_offset_loss=dict(
                type="SmoothL1Loss", beta=1.0, reduction="none"
            ),
            loss_weight_dict=rod_loss_weight,
            aux_loss=dict(type="SmoothL1Loss", beta=1.0, reduction="none"),
            gt_name=f"gt_{rod_res_key}",
        )

    elif mode == "val":
        bev_psd_stage2_out_module["postprocess"] = bev_psd_postprocess
        bev_parkingrod_stage2_out_module[
            "postprocess"
        ] = bev_parkingrod_postprocess
        bev_parkingrod_stage2_out_module["head"]["aux_branch"] = False

    elif mode == "deploy":
        bev_psd_stage2_out_module["postprocess"] = dict(
            type="MultiInputSequential",
            modules=[
                # AddDesc should be the last module
                dict(
                    type="AddDesc",
                    per_tensor_desc=get_bev_psd_desc(),
                ),
            ],
        )

        bev_parkingrod_stage2_out_module["head"]["aux_branch"] = False
        bev_parkingrod_stage2_out_module["postprocess"] = dict(
            type="MultiInputSequential",
            modules=[
                # AddDesc should be the last module
                dict(
                    type="AddDesc",
                    per_tensor_desc=get_bev_parkingrod_desc(),
                ),
            ],
        )
    else:
        raise NotImplementedError

    multi_view_module = dict(
        img=dict(
            type="BEVStageOneModule",
            backbone=backbone,
            neck=pafpn_neck,
            head=head,
        ),
        round_img=dict(
            type="BEVStageOneModule",
            backbone=round_backbone,
            neck=round_pafpn_neck,
            head=round_head,
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
            elif key in fisheye_camera_view_names:
                multi_view_module[key] = dict(
                    type="BEVStageOneModule",
                    backbone=round_backbone,
                    neck=round_pafpn_neck,
                    head=deploy_round_head,
                )
            else:
                raise TypeError
        bevfusion_pick_keys = deploy_homo_offset_key

    model = dict(
        type="MultiViewTwoStageBEVModule",
        # view_img_key to module
        multi_view_module=multi_view_module,
        multi_view_collect=multi_view_collect,
        bev_fusion_module=dict(
            type="BEVStageTwoModule",
            bevfusion=bev_fusion[mode],
            bev_fusion_upsample=bev_fusion_upsampling,
            backbone=bev_backbone,
            neck=bev_neck,
            head=bev_head,
        ),
        bevfusion_pick_keys=bevfusion_pick_keys,
    )

    return model


# -------------------------- TRAIN METRIC --------------------------
# get psd metrics and patterns
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
            pred_pattern=f"^.*{psd_head_prefix_name}.*global_classification_loss$",  # noqa
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{psd_head_prefix_name}.*global_offset_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{psd_head_prefix_name}.*global_occupancy_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{psd_head_prefix_name}.*global_slot_type_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{psd_head_prefix_name}.*global_sideness_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{psd_head_prefix_name}.*global_direction_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{psd_head_prefix_name}.*local_classification_loss$",  # noqa
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{psd_head_prefix_name}.*local_offset_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{psd_head_prefix_name}.*local_sline_angle_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{psd_head_prefix_name}.*local_point_type_loss$",
        ),
    ]

    if aux_near_training:
        metrics.extend(
            [
                dict(type="LossShow", name="loss_psd_near_local_cls"),
                dict(type="LossShow", name="loss_psd_near_local_offset"),
                dict(type="LossShow", name="loss_psd_near_local_angle"),
                dict(type="LossShow", name="loss_psd_near_loss_point_type"),
            ]
        )
        per_metric_patterns.extend(
            [
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{psd_head_prefix_name}.*local_classification_loss_near$",  # noqa
                ),
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{psd_head_prefix_name}.*local_offset_loss_near$",  # noqa
                ),
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{psd_head_prefix_name}.*local_sline_angle_loss_near$",  # noqa
                ),
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{psd_head_prefix_name}.*local_point_type_loss_near$",  # noqa
                ),
            ]
        )

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
            label_pattern=f"^.*{task_name}.*annos_{psd_res_key}",
            pred_pattern=f"^.*{psd_head_prefix_name}.*decode_label*",
        ),
    ]
    return val_metrics, val_per_metric_patterns


psd_metrics, psd_per_metric_patterns = (
    get_psd_train_metrics_patterns() if with_psd else ([], [])
)
psd_val_metrics, psd_val_per_metric_patterns = (
    get_psd_val_metrics_patterns() if with_psd else ([], [])
)


# get parkingrod metrics and patterns
def get_rod_train_metrics_patterns():
    metrics = [
        dict(type="LossShow", name="loss_parkingrod_classification"),
        dict(type="LossShow", name="loss_parkingrod_endpoint_offset"),
        dict(type="LossShow", name="loss_parkingrod_slot_01offset"),
        dict(type="LossShow", name="loss_parkingrod_rod2slot_offset"),
    ]

    per_metric_patterns = [
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{rod_head_prefix_name}.*classification_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{rod_head_prefix_name}.*endpoint_offset_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{rod_head_prefix_name}.*slot_01offset_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{rod_head_prefix_name}.*rod2slot_offset_loss$",
        ),
    ]

    return metrics, per_metric_patterns


def get_rod_val_metrics_patterns():
    val_metrics = [
        dict(
            type="ANCBEVParkingrodMetric",
            name="BEVParkingrod",
            rod_distance_threshold=parkingrod_metric_param[
                "rod_distance_threshold"
            ],
            strip_dictance_threshold=parkingrod_metric_param[
                "strip_dictance_threshold"
            ],
            angle_threshold=parkingrod_metric_param["angle_threshold"],
            resolution_m_per_grid=parkingrod_metric_param[
                "resolution_m_per_grid"
            ],
            ego_vcs_range=parkingrod_metric_param["ego_vcs_range"],
            eval_vcs_range=parkingrod_metric_param["eval_vcs_range"],
            dep_intervals=parkingrod_metric_param["dep_intervals"],
            vcs_range=rod_roi_vcs_range,
            eval_result_path=rod_eval_result_path,
            result_prefix="small",
        )
    ]
    val_per_metric_patterns = [
        dict(
            label_pattern=f"^.*{task_name}.*annos_{rod_res_key}",
            pred_pattern=f"^.*{rod_head_prefix_name}.*preds_parkingrod*",
        ),
    ]
    return val_metrics, val_per_metric_patterns


rod_metrics, rod_per_metric_patterns = (
    get_rod_train_metrics_patterns() if with_rod else ([], [])
)
rod_val_metrics, rod_val_per_metric_patterns = (
    get_rod_val_metrics_patterns() if with_rod else ([], [])
)

# combine psd & parkingrod
metrics = psd_metrics + rod_metrics
per_metric_patterns = psd_per_metric_patterns + rod_per_metric_patterns


metric_updater = get_train_metric_updater(
    task_name=task_name,
    metrics=metrics,
    per_metric_patterns=per_metric_patterns,
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
    elif "preds_parkingrod" in key or "annos_bev_parkingrod_obj" in key:
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


val_metrics = psd_val_metrics + rod_val_metrics
val_per_metric_patterns = (
    psd_val_per_metric_patterns + rod_val_per_metric_patterns
)

_val_metric_updater = get_val_metric_updater(
    task_name=task_name,
    metrics=val_metrics,
    per_metric_patterns=val_per_metric_patterns,
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

# psd tb
psd_gt_hm_key_regex = re.compile(f"^.*{task_name}.*gt_bev_psd_weight_hm")
psd_pred_hm_key_regex = re.compile(
    f"^.*{psd_head_prefix_name}.*pred_bev_psd_hm"
)  # noqa

# parkingrod tb
parkingrod_gt_hm_key_regex = re.compile(
    f"^.*{task_name}.*gt_bev_parkingrod_weight_hm"
)
parkingrod_pred_hm_key_regex = re.compile(
    f"^.*{rod_head_prefix_name}.*pred_bev_parkingrod_hm"  # noqa
)

if enable_tensorboard:
    psd_tb_update_func = get_bev_tb_update_func(
        "bev_psd", psd_pred_hm_key_regex, psd_gt_hm_key_regex
    )
    parkingrod_tb_update_func = get_bev_tb_update_func(
        "bev_parkingrod",
        parkingrod_pred_hm_key_regex,
        parkingrod_gt_hm_key_regex,
    )
    tb_update_func_list = [psd_tb_update_func, parkingrod_tb_update_func]

# -------------------------------pack visualzie------------------------------
visualize = []
vis_small = update_content(
    vis_common,
    range_mode="small",
    vis_img_scale=int(
        vis_common["vis_img_scale"] * ipm_output_size[0] / bev_psd_gt_size[0]
    ),
    vcs_range=vcs_range,
    bev_size=bev_psd_gt_size,
    res_key2anno_name={
        "bev_psd": "annos_bev_psd_obj",
    },
    project_pts_to_cameras=True,  # False if do_pack_infer else True,
)
visualize.append(vis_small)
