import copy
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from bev_common import (
    bev_backbone,
    bev_fusion,
    bev_fusion_input_name,
    bev_fusion_input_stride,
    bev_fusion_output_name,
    bev_fusion_test,
    bev_fusion_upsample,
    bev_neck,
    bev_stage1_head,
    bev_stage2_feats_name,
    bevfusion_spatial_resolution,
    cam_standardized_transform,
    camera_view_names,
    default_calib,
    get_dataloader,
    get_homo_transforms,
    loss_weight,
    merge_ped_and_cyc,
    per_view_shape,
    seperate_stage2_among_bev_tasks,
    seperate_stage2_neck_among_bev_tasks,
    spatial_resolution,
    stage2_stride2channels,
    task_loss_weights,
    vcs_origin_coord,
    vcs_range,
)
from bev_dynamic_base import (
    bev_3d_out_size,
    bev_common_transforms,
    eval_setting,
    get_bev3d_sub_category_id_map,
    get_bev3d_tb_update_func,
    get_dataset_list,
    get_inputs,
    get_metric_updater,
    get_metrics_patterns,
    get_real3d_category_id_map,
    get_val_dataset_list,
    get_val_metric_updater,
    max_objs,
    repeat_image,
    save_eval_results,
    save_vis_dir,
    sub_dirs,
    task_num_classes,
    task_use_ignore_mask,
    train_batch_size_per_gpu,
    train_num_workers,
    use_nms,
    val_batch_size_per_gpu,
    val_num_workers,
    vis_setting,
)
from common import (
    backbone,
    bn_kwargs,
    datapaths,
    fpn_neck,
    input_cat,
    model_thresh,
    resize_hw,
    save_prefix,
    view_num,
    vis_tasks_bev,
    with_cam_standiardization,
)

# -------------------------- TASK ---------------------------
task_name = "bev_3d_cyclist_cls"
vis_tasks_bev.append(task_name)

object_type = "cyclist"
classidx2name = {
    0: "PersonRideBicycle",
    1: "PersonRideMotorcycle",
}
bev_batch_size = train_batch_size_per_gpu
enable_tensorboard = True
use_rec = True

# ------------------ BEV3D TRANSFORM SETTING -------------------
num_classes = task_num_classes[task_name]
use_ignore_mask = task_use_ignore_mask[task_name]
cls_dimension = np.array(
    [[1.5518998, 0.73560804, 1.7083853]] * num_classes
)  # Cyclist
cls_hm_kernel = {i: 3 for i in range(num_classes)}

bev3d_target = dict(
    type="ANCBev3dTargetGenerator",
    num_classes=num_classes,
    max_objs=max_objs,
    bev_size=bev_3d_out_size,
    vcs_range=vcs_range,
    cls_dimension=cls_dimension,
    cls_hm_kernel=cls_hm_kernel,
    category2id_map=get_real3d_category_id_map(task_name),  # sub category
    enable_ignore=use_ignore_mask,
)

bev_3d_transforms = [
    dict(
        type="ConvertMultiViewAnnoTo3DV",
        category_id_dict=get_real3d_category_id_map(
            "bev_3d_cyclist"
        ),  # big class vehicle
        camera_list=sub_dirs,
        per_view_shape=per_view_shape,
        pad_value=128,
        using_sub_category=True,
        default_calib=default_calib,
        sub_category_id_dict=get_bev3d_sub_category_id_map(task_name),
    ),
    repeat_image,
    bev_common_transforms["Resize3DV"],
    bev_common_transforms["Crop3DV"],
    bev_common_transforms["ResizeHomo"],
    bev_common_transforms["ToTensor3DV"],
    # bev_common_transforms["Normalize3DV"],
    bev3d_target,
    bev_common_transforms["PrepareDataBEV"],
]
if with_cam_standiardization:
    bev_3d_transforms.insert(5, cam_standardized_transform)
homo_transforms = get_homo_transforms(
    transforms_list=bev_3d_transforms, camera_view_names=camera_view_names
)

train_data_paths = datapaths.multiview_cyclist_3d_detection.train_data_paths
val_data_paths = datapaths.multiview_cyclist_3d_detection.val_data_paths

train_datasets_list = get_dataset_list(
    train_data_paths[0], bev_3d_transforms, homo_transforms
)
val_datasets_list = get_val_dataset_list(
    val_data_paths, bev_3d_transforms, homo_transforms
)


# ----------------------- DATALODER---------------------------
data_loader = get_dataloader(train_datasets_list, train_num_workers)
data_loader["sampler"] = dict(type=torch.utils.data.DistributedSampler)
data_loader["shuffle"] = True
# To fix reload imgrec problem, use persistent_workers.
# persistent_workers option needs num_workers > 0
data_loader["persistent_workers"] = train_num_workers > 0

val_data_loader = get_dataloader(val_datasets_list, val_num_workers)
val_data_loader["sampler"] = dict(type=torch.utils.data.DistributedSampler)
val_data_loader["batch_size"] = val_batch_size_per_gpu

# -------------------------- MODEL --------------------------
train_inputs, val_inputs, test_inputs = get_inputs(num_classes)
inputs = dict(
    train=train_inputs,
    val={},
    test=test_inputs,
)

task_loss_weight = task_loss_weights[task_name]

bev_stage2_target_name = "gt_bev_3d"


def get_model(mode):
    bev_stage2_out_module = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEV3DHead",
            feature_name=bev_stage2_feats_name,
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[2],
            in_channels=stage2_stride2channels[2],
            forward_frame_idx=0,
            head_channels=OrderedDict(
                bev3d_hm=num_classes,
                bev3d_dim=3,  # h, w, l
                bev3d_rot=2,  # cos, sin
                bev3d_ct_offset=2,  # x, y
                bev3d_loc_z=1,  # vcs z axis
            ),
            use_bias=False,
            bn_kwargs=bn_kwargs,
            dw_with_relu=True,
            pw_with_relu=False,
            factor=2,
            group_base=8,
        ),
        head_parser=None,
        loss=dict(
            type="ANCBEV3DLoss",
            loss_weights=dict(
                bev3d_hm=1.0 * loss_weight * task_loss_weight,
                bev3d_dim=1.0 * loss_weight * task_loss_weight,
                bev3d_rot=3.0 * loss_weight * task_loss_weight,
                bev3d_ct_offset=1.5 * loss_weight * task_loss_weight,
                bev3d_loc_z=1.5 * loss_weight * task_loss_weight,
            ),
            gamma=1,
            beta=0.33,
        ),
        postprocess=None,
        target=None,
        node_name="bev_stage2_3d_cyclist_cls_head",
    )

    if mode == "train" or mode == "val":
        bev_fusion_module = bev_fusion
        stage1_input_keys = ["img"] if view_num != 6 else ["img", "side_img"]
    if mode == "val":
        bev_stage2_out_module.pop("loss")
        bev_stage2_out_module.update({"convert_to_dict": False})
        bev_stage2_out_module["postprocess"] = dict(
            type="ANCBEV3Decoder",
            topk=100,
            max_pool_kernel=5,
            cls_dimension=cls_dimension,
            vcs_range=vcs_range,
            bev_nms=use_nms,
            bev_3d_out_size=bev_3d_out_size,
            num_classes=num_classes,
            classidx2name=classidx2name,
            add_hm_eps=True,
            nms_thresh=0.5,
            center_offset=True,
            score_threshold=model_thresh["bev_det_thresh"][task_name]
            if model_thresh
            else 0.0,
            node_name="bev_stage2_3d_cyclist_cls_decoder",
        )
    if mode == "test":
        bev_fusion_module = bev_fusion_test
        stage1_input_keys = [f"img_{i}" for i in range(view_num)]
        input_cat.update({"need_cat": False})
        bev_stage2_out_module["target"] = None
        bev_stage2_out_module["loss"] = None
        bev_stage2_out_module["head_parser"] = None
        bev_stage2_out_module["postprocess"] = dict(
            type="MultiInputSequential",
            modules=[
                # # AddDesc should be the last module
                dict(
                    type="AddDesc",
                    per_tensor_desc=get_bev_desc(),
                ),
            ],
        )
    indicator = "cyc" if not merge_ped_and_cyc else "pedcyc"
    if not seperate_stage2_among_bev_tasks:
        cyc_backbone = bev_backbone
    else:
        cyc_backbone = copy.deepcopy(bev_backbone)
        cyc_backbone["node_name"] = f"bev_{indicator}_stage2_back_bone"
    if (
        not seperate_stage2_among_bev_tasks
        and not seperate_stage2_neck_among_bev_tasks
    ):
        cyc_neck = bev_neck
    else:
        cyc_neck = copy.deepcopy(bev_neck)
        cyc_neck["node_name"] = f"bev_{indicator}_stage2_neck"
    return dict(
        type="ListInputModelWraper",
        seq_len=view_num,
        preprocess=input_cat,
        model=dict(
            type="TwoStageBEVModule",
            stage1_input_keys=stage1_input_keys,
            stage1_module=dict(
                type="BEVStageOneModule",
                backbone=backbone,
                neck=fpn_neck,
                head=bev_stage1_head,
            ),
            stage2_module=dict(
                type="BEVStageTwoModule",
                bevfusion=bev_fusion_module,
                backbone=cyc_backbone,
                neck=cyc_neck,
                head=bev_stage2_out_module,
                bev_fusion_upsample=bev_fusion_upsample,
                bev_fusion_output_name=bev_fusion_output_name,
                bev_stage2_feats_name=bev_stage2_feats_name,
            ),
            bev_fusion_input_name=bev_fusion_input_name,
            view_num=view_num,
        ),
    )


# -------------------------- SOLVER --------------------------
metrics, per_metric_patterns = get_metrics_patterns(task_name)
metric_updater = get_metric_updater(metrics, per_metric_patterns, task_name)

# -------------------------EVAL SETTING-----------------------
if save_eval_results:
    save_path = os.path.join(save_prefix, f"{task_name}_pred")
else:
    save_path = None
prcurv_save_path = os.path.join(save_prefix, f"{task_name}_aps")
eval_setting[task_name]["save_path"] = save_path
eval_setting[task_name]["prcurv_save_path"] = prcurv_save_path
val_metric_updater = get_val_metric_updater(
    task_name,
    **eval_setting[task_name],
    use_ignore_mask=use_ignore_mask,
    save_vis_dir=save_vis_dir,
    vis_setting=vis_setting,
    # visibility_intervals=visibility_intervals,
)

# -------------------------- TENSORBOAED --------------------------
if enable_tensorboard:
    tb_update_func = get_bev3d_tb_update_func

# model desc
visible_range = [
    vcs_range[2],
    vcs_range[0],
    vcs_range[3],
    vcs_range[1],
]  # (top, bottom, left, right)

bev_common_desc = dict(
    vcs_origin_coord=vcs_origin_coord,
    bev_stage2_input_resolution=spatial_resolution,
    # NOTE: if modify bev_stage2_input_resolution, need to modify the
    # the "projects/fsd/jarvis/ckpt2hbm_configs/bev-3d.yaml" fsd_bev3d_stage2's
    # input_size to 6x1x(resolution)x(resolution)x2
    bev_stage2_output_resolution=bevfusion_spatial_resolution,
    visible_range=visible_range,
    bev_fusion_input_resolution=[
        resize_hw[0] // bev_fusion_input_stride,
        resize_hw[1] // bev_fusion_input_stride,
    ],
)


# TODO: to better generate desc

if num_classes == 1:
    task_cls_label_desc = ["cyclist"]
elif num_classes == 2:
    task_cls_label_desc = ["PersonRideBicycle", "PersonRideMotorcycle"]
else:
    raise NotImplementedError


def get_bev_desc():
    per_tensor_desc = [
        {
            "task": "bev_3d_cyclist_cls",
            "output_name": "bev_3d_cyclist_cls_heatmap_output",
            "score_threshold": model_thresh["bev_det_thresh"][task_name]
            if model_thresh
            else 0.2,
            "max_pool_kernel": 5,
            "properties": [{"channel_labels": task_cls_label_desc}],
            **bev_common_desc,
        },
        {
            "task": "bev_3d_cyclist_cls",
            "output_name": "bev_3d_cyclist_cls_dimension_output",
            "cls_dimension": [1.5518998, 0.73560804, 1.7083853],
            "properties": [{"channel_labels": ["height", "width", "length"]}],
            **bev_common_desc,
        },
        {
            "task": "bev_3d_cyclist_cls",
            "output_name": "bev_3d_cyclist_cls_rot_output",
            "properties": [{"channel_labels": ["cos", "sin"]}],
            **bev_common_desc,
        },
        {
            "task": "bev_3d_cyclist_cls",
            "output_name": "bev_3d_cyclist_cls_ct_offset_output",
            "properties": [{"channel_labels": ["offset_x", "offset_y"]}],
            **bev_common_desc,
        },
        {
            "task": "bev_3d_cyclist_cls",
            "output_name": "bev_3d_cyclist_cls_loc_z_output",
            "properties": [{"channel_labels": ["z_value"]}],
            **bev_common_desc,
        },
    ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc
