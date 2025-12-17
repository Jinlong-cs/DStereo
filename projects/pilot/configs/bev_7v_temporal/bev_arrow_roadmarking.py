import copy
import os
import re
from collections import OrderedDict

import torch

from hat.data.transforms.auto_3dv import get_roi_vcs_range_box
from hat.utils import Config
from hat.utils.apply_func import _as_list
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
    get_update_metric_func,
    remove_none,
)
from projects.pilot.configs.bev_7v_temporal.bev_discobj_base import (
    bev_common_transforms,
    collect_3dv,
    get_bev_tb_update_func,
    get_inputs,
    get_metrics_patterns,
    get_train_metric_updater,
    get_val_metrics,
    name2group,
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
    bn_kwargs,
    bucket_root,
    camera_view_names,
    do_val_visualize,
    ego_ignore_range,
    head,
    img_ori_size,
    img_resize_wh_size,
    length_of_clip,
    log_freq,
    narrow_backbone,
    narrow_head,
    narrow_pafpn_neck,
    num_views,
    offset_save_path,
    pafpn_neck,
    pipeline_test,
    save_prefix,
    side_backbone,
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
    vcs_plane_heights,
    vcs_range,
)

# -------------------------- task --------------------------

task_name = "bev_arrow_roadmarking"
base_task_name = "bev_disc"
bev_batch_size = train_batch_size_per_gpu
enable_tensorboard = True
use_random_rotation = False
use_rec = True
arrow_head_with_cls = True

# -------------------------- data --------------------------
train_data_version = "temporal_v1_3_0_wide"
val_data_version = "temporal_v1_1_0_wide"

if pipeline_test:
    train_data_version = "pipeline_test"
    val_data_version = "pipeline_test"

dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_discrete_obj"
)

vcs_bbox_area_thresh = 0

use_distorted_offset = True
cal_homo_offset_on_gpu = True

use_psc_rot = True
N_steps_PSC_rot = 3
arrow_psc_phase_factor = 1
rmk_psc_phase_factor = 2
use_norm_rot = True

# ------------------ BEV arrow transformation setting -------------------
arrow_roi_vcs_range = (0.0, -21.2, 51.2, 21.2)
arrow_task_in_stride = 2
arrow_task_out_size = (128, 106)

arrow_category2id_map = {
    "Other": 0,
    "Straight": 1,
    "Turnleft": 2,
    "Turnright": 3,
    "Turnoff": 4,
    "Mergeleft": 5,
    "Mergeright": 6,
    "No_Straight": 7,
    "No_Turnleft": 8,
    "No_Turnright": 9,
    "No_Turnoff": 10,
    "No_Mergeleft": 11,
    "No_Mergeright": 12,
    "Turnleft_Straight": 13,
    "Straight_Turnleft": 13,
    "Turnright_Straight": 14,
    "Straight_Turnright": 14,
    "Turnleft_Turnright": 15,
    "Turnright_Turnleft": 15,
    "Turnleft_Turnoff": 16,
    "Turnoff_Turnleft": 16,
    "Straight_Turnoff": 17,
    "Turnoff_Straight": 17,
    "Straight_Turnright_Turnleft": 18,
    "Straight_Turnleft_Turnright": 18,
    "Turnleft_Straight_Turnright": 18,
    "Turnleft_Turnright_Straight": 18,
    "Turnright_Straight_Turnleft": 18,
    "Turnright_Turnleft_Straight": 18,
}
arrow_id2label = {
    0: "Other",
    1: "Straight",
    2: "Turnleft",
    3: "Turnright",
    4: "Turnoff",
    5: "Mergeleft",
    6: "Mergeright",
    7: "No_Straight",
    8: "No_Turnleft",
    9: "No_Turnright",
    10: "No_Turnoff",
    11: "No_Mergeleft",
    12: "No_Mergeright",
    13: "Straight_Turnleft",
    14: "Straight_Turnright",
    15: "Turnleft_Turnright",
    16: "Turnleft_Turnoff",
    17: "Straight_Turnoff",
    18: "Straight_Turnright_Turnleft",
}
arrow_name2label = {"arrows": "ARROW"}
arrow_num_classes = max(arrow_category2id_map.values()) + 1


# road discrete object
bev_arrow_target = dict(
    type="ANCBevDiscreteWithClsTargetGenerator"
    if arrow_head_with_cls
    else "ANCBevDiscreteTargetGenerator",
    num_classes=arrow_num_classes,
    bev_size=arrow_task_out_size,
    max_objs=150,
    vcs_range=arrow_roi_vcs_range,
    category2id_map=arrow_category2id_map,
    vcs_bbox_area_thresh=vcs_bbox_area_thresh,
    name2label=arrow_name2label,
    visible_threshold=0.2,
    name2group=name2group,
    ego_ignore_range=ego_ignore_range,
    vis_mask_vcs_range_cfg=vismask_vcsrange_cfg,
    use_vis_mask=True,
    psc_phase_factor=arrow_psc_phase_factor,
    N_steps_PSC_rot=N_steps_PSC_rot if use_psc_rot else None,
    return_ignore_obj=False,
    res_key="bev_arrow",
)

arrow_roi_resize_cfg = get_roi_resize_cfg(
    input_size=bevfusion_output_size,
    in_stride=arrow_task_in_stride,
    output_size=arrow_task_out_size,
    ori_vcs_range=vcs_range,
    roi_vcs_range=arrow_roi_vcs_range,
)
arrow_roi_resize = dict(
    type="RoiCropResize",
    in_strides=[2, 4, 8, 16, 32],
    target_stride=arrow_roi_resize_cfg["in_stride"],
    output_size=arrow_roi_resize_cfg["output_size"],
    roi_box=arrow_roi_resize_cfg["roi_box"],
    node_name="bev_arrow_roi_resize",
)

# ------------------ BEV roadmarking transformation setting -------------------
rmk_roi_vcs_range = (-32.0, -21.6, 52.8, 21.6)
rmk_task_in_stride = 2
rmk_task_out_size = (212, 72)

# merge_crosswalk_to_junction == True
rmk_category2id_map = {
    "Stopline": 0,
    "DiamondMarking": 1,
    "InvertedTriangleMarking": 2,
    "SpeedBump": 3,
    "NoParkingLine": 4,
    "ExclusiveLaneSign": 5,
}
rmk_id2label = {
    0: "Stopline",
    1: "DiamondMarking",
    2: "InvertedTriangleMarking",
    3: "SpeedBump",
    4: "NoParkingLine",
    5: "ExclusiveLaneSign",
}
rmk_name2label = {
    "stoplines": "Stopline",
    "diamond_markings": "DiamondMarking",
    "inverted_triangle_markings": "InvertedTriangleMarking",
    "speedbumps": "SpeedBump",
    "noparking_lines": "NoParkingLine",
    "exclusive_lane_signs": "ExclusiveLaneSign",
}
rmk_num_classes = max(rmk_category2id_map.values()) + 1


valid_vcs_range = {
    "Stopline": (-32.0, -32.0, 52.8, 32.0),
    "DiamondMarking": (-32.0, -32.0, 52.8, 32.0),
    "InvertedTriangleMarking": (-32.0, -32.0, 52.8, 32.0),
    "SpeedBump": (-32.0, -32.0, 52.8, 32.0),
    "NoParkingLine": (-32.0, -32.0, 52.8, 32.0),
    "ExclusiveLaneSign": (-32.0, -32.0, 52.8, 32.0),
}

valid_range_percls = {}
for category, valid_vcs in valid_vcs_range.items():
    cls_id = rmk_category2id_map[category]
    # top, left, bottom, right
    vis_mask_crop_roi_box = get_roi_vcs_range_box(
        rmk_task_out_size, rmk_roi_vcs_range, valid_vcs
    )
    valid_range_percls[cls_id] = vis_mask_crop_roi_box


# road discrete object
bev_rmk_target = dict(
    type="ANCBevDiscreteTargetGenerator",
    num_classes=rmk_num_classes,
    bev_size=rmk_task_out_size,
    max_objs=150,
    vcs_range=rmk_roi_vcs_range,
    category2id_map=rmk_category2id_map,
    vcs_bbox_area_thresh=vcs_bbox_area_thresh,
    name2label=rmk_name2label,
    visible_threshold=0.2,
    name2group=name2group,
    ignore_miss_cls=True,
    ignore_miss_clsid=0,
    ego_ignore_range=ego_ignore_range,
    vis_mask_vcs_range_cfg=vismask_vcsrange_cfg,
    use_vis_mask=True,
    valid_vcs_range_percls=None,
    psc_phase_factor=rmk_psc_phase_factor,
    N_steps_PSC_rot=N_steps_PSC_rot if use_psc_rot else None,
    res_key="bev_roadmarking",
)

rmk_roi_resize_cfg = get_roi_resize_cfg(
    input_size=bevfusion_output_size,
    in_stride=rmk_task_in_stride,
    output_size=rmk_task_out_size,
    ori_vcs_range=vcs_range,
    roi_vcs_range=rmk_roi_vcs_range,
)
roadmarking_roi_resize = dict(
    type="RoiCropResize",
    in_strides=[2, 4, 8, 16, 32],
    target_stride=rmk_roi_resize_cfg["in_stride"],
    output_size=rmk_roi_resize_cfg["output_size"],
    roi_box=rmk_roi_resize_cfg["roi_box"],
    node_name="bev_roadmarking_roi_resize",
)

bev_discobj_target_list = [bev_arrow_target, bev_rmk_target]


bev_common_transforms = copy.deepcopy(bev_common_transforms)
val_common_transforms = copy.deepcopy(val_common_transforms)
bev_discobj_transforms = [
    collect_3dv,
    bev_common_transforms["ANCResize3DV"],
    bev_common_transforms["ANCCrop3DV"],
    bev_common_transforms["ANCTemporalHomo"],
    *bev_discobj_target_list,
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
        "bev_discrete_obj_lmdb_path",
        "homo_path",
        "calib_path",
        "bev_occlusion_data_path",
        "camera_module_type",
        "homo_noise",
    ],
    update_trans_info={
        "ANCCollect3DV": "fill_fake_temporal_data",
    },
)

val_template_dataset = copy.deepcopy(template_dataset)

val_template_dataset["num_frames_per_iter"] = val_num_frames_per_iter
val_template_dataset["transforms"][0]["img_idxs"] = list(
    range(val_num_frames_per_iter)
)
val_template_dataset["transforms"][0]["pose_idxs"] = list(
    range(val_num_frames_per_iter + 1)
)

for target_generator in bev_discobj_target_list:
    target_generator_index = val_template_dataset["transforms"].index(
        target_generator
    )
    val_template_dataset["transforms"][target_generator_index][
        "return_ignore_obj"
    ] = True
    if "bev_arrow" in target_generator["res_key"]:
        val_template_dataset["transforms"][target_generator_index][
            "visible_threshold"
        ] = 0.4


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
        data_loader = convert_to_split_dataloader(
            data_loader, temporal_bev=True
        )
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
    num_views, vcs_plane_heights, arrow_task_out_size, arrow_num_classes
)
pop_keys = [
    "gt_bev_discrete_obj",
    "annos_bev_discrete_obj",
]
for pop_key in pop_keys:
    train_inputs.pop(pop_key, None)
    # val_inputs.pop(pop_key, None)


add_keys = dict(
    gt_bev_arrow={
        "bev_discobj_hm": torch.zeros(
            1,
            1,
            arrow_task_out_size[0],
            arrow_task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_wh": torch.zeros(
            1,
            2,
            arrow_task_out_size[0],
            arrow_task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_hm_cls": torch.zeros(
            1,
            arrow_num_classes,
            arrow_task_out_size[0],
            arrow_task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_rot": torch.zeros(
            1,
            3,
            arrow_task_out_size[0],
            arrow_task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_ct_offset": torch.zeros(
            1,
            2,
            arrow_task_out_size[0],
            arrow_task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_weight_hm": torch.zeros(
            1,
            1,
            arrow_task_out_size[0],
            arrow_task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_ignore": torch.zeros(
            1,
            1,
            arrow_task_out_size[0],
            arrow_task_out_size[1],
            dtype=torch.float32,
        ),
    },
    gt_bev_roadmarking={
        "bev_discobj_hm": torch.zeros(
            1,
            1,
            rmk_task_out_size[0],
            rmk_task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_wh": torch.zeros(
            1,
            2,
            rmk_task_out_size[0],
            rmk_task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_hm_cls": torch.zeros(
            1,
            rmk_num_classes,
            rmk_task_out_size[0],
            rmk_task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_rot": torch.zeros(
            1,
            3,
            rmk_task_out_size[0],
            rmk_task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_ct_offset": torch.zeros(
            1,
            2,
            rmk_task_out_size[0],
            rmk_task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_weight_hm": torch.zeros(
            1,
            1,
            rmk_task_out_size[0],
            rmk_task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_ignore": torch.zeros(
            1,
            1,
            rmk_task_out_size[0],
            rmk_task_out_size[1],
            dtype=torch.float32,
        ),
    },
)

train_inputs.update(add_keys)
# val_inputs.update(add_keys)

inputs = dict(
    train=train_inputs,
    val=val_inputs,
    deploy=deploy_inputs,
)

if arrow_head_with_cls:
    # update head_channels
    arrow_head_channels = OrderedDict(
        pred_bev_discobj_hm=1,
        pred_bev_discobj_hm_cls=arrow_num_classes,
        pred_bev_discobj_wh=2,  # w, h
        pred_bev_discobj_rot=3,  # cos, sin
        pred_bev_discobj_ct_offset=2,  # u, v
    )
    # update loss
    arrow_loss_weights = {
        "bev_discobj_hm": 3.0,
        "bev_discobj_hm_cls": 3.0,
        "bev_discobj_hm_cls_aux": 3.0,
        "bev_discobj_wh": 1.5,
        "bev_discobj_ct_offset": 1.5,
        "bev_discobj_rot": 4.5,
    }

rmk_head_channels = OrderedDict(
    pred_bev_discobj_hm=rmk_num_classes,
    pred_bev_discobj_wh=2,  # w, h
    pred_bev_discobj_rot=3,  # cos, sin
    pred_bev_discobj_ct_offset=2,  # u, v
)
# update loss
rmk_loss_weights = {
    "bev_discobj_hm": 1.6,
    "bev_discobj_wh": 0.8,
    "bev_discobj_ct_offset": 0.8,
    "bev_discobj_rot": 2.4,
}


def get_model(mode):
    bev_arrow_head = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEV3DHead",
            in_strides=[2, 4],
            out_strides=[2],
            in_channels=48,
            forward_frame_idx=0,
            head_channels=arrow_head_channels,
            use_varg=False,
            last_conv_kernel_size=3,
            bn_kwargs=bn_kwargs,
            merge_block=True,
            merge_output=False,
        ),
        head_parser=None,
        loss=None,
        postprocess=None,
        target=None,
        prefix="bev_stage2_arrow_head",
        node_name="bev_stage2_arrow_head",
    )

    bev_rmk_head = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEV3DHead",
            in_strides=[2, 4],
            out_strides=[4],
            in_channels=48,
            forward_frame_idx=0,
            head_channels=rmk_head_channels,
            use_varg=False,
            last_conv_kernel_size=3,
            bn_kwargs=bn_kwargs,
            merge_block=True,
            merge_output=False,
        ),
        head_parser=None,
        loss=None,
        postprocess=None,
        target=None,
        prefix="bev_stage2_roadmarking_head",
        node_name="bev_stage2_roadmarking_head",
    )

    bev_head = dict(
        type="MultiHeadOutputModule",
        heads=[bev_arrow_head, bev_rmk_head],
    )

    if mode == "train":
        bev_arrow_head["loss"] = dict(
            type="ANCBEVDiscreteObjectWithClsLoss",
            loss_weights=arrow_loss_weights,
            use_focal_hm_loss=True,
            use_focal_cls_loss=False,
            use_softmax_focal_aux_loss=True,
            use_norm_rot=True,
            gt_name="gt_bev_arrow",
        )
        bev_rmk_head["loss"] = dict(
            type="ANCBEVDiscreteObjectLoss",
            loss_weights=rmk_loss_weights,
            use_focal_hm_loss=True,
            use_norm_rot=True,
            gt_name="gt_bev_roadmarking",
        )
    elif mode == "val":
        bev_arrow_head["postprocess"] = dict(
            type="ANCBEVDiscreteObjectDecoder",
            cls_dimension=None,
            topk=20,
            max_pool_kernel=7,
            vcs_range=arrow_roi_vcs_range,
            ct_nms_config={
                "do_ct_nms": True,
                "ct_dist_scale": 1.0,
            },
            rot_mod_threshold=0.0001,
            use_psc_rot=use_psc_rot,
            N_steps_PSC_rot=N_steps_PSC_rot if use_psc_rot else None,
            psc_phase_factor=arrow_psc_phase_factor,
            use_norm_rot=use_norm_rot,
        )
        bev_rmk_head["postprocess"] = dict(
            type="ANCBEVDiscreteObjectDecoder",
            cls_dimension=None,
            topk=20,
            max_pool_kernel=9,
            vcs_range=rmk_roi_vcs_range,
            ct_nms_config={
                "do_ct_nms": True,
                "ct_dist_scale": 0.5,
                "class_ids": [0, 5],
                "rot_match_threshold": 0.5,
                "kernel_from_singlebox": False,
            },
            rot_mod_threshold=0.0001,
            use_psc_rot=use_psc_rot,
            N_steps_PSC_rot=N_steps_PSC_rot if use_psc_rot else None,
            psc_phase_factor=rmk_psc_phase_factor,
            use_norm_rot=use_norm_rot,
        )
    else:
        pass

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
            bev_fusion_upsample=bev_fusion_upsampling,
            temporal_fusion=temporal_fusion[mode],
            backbone=bev_backbone,
            neck=bev_neck,
            roi_resizes=[arrow_roi_resize, roadmarking_roi_resize],
            head=bev_head,
        ),
    )

    return model


# -------------------------- TRAIN METRIC --------------------------
arrow_metrics, arrow_per_metric_patterns = get_metrics_patterns(
    "arrow_head",
    head_with_cls=True,
)

rmk_metrics, rmk_per_metric_patterns = get_metrics_patterns(
    "roadmarking_head",
)

metrics = arrow_metrics + rmk_metrics
per_metric_patterns = arrow_per_metric_patterns + rmk_per_metric_patterns

metric_updater = get_train_metric_updater(
    task_name=task_name,
    metrics=metrics,
    per_metric_patterns=per_metric_patterns,
    log_freq=log_freq,
)

# -------------------------- VALIDATION SETTING --------------------------

arrow_val_metric_setting = {
    "score_threshold": 0.2,
    "iou_threshold": 0.2,
    "yaw_amplitude": 360,
    "gt_max_depth": 55,
}
rmk_val_metric_setting = {
    "score_threshold": 0.2,
    "iou_threshold": 0.2,
    "yaw_amplitude": 180,
    "gt_max_depth": 55,
    "gt_match_mode": {"Stopline": "ct_rot", "SpeedBump": "ct_rot"},
}

val_metric_updater_list = []
for dataset_key in _as_list(val_data_version):
    save_dir = os.path.join(save_prefix, "eval", dataset_key)
    if not os.path.exists(save_dir):
        # shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    save_metric_path = os.path.join(
        save_dir, f"{task_name}_{training_step}_eval_metric.json"
    )
    confusion_save_path = None
    # confusion_save_path = save_dir

    # set save_vis_path to open visualize
    # save_vis_dir = None
    save_vis_dir = os.path.join(save_dir, "vis")
    if os.path.exists(save_vis_dir):
        # shutil.rmtree(save_vis_dir)
        os.makedirs(save_vis_dir)

    save_pred = None
    # save_prcuv = None
    # save_pred = os.path.join(save_dir, f"{task_name}_pred")
    save_prcuv = os.path.join(save_dir, f"{task_name}_aps")

    arrow_val_metric_patterns = [
        dict(
            pred_ct=f"^.*arrow_head.*pred_bev_discobj_ct",  # noqa
            pred_wh=f"^.*arrow_head.*pred_bev_discobj_wh",  # noqa
            pred_score=f"^.*arrow_head.*pred_bev_discobj_score",  # noqa
            pred_yaw=f"^.*arrow_head.*pred_bev_discobj_rot",  # noqa
            pred_bev_discobj_cls_id=f"^.*arrow_head.*pred_bev_discobj_cls_id",  # noqa
        )
    ]

    arrow_val_metrics = get_val_metrics(
        task_name=task_name,
        id2label=arrow_id2label,
        eval_category_ids=tuple(range(arrow_num_classes))[1:],
        iou_threshold=arrow_val_metric_setting["iou_threshold"],
        score_threshold=arrow_val_metric_setting["score_threshold"],
        gt_max_depth=arrow_val_metric_setting["gt_max_depth"],
        metrics=("dx", "dy", "dxy", "dw", "dh", "drot"),
        save_dir=save_dir,
        save_eval_result_path=save_metric_path,
        yaw_amplitude=arrow_val_metric_setting["yaw_amplitude"],
        match_mode=arrow_val_metric_setting.get("gt_match_mode", None),
        compute_foreground_prec=True,
        vis_image_dir=save_vis_dir,
        annos_key="annos_bev_arrow",
        result_prefix="wide",
    )

    rmk_val_metric_patterns = [
        dict(
            pred_ct=f"^.*roadmarking_head.*pred_bev_discobj_ct",  # noqa
            pred_wh=f"^.*roadmarking_head.*pred_bev_discobj_wh",  # noqa
            pred_score=f"^.*roadmarking_head.*pred_bev_discobj_score",  # noqa
            pred_yaw=f"^.*roadmarking_head.*pred_bev_discobj_rot",  # noqa
            pred_bev_discobj_cls_id=f"^.*roadmarking_head.*pred_bev_discobj_cls_id",  # noqa
        )
    ]

    rmk_val_metrics = get_val_metrics(
        task_name=task_name,
        id2label=rmk_id2label,
        eval_category_ids=tuple(range(rmk_num_classes)),
        iou_threshold=rmk_val_metric_setting["iou_threshold"],
        score_threshold=rmk_val_metric_setting["score_threshold"],
        gt_max_depth=rmk_val_metric_setting["gt_max_depth"],
        metrics=("dx", "dy", "dxy", "dw", "dh", "drot"),
        save_dir=save_dir,
        save_eval_result_path=save_metric_path,
        yaw_amplitude=rmk_val_metric_setting["yaw_amplitude"],
        match_mode=rmk_val_metric_setting.get("gt_match_mode", None),
        compute_foreground_prec=True,
        vis_image_dir=save_vis_dir,
        annos_key="annos_bev_roadmarking",
        result_prefix="wide",
    )

    val_metrics = arrow_val_metrics + rmk_val_metrics
    val_metric_patterns = arrow_val_metric_patterns + rmk_val_metric_patterns

    val_metric_updater = dict(
        type="MetricUpdater",
        metrics=val_metrics,
        metric_update_func=get_update_metric_func(
            "val", task_name, val_metric_patterns
        ),
        step_log_freq=-1,
        epoch_log_freq=1,
        log_prefix=f"Validation_{task_name}_{dataset_key}",
    )
    val_metric_updater_list.append(val_metric_updater)


# -------------------------- tensorboard --------------------------
gt_hm_key_regex = re.compile(
    f"^.*{task_name}.*gt_bev_discrete_obj_bev_discobj_weight_hm"
)
pred_hm_key_regex = re.compile(f"^.*{task_name}.*pred_bev_discobj_hm")

if enable_tensorboard:
    tb_update_func = get_bev_tb_update_func(
        task_name, pred_hm_key_regex, gt_hm_key_regex
    )
