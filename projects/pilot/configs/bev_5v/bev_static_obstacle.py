import copy
import json
import os
import re
from collections import OrderedDict

import torch

from hat.utils import Config
from hat.utils.apply_func import _as_list
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
    get_update_metric_func,
    remove_none,
    update_content,
)
from projects.pilot.configs.bev_5v.bev_discobj_base import (
    bev_common_transforms,
    collect_3dv,
    get_bev_tb_update_func,
    get_inputs,
    get_metrics_patterns,
    get_train_metric_updater,
    get_val_metrics,
    high_sp_resolution,
    high_sp_vcs_range,
    img_scale_before_ipm,
    name2group,
    reformat_compile_vcs_range,
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
    bn_kwargs,
    bucket_root,
    camera_view_names,
    deploy_head,
    deploy_homo_offset_key,
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

task_name = "bev_static_obstacle"
sub_tasks = ["bev_parkingloc", "bev_cementcolumn"]
base_task_name = "bev_disc"
enable_tensorboard = True

# -------------------------- data --------------------------
train_data_version = "v_1_7_1"
val_data_version = ["v1.3.0", "v1.0_open_parking", "EK_v1.0"]

if pipeline_test:
    train_data_version = "pipeline_test"
    val_data_version = "pipeline_test"

dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_discrete_obj"
)

# ------------------ BEV static obstacle transformation setting -------------------
roi_vcs_range = (-12.8, -12.8, 25.6, 12.8)
task_in_stride = 4
task_out_size = (192, 128)

bev_discobj_stage2_output_resolution = (
    abs(roi_vcs_range[2] - roi_vcs_range[0]) / task_out_size[0],
    abs(roi_vcs_range[3] - roi_vcs_range[1]) / task_out_size[1],
)  # (height, witdh)

bev_discobj_desc = dict(
    vcs_origin_coord=vcs_origin_coord,
    bev_stage2_input_resolution=spatial_resolution,
    bev_stage2_output_resolution=bev_discobj_stage2_output_resolution,
    visible_range=reformat_compile_vcs_range(roi_vcs_range),
    warp_offset_range=reformat_compile_vcs_range(vcs_range),
    vcs_plane_heights=vcs_plane_heights,
    fisheye_warp_offset_range=reformat_compile_vcs_range(fisheye_warp_range),
)


smallobj_dilate_cfg = {
    "kernel": (3, 3),
    "dilate_cls_id": [0, 1],
    "iterations": 1,
}

apply_lidar_filted_cfg = {
    "name2annolabel": {
        "comment_columns": "CementColumn",
        "parking_locks": "ParkingLock",
    },
    "cls2disthred": {
        "comment_columns": 2,
        "parking_locks": 1.4,
    },
    "cls2exposure_pts": {
        "comment_columns": 2,
        "parking_locks": 2,
    },
    "lidar_blind_vcs_range": (-3.0, -3.0, 6.0, 3.0),
}

if apply_lidar_filted_cfg:
    lidar_filter = dict(
        type="ANCBevDiscObjLidarFilter",
        lidar_filter_cfg=apply_lidar_filted_cfg,
    )

apply_class_match_cfg = {
    "source_class_id": 0,
    "target_class_id": 0,
    "yaw_match_threshold": 0.1,
    "ct_dist_thresh": 9,
}

if apply_class_match_cfg:
    apply_class_match_cfg["source_class_id"] = 0
    apply_class_match_cfg["target_class_id"] = 0
    class_group_matcher = dict(
        type="ANCBevDiscObjClassMatcher",
        ct_dist_type="width_ct_dist",
        class_match_cfg=apply_class_match_cfg,
    )


category_decouple_config = {
    "parking_lock": {
        "category2id_map": {
            "OpenParkingLock": 0,
            "CloseParkingLock": 1,
        },
        "name2label": {
            "parking_locks": "ParkingLock",
        },
        "smallobj_dilate_cfg": smallobj_dilate_cfg,
        "lidar_filter": lidar_filter,
    },
    "cement_column": {
        "category2id_map": {
            "CementColumn": 0,
        },
        "name2label": {
            "comment_columns": "CementColumn",
        },
        "smallobj_dilate_cfg": None,
        "class_group_matcher": class_group_matcher
        if apply_class_match_cfg
        else None,
        "lidar_filter": lidar_filter,
    },
}

pl_num_classes = (
    max(category_decouple_config["parking_lock"]["category2id_map"].values())
    + 1
)
cc_num_classes = (
    max(category_decouple_config["cement_column"]["category2id_map"].values())
    + 1
)

vcs_bbox_area_thresh = 0

parkinglock_id2label = {
    0: "ParkingLock_Open",
    1: "ParkingLock_Close",
}
cementcolumn_id2label = {
    0: "CementColumn",
}

use_distorted_offset = True
cal_homo_offset_on_gpu = True

psc_phase_factor = 2


pl_bev_discobj_target = dict(
    type="ANCBevDiscreteWithClsTargetGenerator",
    num_classes=pl_num_classes,
    bev_size=task_out_size,
    max_objs=100,
    vcs_range=roi_vcs_range,
    category2id_map=category_decouple_config["parking_lock"][
        "category2id_map"
    ],
    vcs_bbox_area_thresh=vcs_bbox_area_thresh,
    name2label=category_decouple_config["parking_lock"]["name2label"],
    visible_threshold=1e-6,
    vis_mask_expand_scale=1.5,
    name2group=name2group,
    smallobj_dilate_cfg=category_decouple_config["parking_lock"][
        "smallobj_dilate_cfg"
    ],
    lidar_filter=category_decouple_config["parking_lock"]["lidar_filter"],
    vis_mask_vcs_range_cfg=vismask_vcsrange_cfg,
    use_vis_mask=True,
    res_key="bev_parkinglock",
    return_ignore_obj=False,
)

cc_bev_discobj_target = dict(
    type="ANCBevDiscreteTargetGenerator",
    num_classes=cc_num_classes,
    bev_size=task_out_size,
    max_objs=100,
    vcs_range=roi_vcs_range,
    category2id_map=category_decouple_config["cement_column"][
        "category2id_map"
    ],
    vcs_bbox_area_thresh=vcs_bbox_area_thresh,
    name2label=category_decouple_config["cement_column"]["name2label"],
    visible_threshold=1e-6,
    vis_mask_expand_scale=1.5,
    name2group=name2group,
    smallobj_dilate_cfg=None,
    class_group_matcher=category_decouple_config["cement_column"][
        "class_group_matcher"
    ],
    lidar_filter=category_decouple_config["cement_column"]["lidar_filter"],
    vis_mask_vcs_range_cfg=vismask_vcsrange_cfg,
    use_vis_mask=True,
    res_key="bev_cementcolumn",
    return_ignore_obj=False,
)


bev_discobj_transforms = [
    collect_3dv,
    bev_common_transforms["ANCResize3DV"],
    bev_common_transforms["ANCCrop3DV"],
    None,
    pl_bev_discobj_target,
    cc_bev_discobj_target,
    vis_common_transforms
    if do_val_visualize
    else None,  # val_common_transforms if do_val_visualize else None,
    bev_common_transforms["ANCToTensor3DV"],
    bev_common_transforms["ANCPrepareDataBEV"],
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
    temporal_bev=False,
    length_of_clip=None,
    train_num_frames_per_iter=None,
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
    do_val_visualize=do_val_visualize,
)


val_template_dataset = copy.deepcopy(template_dataset)

if do_val_visualize:
    val_template_dataset["homo_gen"]["return_offset_in_meta_info"] = True
    homo_gen_high_sp = copy.deepcopy(val_template_dataset["homo_gen"])
    homo_transforms = copy.deepcopy(homo_gen_high_sp["homo_transforms"])
    for cam in homo_transforms:
        homo_transforms[cam] = {"Resize": (540, 960), "Crop": (0, 0, 512, 960)}
    homo_gen_high_sp.update(
        vcs_range=high_sp_vcs_range,
        spatial_resolution=(high_sp_resolution, high_sp_resolution),
        H_persp_view_scale=img_scale_before_ipm,
        homo_transforms=homo_transforms,
    )
    val_template_dataset["homo_gen_high_sp"] = homo_gen_high_sp

# ----------------------- DATALODER---------------------------
data_yaml = os.path.join(dataset_dir, f"{task_name}_dataset.yaml")


def get_train_dataloader():
    data_dict = get_data_dict(data_yaml)
    data_dict = join_path(
        bucket_root, data_dict, ["camera_module_type", "camera_view_names"]
    )

    url = os.path.join(dataset_dir, f"{task_name}_train_version.py")
    train_dataset_info = Config.fromfile(url)

    train_dataset_dict = train_dataset_info[train_data_version]

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
    data_dict = get_data_dict(data_yaml)
    data_dict = join_path(
        bucket_root, data_dict, ["camera_module_type", "camera_view_names"]
    )
    val_url = os.path.join(dataset_dir, f"{task_name}_val_version.yaml")
    val_data_version_dict = get_data_dict(val_url)
    val_dataset_dict = {
        val_version: val_data_version_dict[val_version]
        for val_version in _as_list(val_data_version)
    }

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
    num_views, vcs_plane_heights, task_out_size, pl_num_classes
)

pop_keys = [
    "gt_bev_discrete_obj",
    "annos_bev_discrete_obj",
]
for pop_key in pop_keys:
    train_inputs.pop(pop_key, None)
    val_inputs.pop(pop_key, None)

# add_keys = [
#     "gt_bev_parkinglock",
#     "gt_bev_cementcolumn",
#     "annos_bev_parkinglock",
#     "annos_bev_cementcolumn",
# ]
add_keys = dict(
    gt_bev_parkinglock={
        "bev_discobj_hm": torch.zeros(
            1,
            1,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_wh": torch.zeros(
            1,
            2,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_hm_cls": torch.zeros(
            1,
            pl_num_classes,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_rot": torch.zeros(
            1,
            2,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_ct_offset": torch.zeros(
            1,
            2,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_weight_hm": torch.zeros(
            1,
            1,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_ignore": torch.zeros(
            1,
            1,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
    },
    gt_bev_cementcolumn={
        "bev_discobj_instances": torch.zeros(
            1,
            cc_num_classes,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_hm": torch.zeros(
            1,
            cc_num_classes,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_wh": torch.zeros(
            1,
            2,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_hm_cls": torch.zeros(
            1,
            cc_num_classes,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_rot": torch.zeros(
            1,
            2,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_ct_offset": torch.zeros(
            1,
            2,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_weight_hm": torch.zeros(
            1,
            1,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_ignore": torch.zeros(
            1,
            1,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
    },
)


train_inputs.update(add_keys)
val_inputs.update(add_keys)

inputs = dict(
    train=train_inputs,
    val=val_inputs,
    deploy=deploy_inputs,
)

parkinglock_head_channels = OrderedDict(
    pred_bev_discobj_hm=1,
    pred_bev_discobj_hm_cls=pl_num_classes,
    pred_bev_discobj_wh=2,  # w, h
    pred_bev_discobj_rot=2,  # cos, sin
    pred_bev_discobj_ct_offset=2,  # u, v
)

cementcolumn_head_channels = OrderedDict(
    pred_bev_discobj_hm=cc_num_classes,
    pred_bev_discobj_wh=2,  # w, h
    pred_bev_discobj_rot=2,  # cos, sin
    pred_bev_discobj_ct_offset=2,  # u, v
)

# update loss
parkinglock_loss_weights = {
    "bev_discobj_hm": 6.0,
    "bev_discobj_hm_cls": 3.0,
    "bev_discobj_hm_cls_aux": 3.0,
    "bev_discobj_wh": 1.5,
    "bev_discobj_ct_offset": 1.5,
    "bev_discobj_rot": 3.0,
}

cementcolumn_loss_weights = {
    "bev_discobj_hm": 6.0,
    "bev_discobj_wh": 1.5,
    "bev_discobj_ct_offset": 2.25,
    "bev_discobj_rot": 3.0,
    "bev_discobj_group_rel": 3.0,
}


def get_bev_parkinglock_desc():
    per_tensor_desc = [
        {
            "task": "bev_discobj_parkinglock",
            "output_name": "bev_discobj_parkinglock_heatmap_output",
            "score_threshold": task_val_metric_setting["score_threshold"],
            "max_pool_kernel": 7,
            "properties": [{"channel_labels": ["heatmap"]}],
            **bev_discobj_desc,
        },
        {
            "task": "bev_discobj_parkinglock",
            "output_name": "bev_discobj_parkinglock_classification_output",
            "properties": [
                {
                    "channel_labels": [
                        parkinglock_id2label[id]
                        for id in range(len(parkinglock_id2label))
                    ]
                }
            ],
            **bev_discobj_desc,
        },
        {
            "task": "bev_discobj_parkinglock",
            "output_name": "bev_discobj_parkinglock_dimension_output",
            "properties": [{"channel_labels": ["width", "height"]}],
            **bev_discobj_desc,
        },
        {
            "task": "bev_discobj_parkinglock",
            "output_name": "bev_discobj_parkinglock_rot_output",
            "properties": [{"channel_labels": ["cos", "sin"]}],
            **bev_discobj_desc,
        },
        {
            "task": "bev_discobj_parkinglock",
            "output_name": "bev_discobj_parkinglock_ct_offset_output",
            "properties": [{"channel_labels": ["offset_x", "offset_y"]}],
            **bev_discobj_desc,
        },
    ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


def get_bev_cementcolumn_desc():
    per_tensor_desc = [
        {
            "task": "bev_discobj_cementcolumn",
            "output_name": "bev_discobj_cementcolumn_heatmap_output",
            "score_threshold": task_val_metric_setting["score_threshold"],
            "max_pool_kernel": 7,
            "properties": [
                {
                    "channel_labels": [
                        cementcolumn_id2label[id]
                        for id in range(len(cementcolumn_id2label))
                    ]
                }
            ],
            **bev_discobj_desc,
        },
        {
            "task": "bev_discobj_cementcolumn",
            "output_name": "bev_discobj_cementcolumn_dimension_output",
            "properties": [{"channel_labels": ["width", "height"]}],
            **bev_discobj_desc,
        },
        {
            "task": "bev_discobj_cementcolumn",
            "output_name": "bev_discobj_cementcolumn_rot_output",
            "properties": [{"channel_labels": ["cos", "sin"]}],
            **bev_discobj_desc,
        },
        {
            "task": "bev_discobj_cementcolumn",
            "output_name": "bev_discobj_cementcolumn_ct_offset_output",
            "properties": [{"channel_labels": ["offset_x", "offset_y"]}],
            **bev_discobj_desc,
        },
    ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


def get_model(mode):
    bev_parkinglock_head = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEV3DHead",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[4],
            in_channels=48,
            forward_frame_idx=0,
            head_channels=parkinglock_head_channels,
            use_varg=False,
            last_conv_kernel_size=3,
            bn_kwargs=bn_kwargs,
            merge_block=True,
            merge_output=False,
            quant_config={"pred_bev_discobj_hm_cls": "qint16"},
        ),
        head_parser=None,
        loss=None,
        postprocess=None,
        target=None,
        prefix="bev_stage2_parkinglock_small_head",
        node_name="bev_stage2_parkinglock_small_head",
    )

    bev_cementcolumn_head = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEV3DHead",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[4],
            in_channels=48,
            forward_frame_idx=0,
            head_channels=cementcolumn_head_channels,
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
        prefix="bev_stage2_cementcolumn_small_head",
        node_name="bev_stage2_cementcolumn_small_head",
    )
    bev_head = dict(
        type="MultiHeadOutputModule",
        heads=[bev_parkinglock_head, bev_cementcolumn_head],
    )

    if mode == "train":
        bev_parkinglock_head["loss"] = dict(
            type="ANCBEVDiscreteObjectWithClsLoss",
            loss_weights=parkinglock_loss_weights,
            use_focal_hm_loss=True,
            use_focal_cls_loss=False,
            use_softmax_focal_aux_loss=True,
            gt_name="gt_bev_parkinglock",
        )

        bev_cementcolumn_head["loss"] = dict(
            type="ANCBEVDiscreteObjectLoss",
            loss_weights=cementcolumn_loss_weights,
            use_focal_hm_loss=True,
            group_rel_loss=dict(
                type="ANCBEVDiscObjRelativeLoss",
                loss_weights=dict(
                    bev_discobj_rel_loc=0.2,
                    bev_discobj_rel_rot=0.5,
                ),
                gt_name="gt_bev_cementcolumn",
            ),
            class_weights=[1],
            gt_name="gt_bev_cementcolumn",
        )

    elif mode == "val":
        bev_parkinglock_head["postprocess"] = dict(
            type="ANCBEVDiscreteObjectDecoder",
            cls_dimension=None,
            topk=20,
            max_pool_kernel=7,
            vcs_range=roi_vcs_range,
        )
        bev_cementcolumn_head["postprocess"] = dict(
            type="ANCBEVDiscreteObjectDecoder",
            cls_dimension=None,
            topk=20,
            max_pool_kernel=7,
            vcs_range=roi_vcs_range,
        )
    elif mode == "deploy":
        bev_parkinglock_head["convert_to_dict"] = False
        bev_parkinglock_head["postprocess"] = dict(
            type="MultiInputSequential",
            modules=[
                # AddDesc should be the last module
                dict(
                    type="AddDesc",
                    per_tensor_desc=get_bev_parkinglock_desc(),
                ),
            ],
        )

        bev_cementcolumn_head["convert_to_dict"] = False
        bev_cementcolumn_head["postprocess"] = dict(
            type="MultiInputSequential",
            modules=[
                # AddDesc should be the last module
                dict(
                    type="AddDesc",
                    per_tensor_desc=get_bev_cementcolumn_desc(),
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

parkinglock_metrics, parkinglock_per_metric_patterns = get_metrics_patterns(
    "parkinglock",
    head_with_cls=True,
)

cementcolumn_metrics, cementcolumn_per_metric_patterns = get_metrics_patterns(
    "cementcolumn",
    use_class_match=True,
)

metrics = parkinglock_metrics + cementcolumn_metrics
per_metric_patterns = (
    parkinglock_per_metric_patterns + cementcolumn_per_metric_patterns
)

metric_updater = get_train_metric_updater(
    task_name=task_name,
    metrics=metrics,
    per_metric_patterns=per_metric_patterns,
    log_freq=log_freq,
)

# -------------------------- VALIDATION SETTING --------------------------

task_val_metric_setting = {
    "score_threshold": 0.3,
    "iou_threshold": 0.2,
    "yaw_amplitude": 180,
    "gt_max_depth": 25,
    "gt_match_mode": {"ParkingLock_Open": "ct", "ParkingLock_Close": "ct"},
    "ct_match_mode": "distance",
}

val_metric_updater_list = []
for dataset_key in _as_list(val_data_version):
    save_dir = os.path.join(save_prefix, "eval", dataset_key)
    if not os.path.exists(save_dir):
        # shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
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
        os.makedirs(save_vis_dir, exist_ok=True)

    save_pred = None
    # save_prcuv = None
    # save_pred = os.path.join(save_dir, f"{task_name}_pred")
    save_prcuv = os.path.join(save_dir, f"{task_name}_aps")

    parkinglock_val_metric_patterns = [
        dict(
            pred_ct=f"^.*parkinglock_small.*pred_bev_discobj_ct",  # noqa
            pred_wh=f"^.*parkinglock_small.*pred_bev_discobj_wh",  # noqa
            pred_score=f"^.*parkinglock_small.*pred_bev_discobj_score",  # noqa
            pred_yaw=f"^.*parkinglock_small.*pred_bev_discobj_rot",  # noqa
            pred_bev_discobj_cls_id=f"^.*parkinglock_small.*pred_bev_discobj_cls_id",  # noqa
        )
    ]

    parkinglock_val_metrics = get_val_metrics(
        task_name="bev_parkinglock",
        id2label=parkinglock_id2label,
        eval_category_ids=tuple(range(pl_num_classes)),
        iou_threshold=task_val_metric_setting["iou_threshold"],
        score_threshold=task_val_metric_setting["score_threshold"],
        gt_max_depth=task_val_metric_setting["gt_max_depth"],
        metrics=("dx", "dy", "dxy", "dw", "dh", "drot"),
        save_dir=save_dir,
        save_eval_result_path=save_metric_path,
        yaw_amplitude=task_val_metric_setting["yaw_amplitude"],
        match_mode=task_val_metric_setting.get("gt_match_mode", None),
        ct_match_mode=task_val_metric_setting.get("ct_match_mode", None),
        compute_foreground_prec=True,
        ct_matching_thresholds=0.8,
        vis_image_dir=save_vis_dir,
        annos_key="annos_bev_parkinglock",
        result_prefix="small",
    )

    cementcolumn_val_metric_patterns = [
        dict(
            pred_ct=f"^.*cementcolumn_small.*pred_bev_discobj_ct",  # noqa
            pred_wh=f"^.*cementcolumn_small.*pred_bev_discobj_wh",  # noqa
            pred_score=f"^.*cementcolumn_small.*pred_bev_discobj_score",  # noqa
            pred_yaw=f"^.*cementcolumn_small.*pred_bev_discobj_rot",  # noqa
            pred_bev_discobj_cls_id=f"^.*cementcolumn_small.*pred_bev_discobj_cls_id",  # noqa
        )
    ]
    cementcolumn_val_metrics = get_val_metrics(
        task_name="bev_cementcolumn",
        id2label=cementcolumn_id2label,
        eval_category_ids=tuple(range(cc_num_classes)),
        iou_threshold=task_val_metric_setting["iou_threshold"],
        score_threshold=task_val_metric_setting["score_threshold"],
        gt_max_depth=task_val_metric_setting["gt_max_depth"],
        metrics=("dx", "dy", "dxy", "dw", "dh", "drot"),
        save_dir=save_dir,
        save_eval_result_path=save_metric_path,
        yaw_amplitude=task_val_metric_setting["yaw_amplitude"],
        match_mode=task_val_metric_setting.get("gt_match_mode", None),
        compute_foreground_prec=True,
        vis_image_dir=save_vis_dir,
        annos_key="annos_bev_cementcolumn",
        result_prefix="small",
    )

    val_metrics = parkinglock_val_metrics + cementcolumn_val_metrics
    val_metric_patterns = (
        parkinglock_val_metric_patterns + cementcolumn_val_metric_patterns
    )

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

# --------------------------- visualize ---------------------------
visualize = []
vis_small = update_content(
    vis_common,
    range_mode="small",
    vcs_range=vcs_range,
    bev_size=ipm_output_size,
    res_key2anno_name={
        "bev_parkinglock_small": "annos_bev_parkinglock_small",
        "bev_cementcolumn_small": "annos_bev_cementcolumn_small",
    },
    project_pts_to_cameras=True,
    visual_interval=1,
    out_ori_img_size=(2048 // 2, 1280 // 2),
)
visualize.append(vis_small)
