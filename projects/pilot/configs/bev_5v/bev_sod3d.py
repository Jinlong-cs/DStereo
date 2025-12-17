import json
import os
import re
from collections import OrderedDict
from copy import deepcopy

import torch
from hatbc.utils import _as_list

from hat.utils.config import Config
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
)
from projects.pilot.configs.bev_5v.bev_discobj_base import (
    COLOR_MAP,
    ID2LABEL,
    bev_common_transforms,
    collect_3dv,
    get_bev_tb_update_func,
    get_metrics_patterns,
    get_train_metric_updater,
    get_val_metrics,
    name2group,
    reformat_compile_vcs_range,
    roi_vcs_range,
    val_common_transforms,
)
from projects.pilot.configs.bev_5v.common import (
    H_persp_view_scale,
    backbone,
    bev_backbone,
    bev_fusion,
    bev_fusion_upsampling,
    bev_neck,
    bevfusion_output_size,
    bn_kwargs,
    bucket_root,
    cal_homo_offset_on_gpu,
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
    model_thresh,
    multi_view_collect,
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

cfg_dir = os.path.dirname(__file__)

# -------------------------- TASK ---------------------------
task_name = "bev_sod3d"
category2id_map = {
    "cone": 0,
}
id2label = {
    0: "Cone",
}
name2label = {"sod3d": "SOD3D"}
num_classes = max(category2id_map.values()) + 1
task_loss_weight = 1.0
smallobj_dilate_cfg = dict(
    hm_size_floor=(3, 3),  # (w, h)
    hm_alpha=2,
    kernel=(1, 1),
    dilate_cls_id=[0],
    iterations=1,
)
task_in_stride = 4
task_out_size = (
    int(bevfusion_output_size[0] / task_in_stride),
    int(bevfusion_output_size[1] / task_in_stride),
)

enable_tensorboard = True

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


# ------------------ DATASET SETTING -------------------
# data version
train_data_version = "train_auto_gt_v1_0_3_small_filtered"
val_data_version = [
    "val_auto_gt_v1_0_2_small",
    "val_auto_gt_v1_0_2_small_parking",
    "val_auto_gt_v1_0_2_small_parking_ground",
    "val_auto_gt_v1_0_2_small_parking_underground",
    "val_auto_gt_v1_0_2_small_parking_filtered",
    "val_auto_gt_v1_0_2_small_parking_ground_filtered",
    "val_auto_gt_v1_0_2_small_parking_underground_filtered",
]
if pipeline_test:
    train_data_version = "pipeline_test"
    val_data_version = "pipeline_test"

# train data
dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_discrete_obj"
)

bev_common_transforms = deepcopy(bev_common_transforms)
# if "bev_occlusion_mask" in collect_3dv["load_data_types"]:
#     collect_3dv["load_data_types"].remove("bev_occlusion_mask")
bev_sod3d_target = dict(
    type="ANCBevDiscreteTargetGenerator",
    num_classes=num_classes,
    bev_size=task_out_size,
    max_objs=300,
    vcs_range=roi_vcs_range,
    category2id_map=category2id_map,
    vcs_bbox_area_thresh=0,
    name2label=name2label,
    visible_threshold=0.2,
    name2group=name2group,
    smallobj_dilate_cfg=smallobj_dilate_cfg,
    return_ignore_obj=False,
    use_vis_mask=False,
    res_key="bev_discrete_obj",
)
bev_sod3d_transforms = [
    collect_3dv,
    bev_common_transforms["ANCResize3DV"],
    bev_common_transforms["ANCCrop3DV"],
    bev_sod3d_target,
    val_common_transforms if do_val_visualize else None,
    bev_common_transforms["ANCToTensor3DV"],
    bev_common_transforms["ANCPrepareDataBEV"],
]
bev_sod3d_transforms = remove_none(bev_sod3d_transforms)

use_distorted_offset = True
template_dataset = get_template_dataset(
    img_load_size=list(img_resize_wh_size.values()),
    camera_view_names=camera_view_names,
    per_view_shape=img_ori_size,
    transforms=bev_sod3d_transforms,
    homo_transforms=get_homo_transforms(
        transforms_list=bev_sod3d_transforms,
        camera_view_names=camera_view_names,
    ),
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


# for val, not dilate small obj gt
val_template_dataset = deepcopy(template_dataset)


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
        dataset_dict=train_dataset_dict,
        location_names=None,
        template_dataset=template_dataset,
        data_dict=data_dict,
        update_bev_dataset=update_bev_dataset,
        homo_noise=None,
        global_sample_interval=train_global_sample_interval,
    )
    data_loader = get_dataloader(
        datasets=train_datasets,
        num_workers=train_num_workers,
        batch_size_per_gpu=train_batch_size_per_gpu,
        shuffle=True,
        persistent_workers=True,
    )
    if use_split_dataloader:
        data_loader = convert_to_split_dataloader(data_loader)
    return data_loader


def get_val_dataloader():
    data_dict = get_data_dict(data_yaml)
    data_dict = join_path(
        bucket_root, data_dict, ["camera_module_type", "camera_view_names"]
    )
    # val data
    val_dataset_info = get_data_dict(
        url=os.path.join(dataset_dir, f"{task_name}_val_version.yaml")
    )
    val_dataset_dict = {
        val_version: val_dataset_info[val_version]
        for val_version in _as_list(val_data_version)
    }
    val_dataset_list = get_dataset_list(
        dataset_dict=list(val_dataset_dict.values()),
        location_names=None,
        template_dataset=val_template_dataset,
        data_dict=data_dict,
        update_bev_dataset=update_bev_dataset,
        homo_noise=None,
        repeat_dataset_times=1,
    )
    # val dataloader
    val_data_loader_list = get_dataloader_list(
        datasets=val_dataset_list,
        num_workers=val_num_workers,
        batch_size_per_gpu=val_batch_size_per_gpu,
        shuffle=False,
        persistent_workers=False,
    )
    return val_data_loader_list


# -------------------------- MODEL --------------------------
train_inputs = dict(
    gt_bev_discrete_obj={
        "bev_discobj_hm": torch.zeros(
            1,
            1,
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_wh": torch.zeros(
            1,
            3,  # sod3d with 3 dim
            task_out_size[0],
            task_out_size[1],
            dtype=torch.float32,
        ),
        "bev_discobj_hm_cls": torch.zeros(
            1,
            num_classes,
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
inputs = dict(
    train=train_inputs,
    val={},
    deploy={},
)


def get_desc():
    task_common_desc = deepcopy(bev_discobj_desc)
    per_tensor_desc = [
        {
            "task": "bev_discobj_sod3d",
            "output_name": "bev_discobj_sod3d_heatmap_output",
            "score_threshold": model_thresh.get(task_name, {}).get(
                "score_threshold", 0.2
            ),
            "max_pool_kernel": 5,
            "properties": [
                {
                    "channel_labels": [
                        id2label[id] for id in range(len(id2label))
                    ]
                }
            ],
            **task_common_desc,
        },
        {
            "task": "bev_discobj_sod3d",
            "output_name": "bev_discobj_sod3d_dimension_output",
            "properties": [{"channel_labels": ["width", "height", "dim_z"]}],
            **task_common_desc,
        },
        {
            "task": "bev_discobj_sod3d",
            "output_name": "bev_discobj_sod3d_rot_output",
            "properties": [{"channel_labels": ["cos", "sin"]}],
            **task_common_desc,
        },
        {
            "task": "bev_discobj_sod3d",
            "output_name": "bev_discobj_sod3d_ct_offset_output",
            "properties": [{"channel_labels": ["offset_x", "offset_y"]}],
            **task_common_desc,
        },
    ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]

    return per_tensor_desc


def get_model(mode):

    bev_head = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEV3DHead",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[task_in_stride],
            in_channels=48,
            forward_frame_idx=0,
            head_channels=OrderedDict(
                pred_bev_discobj_hm=1,
                pred_bev_discobj_wh=3,
                pred_bev_discobj_rot=2,
                pred_bev_discobj_ct_offset=2,
            ),
            use_varg=False,
            last_conv_kernel_size=3,
            bn_kwargs=bn_kwargs,
            merge_block=True,
            merge_output=False,
        ),
        loss=None,
        target=None,
        node_name="bev_stage2_sod3d_small_head",
    )

    if mode == "train":
        bev_head["loss"] = dict(
            type="ANCBEVDiscreteObjectLoss",
            loss_weights=dict(
                bev_discobj_hm=1.0 * task_loss_weight,
                bev_discobj_wh=1.0 * task_loss_weight,
                bev_discobj_ct_offset=1.0 * task_loss_weight,
                bev_discobj_rot=0.5 * task_loss_weight,
            ),
            use_focal_hm_loss=True,
            gt_name="gt_bev_discrete_obj",
            use_target_ignore=True,
        )
    elif mode == "val":
        bev_head["postprocess"] = dict(
            type="ANCBEVDiscreteObjectDecoder",
            cls_dimension=None,
            topk=40,
            max_pool_kernel=5,
            vcs_range=roi_vcs_range,
        )
    elif mode == "deploy":
        bev_head["loss"] = None
        bev_head["target"] = None
        bev_head["postprocess"] = dict(
            type="MultiInputSequential",
            modules=[
                dict(
                    type="AddDesc",
                    per_tensor_desc=get_desc(),
                ),
            ],
        )
        bev_head["convert_to_dict"] = False
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
metrics, per_metric_patterns = get_metrics_patterns(
    task_name=task_name,
    head_with_cls=False,
    use_class_match=False,
)
metric_updater = get_train_metric_updater(
    task_name=task_name,
    metrics=metrics,
    per_metric_patterns=per_metric_patterns,
    log_freq=log_freq,
)


# ------------------------- VALIDATION SETTING-----------------------
val_metric_updater_list = []
for dataset_key in _as_list(val_data_version):
    save_dir = os.path.join(save_prefix, "eval", task_name, dataset_key)
    os.makedirs(save_dir, exist_ok=True)
    save_metric_path = os.path.join(
        save_dir, f"{task_name}_{training_step}_eval_metric.json"
    )

    save_vis_dir = os.path.join(save_dir, "vis")
    if os.path.exists(save_vis_dir):
        os.makedirs(save_vis_dir, exist_ok=True)

    val_metrics = get_val_metrics(
        task_name=task_name,
        id2label=id2label,
        eval_category_ids=tuple(range(num_classes)),
        iou_threshold=0.2,
        score_threshold=0.2,
        gt_max_depth=25,
        metrics=("dx", "dy", "dxy", "dw", "dh", "drot"),
        save_dir=save_dir,
        save_eval_result_path=save_metric_path,
        yaw_amplitude=180,
        match_mode="ct",
        ct_match_mode="percentage",
        compute_foreground_prec=True,
        ct_matching_thresholds={2.0: 0.1, 5.0: 0.2, 10.0: 0.3},
        vis_image_dir=save_vis_dir,
        annos_key="annos_bev_discrete_obj",
    )

    pred_patterns = [
        dict(
            pred_ct="^.*pred_bev_discobj_ct",
            pred_wh="^.*pred_bev_discobj_wh",
            pred_score="^.*pred_bev_discobj_score",
            pred_yaw="^.*pred_bev_discobj_rot",
            pred_bev_discobj_cls_id="^.*pred_bev_discobj_cls_id",
        )
    ]

    val_metric_updater = dict(
        type="MetricUpdater",
        metrics=val_metrics,
        metric_update_func=get_update_metric_func(
            mode="val",
            task_name=task_name,
            pred_patterns=pred_patterns,
        ),
        step_log_freq=-1,
        epoch_log_freq=1,
        log_prefix=f"Validation_{task_name}_{dataset_key}",
    )
    val_metric_updater_list.append(val_metric_updater)


# -------------------------- TENSORBOAED --------------------------
if enable_tensorboard:
    tb_update_func = get_bev_tb_update_func(
        task_name=task_name,
        pred_hm_key_regex=re.compile(
            f"^.*{task_name}.*gt_bev_discrete_obj_bev_discobj_weight_hm"
        ),
        gt_hm_key_regex=re.compile(f"^.*{task_name}.*pred_bev_discobj_hm"),
    )

# --------------------------- PACK VIS ----------------------------
visualize_callbacks = [
    dict(
        type="ANCBevObjVisualizeCB",
        output_dir=os.path.join(save_prefix, "visualize", "bev_sod3d"),
        task="bev_sod3d",
        prefix="OutputModule_predict",
        score_threshold=0.2,
        vis_img_scale=1,
        bev_size=ipm_output_size,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        camera_layouts=None,
        res_key2anno_name={
            "bev_sod3d": "annos_bev_discrete_obj",
        },
        color_map=COLOR_MAP,
        id2label=ID2LABEL,
        visual_interval=1,
        vis_ego=True,
        project_pts_to_cameras=False,
    )
]
