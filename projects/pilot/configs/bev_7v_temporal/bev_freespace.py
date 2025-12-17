import json
import os
import re
from copy import deepcopy

import numpy as np
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
    VIS_COLORS,
    common_desc,
    freespace_auto_remap_dict,
    freespace_mannual_remap_dict,
    get_rle_padding,
    lidar_seg_remap_dict,
    raw_gt_res_vcsrange_cfg,
    roi_resize,
    roi_vcs_origin_coord,
    roi_vcs_range,
    stage2_stride2channels,
    task_in_stride,
    task_out_resolution,
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
    deploy_head,
    deploy_narrow_head,
    deploy_side_head,
    deploy_stage2_inputs_key,
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
    train_num_frames_per_iter,
    train_num_workers,
    training_step,
    use_distorted_offset,
    use_split_dataloader,
    val_batch_size_per_gpu,
    val_num_frames_per_iter,
    val_num_workers,
    vcs_plane_heights,
    vcs_range,
)

cfg_dir = os.path.dirname(__file__)

# -------------------------- TASK ---------------------------
task_name = "bev_freespace"
pixelwise_weight_diameter = 3.4
use_rle = True
output_confidence = False
enable_tensorboard = True


# ------------------ DATASET SETTING -------------------
# data version
train_data_version = "v6_1_1_temporal"
val_data_version = [
    "v3.7.0_all_temporal_driving",
    "v2.1.4_small_obstacle_total_sequence",
    "v3.7.0_all_temporal_parking",
]
if pipeline_test:
    train_data_version = "pipeline_test"
    val_data_version = "pipeline_test"

# train data
dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_elevation"
)

bev_common_transforms = deepcopy(common_transforms)
collect_3dv = bev_common_transforms["ANCCollect3DV"]
collect_3dv[f"gt_{task_name}_idxs"] = [0]
load_data_types = [
    f"gt_{task_name}",
    "timestamp",
    "img_name",
    "pose",
    f"{task_name}_mask",
]
load_data_types = remove_none(load_data_types)
collect_3dv["load_data_types"] = load_data_types
# bev_freespace_generator
bev_freespace_generator = dict(
    type="ANCBEVFreespaceGenerator",
    vcs_range=roi_vcs_range,
    raw_gt_res_vcsrange_cfg=raw_gt_res_vcsrange_cfg,
    target_size=task_out_size,
    lidar_seg_remap_dict=lidar_seg_remap_dict,
    freespace_mannual_remap_dict=freespace_mannual_remap_dict,
    freespace_auto_remap_dict=freespace_auto_remap_dict,
    ele_freespace_mannual_remap_dict=None,
    ele_freespace_auto_remap_dict=None,
    occupancy_output_freespace=True,
    ignore_index=255,
    global_dilate_cfg=None,
    smallobj_dilate_cfg=dict(
        use_dilate=True,
        kernel=5,
        autolabel_area_range=(4, 100),  # measued in pixel num
        manuallabel_area_range=(1, 100),
    ),
    median_blur_seg_height=True,
    use_extra_mask=True,
    output_extra_obstacle=True,
    output_lidardet=True,
    output_bev_weight_map=True,
)
bev_transforms = [
    collect_3dv,
    bev_common_transforms["ANCResize3DV"],
    bev_common_transforms["ANCCrop3DV"],
    bev_common_transforms["ANCTemporalHomo"],
    bev_freespace_generator,
    bev_common_transforms["ANCToTensor3DV"],
    bev_common_transforms["ANCPrepareTempoDataBEV"],
    dict(type="ANCSetTemporalClearFlag", clr_mode="clip"),
    dict(type="AddKeys", kv={"return_latest_flag": True}),
    dict(type="AddKeys", kv={"task_name": task_name}),
]
bev_transforms = remove_none(bev_transforms)

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
        f"{task_name}_data_path",
        f"{task_name}_mask_path",
        "homo_path",
        "calib_path",
        "camera_module_type",
        "homo_noise",
    ],
    update_trans_info={
        "ANCCollect3DV": "fill_fake_temporal_data",
        "ANCBEVFreespaceGenerator": "ego_region_range",
    },
)


# for val, not dilate small obj gt
val_template_dataset = deepcopy(template_dataset)
generator_index = val_template_dataset["transforms"].index(
    bev_freespace_generator
)
val_template_dataset["transforms"][generator_index]["global_dilate_cfg"] = None
val_template_dataset["transforms"][generator_index][
    "smallobj_dilate_cfg"
] = None
val_template_dataset["transforms"][generator_index][
    "median_blur_seg_height"
] = False
val_template_dataset["num_frames_per_iter"] = val_num_frames_per_iter
val_template_dataset["transforms"][0]["img_idxs"] = list(
    range(val_num_frames_per_iter)
)
val_template_dataset["transforms"][0]["pose_idxs"] = list(
    range(val_num_frames_per_iter + 1)
)

# ----------------------- DATALODER---------------------------
data_yaml = os.path.join(dataset_dir, f"{task_name}_sequence_dataset.yaml")


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
    )

    data_loader = get_dataloader(
        datasets=train_datasets,
        num_workers=train_num_workers,
        batch_size_per_gpu=train_batch_size_per_gpu,
        shuffle=True,
        persistent_workers=True,
    )
    if use_split_dataloader:
        data_loader = convert_to_split_dataloader(
            dataloader=data_loader, temporal_bev=True
        )
    return data_loader


def get_val_dataloader():
    data_dict = get_data_dict(data_yaml)
    data_dict = join_path(
        bucket_root, data_dict, ["camera_module_type", "camera_view_names"]
    )

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
    gt_bev_freespace=dict(
        gt_bev_freespace=torch.zeros(
            (1, task_out_size[0], task_out_size[1]), dtype=torch.long
        ),
        gt_bev_extra_obstacle=torch.zeros(
            (1, task_out_size[0], task_out_size[1]), dtype=torch.long
        ),
        gt_lidardet_veh_mask=torch.zeros(
            (1, task_out_size[0], task_out_size[1]), dtype=torch.long
        ),
        bev_weight_map=torch.zeros(
            (1, task_out_size[0], task_out_size[1]), dtype=torch.float32
        ),
    )
)

inputs = dict(
    train=train_inputs,
    val={},
    deploy={},
)


def get_desc():
    task_common_desc = deepcopy(common_desc)
    per_tensor_desc = []
    if use_rle:
        task_common_desc["unpad_shape"] = task_out_size

    bev_freespace_desc = dict(
        task=task_name,
        output_name="freespace_cls",
        num_classes=2,
        labels=[
            dict(
                class_name="freespace_yes",
                label_id=0,
            ),
            dict(
                class_name="freespace_no",
                label_id=1,
            ),
        ],
        **task_common_desc,
    )
    per_tensor_desc.append(bev_freespace_desc)
    if output_confidence:
        bev_freespace_conf_desc = dict(
            task=task_name,
            output_name="freespace_prob",
            **task_common_desc,
        )
        per_tensor_desc.append(bev_freespace_conf_desc)

    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]

    return per_tensor_desc


argmax_postprocess = dict(
    type="ArgmaxPostprocess",
    data_name=f"pred_{task_name}_frame0",
    dim=1,
    keepdim=True if training_step == "int_infer" else False,
)


def get_model(mode):

    bev_head = dict(
        type="OutputModule",
        head=dict(
            type="PixelHead",
            in_strides=[2],
            out_strides=[task_in_stride],
            stride2channels=stage2_stride2channels,
            bn_kwargs=bn_kwargs,
            forward_frame_idxs=[0],
            dequant_out=True,
            out_nums=[2],
            output_name=[f"pred_{task_name}"],
            quanti_last_conv=True,
        ),
        head_parser=None,
        loss=None,
        postprocess=None,
        target=None,
        node_name="bev_stage2_freespace_head",
    )

    if mode == "train":
        bev_head["loss"] = dict(
            type="ANCBEVFreespaceLoss",
            cls_loss_name=f"loss_{task_name}",
            cls_loss_cfg=dict(
                type="CrossEntropyLoss",
                reduction="mean",
                loss_weight=6.0,
                ignore_index=255,
                use_sigmoid=True,
            ),
            use_pixelwise_weight=True,
            pixelwise_weight_size=(
                int(pixelwise_weight_diameter / max(task_out_resolution) / 2)
                * 2
                + 1
            ),
            boundary_loss_cfg=dict(
                use_boundary_loss=True,
                boundary_winsize_half=3,
                boundary_loss_weight=1.0,
            ),
        )
        bev_head["postprocess"] = [argmax_postprocess]
    elif mode == "val":
        bev_head["postprocess"] = [argmax_postprocess]
    elif mode == "deploy":
        postprocess_modules = []
        if output_confidence:
            postprocess_modules.append(
                dict(
                    type="MaxPostProcess",
                    data_names=[f"pred_{task_name}_frame0"],
                    out_names=[
                        [
                            f"pred_{task_name}_conf_frame0",  # score
                        ],
                    ],
                    dim=1,
                    keepdim=True if training_step == "int_infer" else False,
                    return_indices=False,
                )
            )
            postprocess_modules.append(
                dict(
                    type="DequantModule",
                    data_names=[
                        f"pred_{task_name}_conf_frame0",
                    ],
                )
            )
        if use_rle:
            use_rle_pad, padding, _ = get_rle_padding(size=task_out_size)
            if use_rle_pad:
                postprocess_modules.append(
                    dict(
                        type="ANCPadPostprocess",
                        data_name=f"pred_{task_name}_frame0",
                        padding=padding,
                    )
                )
            postprocess_modules.append(argmax_postprocess)
            postprocess_modules.append(
                dict(
                    type="RLEPostprocess",
                    data_name=f"pred_{task_name}_frame0",
                    dtype=torch.int8,
                )
            )
        else:
            postprocess_modules.append(argmax_postprocess)
        postprocess_modules.append(
            dict(
                type="AddDesc",
                per_tensor_desc=get_desc(),
            )
        )

        bev_head["loss"] = None
        bev_head["target"] = None
        bev_head["postprocess"] = dict(
            type="MultiInputSequential",
            modules=postprocess_modules,
        )
        bev_head["head"]["dequant_out"] = False
        bev_head["convert_to_dict"] = False
    else:
        raise NotImplementedError

    # module config
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
            roi_resizes=roi_resize,
            head=bev_head,
        ),
        bevfusion_pick_keys=bevfusion_pick_keys,
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
        dict(
            type="LossShow",
            name="loss_freespace_boundary",
        ),
    ],
    metric_update_func=update_metric_using_regex(
        [
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}.*loss_{task_name}$",
            ),
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}.*loss_freespace_boundary$",
            ),
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


# ------------------------- VALIDATION SETTING-----------------------
eval_setting = {
    "eval_vcs_range": roi_vcs_range,
    "smallobj_area_thresh_ub": 12.0,  # measured in m^2
    "smallobj_area_thresh_lb": 0.04,  # measured in m^2
    "smallobj_background_dilate_k": 9,  # in pixel
    "depth_intervals": (-30.4, 0, 20, 50, 80, 105),
    "dist_intervals": (0, 3, 10, 20),  # boundary point
    "pr_cfg": {
        "area_intervals": (0.5, 1.0, 4.0, 12.0),  # measured in m^2
        "dist_interval_dxy_threshs": {
            (0, 5): 0.5,
            (5, 10): 1.0,
            (10, 15): 1.0,
            (15, 20): 1.5,
            (20, 100): 2.0,
            (100, 250): 3.0,
            (250, 1000): 8.0,
        },  # object center dxy thresh for pr matching
    },
    "boundary_cfg": {
        "dist_interval_dxy_threshs": {
            (0, 3): 10.0,
            (3, 10): 10.0,
            (10, 20): 10.0,
        },  # dxy thresh for boundary point matching along ray
    },
    "metric_modes": ["iou", "boundary_error", "pr"],
}

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
                type="ANCBEVFreespaceMetric",
                name="FreespaceMetric",
                seg_class=["0", "1"],
                bev_size=task_out_size,
                vcs_range=roi_vcs_range,
                task_name=task_name,
                gt_name=f"gt_{task_name}",
                pred_name=pred_name,
                ignore_index=255,
                save_metric_path=save_metric_path,
                eval_cfg=eval_setting,
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
def tb_update_func(writer, model_outs, global_step_id, **kwargs):
    pred_regex = re.compile(f"^.*pred_{task_name}_frame0_0")
    target_regex = re.compile(f"^.*target_gt_{task_name}_gt_{task_name}")
    if task_name in list(model_outs.keys())[0]:
        for k, v in model_outs.items():
            if pred_regex.match(k) or target_regex.match(k):
                v = v.detach().cpu().numpy()[0, :, :]
                img_vis = np.zeros([*(v.shape), 3], np.uint8)
                img_vis[v == 0] = [127, 255, 127]  # light green, freespace_yes
                img_vis[v == 1] = [127, 127, 255]  # light blue, freespace_no
                img_vis[v == 255] = [127, 127, 127]  # gray, ignore
                writer.add_images(
                    k, img_vis, global_step_id, dataformats="HWC"
                )


# --------------------------- PACK VIS ----------------------------
visualize_callbacks = [
    dict(
        type="ANCBevOccupancyVisualize",
        output_dir=os.path.join(save_prefix, "visualize", "bev_freespace"),
        prefix="OutputModule_predict",
        color_map={"bev_freespace": VIS_COLORS},
        vcs_origin_coord=roi_vcs_origin_coord,
        img_stitcher=dict(
            type="ANCBEVImgStitcher",
            per_extra_img_size=(960, 512),
            camera_view_names=camera_view_names,
            camera_layouts=None,
        ),
    )
]
