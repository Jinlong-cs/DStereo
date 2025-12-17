import json
import os
from copy import deepcopy

import torch
from hatbc.utils import _as_list

from hat.callbacks.metric_updater import update_metric_using_regex
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
)
from projects.pilot.configs.bev_5v.bev_elevation_base import (
    VIS_COLORS,
    common_desc,
    get_rle_padding,
    raw_gt_res_vcsrange_cfg,
    roi_vcs_origin_coord,
    roi_vcs_range,
    task_in_stride,
    task_out_size,
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
    cal_homo_offset_on_gpu,
    camera_view_names,
    common_transforms,
    deploy_head,
    deploy_homo_offset_key,
    deploy_round_head,
    do_val_visualize,
    fisheye_camera_view_names,
    front_camera_view_names,
    head,
    img_ori_size,
    img_resize_wh_size,
    log_freq,
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
    train_num_workers,
    training_step,
    use_split_dataloader,
    val_batch_size_per_gpu,
    val_num_workers,
    vcs_plane_heights,
    vcs_range,
)

cfg_dir = os.path.dirname(__file__)

# -------------------------- TASK ---------------------------
task_name = "bev_vismask"
use_resize_head_parser = True
use_rle = True
output_confidence = False
enable_tensorboard = False


# ------------------ DATASET SETTING -------------------
# train data
train_data_version = "v5_0_2_fisheye"
val_data_version = [
    "v3.7_11v_driving_temporal",
    "v3.7_11v_parking_temporal",
    "v1.0.0_ek_parking_underground",
    # "v1.0.0_ek_parking_ground",
]
if pipeline_test:
    val_data_version = "pipeline_test"
    train_data_version = "pipeline_test"
dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_elevation"
)

# trainforms
bev_common_transforms = deepcopy(common_transforms)
collect_3dv = bev_common_transforms["ANCCollect3DV"]
collect_3dv["load_data_types"] = [
    "gt_bev_elevation_vismask",
    "timestamp",
    "img_name",
]
# bev_vismask_generator
bev_generator = dict(
    type="ANCBEVVisMaskGenerator",
    target_size=task_out_size,
    ignore_index=255,
    vcs_range=roi_vcs_range,
    raw_gt_res_vcsrange_cfg=raw_gt_res_vcsrange_cfg,
    global_dilate_cfg={
        "use_dilate": True,
        "kernel": 5,
        "iterations": 1,
    },
    gt_name="gt_bev_elevation_vismask",
)
bev_transforms = [
    collect_3dv,
    bev_common_transforms["ANCResize3DV"],
    bev_common_transforms["ANCCrop3DV"],
    bev_generator,
    dict(type="ANCToTensor3DV", with_color_imgs=True),
    bev_common_transforms["ANCPrepareDataBEV"],
]


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
    temporal_bev=False,
    length_of_clip=None,
    train_num_frames_per_iter=None,
)

update_bev_dataset = get_update_bev_dataset_func(
    added_info=[
        "bev_elevation_vismask_data_path",
        "homo_path",
        "calib_path",
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
def get_train_dataloader():
    url = os.path.join(dataset_dir, f"{task_name}_train_version.py")
    train_dataset_info = Config.fromfile(url)
    train_dataset_dict = train_dataset_info[train_data_version]

    data_dict = get_data_dict(
        url=os.path.join(dataset_dir, "bev_elevation_dataset.yaml"),
    )
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
        data_loader = convert_to_split_dataloader(data_loader)
    return data_loader


def get_val_dataloader():
    data_dict = get_data_dict(
        url=os.path.join(dataset_dir, "bev_elevation_dataset.yaml"),
    )
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
train_inputs = {
    "gt_bev_elevation_vismask": {
        "vismask": torch.zeros(
            (1, task_out_size[0], task_out_size[1]), dtype=torch.long
        ),
        "agent": torch.zeros(
            (1, task_out_size[0], task_out_size[1]), dtype=torch.long
        ),
    }
}
val_inputs = deepcopy(train_inputs) if training_step != "pack_infer" else {}
deploy_inputs = {}

inputs = dict(
    train=train_inputs,
    val=val_inputs,
    deploy=deploy_inputs,
)


def get_desc():
    task_common_desc = deepcopy(common_desc)
    per_tensor_desc = []
    if use_rle:
        task_common_desc["unpad_shape"] = task_out_size

    bev_desc_vismask = dict(
        task=task_name,
        num_classes=2,
        **task_common_desc,
    )
    per_tensor_desc.append(bev_desc_vismask)
    if output_confidence:
        bev_desc_vismask_conf = dict(
            task=task_name,
            output_name="vismask_prob",
            **task_common_desc,
        )
        per_tensor_desc.append(bev_desc_vismask_conf)

    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]

    return per_tensor_desc


def get_model(mode):

    bev_head = dict(
        type="OutputModule",
        head=dict(
            type="PixelHead",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[task_in_stride],
            stride2channels={
                2: 48,
                4: 48,
                8: 96,
                16: 192,
                32: 192,
                64: 192,
                128: 192,
                256: 192,
            },
            bn_kwargs=bn_kwargs,
            forward_frame_idxs=[0],
            dequant_out=False
            if use_resize_head_parser or training_step == "int_infer"
            else True,
            out_nums=[2],
            output_name=[f"pred_{task_name}"],
            quanti_last_conv=True,
        ),
        head_parser=dict(
            type="MultiInputSequential",
            modules=[
                dict(
                    type="ResizeParser",
                    data_name=f"pred_{task_name}_frame0",
                    use_plugin_interpolate=True,
                    dequant_out=(
                        False if training_step == "int_infer" else True
                    ),
                    resize_kwargs=dict(size=task_out_size, mode="bilinear"),
                )
            ],
        )
        if use_resize_head_parser
        else None,
        loss=None,
        postprocess=None,
        target=None,
        node_name="bev_stage2_vismask_small_head",
    )

    argmax_postprocess = dict(
        type="ArgmaxPostprocess",
        data_name=f"pred_{task_name}_frame0",
        dim=1,
        keepdim=True if training_step == "int_infer" else False,
    )

    if mode == "train":
        bev_head["loss"] = dict(
            type="ANCBevVismaskLoss",
            vismask_loss_cfg=dict(
                type="CrossEntropyLoss",
                reduction="mean",
                loss_weight=0.24,
                class_weight=[1, 1],
                ignore_index=255,
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
            )
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
                type="ANCBevVismaskMetric",
                bev_size=task_out_size,
                vcs_range=roi_vcs_range,
                task_name=task_name,
                gt_name="gt_bev_elevation_vismask",
                pred_name=pred_name,
                eval_vcs_range=None,
                save_dir=save_dir,
                save_metric_path=save_metric_path,
                vis=False,
                vis_interval=100,
                result_prefix="small",
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

# --------------------------- PACK VIS ----------------------------
visualize_callbacks = [
    dict(
        type="ANCBevOccupancyVisualize",
        output_dir=os.path.join(save_prefix, "visualize", "bev_vismask"),
        prefix="OutputModule_predict",
        color_map={"bev_vismask": VIS_COLORS},
        vcs_origin_coord=roi_vcs_origin_coord,
        img_stitcher=dict(
            type="ANCBEVImgStitcher",
            per_extra_img_size=(960, 512),
            camera_view_names=camera_view_names,
            camera_layouts=None,
        ),
    )
]
