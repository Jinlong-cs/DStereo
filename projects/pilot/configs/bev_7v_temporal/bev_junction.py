import copy
import json
import os
import re
from collections import OrderedDict

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
    reformat_compile_vcs_range,
    remove_none,
)
from projects.pilot.configs.bev_7v_temporal.bev_discobj_base import (
    COLOR_MAP,
    ID2LABEL,
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
    deploy_head,
    deploy_narrow_head,
    deploy_side_head,
    deploy_stage2_inputs_key,
    do_val_visualize,
    front_camera_view_names,
    head,
    img_ori_size,
    img_resize_wh_size,
    ipm_output_size,
    length_of_clip,
    log_freq,
    model_thresh,
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
task_name = "bev_junction"
base_task_name = "bev_disc"
enable_tensorboard = True
use_random_rotation = False
use_rec = True
merge_crosswalk_to_junction = True
# -------------------------- data --------------------------
train_data_version = "temporal_v1_3_0_wide_merge"
val_data_version = "temporal_v1_1_0_wide"

if pipeline_test:
    train_data_version = "pipeline_test"
    val_data_version = "pipeline_test"

dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_discrete_obj"
)

# ------------------ BEV junction transformation setting -------------------
roi_vcs_range = (-32.0, -51.2, 102, 51.2)
task_in_stride = 4
task_out_size = (134, 128)

bev_discobj_junction_stage2_output_resolution = (
    abs(roi_vcs_range[2] - roi_vcs_range[0]) / task_out_size[0],
    abs(roi_vcs_range[3] - roi_vcs_range[1]) / task_out_size[1],
)  # (height, witdh)

category2id_map = {
    "Junction": 0,
    "Crosswalk": 1,
}
id2label = {
    0: "Junction",
    1: "Crosswalk",
}
name2label = {
    "intersections": "Junction",
    "crosswalks": "Crosswalk",
}

num_classes = max(category2id_map.values()) + 1
vcs_bbox_area_thresh = 0

use_distorted_offset = True
cal_homo_offset_on_gpu = True

use_psc_rot = True
N_steps_PSC_rot = 3
psc_phase_factor = 2
use_norm_rot = True


# road discrete object
bev_discobj_target = dict(
    type="ANCBevDiscreteTargetGenerator",
    num_classes=num_classes,
    bev_size=task_out_size,
    max_objs=150,
    vcs_range=roi_vcs_range,
    category2id_map=category2id_map,
    vcs_bbox_area_thresh=vcs_bbox_area_thresh,
    name2label=name2label,
    visible_threshold=0.1,
    name2group=name2group,
    ignore_miss_cls=True,
    ignore_miss_clsid=0,
    vis_mask_vcs_range_cfg=vismask_vcsrange_cfg,
    use_vis_mask=True,
    psc_phase_factor=psc_phase_factor,
    N_steps_PSC_rot=N_steps_PSC_rot if use_psc_rot else None,
    return_ignore_obj=False,
)

bev_discobj_target_list = []
bev_discobj_target_list.append(bev_discobj_target)


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


roi_resize_cfg = get_roi_resize_cfg(
    input_size=bevfusion_output_size,
    in_stride=task_in_stride,
    output_size=task_out_size,
    ori_vcs_range=vcs_range,
    roi_vcs_range=roi_vcs_range,
)
junction_roi_resize = dict(
    type="RoiCropResize",
    in_strides=[2, 4, 8, 16, 32],
    target_stride=roi_resize_cfg["in_stride"],
    output_size=roi_resize_cfg["output_size"],
    roi_box=roi_resize_cfg["roi_box"],
    node_name="junction_roi_resize",
)

# -------------------------- model --------------------------
train_inputs, val_inputs, deploy_inputs = get_inputs(
    num_views, vcs_plane_heights, task_out_size, num_classes
)
inputs = dict(
    train=train_inputs,
    val=val_inputs,
    deploy=deploy_inputs,
)

task_loss_weights = {
    "bev_junction": 0.02,
}
task_attr_loss_weights = {
    "bev_junction": {
        "bev_hm": 2.0,
        "bev_dim": 1.0,
        "bev_ct_offset": 1.0,
        "bev_rot": 4.0,
    },
}
task_loss_weight = task_loss_weights[task_name]
task_attr_loss_weight = task_attr_loss_weights[task_name]
loss_weights = dict(
    bev_discobj_hm=task_attr_loss_weight["bev_hm"] * task_loss_weight,
    bev_discobj_wh=task_attr_loss_weight["bev_dim"] * task_loss_weight,
    bev_discobj_ct_offset=task_attr_loss_weight["bev_ct_offset"]
    * task_loss_weight,
    bev_discobj_rot=task_attr_loss_weight["bev_rot"] * task_loss_weight * 0.75
    if use_psc_rot and N_steps_PSC_rot is not None
    else task_attr_loss_weight["bev_rot"] * task_loss_weight,
)

bev_common_desc = dict(
    vcs_origin_coord=vcs_origin_coord,
    bev_stage2_input_resolution=spatial_resolution,
    bev_stage2_output_resolution=bev_discobj_junction_stage2_output_resolution,
    visible_range=reformat_compile_vcs_range(roi_vcs_range),
    warp_offset_range=reformat_compile_vcs_range(vcs_range),
    vcs_plane_heights=vcs_plane_heights,
)

task_val_metric_setting = {
    "score_threshold": 0.2,
    "iou_threshold": 0.2,
    "yaw_amplitude": 180,
    "gt_max_depth": 105,
}

junction_decode_setting = {
    "topk": 20,
    "max_pool_kernel": 15,
    "do_ct_nms": True,
    "ct_dist_scale": 0.5,
    "class_ids": [1],
    "rot_match_threshold": 0.5,
    "kernel_from_singlebox": False,
}
for update_key, update_value in model_thresh.get(task_name, {}).items():
    task_val_metric_setting[update_key] = update_value


def get_desc(
    use_psc_rot=False,
    N_steps_PSC_rot=None,
    psc_phase_factor=1,
    use_norm_rot=False,
):
    apply_metric_setting = task_val_metric_setting
    apply_decode_setting = junction_decode_setting
    apply_common_desc = copy.deepcopy(bev_common_desc)

    if use_psc_rot and N_steps_PSC_rot is not None:
        channel_labels = [f"cos_idx{idx}" for idx in range(N_steps_PSC_rot)]
        postprocess_rot_desc = {
            "task": "bev_discobj_junction",
            "output_name": "bev_discobj_junction_rot_output",
            "properties": [{"channel_labels": channel_labels}],
            "rot_mod_threshold": 0.0001,
            "use_norm_rot": int(use_norm_rot),
            "psc_phase_factor": psc_phase_factor,
            **apply_common_desc,
        }
    else:
        postprocess_rot_desc = {
            "task": "bev_discobj_junction",
            "output_name": "bev_discobj_junction_rot_output",
            "properties": [{"channel_labels": ["cos", "sin"]}],
            "use_norm_rot": int(use_norm_rot),
            "psc_phase_factor": 1,
            **apply_common_desc,
        }
    per_tensor_desc = [
        {
            "task": "bev_discobj_junction",
            "output_name": "bev_discobj_junction_heatmap_output",
            "score_threshold": apply_metric_setting["score_threshold"],
            "max_pool_kernel": apply_decode_setting["max_pool_kernel"],
            "do_ct_nms": int(apply_decode_setting.get("do_ct_nms", False)),
            "ct_dist_scale": apply_decode_setting.get("ct_dist_scale", 1.0),
            "class_ids": apply_decode_setting.get("class_ids", [0]),
            "rot_match_threshold": apply_decode_setting.get(
                "rot_match_threshold", 0.5
            ),
            "kernel_from_singlebox": int(
                apply_decode_setting.get("kernel_from_singlebox", True)
            ),
            "properties": [
                {
                    "channel_labels": [
                        id2label[id] for id in range(len(id2label))
                    ]
                }
            ],
            **apply_common_desc,
        },
        {
            "task": "bev_discobj_junction",
            "output_name": "bev_discobj_junction_dimension_output",
            "properties": [{"channel_labels": ["width", "height"]}],
            **apply_common_desc,
        },
        postprocess_rot_desc,
        {
            "task": "bev_discobj_junction",
            "output_name": "bev_discobj_junction_ct_offset_output",
            "properties": [{"channel_labels": ["offset_x", "offset_y"]}],
            **apply_common_desc,
        },
    ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


def get_model(mode):
    bev_head = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEV3DHead",
            in_strides=[task_in_stride],
            out_strides=[task_in_stride],
            in_channels=48,
            forward_frame_idx=0,
            head_channels=OrderedDict(
                pred_bev_discobj_hm=num_classes,
                pred_bev_discobj_wh=2,  # w, h
                pred_bev_discobj_rot=N_steps_PSC_rot
                if use_psc_rot and N_steps_PSC_rot is not None
                else 2,  # cos, sin
                pred_bev_discobj_ct_offset=2,  # u, v
            ),
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
        node_name="bev_stage2_junction_head",
    )
    if mode == "train":
        bev_head["loss"] = dict(
            type="ANCBEVDiscreteObjectLoss",
            loss_weights=loss_weights,
            use_focal_hm_loss=True,
            gt_name="gt_bev_discrete_obj",
            use_norm_rot=use_norm_rot,
        )
    elif mode == "val":
        bev_head["postprocess"] = dict(
            type="ANCBEVDiscreteObjectDecoder",
            cls_dimension=None,
            topk=20,
            max_pool_kernel=15,
            vcs_range=roi_vcs_range,
            rot_mod_threshold=0.0001,
            use_psc_rot=use_psc_rot,
            N_steps_PSC_rot=N_steps_PSC_rot if use_psc_rot else None,
            psc_phase_factor=psc_phase_factor,
            use_norm_rot=use_norm_rot,
        )
    else:
        bev_head["convert_to_dict"] = False
        bev_discobj_junction_desc = get_desc(
            use_psc_rot=use_psc_rot,
            N_steps_PSC_rot=N_steps_PSC_rot,
            psc_phase_factor=psc_phase_factor,
            use_norm_rot=use_norm_rot,
        )
        postprocess_modules = []
        postprocess_modules.append(
            dict(
                type="AddDesc",
                per_tensor_desc=bev_discobj_junction_desc,
            )
        )
        bev_head["postprocess"] = dict(
            type="MultiInputSequential", modules=postprocess_modules
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
            bev_fusion_upsample=bev_fusion_upsampling,
            temporal_fusion=temporal_fusion[mode],
            backbone=bev_backbone,
            neck=bev_neck,
            roi_resizes=junction_roi_resize,
            head=bev_head,
        ),
        bevfusion_pick_keys=bevfusion_pick_keys,
    )

    return model


# -------------------------- TRAIN METRIC --------------------------
metrics, per_metric_patterns = get_metrics_patterns(
    task_name,
)

metric_updater = get_train_metric_updater(
    task_name=task_name,
    metrics=metrics,
    per_metric_patterns=per_metric_patterns,
    log_freq=log_freq,
)

# -------------------------- VALIDATION SETTING --------------------------
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
    val_metric_patterns = [
        dict(
            pred_ct=f"^.*pred_bev_discobj_ct",  # noqa
            pred_wh=f"^.*pred_bev_discobj_wh",  # noqa
            pred_score=f"^.*pred_bev_discobj_score",  # noqa
            pred_yaw=f"^.*pred_bev_discobj_rot",  # noqa
            pred_bev_discobj_cls_id=f"^.*pred_bev_discobj_cls_id",  # noqa
        )
    ]

    val_metrics = get_val_metrics(
        task_name=task_name,
        id2label=id2label,
        eval_category_ids=tuple(range(num_classes)),
        iou_threshold=task_val_metric_setting["iou_threshold"],
        score_threshold=task_val_metric_setting["score_threshold"],
        gt_max_depth=task_val_metric_setting["gt_max_depth"],
        metrics=("dx", "dy", "dxy", "dw", "dh", "drot"),
        save_dir=save_dir,
        save_eval_result_path=save_metric_path,
        yaw_amplitude=task_val_metric_setting["yaw_amplitude"],
        match_mode=task_val_metric_setting.get("gt_match_mode", None),
        compute_foreground_prec=False,
        vis_image_dir=save_vis_dir,
        result_prefix="wide",
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


# if (
#     merge_crosswalk_to_junction
#     and not temporal_bev
#     and not pipeline_test
#     and training_step != "int_infer"
# ):
#     assert len(val_metric_updater_list) == 2
#     _val_metric_updater_junction = copy.deepcopy(val_metric_updater_list[0])
#     for idx, val_metrics in enumerate(_val_metric_updater_junction["metrics"]):
#         _val_metrics = copy.deepcopy(val_metrics)
#         _val_metrics["eval_category_ids"] = (0,)
#         _val_metric_updater_junction["metrics"][idx] = _val_metrics
#     val_metric_updater_list[0] = _val_metric_updater_junction

#     _val_metric_updater_crosswalk = copy.deepcopy(val_metric_updater_list[1])
#     for idx, val_metrics in enumerate(
#         _val_metric_updater_crosswalk["metrics"]
#     ):
#         _val_metrics = copy.deepcopy(val_metrics)
#         _val_metrics["eval_category_ids"] = (1,)
#         _val_metric_updater_crosswalk["metrics"][idx] = _val_metrics
#     val_metric_updater_list[1] = _val_metric_updater_crosswalk

# -------------------------- tensorboard --------------------------

gt_hm_key_regex = re.compile(
    f"^.*{task_name}.*gt_bev_discrete_obj_bev_discobj_weight_hm"
)
pred_hm_key_regex = re.compile(f"^.*{task_name}.*pred_bev_discobj_hm")

if enable_tensorboard:
    tb_update_func = get_bev_tb_update_func(
        task_name, pred_hm_key_regex, gt_hm_key_regex
    )

# --------------------------- PACK VIS ----------------------------
visualize_callbacks = [
    dict(
        type="ANCBevObjVisualizeCB",
        output_dir=os.path.join(save_prefix, "visualize", "bev_junction"),
        task="bev_junction",
        prefix="OutputModule_predict",
        score_threshold=0.2,
        vis_img_scale=1,
        bev_size=ipm_output_size,
        vcs_range=vcs_range,
        camera_view_names=camera_view_names,
        camera_layouts=None,
        res_key2anno_name={
            "bev_junction": "annos_bev_discrete_obj",
        },
        color_map=COLOR_MAP,
        id2label=ID2LABEL,
        visual_interval=1,
        vis_ego=True,
        project_pts_to_cameras=False,
    )
]
