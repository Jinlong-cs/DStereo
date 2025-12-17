import copy
import json
import os

import cv2
import torch

from hat.callbacks.metric_updater import update_metric_using_regex
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
    fisheye_block_warp_padding,
    fisheye_camera_view_names,
    fisheye_warp_range,
    front_block_warp_padding,
    front_camera_view_names,
    head,
    img_ori_size,
    img_resize_wh_size,
    is_local_train,
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

task_name = "online_mapping"

om_head_groups = {
    "lane": {
        "multi": True,
        "group": "lane",
        "key": "solid_lanes#lane_type",
        "loss_weight": 3,
        "max_patch_segment": 2,
        "loss_type": "ce_loss",
        "cls_remap": {
            "solid": 1,
            "dotted": 1,
            "dashed": 1,
            "solid_fishbone": 1,
            "ldotted_fishbone": 1,
            "boots_dots": 1,
            "other": 1,
            "unknown": 1,
        },
        "cls_list": ["lane"],
        "cls_colors": ["blue"],
        "cls_thr": 0.5,
    },
    "roadedge": {
        "multi": True,
        "group": "roadedge",
        "key": "roadedges#curb_category",
        "max_patch_segment": 1,
        "loss_type": "ce_loss",
        "cls_remap": {
            "groundside": 1,
            "roadside": 1,
            "cone": 1,
            "water_horse": 1,
            "guardrail": 1,
            "other": 1,
            "unknown": 1,
        },
        "cls_list": ["roadside"],
        "cls_colors": ["red"],
        "cls_thr": 0.5,
    },
    "lane_direction": {
        "group": "lane",
        "loss_weight": 0.5,
        "add_cost": False,
        "auto_class_weight": False,
        "global_auto_class_weight": [0.75, 2.0],
        "key": "solid_lanes#lane_direction",
        "cls_remap": {"unidirectional": 0, "bidirectional": 4},
        "cls_list": ["up", "down", "left", "right", "bidirection"],
        "cls_colors": ["white", "red", "green", "blue", "yellow"],
    },
    "roadedge_subtype": {
        "group": "roadedge",
        "key": "roadedges#curb_category",
        "loss_weight": 1.0,
        "auto_class_weight": False,
        "global_auto_class_weight": [1.0, 2.5],
        "cls_remap": {
            "groundside": 0,
            "roadside": 1,
            "cone": 2,
            "water_horse": 3,
            "guardrail": 4,
            "other": 5,
            "unknown": -1,
        },
        "cls_list": [
            "groundside",
            "roadside",
            "cone",
            "water_horse",
            "guardrail",
            "other",
        ],
        "cls_colors": [
            "white",
            "red",
            "blue",
            "orange",
            "green",
            "yellow",
        ],
    },
    "lane_color": {
        "group": "lane",
        "key": "solid_lanes#lane_color",
        "loss_weight": 0.5,
        "add_cost": False,
        "auto_class_weight": False,
        "global_auto_class_weight": [0.75, 2.0],
        "cls_remap": {
            "white": 0,
            "yellow": 1,
            "blue": 2,
            "orange": 2,
            "red": 2,
            "other": 2,
        },
        "cls_list": ["white", "yellow", "other"],
        "cls_colors": ["white", "yellow", "blue"],
        "only_focus_main_roi": True,
    },
    "lane_wide": {
        "group": "lane",
        "key": "solid_lanes#lane_width",
        "loss_weight": 0.5,
        "add_cost": False,
        "auto_class_weight": False,
        "global_auto_class_weight": [1.0, 3.0],
        "cls_remap": {"normal": 0, "wide": 1},
        "cls_list": ["normal", "wide"],
        "cls_colors": ["red", "blue"],
        "only_focus_main_roi": True,
    },
    "lane_double": {
        "group": "lane",
        "key": "solid_lanes#lane_flag",
        "loss_weight": 1.0,
        "add_cost": False,
        "auto_class_weight": False,
        "global_auto_class_weight": [1.0, 2.0],
        "cls_remap": {"single": 0, "double": 1, "triple": 1},
        "cls_list": ["single", "double"],
        "cls_colors": ["red", "blue"],
        "only_focus_main_roi": True,
    },
    "lane_property": {
        "group": "lane",
        "key": "solid_lanes#lane_properties",
        "loss_weight": 0.5,
        "add_cost": False,
        "auto_class_weight": False,
        "global_auto_class_weight": [1.0, 2.0],
        "cls_remap": {
            "general": 0,
            "stay": 1,
            "tide": 2,
            "three_color": 3,
            "bus": 4,
            "other": 5,
        },
        "cls_list": [
            "general",
            "stay",
            "tide",
            "three_color",
            "bus",
            "other",
        ],
        "cls_colors": [
            "white",
            "red",
            "blue",
            "orange",
            "green",
            "yellow",
        ],
        "only_focus_main_roi": True,
    },
    "lane_dashed": {
        "group": "lane",
        "key": "solid_lanes#lane_type",
        "loss_weight": 3,
        "add_cost": True,
        "auto_class_weight": False,
        "global_auto_class_weight": [1.0, 1.5],
        "cls_remap": {
            "solid": 0,
            "dotted": 1,
            "dashed": 1,
            "solid_fishbone": 0,
            "ldotted_fishbone": 1,
            "boots_dots": 1,
            "other": -1,
            "unknown": -1,
        },
        "cls_list": ["solid", "dashed"],
        "cls_colors": ["red", "blue"],
        "only_focus_main_roi": True,
    },
}

embedding_dim = 4
cluster_alg = "dbscan"
cluster_cfgs = {
    "dbscan": {
        "cluster_bw": 0.7,
    },
    "ogc": {
        "cluster_bw": 0.95,
        "pose_weights": {"lane": 0.1, "roadedge": 0.1},
    },
    "sgc": {
        "cluster_bw": 0.85,
    },
}
block_warp_padding = [front_block_warp_padding] + fisheye_block_warp_padding

om_head_groups_update = {
    "lane": {
        "cls_thr": model_thresh.get(task_name, {}).get("lane", 0.85),
    },
    "roadedge": {
        "cls_thr": model_thresh.get(task_name, {}).get("roadedge", 0.6),
    },
}

bev_om_head_cfg = {
    "main_repeat": 1,
    "sub_repeat": 1,
    "align_channel": 96,
    "repeat_channel": 96,
    "enhance_repeat_times": -1,
}
om_dice_loss = dict(type="ANCDiceLoss")
# need train one more
# dict(
#         type="ANCDiceLoss",
#     )

for om_head in om_head_groups_update:
    for key in om_head_groups_update[om_head]:
        om_head_groups[om_head][key] = om_head_groups_update[om_head][key]

in_stride = (8, 16)
out_size = ((96, 64), (48, 32))

if is_list_of_type(out_size, tuple):
    head_out_size = out_size[0]
roi_vcs_range = (-12.8, -12.8, 25.6, 12.8)

om_stage2_output_resolution = (
    abs(roi_vcs_range[2] - roi_vcs_range[0]) / head_out_size[0],
    abs(roi_vcs_range[3] - roi_vcs_range[1]) / head_out_size[1],
)  # (height, witdh)


# config model output head names and dims
out_nums_list = []
out_repeat_list = []
output_name_list = []
gt_shape_group = {}
multi_head_keys = ["cls", "instance", "r", "sin", "cos"]
om_out_h, om_out_w = head_out_size[:2]
for om_head, head_infos in om_head_groups.items():
    group = head_infos["group"]
    group_head = om_head_groups[group]
    loss_type = group_head.get("loss_type", "ce_loss")
    max_patch_segment = group_head.get("max_patch_segment", 1)
    out_channels = max([v for _, v in head_infos["cls_remap"].items()]) + 1
    multi_head = head_infos.get("multi", False)
    if multi_head:
        if loss_type == "focal_loss":
            # ignore background channel
            out_channels = out_channels - 1
        # multi_head with focal_loss has no background channel
        head_names = [
            f"pred_online_mapping_{key}_{group}" for key in multi_head_keys
        ]
        head_dims = [out_channels, embedding_dim, 1, 1, 1]
        head_repeats = [bev_om_head_cfg["main_repeat"]] * 5
        head_dims = [v * max_patch_segment for v in head_dims]
    else:
        head_names = [f"pred_online_mapping_{om_head}_{group}"]
        head_repeats = [bev_om_head_cfg["sub_repeat"]]
        head_dims = [out_channels * max_patch_segment]
    out_nums_list += head_dims
    output_name_list += head_names
    out_repeat_list += head_repeats
    # target dims
    if max_patch_segment > 1:
        gt_shape_group[om_head] = (-1, 2, om_out_h, om_out_w)
    else:
        gt_shape_group[om_head] = (-1, om_out_h, om_out_w)

select_strides = _as_list(in_stride)


# -------------------------- LOSS ---------------------------
# Om loss
def get_head_cls_loss(head_groups, main_cls_loss_weight, sub_cls_loss_weight):
    head_cls_loss = {}
    for om_head, head_infos in head_groups.items():
        group = head_infos["group"]
        group_head = head_groups[group]
        loss_type = group_head.get("loss_type", "ce_loss")
        max_patch_segment = group_head.get("max_patch_segment", 1)
        assert max_patch_segment == 1 or max_patch_segment == 2
        # check dilate
        if map_dilate < 1:
            head_groups[om_head]["dilate"] = False
        # get cls num
        num_class = max([v for _, v in head_infos["cls_remap"].items()]) + 1
        multi_head = head_infos.get("multi", False)
        if head_infos.get("dilate", True):
            head_infos["auto_class_weight"] = False
        if loss_type == "ce_loss":
            auto_class_weight = head_infos.get("auto_class_weight", True)
            global_auto_class_weight = head_infos.get(
                "global_auto_class_weight", None
            )
            assert (not auto_class_weight) or (
                global_auto_class_weight is None
            )
            ignore_index = head_infos.get("ignore_index", -1)
            cls_loss = dict(
                type="CrossEntropyLoss",
                reduction="none",
                loss_name="loss_cls",
                loss_weight=main_cls_loss_weight
                if multi_head
                else sub_cls_loss_weight,
                class_weight=None,
                auto_class_weight=auto_class_weight,
                weight_min=0.5,
                weight_noobj=0.75,
                num_class=num_class,
                ignore_index=ignore_index,
            )
        elif loss_type == "focal_loss":
            cls_loss = dict(
                type="ANCOMFocalLoss",
                loss_name="loss_cls",
                num_classes=num_class if multi_head else num_class + 1,
                alpha=0.5,
                gamma=0.0,
                gt_offset=1 if multi_head else 0,
                loss_weight=main_cls_loss_weight
                if multi_head
                else sub_cls_loss_weight,
                reduction="none",
            )
        else:
            raise NotImplementedError
        head_cls_loss[om_head] = cls_loss

    return head_cls_loss


map_dilate = 1
main_cls_loss_weight = 1.6
sub_cls_loss_weight = 1.6
head_cls_loss = get_head_cls_loss(
    om_head_groups, main_cls_loss_weight, sub_cls_loss_weight
)

adaptive_loss_weight = True
global_ignore_index = -1
assign_near_instance_cfg = None
merge_instance_cfg = None
om_roi_weight_cfg = None

# -------------------------- Metric -------------------------
metrics = []
per_metric_patterns = []
metrics += [
    dict(type="LossShow", name="loss_cls_all"),
    dict(type="LossShow", name="loss_instance_inner_all"),
    dict(type="LossShow", name="loss_instance_outer_all"),
    dict(type="LossShow", name="loss_r_all"),
    dict(type="LossShow", name="loss_angle_all"),
]

per_metric_patterns += [  # corresponding to metrics
    dict(
        label_pattern=None,
        pred_pattern=f"^.*{task_name}_head.*loss_cls_all$",
    ),
    dict(
        label_pattern=None,
        pred_pattern=f"^.*{task_name}_head.*loss_instance_inner_all$",
    ),
    dict(
        label_pattern=None,
        pred_pattern=f"^.*{task_name}_head.*loss_instance_outer_all$",
    ),
    dict(
        label_pattern=None,
        pred_pattern=f"^.*{task_name}_head.*loss_r_all$",
    ),
    dict(
        label_pattern=None,
        pred_pattern=f"^.*{task_name}_head.*loss_angle_all$",
    ),
]
if om_dice_loss is not None:
    metrics.append(dict(type="LossShow", name="loss_cls_dice_all"))
    per_metric_patterns.append(
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}_head.*loss_cls_dice_all$",
        )
    )

if adaptive_loss_weight:
    metrics += [
        dict(type="LossShow", name="loss_s_cls"),
        dict(type="LossShow", name="loss_s_instance"),
        dict(type="LossShow", name="loss_s_r"),
        dict(type="LossShow", name="loss_s_angle"),
    ]
    per_metric_patterns += [
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}_head__loss_s_cls$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}_head__loss_s_instance$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}_head__loss_s_r$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}_head__loss_s_angle$",
        ),
    ]
    if om_dice_loss is not None:
        metrics.append(dict(type="LossShow", name="loss_s_cls_dice"))
        per_metric_patterns.append(
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_head__loss_s_cls_dice$",
            )
        )

for om_head, head_infos in om_head_groups.items():
    multi_head = head_infos.get("multi", False)
    group = head_infos["group"]
    suffix = f"{om_head}_{group}"
    if not multi_head:
        metrics += [dict(type="LossShow", name=f"loss_{suffix}")]
        per_metric_patterns += [
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_head__loss_{suffix}$",
            ),
        ]
        if adaptive_loss_weight:
            metrics += [dict(type="LossShow", name=f"loss_s_{suffix}")]
            per_metric_patterns += [
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{task_name}_head__loss_s_{suffix}$",
                ),
            ]
metric_updater = dict(
    type="MetricUpdater",
    metrics=metrics,
    metric_update_func=update_metric_using_regex(per_metric_patterns),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


# ----------------------------- DESC ---------------------------
bev_common_desc = dict(
    vcs_origin_coord=vcs_origin_coord,
    spatial_resolution=[1.6, 1.6],
    bev_stage2_input_resolution=spatial_resolution,
    bev_stage2_output_resolution=om_stage2_output_resolution,
    visible_range=reformat_compile_vcs_range(roi_vcs_range),
    warp_offset_range=reformat_compile_vcs_range(vcs_range),
    vcs_plane_heights=vcs_plane_heights,
    fisheye_warp_offset_range=reformat_compile_vcs_range(fisheye_warp_range),
)


def get_desc(multi_head_keys):
    per_tensor_desc = []
    apply_om_head_groups = om_head_groups
    apply_common_desc = copy.deepcopy(bev_common_desc)
    for _head, head_infos in apply_om_head_groups.items():
        group = head_infos["group"]
        group_head = apply_om_head_groups[group]
        loss_type = group_head.get("loss_type", "ce_loss")
        max_patch_segment = group_head.get("max_patch_segment", 1)
        num_class = max([v for _, v in head_infos["cls_remap"].items()]) + 1
        multi_head = head_infos.get("multi", False)
        prefix = "online_mapping_"
        sub_head_lists = [_head]
        if multi_head:
            prefix += head_infos["group"] + "_"
            sub_head_lists = multi_head_keys
        for sub_head in sub_head_lists:
            desc = {}
            desc["task"] = prefix + sub_head
            desc["num_classes"] = num_class
            desc["score_threshold"] = head_infos.get("cls_thr", -1.0)
            desc["cls_loss_type"] = loss_type
            desc["max_patch_segment"] = max_patch_segment
            desc["assign_near_instance"] = (
                1 if assign_near_instance_cfg is not None else 0
            )
            desc.update(**apply_common_desc)
            per_tensor_desc.append(desc)
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


# -------------------------- MODEL --------------------------
def get_inputs():
    inputs = dict(
        om_target=dict(
            lane=dict(
                cls=torch.zeros(1, 2, 96, 64, dtype=torch.int64),
                instance=torch.zeros(1, 2, 96, 64, dtype=torch.int64),
                r=torch.zeros(1, 2, 96, 64, dtype=torch.float64),
                sin=torch.zeros(1, 2, 96, 64, dtype=torch.float64),
                cos=torch.zeros(1, 2, 96, 64, dtype=torch.float64),
                dilate=torch.zeros(1, 2, 96, 64, dtype=torch.int64),
                weight=torch.zeros(1, 2, 96, 64, dtype=torch.float64),
                main_roi_mask=torch.zeros(1, 2, 96, 64, dtype=torch.float64),
                lane_direction=torch.full(
                    (1, 2, 96, 64), -1, dtype=torch.int64
                ),
                lane_color=torch.full((1, 2, 96, 64), -1, dtype=torch.int64),
                lane_wide=torch.full((1, 2, 96, 64), -1, dtype=torch.int64),
                lane_double=torch.full((1, 2, 96, 64), -1, dtype=torch.int64),
                lane_property=torch.full(
                    (1, 2, 96, 64), -1, dtype=torch.int64
                ),
                lane_dashed=torch.full((1, 2, 96, 64), -1, dtype=torch.int64),
            ),
            roadedge=dict(
                cls=torch.zeros(1, 96, 64, dtype=torch.int64),
                instance=torch.zeros(1, 96, 64, dtype=torch.int64),
                r=torch.zeros(1, 96, 64, dtype=torch.float64),
                sin=torch.zeros(1, 96, 64, dtype=torch.float64),
                cos=torch.zeros(1, 96, 64, dtype=torch.float64),
                dilate=torch.zeros(1, 96, 64, dtype=torch.int64),
                weight=torch.zeros(1, 96, 64, dtype=torch.float64),
                main_roi_mask=torch.zeros(1, 96, 64, dtype=torch.float64),
                roadedge_subtype=torch.full(
                    (1, 96, 64), -1, dtype=torch.int64
                ),
            ),
        ),
    )
    return inputs


inputs = get_inputs()
val_inputs = copy.deepcopy(inputs) if training_step != "pack_infer" else {}
inputs = dict(
    train=inputs,
    val=val_inputs,
    deploy={},
)


def get_model(mode):
    bev_head = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEVOMHead",
            in_strides=[2, 4, 8, 16, 32],
            select_strides=select_strides,
            out_stride=8,
            align_channels=bev_om_head_cfg["align_channel"],
            stride2channels=bev_neck["out_stride2channels"],
            out_nums=out_nums_list,
            output_name=output_name_list,
            quanti_last_conv=True,
            dequant_out=True,
            repeat_times=out_repeat_list,
            repeat_channels=bev_om_head_cfg["repeat_channel"],
            enhance_repeat_times=bev_om_head_cfg["enhance_repeat_times"],
            bn_kwargs=bn_kwargs,
        ),
        head_parser=None,
        loss=None,
        postprocess=None,
        target=dict(
            type="ANCBEVTarget",
            gt_name=[
                "om_target",
            ],
            bev_height=om_out_h,
            bev_width=om_out_w,
            gt_shape=gt_shape_group,
        ),
        node_name="bev_stage2_om_small_head",
        prefix="online_mapping_head",
    )
    if mode == "train":
        bev_head["loss"] = dict(
            type="ANCOnlineMappingLoss",
            head_groups=om_head_groups,
            head_cls_loss=head_cls_loss,
            head_cls_dice_loss=om_dice_loss,
            instance_loss=dict(
                type="ANCEmbeddingLoss",
                embed_dim=embedding_dim,
                inner_loss_weight=0.8,
                outer_loss_weight=0.8,
            ),
            reg_loss_weight=1.25 * 0.8,
            adaptive_loss_weight=adaptive_loss_weight,
            phi_loss_proportion_r=True,
            global_ignore_index=global_ignore_index,
            cpts_loss=None,
            assign_near_instance_cfg=assign_near_instance_cfg,
        )
    elif mode == "val":
        bev_head["postprocess"] = dict(
            type="ANCOnlineMappingDecoder",
            head_groups=om_head_groups,
            embedding_dim=embedding_dim,
            out_size=head_out_size,
            vcs_range=roi_vcs_range,
            use_seq=cluster_alg != "sgc",
            use_seq_post=False,
            decoder_output_dir=decoder_output_dir,
            cluster_alg="dbscan",
            cluster_cfg=cluster_cfgs[cluster_alg],
            cluster_split_channel=assign_near_instance_cfg is not None,
            cpts_decoder=None,
        )
    else:
        bev_head["convert_to_dict"] = False
        bev_head["target"] = None
        multi_head_keys = ["cls", "instance", "r", "sin", "cos"]
        bev_head["postprocess"] = dict(
            type="MultiInputSequential",
            modules=[
                # # AddDesc should be the last module
                dict(
                    type="AddDesc",
                    per_tensor_desc=get_desc(multi_head_keys),
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


# ------------------ TRAIN DATASET SETTING -------------------
train_data_version = "v5_6_0_mixed_views_select_small"
val_data_version = [
    "v4_0_0_ls_11v_parking",
    "v5_0_0_11v_driving",
    "v4_0_0_11v_parking",
    "v1_0_0_ek_parking_underground",
    "v1_0_0_ek_parking_lot",
]

if pipeline_test:
    train_data_version = "pipeline_test"
    val_data_version = "pipeline_test"

dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_om"
)

task_name_list = [
    task_name + "_" + val_version for val_version in _as_list(val_data_version)
]

# om
om_view_bev_size = (
    head_out_size[0] * 8,
    head_out_size[1] * 8,
)
om_short_filter_cfg = None
om_roadedge_occ_cfg = {
    "anchor_points": [-2, 1, 4],
    # whether use open/close curb
    "check_close": False,
    "black_list": [],
    "sample_res": 0.5,
}
dilate_weight = {
    "lane": {
        "background": 0.6,
        "dilated": 1.2,
        "foreground": 1.5,
    },
    "roadedge": {
        "background": 0.6,
        "dilated": 1.2,
        "foreground": 1.5,
    },
}

om_view_cols = 3
om_view_sub_head = True
visualize_output_dir = None  # data gt label visualization

rpy_pers_aug = True

load_data_types = [
    "timestamp",
    "img_name",
    "gt_online_mapping",
    "pack_dir",
    "img_paths",
    "origin_imgs",
]

load_data_types = remove_none(load_data_types)

bev_common_transforms = copy.deepcopy(common_transforms)

collect_3dv = bev_common_transforms["ANCCollect3DV"]
collect_3dv["gt_om_idx"] = 0
collect_3dv["load_data_types"] = load_data_types
om_target = dict(
    type="ANCOnlineMappingTargetGenerator",
    head_groups=om_head_groups,
    vcs_range=roi_vcs_range,
    out_size=head_out_size,
    roi_weight_cfg=om_roi_weight_cfg,
    view_bev_size=om_view_bev_size,
    roadedge_occ_cfg=om_roadedge_occ_cfg,
    map_dilate=map_dilate,
    dilate_weight=dilate_weight,
    split_close_roadedge=False,
    view_cols=om_view_cols,
    view_sub_head=om_view_sub_head,
    visualize_output_dir=visualize_output_dir,
    global_ignore_index=global_ignore_index,
    assign_near_instance_cfg=assign_near_instance_cfg,
    om_short_filter_cfg=om_short_filter_cfg,
    block_warp_padding=block_warp_padding,
    merge_instance_cfg=merge_instance_cfg,
)

om_target_list = []
om_target_list.append(om_target)

om_rpy_transforms = None
if rpy_pers_aug:
    om_rpy_transforms = dict(
        type="ANCSetRPYAugParam",
        prob=0.4,
        aug_degree=1,
    )
om_transforms = [
    collect_3dv,
    *om_target_list,
    bev_common_transforms["ANCResize3DV"],
    bev_common_transforms["ANCCrop3DV"],
    bev_common_transforms["ANCClassRemap"],
    bev_common_transforms["ANCToTensor3DV"],
    om_rpy_transforms if training_step == "train" else None,
]

om_transforms.append(common_transforms["ANCPrepareDataBEV"])
om_transforms = remove_none(om_transforms)

homo_transforms = get_homo_transforms(
    transforms_list=om_transforms, camera_view_names=camera_view_names
)
template_dataset = get_template_dataset(
    img_load_size=list(img_resize_wh_size.values()),
    camera_view_names=camera_view_names,
    per_view_shape=img_ori_size,
    transforms=om_transforms,
    homo_transforms=homo_transforms,
    spatial_resolution=spatial_resolution,
    H_persp_view_scale=H_persp_view_scale,
    vcs_range=vcs_range,
    use_distorted_offset=True,
    vcs_plane_heights=vcs_plane_heights,
    cal_homo_offset_on_gpu=cal_homo_offset_on_gpu,
    offset_save_path=offset_save_path,
    temporal_bev=False,
    length_of_clip=None,
    train_num_frames_per_iter=None,
)
train_template_dataset = template_dataset
val_template_dataset = copy.deepcopy(template_dataset)
for transform_dict in train_template_dataset["transforms"]:
    if transform_dict["type"] == "ANCCollect3DV":
        if (
            "origin_imgs" in transform_dict["load_data_types"]
            and visualize_output_dir is None
        ):
            transform_dict["load_data_types"].remove("origin_imgs")

update_bev_dataset = get_update_bev_dataset_func(
    added_info=[
        "homo_path",
        "calib_path",
        "bev_static_lmdb_path",
        "camera_module_type",
        "homo_noise",
        "bev_occlusion_data_path",
    ],
    update_trans_info={
        "ANCCollect3DV": "fill_fake_temporal_data",
    },
    do_val_visualize=do_val_visualize,
)
train_homo_noise = {
    "noise_value": (0.2, 0.2, 0.2, 0.0, 0.0, 0.04),
    # It is the exact value of generated noise when the noise type is
    # 'specific_cam'. When the noise type is 'random_cam' or 'random_vcs',
    # it is the upper bound value of generated random noises. The format is
    # (roll, pitch, yaw, x, y, z), the units are degrees and meters.
    "noise_type": "random_cam",
    "noise_view_names": camera_view_names,
}
enable_homo_noise = False


def get_train_dataloader():
    train_url = os.path.join(dataset_dir, "om_train_version.yaml")
    train_data_version_dict = get_data_dict(train_url)
    train_dataset_dict = train_data_version_dict[train_data_version]
    om_global_sample = 1
    # global resample
    for loc in train_dataset_dict:
        for plate in train_dataset_dict[loc]:
            for item in train_dataset_dict[loc][plate]:
                if om_global_sample < 2:
                    continue
                sample = om_global_sample
                if train_dataset_dict[loc][plate][item] is None:
                    train_dataset_dict[loc][plate][item] = {}
                if "sample_interval" in train_dataset_dict[loc][plate][item]:
                    k = train_dataset_dict[loc][plate][item]["sample_interval"]
                    if isinstance(k, (list, tuple)):
                        new_s = [k_i * sample for k_i in k]
                    else:
                        new_s = sample * k
                    train_dataset_dict[loc][plate][item][
                        "sample_interval"
                    ] = new_s
                else:
                    train_dataset_dict[loc][plate][item][
                        "sample_interval"
                    ] = sample

    base_train_data_file_list = [
        "om_dataset_driving.yaml",
        "om_dataset_parking.yaml",
        "om_dataset_pipeline_test.yaml",
    ]
    train_all_data_dict = {}
    for base_file in base_train_data_file_list:
        url_base = os.path.join(dataset_dir, base_file)
        base_data = get_data_dict(url_base)
        for local, local_dict in base_data.items():
            if local not in train_all_data_dict:
                train_all_data_dict[local] = {}
            for vehicle_type, vehicle_dict in local_dict.items():
                if vehicle_type not in train_all_data_dict[local]:
                    train_all_data_dict[local][vehicle_type] = {}
                train_all_data_dict[local][vehicle_type].update(vehicle_dict)

    train_all_data_dict = join_path(
        bucket_root,
        train_all_data_dict,
        ["camera_module_type", "camera_view_names"],
    )

    train_datasets = get_datasets(
        train_dataset_dict,
        None,
        train_template_dataset,
        train_all_data_dict,
        update_bev_dataset,
        homo_noise=train_homo_noise if enable_homo_noise else None,
    )
    data_loader = get_dataloader(
        datasets=train_datasets,
        num_workers=train_num_workers,
        batch_size_per_gpu=train_batch_size_per_gpu,
        shuffle=True,
        persistent_workers=train_num_workers > 0,
    )
    if is_local_train:
        data_loader["shuffle"] = False
    # To fix reload imgrec problem, use persistent_workers.
    # persistent_workers option needs num_workers > 0
    data_loader["persistent_workers"] = train_num_workers > 0
    if use_split_dataloader:
        data_loader = convert_to_split_dataloader(data_loader)
    return data_loader


# ------------------ VAL DATASET SETTING -------------------
eval_root_dir = os.path.join(save_prefix, task_name, training_step)
# eval_root_dir = "/mnt/data-1/data/kefan.chen/evaluate"
val_ckpt = f"{task_name}_val_result"
profile_path = os.path.join(cfg_dir, "../datasets/bev_om/om_eval_profile.yaml")
eval_result_file = (
    f"{eval_root_dir}/{val_ckpt}/eval_result/eval_online_mapping.json"
)
save_infer_output_dir = f"{eval_root_dir}/{val_ckpt}/infer_file"
metric_save_path = f"{eval_root_dir}/eval_metric.json"

# save decoder result if not None
decoder_output_dir = None
# decoder_output_dir = f"{eval_root_dir}/{val_ckpt}/decoder_out"

# save visualize result if not None
visualize_infer_output_dir = None
# visualize_infer_output_dir = f"{eval_root_dir}/{val_ckpt}/infer_vis"

visualize_eval_output_dir = None
# visualize_eval_output_dir = f"{eval_root_dir}/{val_ckpt}/eval_vis"

save_raw_model_output = False
expect_timestamp = None

val_with_homo_noise = False


def get_val_dataloader():
    val_url = os.path.join(dataset_dir, "om_val_version.yaml")
    val_data_version_dict = get_data_dict(val_url)
    val_dataset_dict = [
        val_data_version_dict[val_version]
        for val_version in _as_list(val_data_version)
    ]
    base_val_data_file_list = [
        "om_dataset.yaml",
        "om_dataset_temporal.yaml",
        "om_dataset_pipeline_test.yaml",
    ]
    val_all_data_dict = {}
    for base_file in base_val_data_file_list:
        url_base = os.path.join(dataset_dir, base_file)
        base_data = get_data_dict(url_base)
        for local, local_dict in base_data.items():
            if local not in val_all_data_dict:
                val_all_data_dict[local] = {}
            for vehicle_type, vehicle_dict in local_dict.items():
                if vehicle_type not in val_all_data_dict[local]:
                    val_all_data_dict[local][vehicle_type] = {}
                val_all_data_dict[local][vehicle_type].update(vehicle_dict)
    val_all_data_dict = join_path(
        bucket_root,
        val_all_data_dict,
        ["camera_module_type", "camera_view_names"],
    )
    val_datasets = get_dataset_list(
        val_dataset_dict,
        None,
        val_template_dataset,
        val_all_data_dict,
        update_bev_dataset,
        homo_noise=None,
        repeat_dataset_times=3 if val_with_homo_noise else 1,
    )
    val_data_loader_list = get_dataloader_list(
        val_datasets, val_num_workers, val_batch_size_per_gpu, False, False
    )
    return val_data_loader_list


# ------------------ METRIC --------------------------------
# om
om_metric_cfg = {
    "cd_list": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "cd_threshold": 0.5,
    "thread_num": 32,
    "upsample_value": -1,
    "black_list": ["lane_other"],
    "region": {
        "all": [-12.8, 12.8, -12.8, 25.6],
        "roi": [-10.0, 10.0, -10.0, 20.0],
        "roi_-10_0": [-10.0, 10.0, -10, 0],
        "roi_0_20": [-10.0, 10.0, 0, 20],
        "roi_0_10": [-10.0, 10.0, 0, 10],
        "roi_10_20": [-10.0, 10.0, 10, 20],
    },
}


basic_metric_state_names = [
    "num_gt_pt",
    "num_pred_pt",
    "cd_g2p",
    "cd_p2g",
    "num_gt_lane",
    "num_pred_lane",
    "num_tp_lane",
    "num_gt_pt_inst_list",
    "num_pred_pt_inst_list",
    "cd_g2p_inst_list",
    "cd_p2g_inst_list",
]

metric_state_names = copy.deepcopy(basic_metric_state_names)

for om_head, head_infos in om_head_groups.items():
    multi_head = head_infos.get("multi", False)
    if not multi_head:
        metric_state_names += [
            f"gt_{om_head}_cls_inst",
            f"pred_{om_head}_cls_inst",
            f"gt_{om_head}_cls_pts",
            f"pred_{om_head}_cls_pts",
        ]
val_metrics = []
val_per_metric_patterns = []

val_metrics += [
    dict(
        type="ANCOnlineMappingMetric",
        name="OnlineMapping",
        metric_state_names=metric_state_names,
        embedding_dim=embedding_dim,
        head_groups=om_head_groups,
        metric_save_path=metric_save_path,
        eval_result_file=eval_result_file,
        save_infer_output_dir=save_infer_output_dir,
        save_raw_model_output=save_raw_model_output,
        visualize_infer_output_dir=visualize_infer_output_dir,
        visualize_eval_output_dir=visualize_eval_output_dir,
        profile_path=profile_path,
        vcs_range=roi_vcs_range,  # (bottom, right, top, left)
        out_size=head_out_size,
        view_bev_size=om_view_bev_size,
        image_size=(320, 512),
        cd_threshold=0.5,
        do_eval=True,
        analysis=False,
        fine_metric_cfg=om_metric_cfg,
        view_cols=om_view_cols,
        view_sub_head=om_view_sub_head,
        result_prefix="small",
    ),
]

label_pattern_list = [
    "^.*online_mapping_image_files",
    "^.*online_mapping_origin_imgs",
    "^.*online_mapping_om_target$",
]
val_per_metric_patterns += [  # corresponding to metrics
    dict(
        label_pattern=label_pattern_list,
        pred_pattern=f"^.*{task_name}_head_predict_om.*",
    ),
]


def om_val_flat_condition(key, values):
    """
    Validation need annotations and flating it will result in a bug with more
    than 255 parameters. So this function will not flat annotations
    """
    non_flat_list = [
        "om_predict",
        "target",
        "image_files",
        "origin_imgs",
        "crosspt_predict",
    ]
    for name in non_flat_list:
        if name in key:
            return False
    return True


def repeat_metric_updater_by_name(
    task_name_list, val_metric_updater, replace_key_list
):
    """Replace the save_path involved in val_metric_updater
    to ensure that the results are saved in different path.
    Return a list of val_metric_updater whose length is equal to len(task_name_list) # noqa
    The replacement rules are as follows:
    1.replace_value is None
        return None
    2.replace_value is a/b
        return a/taskname_b
    3.replace_value is a
        return taskname_a

    Args:
        task_name_list: Eval version name,used to generate new save_path.
        val_metric_updater: MetricUpdater Callback.
        replace_key_list: replace list. e.g."save_dir","name"...

    Returns: [val_metric_updater_1,val_metric_updater_2].
    """
    val_metric_updater_list = []
    for task_name in task_name_list:
        _val_metric_updater = copy.deepcopy(val_metric_updater)
        for idx, val_metrics in enumerate(_val_metric_updater["metrics"]):
            _val_metrics = copy.deepcopy(val_metrics)
            for replace_key in replace_key_list:
                # assert replace_key in _val_metrics
                if replace_key not in _val_metrics:
                    continue
                replace_value = _val_metrics[replace_key]
                if not replace_value:
                    continue
                base_dir, filename = os.path.split(replace_value)
                replace_value = os.path.join(
                    base_dir, task_name + "_" + filename
                )
                _val_metrics[replace_key] = replace_value
            _val_metric_updater["metrics"][idx] = _val_metrics
        val_metric_updater_list.append(_val_metric_updater)
    return val_metric_updater_list


val_flat_condition = om_val_flat_condition

_val_metric_updater = dict(
    type="MetricUpdater",
    metrics=val_metrics,
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=val_per_metric_patterns,
        flat_condition=val_flat_condition,
    ),
    step_log_freq=-1,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

replace_key_list = [
    "eval_result_file",
    "save_infer_output_dir",
    "visualize_infer_output_dir",
    "visualize_eval_output_dir",
    "metric_save_path",
    "name",
]
val_metric_updater_list = repeat_metric_updater_by_name(
    task_name_list, _val_metric_updater, replace_key_list
)


# -------------------------- PACK_INFER_SAVE --------------------------
save_pack_infer = dict(
    type="SaveOnlineMappingConsistencyResult",
    output_dir=os.path.join(save_prefix, "inference", "online_mapping"),
    output_key="online_mapping_head_predict_om_predict",
)

# --------------------------- PACK VIS ----------------------------
visualize_callbacks = [
    dict(
        type="ANCOMVisualize",
        output_dir=os.path.join(save_prefix, "visualize", "online_mapping"),
        task="online_mapping",
        prefix="online_mapping_head_predict_om_predict",
        cluster_alg=cluster_alg,
        out_size=head_out_size,
        view_bev_size=om_view_bev_size,
        vcs_range=roi_vcs_range,
        head_groups=om_head_groups,
        extra_img=dict(
            type="ANCBEVImgStitcher",
            per_extra_img_size=(960, 512),
            camera_view_names=camera_view_names,
            camera_layouts=None,
            interpolation=cv2.INTER_NEAREST,
        ),
    )
]
