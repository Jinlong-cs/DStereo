import copy
import json
import os

import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.utils.apply_func import _as_list, is_list_of_type
from hat.utils.filesystem import join_path
from projects.pilot.configs.bev_7v.base import (
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
)
from projects.pilot.configs.bev_7v.common import (
    H_persp_view_scale,
    backbone,
    bev_backbone,
    bev_fusion,
    bev_fusion_upsampling,
    bev_neck,
    bevfusion_output_size,
    block_warp_padding,
    bn_kwargs,
    bucket_root,
    cal_homo_offset_on_gpu,
    camera_view_names,
    common_transforms,
    deploy_head,
    deploy_homo_offset_key,
    deploy_mode,
    deploy_narrow_head,
    deploy_side_head,
    front_camera_view_names,
    head,
    img_ori_size,
    img_resize_wh_size,
    is_local_train,
    log_freq,
    multi_view_collect,
    narrow_backbone,
    narrow_camera_view_names,
    narrow_head,
    narrow_pafpn_neck,
    offset_save_path,
    pafpn_neck,
    side_backbone,
    side_camera_view_names,
    side_head,
    side_pafpn_neck,
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

# -------------------------- TASK ---------------------------
if not deploy_mode:
    raise NotImplementedError(f'"{__file__}": just for deploy or pack_infer.')

task_name = "bev_crosspoint"
merge_crosspoint = True

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

om_head_groups_update = {
    "lane": {
        "cls_thr": 0.52,
    },
    "roadedge": {
        "cls_thr": 0.4,
    },
}

bev_om_head_cfg = {
    "main_repeat": 1,
    "sub_repeat": 1,
    "align_channel": 96,
    "repeat_channel": 96,
    "enhance_repeat_times": -1,
}

for om_head in om_head_groups_update:
    for key in om_head_groups_update[om_head]:
        om_head_groups[om_head][key] = om_head_groups_update[om_head][key]

in_stride = (2, 4)
out_size = ((166, 128), (83, 64))
cpts_stride = 8

if is_list_of_type(out_size, tuple):
    head_out_size = out_size[0]
roi_vcs_range = (-32.0, -51.2, 100.8, 51.2)

om_stage2_output_resolution = (
    abs(roi_vcs_range[2] - roi_vcs_range[0]) / head_out_size[0],
    abs(roi_vcs_range[3] - roi_vcs_range[1]) / head_out_size[1],
)  # (height, witdh)

# config model output head names and dims
out_nums_list = []
out_repeat_list = []
output_name_list = []
gt_shape_group = {}

cpts_cls_group_map = {"crosspoints": [0, 1, 2, 3, 4, 5], "changepoints": [6]}
if merge_crosspoint:
    for group in cpts_cls_group_map.keys():
        group_names = [
            f"pred_crosspoint_cls_{group}",
            f"pred_crosspoint_x_{group}",
            f"pred_crosspoint_y_{group}",
        ]
        out_channels = len(cpts_cls_group_map[group])
        output_name_list.extend(group_names)
        out_nums_list.extend([out_channels, 1, 1])
        out_repeat_list.extend([bev_om_head_cfg["sub_repeat"]] * 3)

select_strides = _as_list(in_stride)

# task roi resize config
roi_resize_cfg_stride2 = get_roi_resize_cfg(
    input_size=bevfusion_output_size,
    in_stride=select_strides[0],
    output_size=out_size[0],
    ori_vcs_range=vcs_range,
    roi_vcs_range=roi_vcs_range,
)
roi_resize_cfg_stride4 = get_roi_resize_cfg(
    input_size=bevfusion_output_size,
    in_stride=select_strides[1],
    output_size=out_size[1],
    ori_vcs_range=vcs_range,
    roi_vcs_range=roi_vcs_range,
)

om_roi_resize_stride2 = dict(
    type="RoiCropResize",
    in_strides=[2, 4, 8, 16, 32],
    target_stride=roi_resize_cfg_stride2["in_stride"],
    output_size=roi_resize_cfg_stride2["output_size"],
    roi_box=roi_resize_cfg_stride2["roi_box"],
    node_name="bev_om_roi_resize_stride2",
)

om_roi_resize_stride4 = dict(
    type="RoiCropResize",
    in_strides=[2, 4, 8, 16, 32],
    target_stride=roi_resize_cfg_stride4["in_stride"],
    output_size=roi_resize_cfg_stride4["output_size"],
    roi_box=roi_resize_cfg_stride4["roi_box"],
    node_name="bev_om_roi_resize_stride4",
)

om_roi_resize = [om_roi_resize_stride2, om_roi_resize_stride4]


# -------------------------- LOSS ---------------------------
# om loss
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
main_cls_loss_weight = 3
sub_cls_loss_weight = 3
head_cls_loss = get_head_cls_loss(
    om_head_groups, main_cls_loss_weight, sub_cls_loss_weight
)
adaptive_loss_weight = True
global_ignore_index = -1
assign_near_instance_cfg = {
    # for ramp(匝道)
    "assign_blur_window": (3, 3),
    "assign_blur_thresh": 0.5,
    # for four lane and double lane
    "four_lane_dist_thresh": 0.45,
    "double_lane_dist_thresh": 1.05,
    # loss weight cfg
    "near_instance_weight": 1.5,
    "lane_ch2_weight": {
        "background": 1.0,
        "dilated": 1.0,
        "foreground": 0.8,
    },
    "ebd_channel_weight": [1.0, 0.3],
}

cpts_subtype_loss_weight = {
    "crosspoints": [3.0, 2.5, 2.0, 3.0, 1.0, 1.0],
    "changepoints": [1.0],
}

cpts_ignore_index = 255
# -------------------------- Metric -------------------------
metrics = []
per_metric_patterns = []

if merge_crosspoint:
    metrics.extend(
        [
            dict(type="LossShow", name="loss_cpts_cls"),
            dict(type="LossShow", name="loss_cpts_x"),
            dict(type="LossShow", name="loss_cpts_y"),
        ]
    )
    per_metric_patterns.extend(
        [
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_head.*loss_cpts_cls$",
            ),
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_head.*loss_cpts_x$",
            ),
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_head.*loss_cpts_y$",
            ),
        ]
    )
if adaptive_loss_weight:
    if merge_crosspoint:
        metrics.extend(
            [
                dict(type="LossShow", name="loss_s_cpts_cls"),
                dict(type="LossShow", name="loss_s_cpts_x"),
                dict(type="LossShow", name="loss_s_cpts_y"),
            ]
        )
        per_metric_patterns.extend(
            [
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{task_name}_head__loss_s_cpts_cls$",
                ),
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{task_name}_head__loss_s_cpts_x$",
                ),
                dict(
                    label_pattern=None,
                    pred_pattern=f"^.*{task_name}_head__loss_s_cpts_y$",
                ),
            ]
        )

metric_updater = dict(
    type="MetricUpdater",
    metrics=metrics,
    metric_update_func=update_metric_using_regex(per_metric_patterns),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)

cls_group_map = {"crosspoints": [0, 1, 2, 3, 4, 5], "changepoints": [6]}
deploy_conf_thresh = {"crosspoints": 0.4, "changepoints": 0.4}
target_categorys = {
    "merge_start": 0,
    "merge_stop": 1,
    "split_start": 2,
    "split_stop": 3,
    "u_turn": 4,
    "other": 5,
    "changepoint": 6,
}
nms_thresh = (2.5, 1.4)

# ----------------------------- DESC ---------------------------
bev_common_desc = dict(
    vcs_origin_coord=vcs_origin_coord,
    bev_stage2_input_resolution=spatial_resolution,
    bev_stage2_output_resolution=om_stage2_output_resolution,
    visible_range=reformat_compile_vcs_range(roi_vcs_range),
    warp_offset_range=reformat_compile_vcs_range(vcs_range),
    vcs_plane_heights=vcs_plane_heights,
)


def get_desc(head_names):
    per_tensor_desc = []
    for head_name in head_names:
        group = head_name.split("_")[-1]
        desc = {}
        desc["task"] = "bev_crosspoint"
        desc["output_name"] = head_name
        if "cls" in head_name:
            desc["num_classes"] = len(cls_group_map[group])
            desc["score_threshold"] = deploy_conf_thresh[group]
            group_cls_index = cls_group_map[group]
            cls_name_list = list(target_categorys.keys())
            desc["properties"] = (
                {
                    "channel_labels": [
                        cls_name_list[i] for i in group_cls_index
                    ]
                },
            )
        if "_x_" in head_name or "_y_" in head_name:
            desc["nms_thresh"] = list(nms_thresh)
        apply_common_desc = copy.deepcopy(bev_common_desc)
        desc.update(**apply_common_desc)
        per_tensor_desc.append(desc)
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


# -------------------------- MODEL --------------------------
def get_inputs():
    inputs = dict(
        om_target=dict(
            lane=dict(
                cls=torch.zeros(1, 2, 166, 128, dtype=torch.int64),
                instance=torch.zeros(1, 2, 166, 128, dtype=torch.int64),
                r=torch.zeros(1, 2, 166, 128, dtype=torch.float64),
                sin=torch.zeros(1, 2, 166, 128, dtype=torch.float64),
                cos=torch.zeros(1, 2, 166, 128, dtype=torch.float64),
                dilate=torch.zeros(1, 2, 166, 128, dtype=torch.int64),
                weight=torch.zeros(1, 2, 166, 128, dtype=torch.float64),
                main_roi_mask=torch.zeros(1, 2, 166, 128, dtype=torch.float64),
                lane_direction=torch.full(
                    (1, 2, 166, 128), -1, dtype=torch.int64
                ),
                lane_color=torch.full((1, 2, 166, 128), -1, dtype=torch.int64),
                lane_wide=torch.full((1, 2, 166, 128), -1, dtype=torch.int64),
                lane_double=torch.full(
                    (1, 2, 166, 128), -1, dtype=torch.int64
                ),
                lane_property=torch.full(
                    (1, 2, 166, 128), -1, dtype=torch.int64
                ),
                lane_dashed=torch.full(
                    (1, 2, 166, 128), -1, dtype=torch.int64
                ),
            ),
            roadedge=dict(
                cls=torch.zeros(1, 166, 128, dtype=torch.int64),
                instance=torch.zeros(1, 166, 128, dtype=torch.int64),
                r=torch.zeros(1, 166, 128, dtype=torch.float64),
                sin=torch.zeros(1, 166, 128, dtype=torch.float64),
                cos=torch.zeros(1, 166, 128, dtype=torch.float64),
                dilate=torch.zeros(1, 166, 128, dtype=torch.int64),
                weight=torch.zeros(1, 166, 128, dtype=torch.float64),
                main_roi_mask=torch.zeros(1, 166, 128, dtype=torch.float64),
                roadedge_subtype=torch.full(
                    (1, 166, 128), -1, dtype=torch.int64
                ),
            ),
        ),
    )
    if merge_crosspoint:
        inputs["crosspoint_target"] = dict(
            crosspoints=dict(
                x=torch.zeros(1, 1, 166, 128, dtype=torch.float64),
                y=torch.zeros(1, 1, 166, 128, dtype=torch.float64),
                cls=torch.zeros(1, 6, 166, 128, dtype=torch.float64),
            ),
            changepoints=dict(
                x=torch.zeros(1, 1, 166, 128, dtype=torch.float64),
                y=torch.zeros(1, 1, 166, 128, dtype=torch.float64),
                cls=torch.zeros(1, 1, 166, 128, dtype=torch.float64),
            ),
        )
    # if om_rpy_transforms:
    #     inputs["aug_transforms"] = None
    return inputs


inputs = get_inputs()
inputs = dict(
    train=inputs,
    val={},
    deploy={},
)


def get_model(mode):
    # assert mode == "deploy", "Bev crosspoint file is only used for compile!!!"

    bev_head = dict(
        type="OutputModule",
        head=dict(
            type="ANCBEVOMHead",
            in_strides=[2],
            select_strides=select_strides,
            out_stride=2,
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
        target=None,
        node_name="bev_stage2_crosspoint_head",
        prefix="crosspoint_head",
    )
    if mode == "train":
        bev_head["loss"] = dict(
            type="ANCOnlineMappingLoss",
            head_groups=om_head_groups,
            head_cls_loss=head_cls_loss,
            instance_loss=dict(
                type="ANCEmbeddingLoss",
                embed_dim=embedding_dim,
                inner_loss_weight=1.5,
                outer_loss_weight=1.5,
            ),
            head_cls_dice_loss=dict(
                type="ANCDiceLoss",
            ),
            reg_loss_weight=1.25 * 1.5,
            adaptive_loss_weight=adaptive_loss_weight,
            phi_loss_proportion_r=True,
            global_ignore_index=global_ignore_index,
            # cpts_loss=cpts_loss if merge_crosspoint else None,
            cpts_loss=dict(
                type="ANCCrossPointLoss",
                group_loss_weight=dict(crosspoints=1.0, changepoints=0.7),
                reg_loss=dict(
                    type="SmoothL1Loss",
                    beta=1.0 / 9.0,
                    reduction="none",
                ),
                cls_loss_weight=1.5,
                focal_loss_beta=0.5,
                reg_loss_weight=1.5,
                subtype_loss_weight=cpts_subtype_loss_weight,
                empty_sample_weight=0.5,
                ignore_index=cpts_ignore_index,
                cpts_dice_loss=True,
            )
            if merge_crosspoint
            else None,
            assign_near_instance_cfg=assign_near_instance_cfg,
        )
    elif mode == "val":
        bev_head["postprocess"] = dict(
            type="ANCBEVCrosspointDecoder",
            stride=8,
            cls_group_map=cls_group_map,
            bev_size=crosspoint_target_size,
            vcs_range=roi_vcs_range,
            ap_score_thresh=cpts_crosspt_metric_cfg["ap_score_thresh"],
            nms_thresh=nms_thresh,
            ignore_index=cpts_ignore_index,
        )
    else:
        bev_head["convert_to_dict"] = False

        head_names = [
            [
                f"pred_crosspoint_cls_{group}",
                f"pred_crosspoint_x_{group}",
                f"pred_crosspoint_y_{group}",
            ]
            for group in cls_group_map
        ]
        head_names = [item for sublist in head_names for item in sublist]

        bev_head["postprocess"] = dict(
            type="MultiInputSequential",
            modules=[
                # # AddDesc should be the last module
                dict(
                    type="AddDesc",
                    per_tensor_desc=get_desc(head_names),
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
            roi_resizes=om_roi_resize,
            head=bev_head,
        ),
        bevfusion_pick_keys=bevfusion_pick_keys,
    )

    return model


# ------------------ DATASET SETTING -------------------
train_data_version = "pipeline_test"
val_data_version = "pipeline_test"

dataset_dir = os.path.join(
    f"{os.path.dirname(__file__)}", "..", "datasets", "bev_om"
)

# om
stride = 8
roi_vcs_range_main = (-32.0, -13.2, 100.8, 13.2)
om_view_bev_size = (
    head_out_size[0] * 8,
    head_out_size[1] * 8,
)
om_roadedge_occ_cfg = {
    "anchor_points": [-2, 2, 6],
    "check_close": False,
    "black_list": ["solid_lanes"],
    "sample_res": 1.0,
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
# crosspoint
cpts_target_categorys = {
    "merge_start": 0,
    "merge_stop": 1,
    "split_start": 2,
    "split_stop": 3,
    "u_turn": 4,
    "other": 5,
    "changepoint": 6,
}
cpts_vismask_vcs_range_cfg = {
    (1024, 1024): (-30.0, -51.2, 72.4, 51.2),  # spatial res is 0.1m/pixel
    (2048, 1536): (-51.2, -76.8, 153.6, 76.8),  # spatial res is 0.1m/pixel
    (2048, 1600): (-51.2, -80.0, 153.6, 80.0),  # spatial res is 0.1m/pixel
    # resize vismask in packed lmdb
    (512, 512): (-30.0, -51.2, 72.4, 51.2),  # spatial res is 0.2m/pixel
    (1024, 768): (-51.2, -76.8, 153.6, 76.8),  # spatial res is 0.2m/pixel
    (1024, 800): (-51.2, -80.0, 153.6, 80.0),  # spatial res is 0.2m/pixel
    (448, 512): (-31.8, -76.8, 102.6, 76.8),
}
cpts_cls_group_diameter = {"crosspoints": (7, 7), "changepoints": (3, 3)}
cpts_use_vismask = True
rpy_pers_aug = False
crosspoint_target_size = (
    head_out_size[0] * stride,
    head_out_size[1] * stride,
)


load_data_types = [
    "timestamp",
    "img_name",
    "gt_online_mapping",
    "pack_dir",
    "img_paths",
    "origin_imgs",
]
if merge_crosspoint:
    load_data_types.append("gt_bev_crosspoint")
    if cpts_use_vismask:
        load_data_types.append("bev_occlusion_mask")
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
    vcs_range_main=roi_vcs_range_main,
    view_bev_size=om_view_bev_size,
    roadedge_occ_cfg=om_roadedge_occ_cfg,
    map_dilate=map_dilate,
    dilate_weight=dilate_weight,
    roi_weight=2 if roi_vcs_range_main is not None else None,
    split_close_roadedge=True,
    view_cols=om_view_cols,
    view_sub_head=om_view_sub_head,
    visualize_output_dir=visualize_output_dir,
    global_ignore_index=global_ignore_index,
    assign_near_instance_cfg=assign_near_instance_cfg,
    block_warp_padding=block_warp_padding,
)
if merge_crosspoint:
    crosspoint_target = dict(
        type="ANCCrossPointTargetGenerator",
        stride=cpts_stride,
        target_categorys=cpts_target_categorys,
        cls_group_map=cpts_cls_group_map,
        vcs_range=roi_vcs_range,
        use_vismask=cpts_use_vismask,
        vismask_vcs_range_cfg=cpts_vismask_vcs_range_cfg,
        bev_size=tuple([i * cpts_stride for i in head_out_size]),
        ignore_index=cpts_ignore_index,
        ignore_filter_length=5.0,
        gaussian_diameter=cpts_cls_group_diameter,
        gaussian_sigma=(1, 1),
        use_om_aux_loss=False,
    )
om_target_list = []
om_target_list.append(om_target)
if merge_crosspoint:
    om_target_list.append(crosspoint_target)

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
    om_rpy_transforms,
]
om_transforms.append(bev_common_transforms["ANCPrepareDataBEV"])

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
task_name_list = [
    task_name + "_" + val_version for val_version in _as_list(val_data_version)
]

if is_local_train:
    save_prefix = "tmp_output"
else:
    save_prefix = "/job_data/models/"
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

cpts_eval_result_dir = f"{eval_root_dir}/{val_ckpt}/cpts_eval_result"
cpts_metric_save_path = f"{eval_root_dir}/cpts_eval_metric.json"
cpts_save_infer_output_dir = f"{eval_root_dir}/{val_ckpt}/cpts_infer_file"

cpts_visualize_output_dir = None

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
        "roi": [-50, 50, -30, 100],
        "host": [-4, 4, -30, 100],
        "host_next": [-8, 8, -30, 100],
        "host_next_-30_0": [-8, 8, -30, 0],
        "host_next_0_100": [-8, 8, 0, 100],
        "host_next_0_25": [-8, 8, 0, 25],
        "host_next_25_50": [-8, 8, 25, 50],
        "host_next_50_100": [-8, 8, 50, 100],
        "y_0_25": [-25, 25, -30, 100],
        "y_25_50": [[-50, -25, -30, 100], [25, 50, -30, 100]],
    },
}

# crosspoint
cpts_subtype_weights = []
for weight in cpts_subtype_loss_weight.values():
    cpts_subtype_weights.extend(weight)

cpts_crosspt_metric_cfg = {
    "dist_threshold": 1.5,
    "dist_percent_thresh": 0.05,  # 5% distance error
    "score_thresh": 0.4,
    "ap_score_thresh": 0.05,
    "region": {
        "all": [-50.0, 50.0, -30.0, 100.0],
        "all-wide-15": [-15, 15, -30.0, 100.0],
        "forward": [-15, 15, 0, 100.0],
        "forward_next": [-8, 8, 0, 100.0],
        "forward-0-20": [-15, 15, 0, 20],
        "forward-20-50": [-15, 15, 20, 50],
        "forward-50-70": [-15, 15, 50, 70],
        "forward-70-100": [-15, 15, 70, 100],
        "backward": [-15, 15, -30.0, 0],
        "backward-15-0": [-15, 15, -15.0, 0],
        "backward-30-15": [-15, 15, -30.0, -15.0],
        "small-range-all": [-10.0, 10.0, -10.0, 20.0],
        "small-range-forward": [-10.0, 10.0, 0, 20.0],
        "small-range-forward-0-10": [-10.0, 10.0, 0, 10.0],
        "small-range-forward-10-20": [-10.0, 10.0, 10.0, 20.0],
        "small-range-backward": [-10.0, 10.0, -10.0, 0],
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
        result_prefix="wide",
    ),
]
if merge_crosspoint:
    val_metrics += [
        dict(
            type="ANCCrossPointMetric",
            name="CrossPoint",
            stride=cpts_stride,
            target_categorys=cpts_target_categorys,
            cls_group_map=cpts_cls_group_map,
            metric_save_path=cpts_metric_save_path,
            subtype_weights=cpts_subtype_weights,
            eval_result_dir=cpts_eval_result_dir,
            save_infer_output_dir=cpts_save_infer_output_dir,
            visualize_output_dir=cpts_visualize_output_dir,
            metric_cfg=cpts_crosspt_metric_cfg,
            vcs_range=roi_vcs_range,  # (bottom, right, top, left)
            bev_size=om_view_bev_size,
            image_size=(320, 512),
            ignore_index=cpts_ignore_index,
            # multi_views_type=map_views_to_visual_type(views_dist),
            lane_gt_dir=None,
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
if merge_crosspoint:
    cpts_label_pattern_list = [
        "^.*online_mapping_image_files",
        "^.*online_mapping_origin_imgs",
        "^.*online_mapping_crosspoint_target$",
    ]
    val_per_metric_patterns.append(
        dict(
            label_pattern=cpts_label_pattern_list,
            pred_pattern=f"^.*{task_name}_head_predict_crosspt.*",
        )
    )


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
    type="SaveCrossPointConsistencyResult",
    output_dir=os.path.join(save_prefix, "inference", "bev_crosspoint"),
    output_key="crosspoint_head_predict_crosspt_predict",
    cls_score_threshold=deploy_conf_thresh,
)

# --------------------------- PACK VIS ----------------------------
visualize_callbacks = [
    dict(
        type="ANCCrossPointVisualize",
        output_dir=os.path.join(save_prefix, "visualize", "bev_crosspoint"),
        task="bev_crosspoint",
        prefix="crosspoint_head_predict_crosspt_predict",
        stride=8,
        target_categorys=cpts_target_categorys,
        bev_size=om_view_bev_size,
        vcs_range=roi_vcs_range,
        extra_img=dict(
            type="ANCBEVImgStitcher",
            per_extra_img_size=(960, 512),
            camera_view_names=camera_view_names,
            camera_layouts=None,
        ),
    )
]
