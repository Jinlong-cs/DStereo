import numpy as np
import torch

from hat.data.transforms.auto_3dv import ANCOnlineMappingTargetGenerator
from hat.models.task_modules.bev.online_mapping_loss import (
    ANCEmbeddingLoss,
    ANCOnlineMappingLoss,
)
from hat.registry import build_from_registry

assign_near_instance_cfg = {
    "assign_blur_window": (3, 3),
    "assign_blur_thresh": 0.6,
    "multi_ins_grid_thresh": 6,
    "multi_lane_ignore_thresh": 8,
    "near_instance_weight": 1.4,
    "lane_ch2_weight": {
        "background": 1.0,
        "dilated": 1.0,
        "foreground": 0.95,
    },
    "ebd_channel_weight": [1.0, 0.3],
}
om_roadedge_occ_cfg = {
    "anchor_points": [-2, 1, 4],
    "check_close": False,
    "black_list": [],
    "sample_res": 0.5,
}
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


def get_head_cls_loss(head_groups, main_cls_loss_weight, sub_cls_loss_weight):
    head_cls_loss = {}
    for head, head_infos in head_groups.items():
        group = head_infos["group"]
        group_head = head_groups[group]
        loss_type = group_head.get("loss_type", "ce_loss")
        max_patch_segment = group_head.get("max_patch_segment", 1)
        assert max_patch_segment == 1 or max_patch_segment == 2

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
                weight_noobj=1,
                num_class=num_class,
                ignore_index=ignore_index,
            )
        elif loss_type == "focal_loss":
            cls_loss = dict(
                type="OMFocalLoss",
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
        head_cls_loss[head] = cls_loss

    return head_cls_loss


def get_fake_input(embedding_dim=4):
    bs = 1
    head_out_size = [64, 64]
    view_bev_size = (512, 512)
    vcs_range = (-30.0, -51.2, 72.4, 51.2)

    # get fake prediction
    out_nums_list = []
    output_name_list = []
    multi_head_keys = ["cls", "instance", "r", "sin", "cos"]
    om_out_h, om_out_w = head_out_size[:2]
    for head, head_infos in om_head_groups.items():
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
            head_dims = [v * max_patch_segment for v in head_dims]
        else:
            head_names = [f"pred_online_mapping_{head}_{group}"]
            head_dims = [out_channels * max_patch_segment]
        out_nums_list += head_dims
        output_name_list += head_names

    om_predict = {}
    for head_name, dim in zip(output_name_list, out_nums_list):
        om_predict[head_name + "_frame0"] = [
            torch.ones([bs, dim, om_out_h, om_out_w])
        ]

    # get fake target
    data = {
        "pack_dir": "pack",
        "img_paths": "om.jpg",
        "gt_online_mapping": {
            "data_version": "v1.0",
            "solid_lanes": [
                {
                    "lane_type": "solid",
                    "lane_direction": "unidirectional",
                    "lane_flag": "single",
                    "lane_width": "normal",
                    "lane_color": "unknown",
                    "lane_properties": "unknown",
                    "pts": np.array(
                        [[197, 15, 0, 145, 15, 0], [145, 15, 0, 117, 15, 0]],
                        np.float64,
                    ),
                }
            ],
            "roadedges": [
                {
                    "curb_category": "unknown",
                    "pts": np.array(
                        [[36, -71, 0, 35, -26, 0], [35, -26, 0, 34, -21, 0]],
                        np.float64,
                    ),
                }
            ],
        },
        "homography": torch.randn((1, 6, 3, 3)),
    }

    om_target_generator = ANCOnlineMappingTargetGenerator(
        out_size=head_out_size,
        head_groups=om_head_groups,
        vcs_range=vcs_range,
        view_bev_size=view_bev_size,
        roadedge_occ_cfg=om_roadedge_occ_cfg,
        visualize_output_dir=None,
        assign_near_instance_cfg=assign_near_instance_cfg,
    )
    om_gt = om_target_generator(data)

    for group in ["lane", "roadedge"]:
        for k, v in om_gt["om_target"][group].items():
            if v.shape[0] == 2:
                v = v[None]
            om_gt["om_target"][group][k] = torch.from_numpy(v)

    return om_predict, om_gt


def test_om_loss():
    main_cls_loss_weight = 2.0
    sub_cls_loss_weight = 2.0
    om_embedding_inner_loss_weight = 2.0
    om_embedding_outer_loss_weight = 2.0
    embedding_dim = 4
    reg_loss_weight = 1.5
    adaptive_loss_weight = True
    phi_loss_proportion_r = True
    global_ignore_index = -1

    head_cls_loss = get_head_cls_loss(
        om_head_groups, main_cls_loss_weight, sub_cls_loss_weight
    )
    for key, value in head_cls_loss.items():
        head_cls_loss[key] = build_from_registry(value)

    loss = ANCOnlineMappingLoss(
        head_groups=om_head_groups,
        head_cls_loss=head_cls_loss,
        instance_loss=ANCEmbeddingLoss(
            embed_dim=embedding_dim,
            inner_loss_weight=om_embedding_inner_loss_weight,
            outer_loss_weight=om_embedding_outer_loss_weight,
        ),
        reg_loss_weight=reg_loss_weight,
        adaptive_loss_weight=adaptive_loss_weight,
        phi_loss_proportion_r=phi_loss_proportion_r,
        global_ignore_index=global_ignore_index,
        assign_near_instance_cfg=assign_near_instance_cfg,
    )

    pred, target = get_fake_input()
    loss(pred, target)
