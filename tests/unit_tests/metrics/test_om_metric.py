import numpy as np
import torch

from hat.metrics.bev.online_mapping import ANCOnlineMappingMetric
from hat.metrics.bev.online_mapping_utils import (
    get_target_categorys,
    get_vcs_lane,
)

om_head_groups = {
    "lane": {
        "multi": True,
        "group": "lane",
        "key": "solid_lanes#lane_type",
        "loss_weight": 3,
        "cls_remap": {
            "solid": 1,
            "dotted": 2,
            "dashed": 2,
            "solid_fishbone": 1,
            "ldotted_fishbone": 2,
            "boots_dots": 2,
            "other": 3,
            "unknown": 3,
        },
        "cls_list": ["solid", "dashed", "other"],
        "cls_colors": ["red", "blue", "green"],
        "cls_thr": 0.5,
    },
    "roadedge": {
        "multi": True,
        "group": "roadedge",
        "key": "roadedges#curb_category",
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
        "cls_colors": ["green"],
        "cls_thr": 0.5,
    },
    "lane_direction": {
        "group": "lane",
        "loss_weight": 2.0,
        "add_cost": True,
        "auto_class_weight": False,
        "global_auto_class_weight": [0.75, 2.0],
        "key": "solid_lanes#lane_direction",
        "cls_remap": {"unidirectional": 0, "bidirectional": 4},
        "cls_list": ["up", "down", "left", "right", "bidirection"],
        "cls_colors": ["white", "red", "green", "blue", "yellow"],
    },
    "roadedge_subtype": {
        "group": "roadedge",
        "loss_type": "ce_loss",
        "key": "roadedges#curb_category",
        "loss_weight": 2.0,
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
        "cls_colors": ["white", "red", "blue", "orange", "green", "yellow"],
    },
    "lane_color": {
        "group": "lane",
        "key": "solid_lanes#lane_color",
        "loss_weight": 2.0,
        "add_cost": True,
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
    },
    "lane_wide": {
        "group": "lane",
        "key": "solid_lanes#lane_width",
        "loss_weight": 2.0,
        "add_cost": True,
        "auto_class_weight": False,
        "global_auto_class_weight": [1.0, 3.0],
        "cls_remap": {"normal": 0, "wide": 1},
        "cls_colors": ["red", "blue"],
    },
    "lane_double": {
        "group": "lane",
        "key": "solid_lanes#lane_flag",
        "loss_weight": 2.0,
        "add_cost": True,
        "auto_class_weight": False,
        "global_auto_class_weight": [1.0, 2.0],
        "cls_remap": {"single": 0, "double": 1, "triple": 1},
        "cls_list": ["single", "double"],
        "cls_colors": ["red", "blue"],
    },
    "lane_property": {
        "group": "lane",
        "key": "solid_lanes#lane_properties",
        "loss_weight": 1.0,
        "add_cost": True,
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
        "cls_colors": ["white", "red", "blue", "orange", "green", "yellow"],
    },
}


def get_fake_inputs(head_groups, out_h, out_w):
    vcs_range = (-30.0, -51.2, 72.4, 51.2)
    bottom, right, top, left = vcs_range
    scope_h = top - bottom
    scope_w = left - right
    res_h = scope_h / out_h
    res_w = scope_w / out_w
    target_categorys = get_target_categorys(head_groups)

    pred_attr2shape = {
        "prob": [1, out_h, out_w],
        "cls": [1, out_h, out_w],
        "instance": [1, out_h, out_w],
        "r": [1, out_h, out_w],
        "sin": [1, out_h, out_w],
        "cos": [1, out_h, out_w],
        "embedding": [1, out_h, out_w, 4],
        "offset": [1, out_h, out_w, 2],
        "direction": [1, out_h, out_w],
        "lane_direction": [1, out_h, out_w, 5],
    }

    image_files = [
        [
            "root/site/pack/xxx.jpg",
            "root/site/pack/xxx.jpg",
            "root/site/pack/xxx.jpg",
            "root/site/pack/xxx.jpg",
            "root/site/pack/xxx.jpg",
            "root/site/pack/xxx.jpg",
        ]
    ]

    origin_imgs = [[np.zeros([1, out_h, out_w, 3]) for _ in image_files[0]]]

    gt_stats = {}
    for head, head_infos in head_groups.items():
        group = head_infos["group"]
        group_head = head_groups[group]
        max_patch_segment = group_head.get("max_patch_segment", 1)
        if max_patch_segment == 2:
            head_shape = (1, 2, out_h, out_w)
        else:
            head_shape = (1, out_h, out_w)
        multi_head = head_infos.get("multi", False)
        if group not in gt_stats:
            gt_stats[group] = {}
        if multi_head:
            gt_stats[group]["cls"] = torch.zeros(head_shape, dtype=np.int)
            gt_stats[group]["instance"] = torch.zeros(
                head_shape, dtype=np.float
            )
            gt_stats[group]["r"] = torch.zeros(head_shape, dtype=np.float)
            gt_stats[group]["sin"] = torch.zeros(head_shape, dtype=np.float)
            gt_stats[group]["cos"] = torch.zeros(head_shape, dtype=np.float)
        else:
            # if has background cls
            gt_stats[group][head] = torch.full(head_shape, -1, dtype=np.int)

    # generate fake results data
    pred_stats = {}
    for head, head_infos in om_head_groups.items():
        group = head_infos["group"]
        group_head = om_head_groups[group]
        max_patch_segment = group_head.get("max_patch_segment", 1)
        multi_head = head_infos.get("multi", False)
        group = head_infos["group"]
        if group not in pred_stats:
            pred_stats[group] = {}
        if multi_head:
            for attr, shape in pred_attr2shape.items():
                if shape is not None:
                    shape[1] = shape[1] * max_patch_segment
                    value = np.zeros(shape, dtype=np.float)
                else:
                    value = np.array([[None]])
                pred_stats[group][attr] = value
        else:
            shape = pred_attr2shape["cls"]
            shape[1] = shape[1] * max_patch_segment
            for attr in ["cls", "prob"]:
                key = f"{head}_{attr}"
                value = np.zeros(shape, dtype=np.float)
                pred_stats[group][key] = value

    pred_lanes_raw, pred_lanes_seq, _ = get_vcs_lane(
        pred_stats,
        head_groups=om_head_groups,
        out_h=out_h,
        out_w=out_w,
        top=top,
        left=left,
        res_h=res_h,
        res_w=res_w,
        target_categorys=target_categorys,
        embedding_dim=4,
        use_seq_post=False,
    )
    pred_stats_gather = [
        {
            "pred_lanes_raw": pred_lanes_raw,
            "pred_lanes_seq": pred_lanes_seq,
            "pred_stats": pred_stats,
        }
    ]

    inputs = []
    inputs.append(image_files)
    inputs.append(origin_imgs)
    inputs.append(gt_stats)
    inputs.append(pred_stats_gather)
    return inputs


def test_om_metric():
    # update metric names
    metric_state_names = [
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

    for head, head_infos in om_head_groups.items():
        multi_head = head_infos.get("multi", False)
        if not multi_head:
            metric_state_names += [
                f"gt_{head}_cls_inst",
                f"pred_{head}_cls_inst",
                f"gt_{head}_cls_pts",
                f"pred_{head}_cls_pts",
            ]

    om_metric = ANCOnlineMappingMetric(
        name="OnlineMapping",
        metric_state_names=metric_state_names,
        embedding_dim=4,
        head_groups=om_head_groups,
        eval_result_file=None,
        save_infer_output_dir=None,
        save_raw_model_output=None,
        visualize_infer_output_dir=None,
        visualize_eval_output_dir=None,
        vcs_range=(-30.0, -51.2, 72.4, 51.2),
        out_size=(64, 64),
        view_bev_size=(512, 512),
        image_size=(320, 512),
        cd_threshold=0.5,
        do_eval=True,
        analysis=False,
    )

    inputs = get_fake_inputs(om_head_groups, 64, 64)
    om_metric.update(*inputs)
