import copy
import os

import cv2
import numpy as np
import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.models.losses.real3d_losses import sigmoid_and_clip
from projects.pilot.configs.bev_7v_temporal.base import remove_none
from projects.pilot.configs.bev_7v_temporal.common import (
    block_warp_padding,
    common_transforms,
    ipm_output_size,
    model_thresh,
    save_prefix,
    vcs_plane_heights,
    vcs_range,
    views_dist,
)

# --------------------------CCONFIG SETTING ---------------------


bev_discobj_task_list = [
    "bev_arrow",
    "bev_crosswalk",
    "bev_stopline",
    "bev_junction",
    "bev_roadmarking",
    "bev_static_obstacle",
    "bev_sod3d",
]

task_list_for_head_output_cfg = [
    "bev_arrow",
    "bev_crosswalk",
    "bev_stopline",
    "bev_junction",
    "bev_roadmarking",
    "bev_psd",
    "bev_static_obstacle",
    "bev_parkingrod",
    "bev_sod3d",
]

class_valid_vcs_range = (
    {"bev_arrow": None, "bev_junction": None, "bev_roadmarking": None},
)


def update_task_attr_config(attr_config, base_config, task_list):
    for task_name in task_list:
        task_base_config = copy.deepcopy(base_config)
        if attr_config.get(task_name, None) is not None:
            if isinstance(task_base_config, dict):
                task_base_config.update(attr_config[task_name])
            else:
                task_base_config = attr_config[task_name]
        attr_config[task_name] = task_base_config
    return attr_config


# discobj_decode_setting = {
#     "bev_arrow": {
#         "topk": 20,
#         "max_pool_kernel": 7,
#         "do_ct_nms": True,
#         "ct_dist_scale": 1.0,
#     },
#     "bev_junction": {
#         "topk": 20,
#         "max_pool_kernel": 15,
#         "do_ct_nms": True,
#         "ct_dist_scale": 0.5,
#         "class_ids": [1],
#         "rot_match_threshold": 0.5,
#         "kernel_from_singlebox": False,
#     },
#     "bev_roadmarking": {
#         "topk": 20,
#         "max_pool_kernel": 9,
#         "do_ct_nms": True,
#         "ct_dist_scale": 0.5,
#         "class_ids": [4],
#         "rot_match_threshold": 0.5,
#         "kernel_from_singlebox": False,
#     },
# }

# discobj_metric_setting = {
#     "bev_arrow": {
#         "score_threshold": model_thresh.get("bev_arrow", {}).get(
#             "score_threshold", 0.2
#         )
#         if model_thresh
#         else 0.2,
#         "iou_threshold": model_thresh.get("bev_arrow", {}).get(
#             "iou_threshold", 0.2
#         )
#         if model_thresh
#         else 0.2,
#         "yaw_amplitude": 360,
#         "gt_max_depth": 55,
#     },
#     "bev_junction": {
#         "score_threshold": model_thresh.get("bev_junction", {}).get(
#             "score_threshold", 0.2
#         )
#         if model_thresh
#         else 0.2,
#         "iou_threshold": model_thresh.get("bev_junction", {}).get(
#             "iou_threshold", 0.2
#         )
#         if model_thresh
#         else 0.2,
#         "yaw_amplitude": 180,
#         "gt_max_depth": 105,
#     },
#     "bev_roadmarking": {
#         "score_threshold": model_thresh.get("bev_roadmarking", {}).get(
#             "score_threshold", 0.2
#         )
#         if model_thresh
#         else 0.2,
#         "iou_threshold": model_thresh.get("bev_roadmarking", {}).get(
#             "iou_threshold", 0.2
#         )
#         if model_thresh
#         else 0.2,
#         "yaw_amplitude": 180,
#         "gt_max_depth": 55,
#         "gt_match_mode": {"Stopline": "ct_rot", "SpeedBump": "ct_rot"},
#     },
# }

name2group = {
    "arrows": "det",
    "crosswalks": "det",
    "speedbumps": "det",
    "noparking_lines": "det",
    "diamond_markings": "det",
    "inverted_triangle_markings": "det",
    "exclusive_lane_signs": "det",
    "parking_locks": "det",
    "comment_columns": "det",
    "intersections": "det",
    "stoplines": "det",
}

# -------------------------- data --------------------------

# set transforms in bev_task config
load_data_types = [
    "timestamp",
    "img_name",
    "gt_bev_discrete_obj",
    "pose",
]
bev_common_transforms = copy.deepcopy(common_transforms)
collect_3dv = bev_common_transforms["ANCCollect3DV"]
collect_3dv["gt_bev_discobj_idx"] = 0
collect_3dv["load_data_types"] = load_data_types

val_common_transforms = dict(
    type="ANCVisualizeIpm",
    bev_size=ipm_output_size,
    view_idxs=[0, 1, 2, 3, 4, 5],
    img_scale=1,
    block_warp_padding=block_warp_padding,
    meta_name="meta_info",
    enable_vis=True,
    vcs_plane_heights=vcs_plane_heights,
    return_name="ipm",
)


# Note all raw gt saved with 0.1m/pixel spatial resolution
# config fot task raw resolution-vcs_range pairs,
# resolution as key, corresponding vcs range as values.
# resolution in (h, w) order, vcs range in (bottom, right, top, left)
# different saved resolution corresponding to data used
# in different period.
vismask_vcsrange_cfg = {
    (192, 128): (-12.80, -12.80, 25.60, 12.80),  # spatial res is 0.2m/pixel
    (448, 512): (-31.8, -76.8, 102.6, 76.8),  # spatial res is 0.6m/pixel
    (512, 512): (-30.0, -51.2, 72.4, 51.2),  # spatial res is 0.1m/pixel
    (1024, 1024): (-30.0, -51.2, 72.4, 51.2),  # spatial res is 0.1m/pixel
    (2048, 1536): (-51.2, -76.8, 153.6, 76.8),  # spatial res is 0.1m/pixel
    (2048, 1600): (-51.2, -80.0, 153.6, 80.0),  # spatial res is 0.1m/pixel
    # resize vismask in packed lmdb
    (1024, 768): (-51.2, -76.8, 153.6, 76.8),  # spatial res is 0.2m/pixel
    (1024, 800): (-51.2, -80.0, 153.6, 80.0),  # spatial res is 0.2m/pixel
}

load_data_types.append("bev_occlusion_mask")

load_data_types = remove_none(load_data_types)


def get_inputs(num_views, vcs_plane_heights, task_out_size, num_classes=1):
    inputs = dict(
        timestamp=torch.randn((1, 1)),
        view=views_dist,
        temporal_info=None,
        meta_info=dict(
            T_vcs2cam=[torch.zeros(1, 1, 4, 4, dtype=torch.float32)]
            * num_views,
            intrinsics=[torch.zeros(1, 1, 3, 3, dtype=torch.float32)]
            * num_views,
            distort_coeffs=[torch.zeros(1, 8, dtype=torch.float32)]
            * num_views,
            transformats=[torch.zeros(1, 1, 3, 3, dtype=torch.float32)]
            * num_views,
            ipm_img_sizes=[torch.zeros(1, 2, dtype=torch.float32)] * num_views,
            img_shape=[torch.zeros(1, 1, 2, dtype=torch.float32)] * num_views,
            fake_homo_flag=torch.zeros(
                1, len(vcs_plane_heights) * num_views, dtype=torch.float32
            ),
            aug_flag=[False],
            homo_offset=torch.randn(
                (num_views * len(vcs_plane_heights),) + ipm_output_size + (2,)
            ),
            homography=torch.randn(
                (1, num_views * len(vcs_plane_heights), 3, 3)
            ),
        ),
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
                2,
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
                3,
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
    val_inputs = {}
    deploy_inputs = {}
    return inputs, val_inputs, deploy_inputs


# -------------------------- solver --------------------------
def get_metrics_patterns(
    task_name,
    head_with_cls=False,
):
    metrics = [
        dict(type="LossShow", name=f"{task_name}_hm_loss"),
        dict(type="LossShow", name=f"{task_name}_wh_loss"),
        dict(type="LossShow", name=f"{task_name}_rot_loss"),
        dict(type="LossShow", name=f"{task_name}_ct_offset_loss"),
    ]

    per_metric_patterns = [  # corresponding to metrics
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}.*loss_bev_discobj_hm$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}.*loss_bev_discobj_wh$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}.*loss_bev_discobj_rot$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}.*loss_bev_discobj_ct_offset$",
        ),
    ]

    if head_with_cls:
        metrics.insert(
            1, dict(type="LossShow", name=f"{task_name}_hm_cls_loss")
        )
        metrics.insert(
            1, dict(type="LossShow", name=f"{task_name}_hm_cls_aux_loss")
        )
        per_metric_patterns.insert(
            1,
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}.*loss_bev_discobj_hm_cls$",
            ),
        )
        per_metric_patterns.insert(
            1,
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}.*loss_bev_discobj_hm_cls_aux$",
            ),
        )

    return metrics, per_metric_patterns


def get_train_metric_updater(
    task_name,
    metrics,
    per_metric_patterns,
    log_freq,
):
    metric_updater = dict(
        type="MetricUpdater",
        metrics=metrics,
        metric_update_func=update_metric_using_regex(
            per_metric_patterns=per_metric_patterns
        ),
        step_log_freq=log_freq,
        epoch_log_freq=1,
        log_prefix=task_name,
        reset_metrics_by="log",
    )
    return metric_updater


# update depth_intervals_setting
base_depth_intervals_setting = (-20, -10, 0, 10, 20, 30, 40, 50, 70)

arrow_depth_intervals = (-20, -10, 0, 10, 20, 30, 40)
junction_depth_intervals = (-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
roadmarking_depth_intervals = (-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
static_obstacle_depth_intervals = base_depth_intervals_setting
sod3d_depth_intervals = base_depth_intervals_setting
eval_depth_intervals = {
    "Other": arrow_depth_intervals,
    "Straight": arrow_depth_intervals,
    "Turnleft": arrow_depth_intervals,
    "Turnright": arrow_depth_intervals,
    "Turnoff": arrow_depth_intervals,
    "Mergeleft": arrow_depth_intervals,
    "Mergeright": arrow_depth_intervals,
    "No_Straight": arrow_depth_intervals,
    "No_Turnleft": arrow_depth_intervals,
    "No_Turnright": arrow_depth_intervals,
    "No_Turnoff": arrow_depth_intervals,
    "No_Mergeleft": arrow_depth_intervals,
    "No_Mergeright": arrow_depth_intervals,
    "Straight_Turnleft": arrow_depth_intervals,
    "Straight_Turnright": arrow_depth_intervals,
    "Turnleft_Turnright": arrow_depth_intervals,
    "Turnleft_Turnoff": arrow_depth_intervals,
    "Straight_Turnoff": arrow_depth_intervals,
    "Straight_Turnright_Turnleft": arrow_depth_intervals,
    "Crosswalk": roadmarking_depth_intervals,
    "Stopline": arrow_depth_intervals,
    "NoParkingLine": arrow_depth_intervals,
    "DiamondMarking": arrow_depth_intervals,
    "InvertedTriangleMarking": arrow_depth_intervals,
    "SpeedBump": arrow_depth_intervals,
    "ExclusiveLaneSign": arrow_depth_intervals,
    "Junction": junction_depth_intervals,
    "CementColumn": static_obstacle_depth_intervals,
    "ParkingLock_Open": static_obstacle_depth_intervals,
    "ParkingLock_Close": static_obstacle_depth_intervals,
    "Cone": sod3d_depth_intervals,
}

base_eval_vcs_range = (vcs_range[0], vcs_range[1], vcs_range[2], vcs_range[3])

old_arrow_eval_vcs_range = (-32.0, -8.0, 52.8, 8.0)
arrow_eval_vcs_range = (0.0, -8.0, 52.8, 8.0)
junction_eval_vcs_range = (-32.0, -51.2, 102.0, 51.2)
roadmarking_eval_vcs_range = (-32.0, -51.2, 102.0, 51.2)
static_obstacle_eval_vcs_range = base_eval_vcs_range
sod3d_eval_vcs_range = base_eval_vcs_range
noparking_eval_vcs_range = (-32.0, -33.0, 52.8, 33.0)

apply_eval_vcs_range = {
    "Other": arrow_eval_vcs_range,
    "Straight": arrow_eval_vcs_range,
    "Turnleft": arrow_eval_vcs_range,
    "Turnright": arrow_eval_vcs_range,
    "Turnoff": arrow_eval_vcs_range,
    "Mergeleft": arrow_eval_vcs_range,
    "Mergeright": arrow_eval_vcs_range,
    "No_Straight": arrow_eval_vcs_range,
    "No_Turnleft": arrow_eval_vcs_range,
    "No_Turnright": arrow_eval_vcs_range,
    "No_Turnoff": arrow_eval_vcs_range,
    "No_Mergeleft": arrow_eval_vcs_range,
    "No_Mergeright": arrow_eval_vcs_range,
    "Straight_Turnleft": arrow_eval_vcs_range,
    "Straight_Turnright": arrow_eval_vcs_range,
    "Turnleft_Turnright": arrow_eval_vcs_range,
    "Turnleft_Turnoff": arrow_eval_vcs_range,
    "Straight_Turnoff": arrow_eval_vcs_range,
    "Straight_Turnright_Turnleft": arrow_eval_vcs_range,
    "Crosswalk": roadmarking_eval_vcs_range,
    "NoParkingLine": noparking_eval_vcs_range,
    "ExclusiveLaneSign": arrow_eval_vcs_range,
    "Stopline": old_arrow_eval_vcs_range,
    "DiamondMarking": old_arrow_eval_vcs_range,
    "InvertedTriangleMarking": old_arrow_eval_vcs_range,
    "SpeedBump": old_arrow_eval_vcs_range,
    "Junction": junction_eval_vcs_range,
    "ParkingLock_Open": static_obstacle_eval_vcs_range,
    "ParkingLock_Close": static_obstacle_eval_vcs_range,
    "CementColumn": static_obstacle_eval_vcs_range,
    "Cone": sod3d_eval_vcs_range,
}


def get_val_metrics(
    task_name,
    id2label,
    eval_category_ids,
    score_threshold,
    gt_max_depth=100,
    metrics=("dx", "dy", "dxy", "dw", "dh", "drot"),
    save_dir=None,
    save_eval_result_path=None,
    match_mode=None,
    ct_match_mode="percentage",
    yaw_amplitude=180,
    ap_score_threshold=0.0,
    iou_threshold=0.2,
    compute_foreground_prec=False,
    vis_image_dir=None,
    annos_key="annos_bev_discrete_obj",
    result_prefix="wide",
):
    val_metrics = [
        dict(
            type="ANCBEVDiscreteObjectEval",
            name=task_name,
            annos_key=annos_key,
            id2label=id2label,
            eval_category_ids=eval_category_ids,
            depth_intervals=eval_depth_intervals,
            eval_vcs_range=apply_eval_vcs_range,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            gt_max_depth=gt_max_depth,
            metrics=metrics,
            save_dir=save_dir,
            save_eval_result_path=save_eval_result_path,
            match_mode=match_mode,
            ct_match_mode=ct_match_mode,
            yaw_amplitude=yaw_amplitude,
            ap_score_threshold=ap_score_threshold,
            dist_intervals=None,  # noqa
            ct_matching_thresholds=None,
            compute_foreground_prec=compute_foreground_prec,
            result_prefix=result_prefix,
            vis_image_dir=vis_image_dir,
            upload_image_2_aidi=False,
        )
    ]

    return val_metrics


# ----------------------------visualize----------------------------
COLOR_MAP = {  # BGR
    "crosswalk": {0: (128, 64, 128)},
    "stopline": {0: (140, 150, 230)},
    "arrow": {
        0: (128, 64, 128),
        1: (200, 200, 128),
        2: (230, 150, 140),
        3: (18, 145, 170),
        4: (0, 0, 230),
        5: (220, 220, 0),
        6: (220, 20, 60),
        7: (70, 130, 180),
        8: (0, 0, 110),
        9: (0, 80, 100),
        10: (190, 153, 153),
        11: (224, 35, 232),
        12: (70, 70, 70),
        13: (129, 187, 89),
        14: (153, 153, 153),
        15: (230, 123, 34),
        16: (34, 237, 242),
        17: (102, 102, 156),
        18: (150, 100, 100),
    },
    "junction": {
        0: (230, 0, 0),
        1: (130, 0, 100),
    },
    "roadmarking": {
        0: (0, 220, 220),  # crosswalk
        1: (60, 220, 20),  # stopline
        2: (180, 70, 130),  # DiamondMarking
        3: (110, 0, 0),  # InvertedTriangleMarking
        4: (100, 0, 80),  # SpeedBump
        5: (153, 190, 153),  # "NoParkingLine",
        6: (232, 244, 35),  # "ExclusiveLaneSign",
    },
    "static_obstacle": {
        0: (100, 0, 80),  # OpenParkingLock
        1: (150, 50, 150),  # CloseParkingLock
        2: (0, 128, 128),  # CementColumn
    },
    "sod3d": {0: (128, 64, 128)},
    "cementcolumn": {
        0: (0, 128, 128),
    },
    "parkinglock": {
        0: (100, 0, 80),  # OpenParkingLock
        1: (150, 50, 150),  # CloseParkingLock
    },
    "psd": {
        0: (255, 0, 0),
        1: (0, 200, 200),
        2: (128, 0, 128),
    },
    "parkingrod": {
        0: (0, 255, 0),
    },
    "parking": {
        0: (255, 0, 0),
        1: (0, 200, 200),
        2: (128, 0, 128),
    },
}

ID2LABEL = {
    "arrow": {
        0: "O",
        1: "S",
        2: "L",
        3: "R",
        4: "TO",
        5: "ML",
        6: "MR",
        7: "NS",
        8: "NL",
        9: "NR",
        10: "NTO",
        11: "NML",
        12: "NMR",
        13: "SL",
        14: "SR",
        15: "LR",
        16: "LT",
        17: "STO",
        18: "SLR",
    },
    "junction": {
        0: "junction",
        1: "crosswalk",
    },
    "roadmarking": {
        0: "Crosswalk",
        1: "Stopline",
        2: "Diamond",
        3: "ITriangle",
        4: "SpeedBump",
        5: "NoParking",
        6: "ExclusiveSign",
    },
    "static_obstacle": {
        0: "Open_PL",
        1: "Close_PL",
        2: "CColumn",
    },
    "sod3d": {0: "Cone"},
    "psd": {
        0: "Vertical",
        1: "Parallel",
        2: "Oblique",
    },
    "parkingrod": {
        0: "parkingrod",
    },
    "parking": {
        0: "Vertical",
        1: "Parallel",
        2: "Oblique",
    },
}


# -------------------------- tensorboard --------------------------
def apply_color(data, norm_with_minmax=True, cmp=cv2.COLORMAP_JET):
    if norm_with_minmax:
        data = (data - data.min()) / (data.max() - data.min()) * 255
    else:
        data *= 255
    data = data.astype("uint8")
    data = cv2.applyColorMap(data, cmp)[:, :, [2, 1, 0]]
    return data


def get_bev_tb_update_func(
    task_name,
    pred_hm_key_regex,
    gt_hm_key_regex,
):
    def bev_tb_update_func(writer, model_outs, global_step_id, **kwargs):
        # model_outs = to_flat_ordered_dict(model_outs)
        if task_name in list(model_outs.keys())[0]:
            for k, v in model_outs.items():
                if pred_hm_key_regex.match(k):
                    hm = (
                        sigmoid_and_clip(v.detach().cpu()).numpy()[0].squeeze()
                    )
                    if hm.ndim > 2:
                        data_vis = apply_color(
                            np.sum(hm, axis=0), norm_with_minmax=False
                        )
                    else:
                        data_vis = apply_color(hm, norm_with_minmax=False)
                    writer.add_images(
                        k, data_vis, global_step_id, dataformats="HWC"
                    )
                if gt_hm_key_regex.match(k):
                    hm = v.detach().cpu().numpy()[0].squeeze()
                    data_vis = apply_color(hm, norm_with_minmax=False)
                    writer.add_images(
                        k, data_vis, global_step_id, dataformats="HWC"
                    )

    return bev_tb_update_func


# -------------------------- PACK_INFER_SAVE --------------------------
task_res_key_cfg = {
    "bev_arrow": ["arrow"],
    "bev_junction": ["junction"],
    "bev_roadmarking": ["roadmarking"],
}
score_thr_cfg = {
    "bev_arrow": model_thresh.get("bev_arrow", {}).get("score_threshold", 0.2),
    "bev_junction": model_thresh.get("bev_junction", {}).get(
        "score_threshold", 0.2
    ),
    "bev_roadmarking": model_thresh.get("bev_roadmarking", {}).get(
        "score_threshold", 0.2
    ),
}
save_pack_infer = dict(
    type="SaveDiscObjConsistencyResult",
    output_dir=os.path.join(save_prefix, "inference", "bev_disc_obj"),
    prefix="OutputModule_predict",
    task_res_key_cfg=task_res_key_cfg,
    scor_thr_cfg=score_thr_cfg,
    merge_crosswalk_to_junction=True,
)
