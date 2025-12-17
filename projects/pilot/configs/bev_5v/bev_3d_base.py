import copy
import os

import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.models.losses.real3d_losses import sigmoid_and_clip
from hat.utils.apply_func import _as_list
from projects.pilot.configs.bev_5v.base import remove_none
from projects.pilot.configs.bev_5v.common import (
    bevfusion_output_size,
    common_transforms,
    log_freq,
    save_prefix,
    vcs_range,
)

# test_image_dir = ""
# test_image_calibration = ""
# test_image_dist_coeffs = ""
# test_attribte_json_path = ""
# test_batch_size_per_gpu = 1
# test_homo_path = ""

# --------------------------BEV3D BASE --------------------------
use_multi_head = False
use_stage1_loss = False
use_distorted_offset = True
# assert use_multi_head, "current bev only support use_multi_head"

# HAT metric setting
save_eval_results = True
save_real3d_results = True
save_vis_dir = None
vis_setting = None
get_pack_dir = None


# ------------------ BEV3D DATASET SETTING -------------------
bev_3d_out_size = (
    bevfusion_output_size[0] // 8,
    bevfusion_output_size[1] // 8,
)
bev_3d_stage2_output_resolution = (
    abs(vcs_range[2] - vcs_range[0]) / (bev_3d_out_size[0] * 2),
    abs(vcs_range[3] - vcs_range[1]) / (bev_3d_out_size[1] * 2),
)  # (height, witdh)

max_objs = 900

load_data_types = [
    "gt_bev_3d",
    "gt_multi_view",
    "img_name",
    "timestamp",
    "pack_dir",
    None,
    "img_paths",
]


bev_common_transforms = copy.deepcopy(common_transforms)
bev_common_transforms["ANCCollect3DV"]["gt_bev_3d_idx"] = 0
bev_common_transforms["ANCCollect3DV"]["load_data_types"] = load_data_types
bev_common_transforms["ANCMultiViewTargetGenerator"] = dict(
    type="ANCMultiViewTargetGenerator",
    occlusion_attribute=False,
    occlusion_attribute_dict=None,
    use_ignore_mask_img=False,
)
bev_common_transforms["ANCApplyMaskOnImg"] = dict(
    type="ANCApplyMaskOnImg",
    use_yuv_format=False,
)


def get_bev_3d_transforms(
    bev3d_target,
    bev_common_transforms,
    load_data_types,
    temporal_bev=False,
    bev3d_rpy_transforms=None,
    use_occlusion_attribute=False,
    occlusion_attribute_dict=None,
    use_ignore_mask_img=False,
):
    bev3d_target = _as_list(bev3d_target)
    bev_common_transforms["ANCMultiViewTargetGenerator"][
        "occlusion_attribute"
    ] = use_occlusion_attribute
    bev_common_transforms["ANCMultiViewTargetGenerator"][
        "occlusion_attribute_dict"
    ] = occlusion_attribute_dict
    bev_common_transforms["ANCMultiViewTargetGenerator"][
        "use_ignore_mask_img"
    ] = use_ignore_mask_img
    collect_3dv = bev_common_transforms["ANCCollect3DV"]
    collect_3dv["gt_bev_3d_idx"] = 0
    collect_3dv["load_data_types"] = copy.deepcopy(load_data_types)
    bev_3d_transforms = [
        collect_3dv,
        bev_common_transforms["ANCResize3DV"],
        bev_common_transforms["ANCCrop3DV"],
        bev_common_transforms["ANCTemporalHomo"] if temporal_bev else None,
        bev_common_transforms["ANCToTensor3DV"],
        bev_common_transforms["ANCMultiViewTargetGenerator"],
        bev_common_transforms["ANCApplyMaskOnImg"]
        if use_ignore_mask_img
        else None,
        bev3d_rpy_transforms,
        *bev3d_target,
    ]
    if temporal_bev:
        bev_3d_transforms.append(
            bev_common_transforms["ANCPrepareTempoDataBEV"]
        )
        bev_3d_transforms.append(
            dict(type="ANCSetTemporalClearFlag", clr_mode="clip")
        )
        bev_3d_transforms.append(
            dict(type="ANCAddKeys", kv={"return_latest_flag": True})
        )
    else:
        bev_3d_transforms.append(bev_common_transforms["ANCPrepareDataBEV"])
    return remove_none(bev_3d_transforms)


# -------------------------- MODEL --------------------------
def get_inputs(task_out_size, num_classes=8):
    inputs = dict(
        gt_bev_3d={
            "bev3d_hm": torch.zeros(
                1,
                num_classes,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_dim": torch.zeros(
                1,
                3,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_rot": torch.zeros(
                1,
                3,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_ct_offset": torch.zeros(
                1,
                2,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_loc_z": torch.zeros(
                1,
                1,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_weight_hm": torch.zeros(
                1,
                1,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_point_pos_mask": torch.zeros(
                1,
                1,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_ignore_mask": torch.zeros(
                1,
                1,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_ignore_obj_cls": torch.tensor([[False]]),
            "bev3d_category_class_weight": torch.zeros(
                1,
                1,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_background_weight": torch.zeros(
                1,
                1,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_roi_weight": torch.zeros(
                1,
                1,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_rot_reweight_mask": torch.zeros(
                1,
                1,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_cls_hm": torch.zeros(
                1,
                num_classes,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_occlusion_hm": torch.zeros(
                1,
                4,
                task_out_size[0],
                task_out_size[1],
                dtype=torch.float32,
            ),
            "bev3d_ignore_occlusion": torch.tensor([[True]]),
        },
        annos_bev_3d={
            "vcs_loc_": torch.zeros(1, max_objs, 3, dtype=torch.float32),
            "vcs_cls": torch.zeros(1, max_objs, dtype=torch.float32),
            "vcs_rot_z_": torch.zeros(1, max_objs, dtype=torch.float32),
            "vcs_dim_": torch.zeros(1, max_objs, 3, dtype=torch.float32),
            "vcs_ignore_": torch.zeros(1, max_objs, dtype=torch.bool),
        },
    )
    val_inputs = {}
    deploy_inputs = {}

    return inputs, val_inputs, deploy_inputs


# -------------------- BEV3D COMMON SOLVER ----------------
def get_metrics_patterns(
    task_name,
    use_category_decouple=False,
    use_occlusion_attribute=False,
):
    metrics = [
        dict(type="LossShow", name=f"{task_name}_hm_loss"),
        dict(type="LossShow", name=f"{task_name}_dim_loss"),
        dict(type="LossShow", name=f"{task_name}_rot_loss"),
        dict(type="LossShow", name=f"{task_name}_ct_offset_loss"),
        dict(type="LossShow", name=f"{task_name}_loc_z_loss"),
    ]

    per_metric_patterns = [  # corresponding to metrics
        dict(
            label_pattern=None, pred_pattern=f"^.*{task_name}.*bev3d_hm_loss$"
        ),
        dict(
            label_pattern=None, pred_pattern=f"^.*{task_name}.*bev3d_dim_loss$"
        ),
        dict(
            label_pattern=None, pred_pattern=f"^.*{task_name}.*bev3d_rot_loss$"
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}.*bev3d_ct_offset_loss$",
        ),
        dict(
            label_pattern=None,
            pred_pattern=f"^.*{task_name}.*bev3d_loc_z_loss$",
        ),
    ]

    if use_category_decouple:
        metrics.insert(
            1, dict(type="LossShow", name=f"{task_name}_cls_hm_loss")
        )
        per_metric_patterns.insert(
            1,
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}.*bev3d_cls_hm_loss$",
            ),
        )
    if use_occlusion_attribute:
        metrics.append(
            dict(type="LossShow", name=f"{task_name}_occlusion_hm_loss")
        )
        per_metric_patterns.append(
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}.*bev3d_occlusion_hm_loss$",
            )
        )

    return metrics, per_metric_patterns


def get_metric_updater(metrics, per_metric_patterns, task_name):
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


def get_val_metrics(
    task_name,
    eval_category_ids,
    score_threshold,
    iou_threshold,
    gt_max_depth,
    save_path,
    use_ignore_mask,
    save_vis_dir,
    vis_setting,
    prcurv_save_path,
    save_metric_path=None,
    visibility_intervals=None,
    depth_intervals=(20, 50, 70),
    eval_vcs_range=None,
    ego_ignore_range=None,
    id2label=None,
    eval_occlusion=False,
    occlusion_ignore_id=-99,
    eval_category_confusion=False,
    confusion_save_path=None,
    eval_small=False,
    num_classes=None,
    name="BEV3D",
    eval_mode="bev_iou",
    let_iou_param=None,
    result_prefix="small",
    eval_category_cls=True,
    group_pred_by_cls=False,
    metric_key=("dx", "dxp", "dy", "dyp", "dxyp", "drot"),
    ct_rot_size_threshold=None,
    base_taggers=None,
    taggers=None,
    category_wise_threshold=True,
    auto_threshold=False,
    distance_wise_mode="depth",
    eval_stability=False,
    metric_save_dir=None,
):
    metric_key = list(copy.deepcopy(metric_key))
    if eval_occlusion:
        metric_key.append("occlusion")
    if eval_category_cls:
        metric_key.append("cls")
    metric_key = tuple(metric_key)
    if not ct_rot_size_threshold:
        ct_rot_size_threshold = {
            1000: {
                "ct": 2,
                "rot": 360,
                "size": 0.2,
            }
        }

    if eval_small:
        anno_name = "annos_bev_3d_small"
    else:
        anno_name = "annos_bev_3d"

    val_metrics = [
        dict(
            type="ANCBEVDetTagEvalV2",
            metric_save_dir=metric_save_dir,
            name=name,
            eval_category_ids=eval_category_ids,
            id2label=id2label,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            gt_max_depth=gt_max_depth,
            save_path=save_path,
            save_real3d_res=False,
            save_vis_dir=save_vis_dir,
            save_metric_path=save_metric_path,
            vis_setting=vis_setting,
            enable_ignore=use_ignore_mask,
            depth_intervals=depth_intervals,
            prcurv_save_path=prcurv_save_path,
            metrics=metric_key,
            visibility_intervals=visibility_intervals,
            eval_vcs_range=eval_vcs_range,
            ego_ignore_range=ego_ignore_range,
            eval_occlusion=eval_occlusion,
            occlusion_ignore_id=occlusion_ignore_id,
            anno_name=anno_name,
            result_prefix=result_prefix,
            eval_mode=eval_mode,
            let_iou_param=let_iou_param,
            eval_category_cls=eval_category_cls,
            ct_rot_size_threshold=ct_rot_size_threshold,
            group_pred_by_cls=group_pred_by_cls,
            base_taggers=base_taggers,
            taggers=taggers,
            category_wise_threshold=category_wise_threshold,
            auto_threshold=auto_threshold,
            distance_wise_mode=distance_wise_mode,
            eval_stability=eval_stability,
        )
    ]

    if eval_category_confusion:
        val_metrics.append(
            dict(
                type="ConfusionMatrixBEV3D",
                name=task_name + "_CategoryConfusionMatrix",
                num_classes=num_classes,
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
                gt_max_depth=gt_max_depth,
                eval_vcs_range=eval_vcs_range,
                enable_ignore=use_ignore_mask,
                ego_ignore_range=ego_ignore_range,
                id2label=id2label,
                pred_keys=(
                    "bev3d_ct",
                    "bev3d_cls_id",
                    "bev3d_score",
                    "bev3d_rot",
                    "bev3d_dim",
                    "bev3d_loc_z",
                ),
                confusion_save_path=confusion_save_path,
                save_score_thr=score_threshold,
                gt_id_key="vcs_cls_",
                pred_id_key="bev3d_cls_id",
                anno_name=anno_name,
                range_mode=result_prefix,
            )
        )

    return val_metrics


# -------------------------- TENSORBOADD --------------------------
def get_gt_hm_vis(data):
    data = data.detach().permute(1, 2, 0).cpu().numpy()
    data *= 255.0
    num_classes = data.shape[2]
    data = data[:, :, tuple(range(num_classes)[::-1])]
    data = data.astype("uint8")
    return data


def get_pred_hm_vis(data):
    data_vis = sigmoid_and_clip(data.detach())
    data_vis = data_vis / data_vis.max()
    data_vis = data_vis * 255.0
    data_vis[data_vis != data_vis.max(dim=1, keepdim=True)[0]] = 0
    data_vis = data_vis[0].permute([1, 2, 0]).cpu().numpy()
    num_classes = data_vis.shape[2]
    data = data[:, :, tuple(range(num_classes)[::-1])]
    data_vis = data_vis.astype("uint8")
    return data_vis


def get_bev3d_tb_update_func(writer, model_outs, global_step_id, **kwargs):
    # model_outs = to_flat_ordered_dict(model_outs)
    if "bev_3d" in list(model_outs.keys())[0]:
        for k, v in model_outs.items():
            if "target_gt_bev_3d_bev3d_hm" in k:
                data_vis = get_gt_hm_vis(v[0])
                writer.add_images(
                    k, data_vis, global_step_id, dataformats="HWC"
                )
            elif "predict_bev3d_hm" in k:
                data_vis = get_pred_hm_vis(v[:1])
                writer.add_images(
                    k, data_vis, global_step_id, dataformats="HWC"
                )
            else:
                continue


# -------------------------- PACK_INFER_SAVE --------------------------
save_pack_infer = dict(
    type="SaveBEV3DConsistencyResult",
    output_dir=os.path.join(save_prefix, "inference", "bev_3d"),
    prefix="OutputModule_predict",
)
