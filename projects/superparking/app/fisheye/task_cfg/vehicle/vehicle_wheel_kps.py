import copy

import numpy as np
import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from ...common import (
    backbone,
    datapaths,
    input_size,
    is_int_infer,
    log_freq,
    neck,
    tasks_batch_size,
    val_only,
)
from ..vdvru_common import (
    bifpn_desc,
    bifpn_quant,
    get_2d_aidi_eval_loaders,
    get_mtfcos3d_module,
    get_rcnn_aidi_eval_info,
    pick_stride_neck,
)
from .common import (
    add_bbox_desc_pp,
    classnames,
    get_roi_module4kps_task,
    mtfos3d_params,
    object_type,
    roi_args,
    roi_feat_extractor,
    val_decoders,
)

task_name = "vehicle_wheel_kps"
task_type = "kps"
batch_size = tasks_batch_size[task_name]

task_roi_args = copy.deepcopy(roi_args)
task_roi_args.update(
    exclude_background=True, num_fg_classes=1, expand_param=1.2
)


def get_model(mode):

    if (not is_int_infer) or (val_only):
        return dict(
            type="TwoStageDetector",
            backbone=backbone,
            neck=dict(
                type="ExtSequential",
                modules=[neck, pick_stride_neck, bifpn_quant],
            ),
            remap_label_key=True,
            rpn_module=get_mtfcos3d_module(
                object_type, mode, **mtfos3d_params
            ),
            roi_module=get_roi_module4kps_task(
                mode, classnames, roi_feat_extractor, task_name, task_roi_args
            ),
        )
    else:
        return dict(
            type="TraceRcnnModule",
            obj_type=object_type,
            quant_module=bifpn_quant,
            fpn_desc=bifpn_desc,
            roi_desc=add_bbox_desc_pp,
            roi_module=get_roi_module4kps_task(
                mode, classnames, roi_feat_extractor, task_name, task_roi_args
            ),
        )


val_decoders[object_type][0].append(task_name)
val_decoders[object_type][1]["task_descs"][task_name] = (
    "pred_kps",
    task_type,
)

# inputs
inputs = dict(
    train=dict(
        gt_boxes=torch.zeros((1, 100, 5)),
        gt_boxes_num=torch.zeros(1),
        im_hw=torch.zeros((1, 2)),
    ),
    val=dict(
        gt_boxes=torch.zeros((1, 100, 5)),
    ),
    test=dict(
        gt_boxes=torch.zeros((1, 100, 5)),
    ),
)


def get_inputs(mode):
    return inputs[mode]


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="rcnn_kps_class_loss"),
        dict(type="LossShow", name="rcnn_kps_reg_loss"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_rcnn_kps_class_loss$",
            ),
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_rcnn_kps_reg_loss$",
            ),
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


def val_update_metric_func(metrics, batch, model_outs):
    batch_data, _ = batch
    preds = model_outs["vehicle"]

    preds_parents, preds_kps = [], []
    for pred in preds:
        preds_parents.append(pred.vehicle_detection)  # noqa B009
        preds_kps.append(getattr(pred, task_name))

    predictions = {"detection": preds_parents, "kps": preds_kps}

    for metric in metrics:
        metric.update(batch_data, predictions)


val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="KpsMetric"),
    ],
    metric_update_func=val_update_metric_func,
    step_log_freq=1000,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)

# ------------------------ dataloader ---------------------------

ds = datapaths.vehicle_wheel_kps
rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

val_rec_paths = [d["rec_path"] for d in ds["val_data_paths"]]
val_anno_paths = [d["anno_path"] for d in ds["val_data_paths"]]
val_sample_weights = [d["sample_weight"] for d in ds["val_data_paths"]]

inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.5
rand_translation_ratio = 0.1

data_loader = dict(
    type="GluonDataLoader",
    __build_recursive=False,
    dataset=[
        dict(
            type="KPSDataset",
            img_rec_path=rec_path_i,
            anno_rec_path=anno_path_i,
            to_rgb=True,
        )
        for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
    ],
    transform=[
        dict(
            type="KPSIterableDetRoITransform",
            kps_num=2,
            # roi transform
            target_wh=input_size[::-1],
            resize_wh=None if input_size is None else input_size[::-1],
            img_scale_range=(0.5, 2.0),
            roi_scale_range=(0.7, 1.0 / 0.7),
            min_sample_num=1,
            max_sample_num=1,
            center_aligned=False,
            inter_method=inter_method,
            use_pyramid=True,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            min_valid_area=100,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=10,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0,
            flip_prob=0.5,
            clip_bbox=False,
            pixel_center_aligned=pixel_center_aligned,
            min_kps_distance=4,
            keep_aspect_ratio=True,
        ),
        dict(
            type="PadKpsData",
            target_wh=input_size[::-1],
            max_gt_boxes_num=200,
            max_ig_regions_num=100,
        ),
        dict(type="CastEx", dtypes=(None,) + (np.float32,) * 3),
        dict(
            type="ToDict",
            keys=("img", "im_hw", "gt_boxes", "gt_boxes_num"),
        ),
    ],
    batch_size=batch_size["train"],
    shuffle=True,
    num_workers=0,
    last_batch="rollover",
)

val_data_loader = dict(
    type="GluonDataLoader",
    __build_recursive=False,
    dataset=[
        dict(
            type="KPSDataset",
            img_rec_path=rec_path_i,
            anno_rec_path=anno_path_i,
            to_rgb=True,
        )
        for rec_path_i, anno_path_i in zip(val_rec_paths, val_anno_paths)
    ],
    transform=[
        dict(
            type="KPSAffineAugTransformer",
            # transform
            target_wh=input_size[::-1],
            center_aligned=True,
            inter_method=10,
            use_pyramid=False,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            rand_translation_ratio=0.0,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0.0,
            rand_scale_range=(1.0, 1.0),
            flip_prob=0.0,
            norm_wh=None,
            norm_scale=None,
            resize_wh=None,
            clip_bbox=False,
            min_valid_area=100,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=10,
            pixel_center_aligned=False,
        ),
        dict(
            type="PadKpsData",
            target_wh=input_size[::-1],
            max_gt_boxes_num=200,
            max_ig_regions_num=100,
        ),
        dict(type="CastEx", dtypes=(None,) + (np.float32,) * 3),
        dict(type="ToDict", keys=("img", "im_hw", "gt_boxes", "gt_boxes_num")),
    ],
    batch_size=batch_size["val"],
    shuffle=False,
    num_workers=1,
    last_batch="keep",
)

(
    aidi_eval_page_label,
    aidi_eval_dataset_id,
    aidi_eval_callback,
    aidi_eval_type,
    old_prediction,
    new_prediction,
) = get_rcnn_aidi_eval_info("vehicle_wheel_kps", "vehicle")

aidi_eval_loader = get_2d_aidi_eval_loaders(
    batch_size["val"], "vehicle_wheel_kps", infer_model_type="rcnn"
)
