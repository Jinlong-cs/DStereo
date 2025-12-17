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
    get_roi_module4cls_task,
    mtfos3d_params,
    object_type,
    roi_args,
    roi_feat_extractor,
    val_decoders,
)

task_type = "classification"
task_name = f"{object_type}_category_{task_type}"
batch_size = tasks_batch_size[task_name]

sub_classnames = [
    "Bus",
    "Small_Medium_Car",
    "Trucks",
    "Motors",
    "Special_vehicle",
    "Tiny_car",
    "Lorry",
    "MiniVan",
]
num_classes = len(sub_classnames)

task_roi_args = copy.deepcopy(roi_args)
task_roi_args.update(num_fg_classes=num_classes)


def get_model(mode):

    if (not is_int_infer) or (val_only):
        return dict(
            type="TwoStageDetector",
            backbone=backbone,
            neck=dict(
                type="ExtSequential",
                modules=[neck, pick_stride_neck, bifpn_quant],
            ),
            # neck=dict(type="ExtSequential", modules=[neck, pick_stride_neck, bifpn_quant]),
            remap_label_key=True,
            rpn_module=get_mtfcos3d_module(
                object_type, mode, **mtfos3d_params
            ),
            roi_module=get_roi_module4cls_task(
                mode,
                classnames,
                roi_feat_extractor,
                sub_classnames,
                task_name,
                task_roi_args,
            ),
        )
    else:
        return dict(
            type="TraceRcnnModule",
            obj_type=object_type,
            quant_module=bifpn_quant,
            fpn_desc=bifpn_desc,
            roi_desc=add_bbox_desc_pp,
            roi_module=get_roi_module4cls_task(
                mode,
                classnames,
                roi_feat_extractor,
                sub_classnames,
                task_name,
                task_roi_args,
            ),
        )

        # return get_roi_module4cls_task(mode, sub_classnames, task_name, task_roi_args)


val_decoders[object_type][0].append(task_name)
val_decoders[object_type][1]["task_descs"][task_name] = (
    "pred_cls",
    task_type,
)

inputs = dict(
    train=dict(
        gt_boxes=torch.zeros((1, 100, 5)),
        gt_boxes_num=torch.zeros(1),
        ig_regions=torch.zeros((1, 110, 5)),
        ig_regions_num=torch.zeros(1),
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
        dict(type="LossShow", name="rcnn_cls_loss"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_rcnn_cls_loss$",
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

    pred_bboxes = []
    for pred in preds:
        det_pred = pred.vehicle_detection
        cls_pred = getattr(pred, task_name)
        pred_bboxes.append(
            torch.cat(
                [
                    det_pred.boxes,
                    cls_pred.cls_idxs[:, None],
                    cls_pred.scores[:, None],
                ],
                dim=1,
            )
        )
    gt_bboxes, gt_classes = [], []
    for gt_boxes, gt_boxes_num in zip(
        batch_data["gt_boxes"],
        batch_data["gt_boxes_num"],
    ):
        gt_boxes = gt_boxes[: gt_boxes_num.item()]
        gt_bboxes.append(gt_boxes[:, :4])
        gt_classes.append(gt_boxes[:, 4] - 1)  # exclude_background

    results = {
        "pred_bboxes": pred_bboxes,
        "gt_bboxes": gt_bboxes,
        "gt_classes": gt_classes,
        "gt_difficult": None,
    }

    for metric in metrics:
        metric.update(results)


val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(
            type="VOCMApMetric",
            num_classes=num_classes,
            class_names=sub_classnames,
        ),
    ],
    metric_update_func=val_update_metric_func,
    step_log_freq=1000,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


ds = datapaths.get("vehicle_category_classification", None)
if ds is None:
    ds = datapaths.vehicle_category

ds = datapaths.vehicle_category
train_rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
train_anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
train_sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

val_rec_paths = [d["rec_path"] for d in ds["val_data_paths"]]
val_anno_paths = [d["anno_path"] for d in ds["val_data_paths"]]
val_sample_weights = [d["sample_weight"] for d in ds["val_data_paths"]]

classname2idxs = list(range(1, num_classes + 1))
inter_method = 10
pixel_center_aligned = False
min_valid_clip_area_ratio = 0.5
rand_translation_ratio = 0.1
batch_size = tasks_batch_size[task_name]

data_loader = dict(
    type="MultiCachedDataLoader",
    __build_recursive=False,
    last_batch="pad",
    batch_size=batch_size["train"],
    num_workers=1,
    shuffle=True,
    chunk_size=32,
    min_prefetch=1,
    max_prefetch=2,
    min_chunk_num=2,
    max_chunk_num=4,
    batched_transform=True,
    skip_batchify=False,
    prefetcher_using_thread=True,
    dataset=dict(
        type="MultiFusedIterableDataset",
        dataset=[
            dict(
                type="SplitDataset",
                dataset=dict(
                    type="LegacyDenseBoxImageRecordDataset",
                    rec_path=rec_path_i,
                    anno_path=anno_path_i,
                    read_only=True,
                    with_seg_label=False,
                    to_rgb=True,
                    as_nd=False,
                ),
                even_split=False,
            )
            for rec_path_i, anno_path_i in zip(
                train_rec_paths, train_anno_paths
            )
        ],
        prob=train_sample_weights,
        balance=True,
    ),
    transform=[
        dict(
            type="LegacyDenseBoxImageRecordDatasetDecoder",
            to_rgb=True,
            as_nd=False,
        ),
        dict(
            type="DecodeDenseBoxDatasetToDetFormat",
            selected_class_ids=classname2idxs,
            lt_point_id=0,
            rb_point_id=2,
        ),
        dict(
            type="IterableDetRoITransform",
            target_wh=input_size[::-1],
            resize_wh=None if input_size is None else input_size[::-1],
            img_scale_range=(0.7, 1.0 / 0.7),
            roi_scale_range=(0.5, 2.0),
            min_sample_num=1,
            max_sample_num=1,
            center_aligned=False,
            inter_method=inter_method,
            use_pyramid=True,
            pyramid_min_step=0.7,
            pyramid_max_step=0.8,
            pixel_center_aligned=pixel_center_aligned,
            min_valid_area=100,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=10,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0,
            flip_prob=0.5,
            clip_bbox=True,
            keep_aspect_ratio=True,
        ),
        dict(
            type="PadDetData",
            target_wh=input_size[::-1],
            max_gt_boxes_num=2000,
            max_ig_regions_num=100,
        ),
        dict(type="CastEx", dtypes=(None,) + (np.float32,) * 5),
        dict(
            type="ToDict",
            keys=(
                "img",
                "im_hw",
                "gt_boxes",
                "gt_boxes_num",
                "ig_regions",
                "ig_regions_num",
            ),
        ),
    ],
)

val_data_loader = dict(
    type="MultiCachedDataLoader",
    __build_recursive=False,
    last_batch="keep",
    batch_size=batch_size["val"],
    num_workers=1,
    shuffle=False,
    chunk_size=32,
    min_prefetch=1,
    max_prefetch=2,
    min_chunk_num=2,
    max_chunk_num=4,
    batched_transform=True,
    skip_batchify=False,
    prefetcher_using_thread=True,
    dataset=dict(
        type="MultiFusedIterableDataset",
        dataset=[
            dict(
                type="SplitDataset",
                dataset=dict(
                    type="LegacyDenseBoxImageRecordDataset",
                    rec_path=rec_path_i,
                    anno_path=anno_path_i,
                    read_only=True,
                    with_seg_label=False,
                    to_rgb=True,
                    as_nd=False,
                ),
                even_split=False,
            )
            for rec_path_i, anno_path_i in zip(val_rec_paths, val_anno_paths)
        ],
        prob=val_sample_weights,
        balance=True,
    ),
    transform=[
        dict(
            type="LegacyDenseBoxImageRecordDatasetDecoder",
            to_rgb=True,
            as_nd=False,
        ),
        dict(
            type="DecodeDenseBoxDatasetToDetFormat",
            selected_class_ids=classname2idxs,
            lt_point_id=0,
            rb_point_id=2,
        ),
        dict(
            type="DetAffineAugTransformer",
            target_wh=input_size[::-1],
            resize_wh=None,
            flip_prob=0.0,
            inter_method=1,
            use_pyramid=False,
            center_aligned=False,
            pixel_center_aligned=pixel_center_aligned,
            min_valid_area=100,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=10,
            clip_bbox=True,
            keep_aspect_ratio=True,
        ),
        dict(
            type="PadDetData",
            target_wh=input_size[::-1],
            max_gt_boxes_num=2000,
            max_ig_regions_num=100,
        ),
        dict(type="CastEx", dtypes=(None,) + (np.float32,) * 5),
        dict(
            type="ToDict",
            keys=(
                "img",
                "im_hw",
                "gt_boxes",
                "gt_boxes_num",
                "ig_regions",
                "ig_regions_num",
            ),
        ),
    ],
)

(
    aidi_eval_page_label,
    aidi_eval_dataset_id,
    aidi_eval_callback,
    aidi_eval_type,
    old_prediction,
    new_prediction,
) = get_rcnn_aidi_eval_info("vehicle_category_classification", "vehicle")

aidi_eval_loader = get_2d_aidi_eval_loaders(
    batch_size["val"],
    "vehicle_category_classification",
    infer_model_type="rcnn",
)
