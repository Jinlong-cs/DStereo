import os
from copy import deepcopy

import torch
from common import (
    backbone,
    batch_size,
    bn_kwargs,
    datapaths,
    fix_channel_neck,
    fpn_neck,
    input_hw,
    inter_method,
    lmdb_data,
    log_freq,
    min_valid_clip_area_ratio,
    model_thresh,
    pixel_center_aligned,
    rand_translation_ratio,
    resize_hw,
)
from rear_detection import (
    anchor_desc,
    anchor_generator,
    anchor_head,
    anchor_pred,
    classnames,
    object_type,
    roi_feat_extractor,
    val_decoders,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import frcnn_roi_det_with_subscore_desc

task_name = "rear_plate_detection"
task_type = "roi_detection"

data_args = dict(legacy_bbox=True)

sub_classnames = ["rear_plate"]
num_classes = len(sub_classnames)

roi_args = dict(
    class_agnostic_reg=True,
    exclude_background=True,
    num_fg_classes=num_classes,
    roi_zoom_scale_wh=(1.0, 1.0),
)


def get_model(mode):
    train_pred = deepcopy(anchor_pred)
    train_pred.update(
        pre_nms_top_k=2000,
        post_nms_top_k=256,
        nms_padding_mode="rollover",
        node_name=f"{task_name}_anchor_pred",
    )

    return dict(
        type="TwoStageDetector",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[fpn_neck, fix_channel_neck]),
        rpn_out_keys=[] if mode != "train" else None,
        rpn_module=dict(
            type="AnchorModule",
            anchor_generator=anchor_generator,
            head=anchor_head,
            postprocess=train_pred if mode == "train" else anchor_pred,
            target=None,
            loss=None,
            desc=anchor_desc if mode != "train" else None,
        ),
        roi_module=dict(
            type="RoIModule",
            output_head_out=True,
            roi_key="pred_boxes",
            target_keys=["gt_boxes", "parent_gt_boxes"],
            target_opt_keys=[
                "parent_gt_boxes_num",
                "parent_ig_regions",
                "parent_ig_regions_num",
            ],
            roi_feat_extractor=roi_feat_extractor,
            head=dict(
                type="ExtSequential",
                modules=[
                    dict(
                        type="RCNNVarGNetHead",
                        with_box_reg=True,
                        num_fg_classes=roi_args["num_fg_classes"],
                        bn_kwargs=bn_kwargs,
                        class_agnostic_reg=roi_args["class_agnostic_reg"],
                        with_background=not roi_args["exclude_background"],
                        roi_out_channel=64,
                        dw_num_filter=128,
                        pw_num_filter=128,
                        pw_num_filter2=96,
                        group_base=8,
                        factor=1,
                        stride_step1=1,
                        ksize_step2=1,
                        node_name=f"{task_name}_roi_head",
                    )
                ]
                + (
                    [
                        dict(
                            type="AddDesc",
                            per_tensor_desc=frcnn_roi_det_with_subscore_desc(
                                task_name="frcnn_roi_detection",
                                class_names=classnames,
                                sub_class_name="rear_plate",
                                label_output_name=f"{task_name}_label",
                                offset_output_name=f"{task_name}_offset",
                                subbox_score_thresh=model_thresh[
                                    "roi_det_thresh"
                                ][object_type]
                                if model_thresh
                                else 0.1,
                            ),
                            node_name=f"{task_name}_roi_desc",
                        )
                    ]
                    if "test" in mode
                    else []
                ),
            ),
            target=dict(
                type="ProposalTargetBinDet",
                matcher=dict(
                    type="MaxIoUMatcher",
                    pos_iou=0.5,
                    neg_iou=0.5,
                    allow_low_quality_match=False,
                    low_quality_match_iou=0.3,
                    legacy_bbox=data_args["legacy_bbox"],
                ),
                ig_region_matcher=dict(
                    type="IgRegionMatcher",
                    num_classes=roi_args["num_fg_classes"] + 1,
                    legacy_bbox=data_args["legacy_bbox"],
                    ig_region_overlap=0.5,
                    exclude_background=roi_args["exclude_background"],
                ),
                label_encoder=dict(
                    type="RCNNBinDetLabelFromMatch",
                    roi_w_zoom_scale=roi_args["roi_zoom_scale_wh"][0],
                    roi_h_zoom_scale=roi_args["roi_zoom_scale_wh"][1],
                    feature_w=8,
                    feature_h=8,
                    num_classes=num_classes,
                    cls_on_hard=False,
                    allow_low_quality_heatmap=True,
                ),
                node_name=f"{task_name}_roi_target",
            )
            if mode == "train"
            else None,
            loss=dict(
                type="RCNNBinDetLoss",
                cls_loss=dict(
                    type="SmoothL1Loss",
                    reduction="mean",
                    hard_neg_mining_cfg=dict(
                        keep_pos=True,
                        neg_ratio=0.5,
                        hard_ratio=0.5,
                    ),
                ),
                reg_loss=dict(type="SmoothL1Loss", reduction="mean"),
                node_name=f"{task_name}_roi_loss",
            )
            if mode == "train"
            else None,
            postprocess=dict(
                type="HeatmapBox2dDecoder",
                score_threshold=model_thresh["roi_det_thresh"][object_type]
                if model_thresh
                else 0.1,
                roi_wh_zoom_scale=roi_args["roi_zoom_scale_wh"],
                node_name=f"{task_name}_roi_decoder",
            )
            if "val" in mode
            else None,
        ),
    )


assert object_type in val_decoders
val_decoders[object_type][0].append(task_name)
val_decoders[object_type][1]["task_descs"][task_name] = (
    "pred_boxes",
    task_type,
)

# inputs
inputs = dict(
    train=dict(
        gt_boxes=torch.zeros((1, 100, 5)),
        gt_boxes_num=torch.zeros(1),
        ig_regions=torch.zeros((1, 110, 5)),
        ig_regions_num=torch.zeros(1),
        im_hw=torch.zeros((1, 2)),
        parent_gt_boxes=torch.zeros((1, 10, 5)),
        parent_gt_boxes_num=torch.zeros(1),
        parent_ig_regions=torch.zeros((1, 110, 5)),
        parent_ig_regions_num=torch.zeros(1),
    ),
    val=dict(),
    test=dict(),
)


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="label_map_loss"),
        dict(type="LossShow", name="offset_loss"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_label_map_loss$",
            ),
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_offset_loss$",
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
    preds = model_outs["rear"]

    pred_bboxes = []
    for pred in preds:
        det_pred = getattr(pred, task_name)
        pred_bboxes.append(
            torch.cat(
                [
                    torch.cat(det_pred.boxes_list),
                    torch.cat(det_pred.cls_idxs_list)[:, None],
                    torch.cat(det_pred.scores_list)[:, None],
                ],
                dim=1,
            )
            if len(det_pred.boxes_list) > 0
            else [],
        )
    gt_bboxes, gt_classes = [], []
    for gt_boxes, gt_boxes_num in zip(
        batch_data["gt_boxes"],
        batch_data["gt_boxes_num"],
    ):
        gt_boxes = gt_boxes[: int(gt_boxes_num.item())]
        gt_boxes = gt_boxes.reshape(-1, 5)
        gt_boxes = gt_boxes[gt_boxes[:, -1] != 0]
        gt_bboxes.append(gt_boxes[:, :4])
        gt_classes.append(gt_boxes[:, 4])

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
            num_classes=1,
            cls_idx_mapping={1: 0},
        ),
    ],
    metric_update_func=val_update_metric_func,
    step_log_freq=1000,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


# data
if lmdb_data:
    ds = datapaths.rear_plate_detection

    classname2idxs = [2]
    data_paths = [d["data_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

    data_loader = dict(
        type=torch.utils.data.DataLoader,
        drop_last=True,
        num_workers=1,  # lmdb
        persistent_workers=True,
        batch_size=batch_size,
        dataset=dict(
            type="DistributedComposeRandomDataset",
            sample_weights=sample_weights,
            datasets=[
                dict(
                    type="DetSeg2DAnnoDataset",
                    idx_path=os.path.join(path, "idx"),
                    img_path=os.path.join(path, "img"),
                    anno_path=os.path.join(path, "anno"),
                    transforms=[
                        dict(
                            type="DetSeg2DAnnoDatasetToROIFormat",
                            selected_class_ids=classname2idxs,
                            lt_point_id=10,
                            rb_point_id=12,
                            parent_lt_point_id=0,
                            parent_rb_point_id=2,
                            use_parent=True,
                            parent_id=1,
                        ),
                        dict(
                            type="ROIDetectionIterableDetRoITransform",
                            # roi transform
                            target_wh=input_hw[::-1],
                            resize_wh=None
                            if resize_hw is None
                            else resize_hw[::-1],
                            img_scale_range=(0.5, 2.0),
                            roi_scale_range=(0.7, 1.0 / 0.7),
                            min_sample_num=1,
                            max_sample_num=1,
                            center_aligned=pixel_center_aligned,
                            inter_method=inter_method,
                            use_pyramid=True,
                            pyramid_min_step=0.7,
                            pyramid_max_step=0.8,
                            min_valid_area=16,
                            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
                            min_edge_size=4,
                            rand_translation_ratio=rand_translation_ratio,
                            rand_aspect_ratio=0.0,
                            rand_rotation_angle=0,
                            flip_prob=0.5,
                            clip_bbox=False,
                            # person center and head center must in image
                            allow_outside_center=True,  # True is better than False
                            pixel_center_aligned=pixel_center_aligned,
                            keep_aspect_ratio=True,
                        ),
                        dict(
                            type="PadRoIDetData",
                            max_gt_boxes_num=200,
                            max_ig_regions_num=100,
                        ),
                    ],
                )
                for path in data_paths
            ],
        ),
    )

    val_data_loader = dict(
        type=torch.utils.data.DataLoader,
        num_workers=0,
        batch_size=batch_size,
        dataset=dict(
            type="DistributedComposeRandomDataset",
            sample_weights=sample_weights,
            datasets=[
                dict(
                    type="DetSeg2DAnnoDataset",
                    idx_path=os.path.join(path, "idx"),
                    img_path=os.path.join(path, "img"),
                    anno_path=os.path.join(path, "anno"),
                    transforms=[
                        dict(
                            type="DetSeg2DAnnoDatasetToROIFormat",
                            selected_class_ids=classname2idxs,
                            lt_point_id=10,
                            rb_point_id=12,
                            parent_lt_point_id=0,
                            parent_rb_point_id=2,
                            use_parent=True,
                            parent_id=1,
                        ),
                        dict(
                            type="ROIDetectionIterableDetRoITransform",
                            # roi transform
                            target_wh=input_hw[::-1],
                            resize_wh=None
                            if resize_hw is None
                            else resize_hw[::-1],
                            img_scale_range=(1.0, 1.0),
                            roi_scale_range=(1.0, 1.0),
                            min_sample_num=1,
                            max_sample_num=1,
                            center_aligned=pixel_center_aligned,
                            inter_method=inter_method,
                            use_pyramid=True,
                            pyramid_min_step=0.7,
                            pyramid_max_step=0.8,
                            min_valid_area=10,
                            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
                            min_edge_size=4,
                            rand_translation_ratio=rand_translation_ratio,
                            rand_aspect_ratio=0.0,
                            rand_rotation_angle=0,
                            flip_prob=0.0,
                            clip_bbox=False,
                            # person center and head center must in image
                            allow_outside_center=True,  # True is better than False
                            pixel_center_aligned=pixel_center_aligned,
                            keep_aspect_ratio=True,
                        ),
                        dict(
                            type="PadRoIDetData",
                            max_gt_boxes_num=200,
                            max_ig_regions_num=100,
                        ),
                    ],
                )
                for path in data_paths
            ],
        ),
    )
else:
    ds = datapaths.rear_plate_detection
    rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
    anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

    classname2idxs = [2]

    data_loader = dict(
        type="MultiCachedDataLoader",
        __build_recursive=False,
        last_batch="pad",
        batch_size=batch_size,
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
                    ),
                    even_split=False,
                )
                for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
            ],
            prob=sample_weights,
            balance=True,
        ),
        transform=[
            dict(
                type="LegacyDenseBoxImageRecordDatasetDecoder",
                to_rgb=True,
            ),
            dict(
                type="DecodeDenseBoxDatasetToPersonHeadDetFormat",
                selected_class_ids=classname2idxs,
                lt_point_id=10,
                rb_point_id=12,
                parent_lt_point_id=0,
                parent_rb_point_id=2,
                use_parent=True,
                parent_id=1,
            ),
            dict(
                type="ROIDetectionIterableDetRoITransform",
                # roi transform
                target_wh=input_hw[::-1],
                resize_wh=None if resize_hw is None else resize_hw[::-1],
                img_scale_range=(0.5, 2.0),
                roi_scale_range=(0.7, 1.0 / 0.7),
                min_sample_num=1,
                max_sample_num=1,
                center_aligned=pixel_center_aligned,
                inter_method=inter_method,
                use_pyramid=True,
                pyramid_min_step=0.7,
                pyramid_max_step=0.8,
                min_valid_area=16,
                min_valid_clip_area_ratio=min_valid_clip_area_ratio,
                min_edge_size=4,
                rand_translation_ratio=rand_translation_ratio,
                rand_aspect_ratio=0.0,
                rand_rotation_angle=0,
                flip_prob=0.5,
                clip_bbox=False,
                # person center and head center must in image
                allow_outside_center=True,  # True is better than False
                pixel_center_aligned=pixel_center_aligned,
                keep_aspect_ratio=True,
            ),
            dict(
                type="PadPersongHeadData",
                target_wh=input_hw[::-1],
                max_gt_boxes_num=200,
                max_ig_regions_num=100,
            ),
        ],
    )
