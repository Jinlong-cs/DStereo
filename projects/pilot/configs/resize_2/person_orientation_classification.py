import os
from copy import deepcopy

import numpy as np
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
    pixel_center_aligned,
    rand_translation_ratio,
    resize_hw,
)
from person_detection import (
    anchor_desc,
    anchor_generator,
    anchor_head,
    anchor_pred,
    classnames,
    object_type,
    roi_feat_extractor,
    roi_head,
    val_decoders,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import frcnn_classification_desc

task_type = "classification"
task_name = f"{object_type}_orientation_{task_type}"


data_args = dict(legacy_bbox=True)

sub_classnames = [
    "back",
    "front",
    "left",
    "left_anterior",
    "left_back",
    "right",
    "right_back",
    "right_front",
]
num_classes = len(sub_classnames)

roi_args = dict(
    exclude_background=True,
    num_fg_classes=num_classes,
    tracking_feat_len=96,
    with_tracking_feat=False,
)


def get_model(mode):
    train_pred = deepcopy(anchor_pred)
    train_pred.update(
        pre_nms_top_k=2000,
        post_nms_top_k=1000,
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
            roi_feat_extractor=roi_feat_extractor,
            head=dict(
                type="ExtSequential",
                modules=[
                    roi_head,
                    dict(
                        type="RCNNVarGNetSplitHead",
                        num_fg_classes=roi_args["num_fg_classes"],
                        bn_kwargs=bn_kwargs,
                        with_background=not roi_args["exclude_background"],
                        in_channel=128,
                        pw_num_filter2=roi_args["tracking_feat_len"],
                        with_box_reg=False,
                        with_tracking_feat=roi_args["with_tracking_feat"],
                        node_name=f"{task_name}_roi_split_head",
                    ),
                ]
                + (
                    [
                        dict(
                            type="AddDesc",
                            per_tensor_desc=frcnn_classification_desc(
                                task_name="frcnn_classification",
                                output_name=task_name,
                                class_names=classnames,
                                desc_id=str(num_classes),
                                with_tracking_feat=roi_args[
                                    "with_tracking_feat"
                                ],
                            ),
                            node_name=f"{task_name}_roi_desc",
                        )
                    ]
                    if "test" in mode
                    else []
                ),
            ),
            target=dict(
                type="ProposalTarget",
                matcher=dict(
                    type="MaxIoUMatcher",
                    pos_iou=0.7,
                    neg_iou=0.7,
                    allow_low_quality_match=False,
                    low_quality_match_iou=0.3,
                    legacy_bbox=data_args["legacy_bbox"],
                ),
                ig_region_matcher=dict(
                    type="IgRegionMatcher",
                    num_classes=roi_args["num_fg_classes"] + 1,
                    ig_region_overlap=0.5,
                    legacy_bbox=data_args["legacy_bbox"],
                    exclude_background=roi_args["exclude_background"],
                ),
                label_encoder=dict(
                    type="MatchLabelSepEncoder",
                    class_encoder=dict(
                        type="OneHotClassEncoder",
                        num_classes=roi_args["num_fg_classes"] + 1,
                        class_agnostic_neg=False,
                        exclude_background=roi_args["exclude_background"],
                    ),
                    cls_use_pos_only=True,
                    cls_on_hard=False,
                ),
                node_name=f"{task_name}_roi_target",
            )
            if mode == "train"
            else None,
            loss=dict(
                type="RCNNCLSLoss",
                cls_loss=dict(
                    type="SoftmaxCELoss",
                    dim=1,
                    reduction="mean",
                    loss_weight=0.25,
                ),
                node_name=f"{task_name}_roi_loss",
            )
            if mode == "train"
            else None,
            postprocess=dict(
                type="SoftmaxRoIClsDecoder",
                cls_name_mapping={
                    i: cls_name for i, cls_name in enumerate(sub_classnames)
                },
                node_name=f"{task_name}_roi_decoder",
            )
            if "val" in mode
            else None,
        ),
    )


val_decoders[object_type][0].append(task_name)
val_decoders[object_type][1]["task_descs"][task_name] = (
    "pred_cls",
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
    ),
    val=dict(),
    test=dict(),
)


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


# data
if lmdb_data:
    ds = datapaths.person_orientation_classification

    classname2idxs = list(range(1, num_classes + 1))
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
                            type="DetSeg2DAnnoDatasetToDetFormat",
                            selected_class_ids=classname2idxs,
                            lt_point_id=0,
                            rb_point_id=2,
                        ),
                        dict(
                            type="IterableDetRoITransform",
                            target_wh=input_hw[::-1],
                            resize_wh=None
                            if resize_hw is None
                            else resize_hw[::-1],
                            img_scale_range=(0.7, 1.0 / 0.7),
                            roi_scale_range=(0.5, 2.0),
                            min_sample_num=1,
                            max_sample_num=1,
                            center_aligned=False,
                            inter_method=inter_method,
                            use_pyramid=True,
                            pyramid_min_step=0.7,
                            pyramid_max_step=0.8,
                            min_valid_area=8,
                            min_valid_clip_area_ratio=0.5,
                            min_edge_size=2,
                            rand_translation_ratio=rand_translation_ratio,
                            rand_aspect_ratio=0.0,
                            rand_rotation_angle=0,
                            flip_prob=0,
                            clip_bbox=True,
                            pixel_center_aligned=pixel_center_aligned,
                            keep_aspect_ratio=True,
                        ),
                        dict(
                            type="PadDetData",
                            max_gt_boxes_num=2000,
                            max_ig_regions_num=100,
                        ),
                    ],
                )
                for path in data_paths
            ],
        ),
    )
else:
    ds = datapaths.person_orientation_classification
    rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
    anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

    classname2idxs = list(range(1, num_classes + 1))

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
                type="DecodeDenseBoxDatasetToDetFormat",
                selected_class_ids=classname2idxs,
                lt_point_id=0,
                rb_point_id=2,
            ),
            dict(
                type="IterableDetRoITransform",
                target_wh=input_hw[::-1],
                resize_wh=None if resize_hw is None else resize_hw[::-1],
                img_scale_range=(0.7, 1.0 / 0.7),
                roi_scale_range=(0.5, 2.0),
                min_sample_num=1,
                max_sample_num=1,
                center_aligned=False,
                inter_method=inter_method,
                use_pyramid=True,
                pyramid_min_step=0.7,
                pyramid_max_step=0.8,
                min_valid_area=8,
                min_valid_clip_area_ratio=0.5,
                min_edge_size=2,
                rand_translation_ratio=rand_translation_ratio,
                rand_aspect_ratio=0.0,
                rand_rotation_angle=0,
                flip_prob=0,
                clip_bbox=True,
                pixel_center_aligned=pixel_center_aligned,
                keep_aspect_ratio=True,
            ),
            dict(
                type="PadDetData",
                target_wh=input_hw[::-1],
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
