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
    input_padding,
    inter_method,
    lmdb_data,
    log_freq,
    pixel_center_aligned,
    rand_translation_ratio,
    resize_hw,
)
from vehicle_detection import (
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
from hat.core.proj_spec.descs import frcnn_kps_desc

task_name = "vehicle_wheel_kps"
task_type = "kps"


data_args = dict(legacy_bbox=True)

roi_args = dict(
    exclude_background=True,
    num_fg_classes=1,
    expand_param=1.2,
)

roi_head = dict(
    type="RCNNVarGNetShareHead",
    bn_kwargs=bn_kwargs,
    roi_out_channel=64,
    gc_num_filter=128,
    pw_num_filter=128,
    pw_num_filter2=128,
    group_base=8,
    factor=1,
    stride=1,
    upscale=True,
    upscale_conv=True,
    node_name=f"{task_name}_roi_share_head",
)


def get_model(mode):
    train_pred = deepcopy(anchor_pred)
    train_pred.update(
        pre_nms_top_k=2000,
        post_nms_top_k=256,
        use_clippings=False,
        nms_padding_mode="rollover",
        bbox_min_hw=(8, 8),
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
                        type="RCNNKPSSplitHead",
                        in_channel=128,
                        points_num=2,
                        node_name=f"{task_name}_roi_split_head",
                    ),
                ]
                + (
                    [
                        dict(
                            type="AddDesc",
                            per_tensor_desc=frcnn_kps_desc(
                                task_name="frcnn_kps_detection",
                                class_names=classnames,
                                label_output_name=f"{task_name}_detection_label",
                                offset_output_name=f"{task_name}_detection_offset",
                                roi_expand_param=roi_args["expand_param"],
                            ),
                            node_name=f"{task_name}_roi_desc",
                        ),
                    ]
                    if "test" in mode
                    else []
                ),
            ),
            target=dict(
                type="ProposalTarget",
                matcher=dict(
                    type="MaxIoUMatcher",
                    pos_iou=0.5,
                    neg_iou=0.5,
                    allow_low_quality_match=False,
                    low_quality_match_iou=0.5,
                    legacy_bbox=data_args["legacy_bbox"],
                ),
                label_encoder=dict(
                    type="RCNNKPSLabelFromMatch",
                    feat_h=8,
                    feat_w=8,
                    kps_num=2,
                    ignore_labels=(0, 3),
                    roi_expand_param=roi_args["expand_param"],
                ),
                node_name=f"{task_name}_roi_target",
            )
            if mode == "train"
            else None,
            loss=dict(
                type="RCNNKPSLoss",
                kps_num=2,
                cls_loss=dict(
                    type="SmoothL1Loss", loss_weight=1, reduction="mean"
                ),
                reg_loss=dict(
                    type="SmoothL1Loss", loss_weight=1, reduction="mean"
                ),
                feat_height=8,
                feat_width=8,
                node_name=f"{task_name}_roi_loss",
            )
            if mode == "train"
            else None,
            postprocess=dict(
                type="KpsDecoder",
                num_kps=2,
                pos_distance=1,
                roi_expand_param=roi_args["expand_param"],
                node_name=f"{task_name}_roi_decoder",
            )
            if "val" in mode
            else None,
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
    val=dict(),
    test=dict(),
)


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


# data
if lmdb_data:
    ds = datapaths.vehicle_wheel_kps
    data_paths = [d["data_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

    data_loader = dict(
        type=torch.utils.data.DataLoader,
        drop_last=True,
        num_workers=1,
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
                            type="KPSIterableDetRoITransform",
                            kps_num=2,
                            # roi transform
                            target_wh=resize_hw[::-1],
                            resize_wh=None
                            if resize_hw is None
                            else resize_hw[::-1],
                            img_scale_range=(0.5, 2.0),
                            roi_scale_range=(0.7, 1.0 / 0.7),
                            min_sample_num=1,
                            max_sample_num=1,
                            center_aligned=False,
                            inter_method=inter_method,
                            use_pyramid=True,
                            pyramid_min_step=0.7,
                            pyramid_max_step=0.8,
                            min_valid_area=64,
                            min_valid_clip_area_ratio=0.5,
                            min_edge_size=8,
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
                            type="PadDetData",
                            max_gt_boxes_num=200,
                            max_ig_regions_num=100,
                        ),
                        dict(
                            type="KPSDetInputPadding",
                            input_padding=input_padding,
                            kps_num=2,
                        ),
                    ],
                )
                for path in data_paths
            ],
        ),
    )
else:
    ds = datapaths.vehicle_wheel_kps
    rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
    anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

    data_loader = dict(
        type="GluonDataLoader",
        __build_recursive=False,
        input_padding=input_padding,
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
                target_wh=resize_hw[::-1],
                resize_wh=None if resize_hw is None else resize_hw[::-1],
                img_scale_range=(0.5, 2.0),
                roi_scale_range=(0.7, 1.0 / 0.7),
                min_sample_num=1,
                max_sample_num=1,
                center_aligned=False,
                inter_method=inter_method,
                use_pyramid=True,
                pyramid_min_step=0.7,
                pyramid_max_step=0.8,
                min_valid_area=64,
                min_valid_clip_area_ratio=0.5,
                min_edge_size=8,
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
                target_wh=resize_hw[::-1],
                max_gt_boxes_num=200,
                max_ig_regions_num=100,
            ),
            dict(type="CastEx", dtypes=(None,) + (np.float32,) * 3),
            dict(
                type="ToDict",
                keys=("img", "im_hw", "gt_boxes", "gt_boxes_num"),
            ),
            dict(
                type="PadKPSGTBeforeLabelGenerator",
                pad_width=input_padding[:2],
            ),
        ],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        last_batch="rollover",
    )
