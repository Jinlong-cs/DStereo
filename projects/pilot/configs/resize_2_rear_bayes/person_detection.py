import json
import os
from collections import OrderedDict
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
    get_det_rpn_out_keys,
    input_hw,
    inter_method,
    lmdb_data,
    log_freq,
    model_thresh,
    pixel_center_aligned,
    rand_translation_ratio,
    resize_hw,
    roi_region,
    test_roi_num,
    val_decoders,
    vanishing_point,
    vis_tasks,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.descs import frcnn_det_desc
from hat.core.proj_spec.detection import (
    classname2id,
    get_class_names_used_in_desc,
    get_det_default_merge_fn_type_and_params,
)
from hat.models.backbones.mixvargenet import MixVarGENetConfig

task_type = "detection"
vis_tasks.append("person")

classnames = ["person"]
classname2idxs = list(map(lambda x: classname2id[x], classnames))
object_type = "_".join(classnames)

task_name = f"{object_type}_{task_type}"

data_args = dict(legacy_bbox=True)
anchor_args = dict(
    feat_strides=[8, 16, 32, 64],
    anchor_wh_groups=[
        [[13, 35], [16, 33], [22, 60], [16, 45], [33, 80]],
        [[68, 163], [50, 122]],
        # [[152, 232]],
        # [[178, 363]],
        [[104, 326], [147, 340], [181, 456], [278, 559]],
        [[278, 559]],
    ],
    num_fg_classes=1,
    exclude_background=True,
)

roi_args = dict(
    class_agnostic_reg=True,
    exclude_background=True,
    num_fg_classes=1,
)

# sharable modules
anchor_generator = dict(
    type="AnchorGenerator",
    feat_strides=anchor_args["feat_strides"],
    anchor_wh_groups=anchor_args["anchor_wh_groups"],
    legacy_bbox=data_args["legacy_bbox"],
    node_name=f"{object_type}_anchor",
)

anchor_head = dict(
    type="RPNVarGNetHead",
    in_channels=32,
    num_channels=[32, 32, 32, 32],
    num_classes=anchor_args["num_fg_classes"],
    num_anchors=[len(_) for _ in anchor_args["anchor_wh_groups"]],
    feat_strides=anchor_args["feat_strides"],
    is_dim_match=False,
    bn_kwargs=bn_kwargs,
    factor=2,
    group_base=16,
    node_name=f"{object_type}_anchor_head",
)

anchor_pred = dict(
    type="AnchorPostProcess",
    num_classes=anchor_args["num_fg_classes"],
    class_offsets=[0] * len(anchor_args["feat_strides"]),
    use_clippings=True,
    image_hw=input_hw,
    input_key="rpn_head_out",
    nms_iou_threshold=0.7,
    nms_margin=0.0,
    box_filter_threshold=model_thresh["rpn_thresh"][object_type]
    if model_thresh
    else 0.0,
    pre_nms_top_k=2000,
    post_nms_top_k=test_roi_num,
    nms_padding_mode="pad_zero",
    bbox_min_hw=(1, 1),
    node_name=f"{object_type}_anchor_pred",
)

merge_fn_type, merge_fn_params = get_det_default_merge_fn_type_and_params()

anchor_desc = dict(
    type="AddDesc",
    strict=False,
    per_tensor_desc=[
        json.dumps(
            dict(
                task="frcnn_detection",
                class_name=get_class_names_used_in_desc(classnames),
                class_agnostic=1,
                score_act_type="identity",
                with_background=0,
                mean=(0, 0, 0, 0),
                std=(1, 1, 1, 1),
                reg_type="rcnn",
                legacy_bbox=int(
                    data_args["legacy_bbox"]
                ),  # use 0/1 instead of False/True
                nms_threshold=0.7,
                roi_regions=roi_region,
                vanishing_point=vanishing_point,
                merge_fn_type=merge_fn_type,
                merge_fn_params=merge_fn_params,
                output_name="rpn",
                score_threshold=[
                    model_thresh["rpn_thresh"][object_type]
                    if model_thresh
                    else 0.0,
                ],
            )
        ),
    ],
    node_name=f"{object_type}_anchor_desc",
)

roi_feat_extractor = dict(
    type="MultiScaleRoIAlign",
    output_size=(8, 8),
    feature_strides=anchor_args["feat_strides"],
    canonical_level=5,
    aligned=None,
    node_name=f"{object_type}_roi_feat_extractor",
)

roi_head = dict(
    type="RCNNMixVarGEShareHead",
    bias=True,
    bn_kwargs=bn_kwargs,
    head_config=[
        MixVarGENetConfig(
            in_channels=32,
            out_channels=64,
            head_op="mixvarge_f2",
            stack_ops=[],
            stride=2,
            extra_downsample_num=0,
        ),
        MixVarGENetConfig(
            in_channels=64,
            out_channels=128,
            head_op="mixvarge_f2",
            stack_ops=[],
            stride=1,
            extra_downsample_num=0,
        ),
        MixVarGENetConfig(
            in_channels=128,
            out_channels=128,
            head_op="mixvarge_f2",
            stack_ops=[],
            stride=1,
            extra_downsample_num=0,
        ),
    ],
    upscale=False,
    node_name="person_roi_share_head",
)


def get_model(mode):
    train_pred = deepcopy(anchor_pred)
    train_pred.update(
        pre_nms_top_k=10000,
        post_nms_top_k=500,
        nms_padding_mode="rollover",
    )

    return dict(
        type="TwoStageDetector",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[fpn_neck, fix_channel_neck]),
        rpn_out_keys=get_det_rpn_out_keys(mode),
        rpn_module=dict(
            type="AnchorModule",
            anchor_generator=anchor_generator,
            head=anchor_head,
            postprocess=train_pred if mode == "train" else anchor_pred,
            target=dict(
                node_name=f"{object_type}_anchor_target",
                type="BBoxTargetGenerator",
                matcher=dict(
                    type="MaxIoUMatcher",
                    pos_iou=0.6,
                    neg_iou=0.45,
                    allow_low_quality_match=True,
                    low_quality_match_iou=0.3,
                    legacy_bbox=data_args["legacy_bbox"],
                ),
                ig_region_matcher=dict(
                    type="IgRegionMatcher",
                    num_classes=anchor_args["num_fg_classes"] + 1,
                    ig_region_overlap=0.5,
                    legacy_bbox=data_args["legacy_bbox"],
                    exclude_background=anchor_args["exclude_background"],
                ),
                label_encoder=dict(
                    type="MatchLabelSepEncoder",
                    class_encoder=dict(
                        type="OneHotClassEncoder",
                        num_classes=anchor_args["num_fg_classes"] + 1,
                        class_agnostic_neg=False,
                        exclude_background=anchor_args["exclude_background"],
                    ),
                    bbox_encoder=dict(
                        type="XYWHBBoxEncoder",
                        legacy_bbox=data_args["legacy_bbox"],
                    ),
                    cls_use_pos_only=False,
                    cls_on_hard=False,
                    reg_on_hard=False,
                ),
            )
            if mode == "train"
            else None,
            loss=dict(
                type="RPNSepLoss",
                cls_loss=dict(
                    type="ElementwiseL2HingeLoss",
                    hard_neg_mining_cfg=dict(
                        keep_pos=True,
                        neg_ratio=0.5,
                        hard_ratio=0.5,
                        min_keep_num=32,
                    ),
                    reduction="mean",
                ),
                reg_loss=dict(
                    type="MSELoss",
                    clip_val=1,
                    loss_weight=32,
                    reduction="mean",
                ),
                node_name=f"{object_type}_anchor_loss",
            )
            if mode == "train"
            else None,
            desc=anchor_desc if mode != "train" else None,
        ),
        roi_module=dict(
            type="RoIModule",
            output_head_out=True,
            roi_key="pred_boxes",
            roi_feat_extractor=roi_feat_extractor,
            target_opt_keys=(
                "ig_regions",
                "ig_regions_num",
                "im_hw",
            ),
            head=dict(
                type="ExtSequential",
                modules=[
                    roi_head,
                    dict(
                        type="RCNNVarGNetSplitHead",
                        num_fg_classes=roi_args["num_fg_classes"],
                        bn_kwargs=bn_kwargs,
                        class_agnostic_reg=roi_args["class_agnostic_reg"],
                        with_background=not roi_args["exclude_background"],
                        in_channel=128,
                        pw_num_filter2=96,
                        with_box_reg=True,
                        reg_channel_base=4,
                        use_bin=False,
                        node_name=f"{task_name}_roi_split_head",
                    ),
                ]
                + (
                    [
                        dict(
                            type="AddDesc",
                            per_tensor_desc=frcnn_det_desc(
                                task_name="frcnn_detection",
                                class_names=classnames,
                                class_agnostic_reg=False,
                                roi_regions=roi_region,
                                vanishing_point=vanishing_point,
                                score_act_type="identity",
                                with_background=not roi_args[
                                    "exclude_background"
                                ],
                                legacy_bbox=data_args["legacy_bbox"],
                                score_threshold_per_class=[
                                    model_thresh["det_thresh"][object_type]
                                    if model_thresh
                                    else 0.5
                                ],
                                nms_threshold=[0.5],
                            ),
                            node_name=f"{task_name}_roi_desc",
                        )
                    ]
                    if "test" in mode
                    else []
                ),
            ),
            target=dict(
                node_name=f"{task_name}_roi_target",
                type="ProposalTarget",
                matcher=dict(
                    type="MaxIoUMatcher",
                    pos_iou=0.5,
                    neg_iou=0.5,
                    allow_low_quality_match=True,
                    low_quality_match_iou=0.3,
                    legacy_bbox=data_args["legacy_bbox"],
                    clip_gt_before_matching=True,
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
                    bbox_encoder=dict(
                        type="XYWHBBoxEncoder",
                        legacy_bbox=data_args["legacy_bbox"],
                    ),
                    cls_use_pos_only=False,
                    cls_on_hard=False,
                    reg_on_hard=False,
                ),
            )
            if mode == "train"
            else None,
            loss=dict(
                type="RCNNLoss",
                cls_loss=dict(
                    type="ElementwiseL2HingeLoss",
                    hard_neg_mining_cfg=dict(
                        keep_pos=True,
                        neg_ratio=0.75,
                        hard_ratio=0.5,
                        min_keep_num=32,
                    ),
                    reduction="mean",
                ),
                reg_loss=dict(
                    type="MSELoss",
                    clip_val=1,
                    loss_weight=16,
                    reduction="mean",
                ),
                class_agnostic_reg=roi_args["class_agnostic_reg"],
                node_name=f"{task_name}_roi_loss",
            )
            if mode == "train"
            else None,
            postprocess=dict(
                type="RCNNDecoder",
                bbox_decoder=dict(
                    type="XYWHBBoxDecoder",
                    legacy_bbox=data_args["legacy_bbox"],
                ),
                cls_act_type="identity",
                cls_name_mapping={
                    i + 1: cls_name for i, cls_name in enumerate(classnames)
                },
                node_name=f"{task_name}_roi_decoder",
            )
            if "val" in mode
            else None,
        ),
    )


val_decoders[object_type] = (
    [task_name],
    dict(
        type="RoIDecoder",
        task_descs=OrderedDict([(task_name, ("pred_boxes", task_type))]),
        im_hw=input_hw,
        clip_by_im_hw=True,
        min_filter_hw=(2, 2),
        nms_threshold=0.5,
        score_threshold=model_thresh["det_thresh"][object_type]
        if model_thresh
        else 0.1,
        node_name=f"{object_type}_decoder",
    ),
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

# metrics
loss_names = [
    "rpn_cls_loss",
    "rpn_reg_loss",
    "rcnn_cls_loss",
    "rcnn_reg_loss",
]
metric_updater = dict(
    type="MetricUpdater",
    metrics=[dict(type="LossShow", name=name) for name in loss_names],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_{name}$",
            )
            for name in loss_names
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


# data
if lmdb_data:
    ds = datapaths.person
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
                            type="DetSeg2DAnnoDatasetToDetFormat",
                            selected_class_ids=classname2idxs,
                            lt_point_id=0,
                            rb_point_id=2,
                        ),
                        dict(
                            type="IterableDetRoITransform",
                            # roi transform
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
                            pixel_center_aligned=pixel_center_aligned,
                            min_valid_area=8,
                            min_valid_clip_area_ratio=0.5,  # noqa
                            min_edge_size=2,
                            rand_translation_ratio=rand_translation_ratio,
                            rand_aspect_ratio=0.0,
                            rand_rotation_angle=0,
                            flip_prob=0.5,
                            reselect_ratio=-1,
                            clip_bbox=True,
                            keep_aspect_ratio=True,
                        ),
                        dict(
                            type="PadDetData",
                            max_gt_boxes_num=300,
                            max_ig_regions_num=100,
                        ),
                    ],
                )
                for path in data_paths
            ],
        ),
    )
else:
    ds = datapaths.person
    rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
    anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

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
                # roi transform
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
                pixel_center_aligned=pixel_center_aligned,
                min_valid_area=8,
                min_valid_clip_area_ratio=0.5,
                min_edge_size=2,
                rand_translation_ratio=rand_translation_ratio,
                rand_aspect_ratio=0.0,
                rand_rotation_angle=0,
                flip_prob=0.5,
                reselect_ratio=-1,
                clip_bbox=True,
                keep_aspect_ratio=True,
            ),
            dict(
                type="PadDetData",
                target_wh=input_hw[::-1],
                max_gt_boxes_num=300,
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
