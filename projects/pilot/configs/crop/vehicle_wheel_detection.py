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
)
from vehicle_detection import (
    anchor_args,
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
from hat.core.proj_spec.descs import frcnn_veh_wheel_detection_desc

task_name = "vehicle_wheel_detection"

classname2idxs = [6]
sub_classnames = ["vehicle_wheel_detection"]
task_type = "roi_detection"
num_classes = len(sub_classnames)

max_subbox_num = 10
subbox_score_thresh = (
    model_thresh["roi_det_thresh"][object_type] if model_thresh else 0.1
)
feature_h = 16
feature_w = 16

data_args = dict(legacy_bbox=True)

roi_args = dict(
    exclude_background=True,
    num_fg_classes=1,
    class_agnostic_reg=True,
    roi_wh_zoom_scale=(1.0, 1.0),
)


def get_model(mode):
    train_pred = deepcopy(anchor_pred)
    train_pred.update(
        pre_nms_top_k=2000,
        post_nms_top_k=128,
        use_clippings=False,
        nms_padding_mode="rollover",
        bbox_min_hw=(1, 1),
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
                "ig_regions",
                "ig_regions_num",
            ],
            roi_feat_extractor=roi_feat_extractor,
            head=dict(
                type="ExtSequential",
                modules=[
                    dict(
                        type="RCNNVarGNetShareHead",
                        bn_kwargs=bn_kwargs,
                        roi_out_channel=64,
                        gc_num_filter=128,
                        pw_num_filter=128,
                        pw_num_filter2=128,
                        group_base=8,
                        factor=1,
                        stride=1,
                        node_name="vehicle_wheel_kps_roi_share_head",
                    ),
                    dict(
                        type="RCNNVarGNetSplitHead",
                        num_fg_classes=roi_args["num_fg_classes"],
                        bn_kwargs=bn_kwargs,
                        class_agnostic_reg=roi_args["class_agnostic_reg"],
                        with_background=not roi_args["exclude_background"],
                        upscale=(feature_w > 8),
                        in_channel=128,
                        pw_num_filter2=96,
                        with_box_reg=True,
                        reg_channel_base=4,
                        use_bin=True,
                        node_name=f"{task_name}_roi_split_head",
                    ),
                ]
                + (
                    [
                        dict(
                            type="AddDesc",
                            per_tensor_desc=frcnn_veh_wheel_detection_desc(
                                task_name="frcnn_roi_detection",
                                class_names=classnames,
                                label_output_name="vehicle_wheel_detection_label",
                                offset_output_name="vehicle_wheel_detection_offset",
                                subbox_score_thresh=subbox_score_thresh,
                                max_subbox_num=max_subbox_num,
                                roi_w_expand_scale=roi_args[
                                    "roi_wh_zoom_scale"
                                ][0],
                                roi_h_expand_scale=roi_args[
                                    "roi_wh_zoom_scale"
                                ][1],
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
                    pos_iou=0.6,
                    neg_iou=0.6,
                    allow_low_quality_match=False,
                    low_quality_match_iou=0.3,
                    legacy_bbox=data_args["legacy_bbox"],
                ),
                # just for parent boxes ignore match
                ig_region_matcher=dict(
                    type="IgRegionMatcher",
                    num_classes=anchor_args["num_fg_classes"] + 1,
                    ig_region_overlap=0.5,
                    legacy_bbox=data_args["legacy_bbox"],
                    exclude_background=anchor_args["exclude_background"],
                ),
                label_encoder=dict(
                    type="RCNNMultiBinDetLabelFromMatch",
                    feature_h=feature_h,
                    feature_w=feature_w,
                    num_classes=num_classes,
                    max_subbox_num=max_subbox_num,
                    cls_on_hard=True,
                    reg_on_hard=True,
                    use_ig_region=True,
                    roi_w_zoom_scale=roi_args["roi_wh_zoom_scale"][0],
                    roi_h_zoom_scale=roi_args["roi_wh_zoom_scale"][1],
                ),
                node_name=f"{task_name}_roi_target",
            )
            if mode == "train"
            else None,
            loss=dict(
                type="RCNNMultiBinDetLoss",
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
                    type="SmoothL1Loss",
                    reduction="mean",
                ),
                node_name=f"{task_name}_roi_loss",
            )
            if mode == "train"
            else None,
            postprocess=dict(
                type="HeatmapMultiBox2dDecoder",
                subbox_score_thresh=subbox_score_thresh,
                nms_iou_thresh=0.01,
                legacy_bbox=data_args["legacy_bbox"],
                roi_wh_zoom_scale=roi_args["roi_wh_zoom_scale"],
                node_name=f"{task_name}_roi_decoder",
            )
            if "val" in mode
            else None,
        ),
    )


val_decoders[object_type][0].append(task_name)
val_decoders[object_type][1]["task_descs"][task_name] = (
    "pred_boxes",
    task_type,
)

# inputs
inputs = dict(
    train=dict(
        im_hw=torch.zeros((1, 2)),
        gt_boxes=torch.zeros((1, 10, max_subbox_num, 5)),
        gt_boxes_num=torch.zeros(1),
        ig_regions=torch.zeros((1, 10, max_subbox_num, 5)),
        ig_regions_num=torch.zeros(1),
        parent_gt_boxes=torch.zeros((1, 10, 5)),
        parent_gt_boxes_num=torch.zeros(1),
        parent_ig_regions=torch.zeros((1, 10, 5)),
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


# data
if lmdb_data:
    ds = datapaths.vehicle_wheel_detection

    use_parent = True
    parent_id = 1

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
                            type="DetSeg2DAnnoDatasetToRoiMultiDetectionFormat",
                            selected_class_ids=classname2idxs,
                            lt_point_id=10,
                            rb_point_id=12,
                            parent_lt_point_id=0,
                            parent_rb_point_id=2,
                            use_parent=use_parent,
                            parent_id=parent_id,
                        ),
                        dict(
                            type="ROIDetectionIterableMultiDetRoITransform",
                            # roi transform
                            target_wh=input_hw[::-1],
                            resize_wh=None,
                            img_scale_range=(0.7, 1.0 / 0.7),
                            roi_scale_range=(0.5, 2.0),
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
                            clip_bbox=True,
                            allow_outside_center=True,  # True is better than False
                            pixel_center_aligned=pixel_center_aligned,
                            keep_aspect_ratio=True,
                            # for multi-subbox
                            max_subbox_num=max_subbox_num,
                            ig_region_match_thresh=0.8,
                            min_valid_sub_area=8,
                            min_sub_edge_size=2,
                        ),
                        dict(
                            type="PadVehicleWheelData",
                            max_gt_boxes_num=100,
                            max_ig_regions_num=100,
                        ),
                    ],
                )
                for path in data_paths
            ],
        ),
    )
else:
    ds = datapaths.vehicle_wheel_detection
    rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
    anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
    sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

    use_parent = True
    parent_id = 1

    data_loader = dict(
        type="GluonDataLoader",
        __build_recursive=False,
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
        transform=[
            dict(
                type="LegacyDenseBoxImageRecordDatasetDecoder",
                to_rgb=True,
            ),
            dict(
                type="DecodeDenseBoxDatasetToRoiMultiDetectionFormat",
                selected_class_ids=classname2idxs,
                lt_point_id=10,
                rb_point_id=12,
                parent_lt_point_id=0,
                parent_rb_point_id=2,
                use_parent=use_parent,
                parent_id=parent_id,
            ),
            dict(
                type="ROIDetectionIterableMultiDetRoITransform",
                # roi transform
                target_wh=input_hw[::-1],
                resize_wh=None,
                img_scale_range=(0.7, 1.0 / 0.7),
                roi_scale_range=(0.5, 2.0),
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
                clip_bbox=True,
                allow_outside_center=True,  # True is better than False
                pixel_center_aligned=pixel_center_aligned,
                keep_aspect_ratio=True,
                # for multi-subbox
                max_subbox_num=max_subbox_num,
                ig_region_match_thresh=0.8,
                min_valid_sub_area=8,
                min_sub_edge_size=2,
            ),
            dict(
                type="PadVehicleWheelData",
                target_wh=input_hw[::-1],
                max_gt_boxes_num=100,
                max_ig_regions_num=100,
            ),
        ],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        last_batch="rollover",
    )
