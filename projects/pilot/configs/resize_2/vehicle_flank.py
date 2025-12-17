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
    inter_method,
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
from hat.core.proj_spec.descs import frcnn_flank_desc

task_name = "vehicle_flank"
task_type = "line"


data_args = dict(legacy_bbox=True)

roi_args = dict(
    exclude_background=True,
    num_fg_classes=1,
    class_agnostic_reg=True,
    with_tracking_feat=False,
)


def get_model(mode):
    train_pred = deepcopy(anchor_pred)
    train_pred.update(
        pre_nms_top_k=2000,
        post_nms_top_k=128,
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
            target_keys=["gt_boxes", "gt_flanks"],
            target_opt_keys=["gt_boxes_num", "gt_flanks_num"],
            roi_feat_extractor=roi_feat_extractor,
            head=dict(
                type="RCNNVarGNetHead",
                with_box_cls=True,
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
                reg_channel_base=4,
                node_name=f"{task_name}_roi_head",
            ),
            target=dict(
                type="ProposalTargetGroundLine",
                matcher=dict(
                    type="MaxIoUMatcher",
                    pos_iou=0.6,
                    neg_iou=0.6,
                    allow_low_quality_match=False,
                    low_quality_match_iou=0.5,
                    legacy_bbox=data_args["legacy_bbox"],
                ),
                label_encoder=dict(
                    type="MatchLabelFlankEncoder",
                    use_cls=True,
                    cls_on_hard=False,
                    reg_on_hard=False,
                    reg_on_inside_only=False,
                ),
                node_name=f"{task_name}_roi_target",
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
                    type="SmoothL1Loss",
                    loss_weight=16,
                    reduction="mean",
                ),
                node_name=f"{task_name}_roi_loss",
            )
            if mode == "train"
            else None,
            head_desc=dict(
                type="AddDesc",
                per_tensor_desc=frcnn_flank_desc(
                    task_name="frcnn_gdl_detection",
                    class_names=classnames,
                    label_output_name="vehicle_ground_line_label",
                    offset_output_name="vehicle_ground_line_reg",
                    feature="undistortion",
                ),
                node_name="vehicle_ground_line_roi_desc",
            )
            if mode != "train"
            else None,
            postprocess=dict(
                type="FlankPointDecoder",
                node_name=f"{task_name}_roi_decoder",
            )
            if "val" in mode
            else None,
        ),
    )


val_decoders[object_type][0].append(task_name)
val_decoders[object_type][1]["task_descs"][task_name] = (
    "pred_flank",
    task_type,
)

# inputs
inputs = dict(
    train=dict(
        gt_boxes=torch.zeros((1, 10, 5)),
        gt_boxes_num=torch.zeros(1),
        gt_flanks=torch.zeros((1, 10, 2, 3)),
        gt_flanks_num=torch.zeros(1),
        ig_regions=torch.zeros((1, 10, 5)),
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
        dict(type="LossShow", name="rcnn_reg_loss"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_rcnn_cls_loss$",
            ),
            dict(
                label_pattern=None,
                pred_pattern=f"^.*{task_name}_rcnn_reg_loss$",
            ),
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


# data
ds = datapaths.vehicle_flank
rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
anno_paths = [d["anno_path"] for d in ds["train_data_paths"]]
sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

data_loader = dict(
    type="GluonDataLoader",
    __build_recursive=False,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    last_batch="rollover",
    dataset=[
        dict(
            type="SplitDataset",
            dataset=dict(
                type="KPSDataset",
                img_rec_path=rec_path_i,
                anno_rec_path=anno_path_i,
                to_rgb=True,
            ),
            even_split=False,
        )
        for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
    ],
    transform=[
        dict(
            type="FlankCornersDetRoiTransform",
            # roi transform
            target_wh=resize_hw[::-1],
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
            min_valid_area=100,
            min_valid_clip_area_ratio=0.3,  # low value for truncation
            min_edge_size=10,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0,
            flip_prob=0.5,
            keep_aspect_ratio=True,
            min_flank_width=10,
            point_clip_mode="inside",  # 'vertical', 'horizon', 'inside'
            is_directed=False,
        ),
        dict(
            type="PadFlankCornersData",
            target_wh=resize_hw[::-1],
            max_gt_boxes_num=200,
            max_ig_regions_num=100,
        ),
        dict(type="CastEx", dtypes=(None,) + (np.float32,) * 7),
        dict(
            type="ToDict",
            keys=(
                "img",
                "im_hw",
                "gt_boxes",
                "gt_boxes_num",
                "gt_flanks",
                "gt_flanks_num",
                "ig_regions",
                "ig_regions_num",
            ),
        ),
    ],
)
