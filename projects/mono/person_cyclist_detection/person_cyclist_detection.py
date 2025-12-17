from copy import deepcopy

import torch
from common import (
    backbone,
    get_anchor_generator,
    get_anchor_head,
    get_anchor_model_desc,
    get_anchor_post_process_cfg,
    get_unet_neck,
    input_hw,
    inter_method,
    log_freq,
    min_valid_clip_area_ratio,
    norm_len,
    norm_method,
    pipeline_test,
    pixel_center_aligned,
    stride2channels,
)
from datasets import datapaths, parse_dataset

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.proj_spec.detection import classname2id
from hat.data.collates.collates import collate_2d

# -------------------------------------------------------
# task define
# ------------------------------------------------------
task_name = "person_cyclist_detection"
num_classes = 2

classnames = ["person_2pe", "cyclist_2pe"]
classname2idxes = list(map(lambda x: classname2id[x], classnames))
# classname2idxes = [1, 7]
# ------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------
inputs = dict(
    train=dict(
        gt_boxes=torch.zeros((1, 100, 14)),
        gt_boxes_num=torch.zeros(1),
        ig_regions=torch.zeros((1, 100, 5)),
        ig_regions_num=torch.zeros(1),
        im_hw=torch.zeros((1, 2)),
    ),
    val=dict(),
    test=dict(),
)

# ------------------------------------------------------------------------------
# data loader
# ------------------------------------------------------------------------------
data_desc = parse_dataset(datapaths, classnames[0])
train_data_desc = data_desc.train_data_desc
val_data_desc = data_desc.val_data_desc
# use val dataset to boost pipeline-test, noqa
train_data_desc = train_data_desc if not pipeline_test else val_data_desc
train_batch_size = train_data_desc.batch_size_per_ctx
rec_paths = train_data_desc.rec_paths
anno_paths = train_data_desc.anno_paths
roi_list_paths = train_data_desc.roi_list_paths
sample_weights = train_data_desc.sample_weights
# data loader

datasets = [
    dict(
        type="DenseboxWithRoilistDataset",
        data_path=rec_path_i,
        anno_path=anno_path_i,
        roi_list_path=roi_list_path_i,
        transforms=[
            dict(
                type="DetSeg2DAnnoDatasetToDetFormat",
                selected_class_ids=classname2idxes,
                lt_point_id=0,
                rb_point_id=2,
                min_edge_size=0.1,
            ),
            dict(
                type="RoiTransformCroper",
                # roi transform
                min_sample_num=1,
                max_sample_num=10,
                roi_crop_parm=dict(
                    norm_len=norm_len,
                    norm_method=norm_method,
                    output_wh=input_hw[::-1],
                    input_wh=None,
                    max_crop_scale=1 / 0.9,
                    min_crop_scale=0.9,
                    max_coord_jitter_ratio=0.7,
                    img_min_scale=0.15,
                    img_max_scale=6,
                    padd_val=0,
                    random_roi_ratio=0.4,
                    restrict_roi_in_center=False,
                    flip_ratio=0.3,
                ),
                img_crop_parm=dict(
                    target_wh=input_hw[::-1],
                    inter_method=inter_method,
                    use_pyramid=True,
                    pyramid_min_step=0.7,
                    pyramid_max_step=0.8,
                    pixel_center_aligned=pixel_center_aligned,
                ),
                bbox_ts_parm=dict(
                    clip=True,
                    min_valid_area=10,
                    min_valid_clip_area_ratio=min_valid_clip_area_ratio,
                    # work on 2pe roi
                    min_edge_size=1,
                    label_type="detection",
                ),
                from_roi_ratio=0.5,
                transforms=[
                    dict(
                        type="PadDetData",
                        max_gt_boxes_num=100,
                        max_ig_regions_num=50,
                    ),
                ],
            ),
        ],
        to_rgb=True,
        task_type="detection",
    )
    for rec_path_i, anno_path_i, roi_list_path_i in zip(
        rec_paths, anno_paths, roi_list_paths
    )
]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="DistributedComposeRandomDataset",
        datasets=datasets,
        sample_weights=sample_weights,
        multi_sample_output=True,
    ),
    collate_fn=collate_2d,
    batch_size=train_batch_size,
    # shuffle=True,
    num_workers=40,
    pin_memory=True,
    persistent_workers=True,
)


# -------------------------------------------------------
# models
# ------------------------------------------------------
def get_model(mode):
    box_filter_threshold = 0.05
    feat_strides = [4]
    anchor_wh_groups = [[[40, 92], [92, 92]]]
    data_args = dict(legacy_bbox=True)
    anchor_args = dict(
        feat_strides=feat_strides,
        anchor_wh_groups=anchor_wh_groups,
        num_fg_classes=num_classes,
        exclude_background=True,
    )
    anchor_generator = get_anchor_generator(
        anchor_args, data_args, task_name=task_name
    )
    anchor_head = get_anchor_head(
        in_channels=[stride2channels[i] for i in anchor_args["feat_strides"]],
        num_channels=[stride2channels[i] for i in anchor_args["feat_strides"]],
        anchor_args=anchor_args,
        task_name=task_name,
    )
    anchor_pred = get_anchor_post_process_cfg(
        anchor_args,
        task_name=task_name,
        box_filter_threshold=box_filter_threshold,
    )

    model_descs = get_anchor_model_desc(
        classnames,
        data_args,
        anchor_args,
        prediction_return_cnt=4,
    )
    unet_neck = get_unet_neck()

    train_pred = deepcopy(anchor_pred)
    train_pred.update(
        pre_nms_top_k=2000,
        post_nms_top_k=100,
        nms_padding_mode="rollover",
    )
    model = dict(
        type="OneStageDetector",
        backbone=backbone,
        neck=unet_neck,
        rpn_out_keys=["pred_boxes_out", "pred_scores", "pred_cls"]
        if mode != "train"
        else None,
        rpn_module=dict(
            type="AnchorModule",
            anchor_generator=anchor_generator,
            head=anchor_head,
            # output_head_out=True,
            postprocess=None
            if mode == "train"
            else anchor_pred,  # anchor_pred,
            target=dict(
                node_name=f"{task_name}_anchor_target",
                type="BBoxTargetGenerator",
                matcher=dict(
                    type="MaxIoUMatcher",
                    pos_iou=0.5,
                    neg_iou=0.35,
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
                    type="SmoothL1Loss",
                    loss_weight=16,
                    reduction="mean",
                ),
                node_name=f"{task_name}_anchor_loss",
            )
            if mode == "train"
            else None,
            desc=model_descs if mode != "train" else None,
        ),
    )
    return model


# ------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------
loss_names = [
    "rpn_cls_loss",
    "rpn_reg_loss",
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
