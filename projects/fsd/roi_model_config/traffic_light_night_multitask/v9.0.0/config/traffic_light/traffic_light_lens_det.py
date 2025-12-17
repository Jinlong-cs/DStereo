from collections import OrderedDict
from copy import deepcopy

import torch
from common import (
    backbone,
    batch_size,
    datapaths,
    get_anchor_generator,
    get_anchor_head,
    get_anchor_model_desc,
    get_anchor_post_process_cfg,
    get_model_track_desc,
    get_unet_neck,
    input_preprocess,
    input_size,
    inter_method,
    log_freq,
    loss_weight,
    mask_in_bpu,
    norm_length,
    norm_method,
    num_worker,
    pixel_center_aligned,
    stride2channels,
    val_decoders,
)

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.data.collates.collates import collate_2d

task_type = "detection"
task_name = f"traffic_light_lens_{task_type}"
classnames = ["traffic_light_lens"]
traffic_lens_color_sub_classnames = ["green", "yellow", "red"]
taffic_lens_type_sub_classnames = [
    "L_Circle",
    "L_Forward",
    "L_Left",
    "L_Right",
    "L_Return",
    "L_Pedestrain",
    "L_Non_Motor",
    "L_Time",
    "L_left_and_return",
    "L_Forward_and_Left",
    "L_Forward_and_Right",
    "L_No_Drive_into",
    "L_Allow_Drive_into",
]
traffic_lens_color_eval_classnames = [
    "green",
    "yellow",
    "red",
    "white",
    "other",
]
traffic_lens_type_eval_classnames = [
    "L_Circle",
    "L_Forward",
    "L_Left",
    "L_Right",
    "L_Return",
    "L_Pedestrain",
    "L_Non_Motor",
    "L_Time",
    "L_Other",
    "L_left_and_return",
    "L_Forward_and_Left",
    "L_Forward_and_Right",
    "L_No_Drive_into",
    "L_Allow_Drive_into",
    "L_unknown",
    "text_of_allow_ped",
    "sign_of_allow_ped",
    "text_of_forbid_ped",
    "sign_of_forbid_ped",
    "REMOVE",
]
num_color_cls = len(traffic_lens_color_sub_classnames)
num_type_cls = len(taffic_lens_type_sub_classnames)
num_eval_color = len(traffic_lens_color_eval_classnames)
num_eval_type = len(traffic_lens_type_eval_classnames)


def get_model(mode):
    box_filter_threshold = 0.05
    out_strides = (4,)
    feat_strides = [4]
    anchor_wh_groups = [[[16, 16], [24, 24], [52, 52]]]
    data_args = dict(legacy_bbox=True)
    anchor_args = dict(
        feat_strides=feat_strides,
        anchor_wh_groups=anchor_wh_groups,
        num_fg_classes=1,
        exclude_background=True,
        attr_list=[["type", num_type_cls], ["color", num_color_cls]],
    )
    anchor_generator = get_anchor_generator(
        anchor_args, data_args, task_name=task_name
    )
    anchor_head = get_anchor_head(
        in_channels=[stride2channels[i] for i in anchor_args["feat_strides"]],
        num_channels=[stride2channels[i] for i in anchor_args["feat_strides"]],
        anchor_args=anchor_args,
        dequant_output=False if mode == "test" else True,
        mode=mode,
        task_name=task_name,
    )
    anchor_pred = get_anchor_post_process_cfg(
        anchor_args,
        task_name=task_name,
        box_filter_threshold=box_filter_threshold,
        nms_iou_threshold=0.2,
        mode=mode,
    )

    model_descs = get_anchor_model_desc(
        classnames, data_args, anchor_args, out_strides, task_name
    )
    unet_neck = get_unet_neck(out_strides=out_strides)

    train_pred = deepcopy(anchor_pred)
    train_pred.update(
        pre_nms_top_k=500,
        post_nms_top_k=100,
        nms_padding_mode="rollover",
    )

    model = dict(
        type="OneStageDetector",
        input_preprocess=input_preprocess,
        backbone=backbone,
        neck=unet_neck,
        rpn_out_keys=["pred_boxes_out", "pred_scores", "pred_cls", "attr_cls"]
        if mode == "val"
        else None,
        rpn_module=dict(
            type="AnchorModule",
            anchor_generator=anchor_generator,
            head=anchor_head,
            postprocess=anchor_pred,
            target=dict(
                node_name=f"{task_name}_anchor_target",
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
                    type="TrafficLensAttrlabelEncoder",
                    class_encoder=dict(
                        type="OneHotClassEncoder",
                        num_classes=anchor_args["num_fg_classes"] + 1,
                        class_agnostic_neg=False,
                        exclude_background=anchor_args["exclude_background"],
                    ),
                    attr_encoder_list=[
                        [
                            attr_type,
                            dict(
                                type="OneHotClassEncoder",
                                num_classes=attr_ch + 1,
                                class_agnostic_neg=False,
                                exclude_background=anchor_args[
                                    "exclude_background"
                                ],
                            ),
                        ]
                        for attr_type, attr_ch in anchor_args["attr_list"]
                    ],
                    bbox_encoder=dict(
                        type="XYWHBBoxEncoder",
                        legacy_bbox=data_args["legacy_bbox"],
                    ),
                    cls_use_pos_only=False,
                    cls_on_hard=False,
                    reg_on_hard=False,
                    attr_use_pos_only=True,
                ),
            )
            if mode == "train"
            else None,
            loss=dict(
                type="TLAttrRPNSepLoss",
                cls_loss=dict(
                    type="ElementwiseL2HingeLoss",
                    hard_neg_mining_cfg=dict(
                        keep_pos=True,
                        neg_ratio=0.5,
                        hard_ratio=0.5,
                        min_keep_num=32,
                    ),
                    loss_weight=loss_weight["lens_cls"],
                    reduction="mean",
                ),
                reg_loss=dict(
                    type="SmoothL1Loss",
                    loss_weight=loss_weight["lens_reg"],
                    reduction="mean",
                ),
                node_name=f"{task_name}_anchor_loss",
            )
            if mode == "train"
            else None,
            desc=model_descs if mode == "test" else None,
        ),
        mask_in_bpu=mask_in_bpu,
        track_feature=True if mode == "test" else False,
        track_desc=get_model_track_desc(task_name) if mode == "test" else None,
    )
    return model


val_decoders[task_type] = (
    [task_name],
    dict(
        type="TLDetAttrDecoder",
        score_threshold_per_class=[0.1],
        task_descs=OrderedDict([(task_name, ("pred_boxes", task_type))]),
        node_name=f"{task_type}_decoder",
    ),
)


inputs = dict(
    train=dict(
        gt_boxes=torch.zeros((1, 100, 7)),
        gt_boxes_num=torch.zeros(1),
        ig_regions=torch.zeros((1, 100, 5)),
        ig_regions_num=torch.zeros(1),
        im_hw=torch.zeros((1, 2)),
    ),
    val=dict(),
    test=dict(),
)


loss_names = ["rpn_cls_loss", "rpn_reg_loss", "type_loss", "color_loss"]
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


ds = datapaths.traffic_light_lens
rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]
anno_paths = [rec.replace(".rec", ".json") for rec in rec_paths]
pb_rec_paths = [d["anno_path"] for d in ds["train_data_paths"]]
anno_idx_paths = [rec_p.replace(".rec", ".rec.idx") for rec_p in rec_paths]
roilist_paths = [d["roi_list_path"] for d in ds["train_data_paths"]]
sample_weights = [d["sample_weight"] for d in ds["train_data_paths"]]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    drop_last=True,
    num_workers=num_worker,
    persistent_workers=True,
    batch_size=batch_size,
    collate_fn=collate_2d,
    dataset=dict(
        type="DistributedComposeRandomDataset",
        sample_weights=sample_weights,
        multi_sample_output=True,
        datasets=[
            dict(
                type="DenseboxWithRoilistDataset",
                data_path=rec_path,
                anno_path=pb_rec_path,
                rec_idx_file_path=anno_idx_path,
                roi_list_path=roi_path,
                transforms=[
                    dict(
                        type="TrafficLens2DToDetFormat",
                        selected_class_ids=[6],
                        lt_point_id=0,
                        rb_point_id=2,
                        min_edge_size=0.1,
                    ),
                    dict(
                        type="RoiTransformCroperWithMask",
                        # roi transform
                        min_sample_num=1,
                        max_sample_num=100,
                        roi_crop_parm=dict(
                            norm_len=norm_length,
                            norm_method=norm_method,
                            output_wh=input_size,
                            input_wh=None,
                            max_crop_scale=1.1,
                            min_crop_scale=0.95,
                            max_coord_jitter_ratio=0.1,
                            img_min_scale=0.01,
                            img_max_scale=100,
                            padd_val=0,
                            random_roi_ratio=0.0,
                            restrict_roi_in_center=False,
                            flip_ratio=0.0,
                        ),
                        img_crop_parm=dict(
                            target_wh=input_size,
                            inter_method=inter_method,
                            use_pyramid=True,
                            pyramid_min_step=0.7,
                            pyramid_max_step=0.8,
                            pixel_center_aligned=pixel_center_aligned,
                        ),
                        bbox_ts_parm=dict(
                            clip=True,
                            min_valid_area=10,
                            min_valid_clip_area_ratio=0.02,
                            # work on 2pe roi
                            min_edge_size=1,
                            label_type=task_type,
                        ),
                        from_roi_ratio=0.0,
                        mask_out_gt_crop_flag=True,
                        mask_in_bpu=mask_in_bpu,
                        extend_ratio=0.1,
                        transforms=[
                            dict(
                                type="PadDetData",
                                max_gt_boxes_num=10,
                                max_ig_regions_num=10,
                            ),
                        ],
                    ),
                ],
                to_rgb=False,
                task_type="detection",
            )
            for rec_path, pb_rec_path, anno_idx_path, roi_path in zip(
                rec_paths, pb_rec_paths, anno_idx_paths, roilist_paths
            )
        ],
    ),
)
