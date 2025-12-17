# import copy
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from common import (  # val_decoders,; neck_out_strides,; neck_stride2channels,
    backbone,
    batch_size,
    datapaths,
    fix_channel_neck,
    fpn_neck,
    input_cat,
    input_hw,
    inter_method,
    log_freq,
    min_valid_clip_area_ratio,
    model_thresh,
    pixel_center_aligned,
    rand_translation_ratio,
    resize_hw,
    roi_region,
    split_mode,
    val_decoders,
    val_transforms,
    vanishing_point,
    view_num,
    vis_tasks_2d,
)

from hat.core.proj_spec.detection import (  # get_class_names_used_in_desc,; get_det_default_merge_fn_type_and_params,
    classname2id,
)

view_num = 1 if split_mode else view_num

task_type = "detection"
classnames = ["vehicle_plate"]
classname2idxs = list(map(lambda x: classname2id[x], classnames))
object_type = "_".join(classnames)
vis_tasks_2d.append("vehicle_plate")
training_step = os.getenv("HAT_TRAINING_STEP")
task_name = f"{object_type}_{task_type}"
# task description
desc_task_name = "detection"
desc_class_names = [task_name]
desc_out_names = [
    "point_coordinate",
    "score",
    "bbox",
    "centerness",
]
score_threshold = 0.5  # you have to think about which threshold is suitable
feat_channels = 24
num_classes = 1
norm_target_bbox = True
INF = 1e8
default_regress_ranges = (
    (-1, 64),
    (64, 128),
    (128, 256),
    (256, INF),
    # (512, INF),
)
task_loss_weight = 0.05
range_multiplier = 0.5
regress_ranges = tuple(
    (
        (x * range_multiplier, y * range_multiplier)
        for x, y in (default_regress_ranges)
    )
)
neck_out_strides = [8, 16, 32, 64]
feature_channles = 32
neck_stride2channels = {
    stride: feature_channles for stride in neck_out_strides
}
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(name="nms", iou_threshold=0.6),
    max_per_img=100,
)
test_transforms = val_transforms
data_args = dict(legacy_bbox=True)

use_iou_replace_ctrness = False
enable_auto_assign = False
# sharable modules


def get_model(mode):
    model = dict(
        type="FCOS",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[fpn_neck, fix_channel_neck]),
        head=dict(
            type="FCOSHead",
            num_classes=num_classes,
            in_strides=neck_out_strides,
            out_strides=neck_out_strides,
            stride2channels=neck_stride2channels,
            feat_channels=feat_channels,
            stacked_convs=2,
            use_sigmoid=True,
            share_bn=False,
            upscale_bbox_pred=not norm_target_bbox,
            node_name=f"{object_type}_fcos_head",
        ),
        targets=dict(
            type="FCOSTarget",
            strides=neck_out_strides,
            regress_ranges=regress_ranges,
            cls_out_channels=num_classes,
            background_label=num_classes,
            use_iou_replace_ctrness=use_iou_replace_ctrness,
            norm_on_bbox=norm_target_bbox,
            node_name=f"{object_type}_fcos_target",
        ),
        loss_cls=dict(
            type="FocalLoss",
            loss_name="loss_cls",
            num_classes=num_classes + 1,
            alpha=0.25,
            gamma=2.0,
            # from_logits=False,
            loss_weight=1.0 * task_loss_weight,
            node_name=f"{object_type}_fcos_loss_cls",
        ),
        loss_reg=dict(
            type="GIoULoss",
            loss_name="loss_bbox",
            loss_weight=1.0 * task_loss_weight,
            node_name=f"{object_type}_fcos_loss_reg",
        ),
        loss_centerness=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            loss_name="loss_centerness"
            if not use_iou_replace_ctrness
            else "loss_iou",  # noqa
            loss_weight=1.0 * task_loss_weight,
            node_name=f"{object_type}_fcos_loss_centerness",
        ),
    )

    if mode != "train":
        model["head"].update(dict(upscale_bbox_pred=True, dequant_output=True))
        model["targets"] = None
        # model["loss"] = None
        model["post_process"] = dict(
            type="FCOSDecoder",
            node_name=f"{object_type}_fcos_decoder",
            num_classes=num_classes,
            strides=neck_out_strides,
            test_cfg=test_cfg,
            nms_sqrt=True if not enable_auto_assign else False,
            meta_data_bool=False,
            label_offset=1,
        )
        if mode == "test":
            model = model
            model["targets"] = None
            # model["loss"] = None
            add_desc_pp = dict(
                type="AddDesc",
                per_tensor_desc=[
                    # num of desc == num of pred output tensors (one tensor per out stride)
                    json.dumps(
                        dict(
                            task=desc_task_name,
                            class_name=desc_class_names,
                            output_name=out_name,
                            stride=s,
                            score_threshold=model_thresh["det_thresh"][
                                task_name
                            ]
                            if model_thresh
                            else score_threshold,
                            roi_regions=roi_region,
                            vanishing_point=vanishing_point,
                            output_id=i,
                        )
                    )
                    for s in neck_out_strides
                    for i in range(view_num)
                    for out_name in desc_out_names
                ],
                node_name=f"{object_type}_fcos_desc",
            )
            filter_module = dict(
                type="FCOSMultiStrideFilter",
                strides=neck_out_strides,
                threshold=-4.59,
                node_name=f"{object_type}_fcos_multistride_filter",
            )
            if training_step == "int_infer":
                model["post_process"] = dict(
                    type="MultiInputSequential",
                    modules=[
                        filter_module,
                        add_desc_pp,
                    ],  # should be the last module
                )
                model["head"]["upscale_bbox_pred"] = False
                model["head"]["dequant_output"] = False
            else:
                model["post_process"] = dict(
                    type="FCOSDecoder",
                    node_name=f"{object_type}_fcos_decoder",
                    num_classes=num_classes,
                    strides=neck_out_strides,
                    test_cfg=test_cfg,
                    nms_sqrt=True if not enable_auto_assign else False,
                    meta_data_bool=False,
                    label_offset=1,
                )
                model["head"]["upscale_bbox_pred"] = True
                model["head"]["dequant_output"] = True
    return dict(
        type="ListInputModelWraper",
        # compile=mode == "test",
        seq_len=view_num,
        preprocess=input_cat,
        model=model,
    )


val_decoders[object_type] = (
    [task_name],
    dict(
        type="RoIDecoder",
        task_descs=OrderedDict([(task_name, ("pred_boxes", task_type))]),
        im_hw=input_hw,
        clip_by_im_hw=False,
        min_filter_hw=None,
        nms_threshold=None,
        score_threshold=model_thresh["det_thresh"][task_name]
        if model_thresh
        else None,
        node_name=f"{object_type}_decoder",
    ),
)


# inputs
inputs = dict(
    train=dict(
        gt_boxes=torch.zeros((view_num, 100, 5)),
        gt_bboxes=torch.zeros((view_num, 100, 4)),
        gt_classes=torch.zeros((view_num, 100)),
        gt_boxes_num=torch.zeros(view_num),
        ig_regions=torch.zeros((view_num, 110, 5)),
        ig_regions_num=torch.zeros(view_num),
        im_hw=torch.zeros((view_num, 2)),
    ),
    val=dict(),
    test=dict(),
)


# metrics
# -------------------------- solver --------------------------
def get_update_metric(is_train=False):
    def update_metric(metrics, batch, model_outs):
        if is_train:
            # model_outs[0].popitem(last=False)
            # losses = [for output in model_outs if 'loss' in output]
            # if len(model_outs[0]) == 4:
            #     model_outs[0] = model_outs[0][1:]
            keys = list(model_outs[0]._asdict().keys())
            loss_keys = []
            for i in range(len(keys)):
                key = keys[i]
                if "loss" in key:
                    loss_keys.append(i)

            # for metric, out in zip(metrics, [model_outs[0][key] for key in loss_keys]):
            for i in range(len(loss_keys)):
                metric = metrics[i]
                out = model_outs[0][loss_keys[i]]
                if is_train:
                    metric.update(out)
        # else:
        #     # reorganize model prediction results
        #     assert len(metrics) == 1
        #     metrics[0].update(coco_metric_reorganize(model_outs[0]))

    return update_metric


metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="loss_cls"),
        dict(type="LossShow", name="loss_bbox"),
        dict(
            type="LossShow",
            name="loss_centerness"
            if not use_iou_replace_ctrness
            else "loss_iou",
        ),  # noqa
    ]
    if not enable_auto_assign
    else [
        dict(type="LossShow", name="loss_pos"),
        dict(type="LossShow", name="loss_neg"),
        dict(type="LossShow", name="loss_center"),
    ],
    metric_update_func=get_update_metric(is_train=True),
    # TODO(min.du, 0.5): 1 should not show up here #
    filter_condition=lambda x: x[1] == task_name,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)


# data
ds = datapaths.plate
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
                    as_nd=False,
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
            min_valid_area=16,
            min_valid_clip_area_ratio=min_valid_clip_area_ratio,
            min_edge_size=4,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=0.0,
            rand_rotation_angle=0,
            flip_prob=0.5,
            reselect_ratio=-1,
            clip_bbox=False,
            keep_aspect_ratio=True,
        ),
        dict(
            type="PadDetData",
            target_wh=input_hw[::-1],
            max_gt_boxes_num=100,
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
        dict(type="TransformBboxToFcosFormat", change_label_bool=True),
    ],
)
