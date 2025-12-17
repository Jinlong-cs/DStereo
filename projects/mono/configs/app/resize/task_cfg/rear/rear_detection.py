import json

from ...common import val_decoders  # noqa
from ...common import (
    datasets,
    fcos_max_per_img,
    log_freq,
    loss_weights,
    rpn_out_strides,
    tasks_batch_size,
    training_step,
    val_only,
)
from ..vdvru_common import get_inputs  # noqa
from ..vdvru_common import (
    backbone,
    enable_auto_assign,
    get_2d_detection_aidi_eval_callback,
    get_2d_detection_aidi_eval_loaders,
    get_2d_detection_dataloader,
    get_2d_detection_dataset,
    get_update_metric,
    neck,
    pick_stride_neck,
    regress_ranges,
    use_iou_replace_ctrness,
)
from .common import fcos_head, object_type

task_type = "detection"
task_class_id = 5
task_name = f"{object_type}_{task_type}"
batch_size = tasks_batch_size[task_name]
num_classes = 1
norm_target_bbox = False if training_step == "int_infer" and val_only else True
loss_weight = loss_weights[task_name]
dataset_name = "vehicle_rear"

post_rpn_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.0,
    nms=dict(name="nms", iou_threshold=0.5),
    max_per_img=fcos_max_per_img,
)


def get_model(mode):
    model = dict(
        type="FCOS",
        backbone=backbone,
        neck=dict(type="ExtSequential", modules=[neck, pick_stride_neck]),
        head=fcos_head,
        targets=dict(
            type="FCOSTarget",
            strides=rpn_out_strides,
            regress_ranges=regress_ranges,
            cls_out_channels=num_classes,
            background_label=num_classes,
            use_iou_replace_ctrness=use_iou_replace_ctrness,
            norm_on_bbox=norm_target_bbox,
            node_name="rear_target",
        )
        if mode == "train"
        else None,
        post_process=dict(
            type="FCOSDecoder",
            node_name=f"{task_name}_fcos_decoder",
            num_classes=num_classes,
            strides=rpn_out_strides,
            test_cfg=post_rpn_cfg,
            nms_sqrt=True if not enable_auto_assign else False,
            filter_score_mul_centerness=enable_auto_assign,
        )
        if mode == "val"
        else None,
        loss_cls=dict(
            type="FocalLoss",
            loss_name="loss_cls",
            num_classes=num_classes + 1,
            alpha=0.25,
            gamma=2.0,
            loss_weight=loss_weight,
            node_name="rear_loss_cls",
        )
        if mode == "train"
        else None,
        loss_reg=dict(
            type="GIoULoss",
            loss_name="loss_bbox",
            loss_weight=loss_weight,
            node_name="rear_loss_reg",
        )
        if mode == "train"
        else None,
        loss_centerness=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            loss_name="loss_centerness"
            if not use_iou_replace_ctrness
            else "loss_iou",  # noqa
            loss_weight=loss_weight,
            node_name="rear_loss_centerness",
        )
        if mode == "train"
        else None,
    )
    return model


# -------------------------- data --------------------------
# data
ds = datasets[object_type]
train_rec_paths = [d["rec"] for d in ds["train"]]
train_anno_paths = [d["pbrec"] for d in ds["train"]]
train_sample_weights = [d["sample_weight"] for d in ds["train"]]

val_rec_paths = [d["rec"] for d in ds["val"]]
val_anno_paths = [d["pbrec"] for d in ds["val"]]
val_sample_weights = [d["sample_weight"] for d in ds["val"]]

data_loader = get_2d_detection_dataloader(
    "train",
    get_2d_detection_dataset(
        "train",
        train_rec_paths,
        train_anno_paths,
        task_class_id,
        train_sample_weights,
    ),
    batch_size,
)

val_data_loader = get_2d_detection_dataloader(
    "val",
    get_2d_detection_dataset(
        "val",
        val_rec_paths,
        val_anno_paths,
        task_class_id,
    ),
    batch_size,
)

# -------------------------- model --------------------------

# test
desc_task_name = "fcos_rear_detection"
desc_class_names = ["rear"]
desc_out_names = [
    "score",
    "bbox",
    "centerness",
]  # in order of FilterByThreshold's output order  # noqa
score_threshold = 0.5
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
                score_threshold=score_threshold,
                nms_threshold=0.5,
                crop_offset=[0.0, 0.0, 0.0, 0.0]
                # **global_desc,
            )
        )
        for out_name in desc_out_names
        for s in rpn_out_strides
    ],
)

# -------------------------- solver --------------------------
task_filter_confition = lambda x: x[1] == task_name
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
    filter_condition=lambda x: x[1] == task_name,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)

classnames = [object_type]
classname_to_idx_map = {}
for class_idx, class_name in enumerate(classnames):
    classname_to_idx_map[class_name] = class_idx

vstg = 0.025  # val_score_thresh_gap
ststart = 0.5  # score thresh start
stend = 0.7  # score thresh end
val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(
            type="VOCMApMetric",
            num_classes=1,
            class_names=classnames,
            iou_thresh=0.5,
            score_threshs=tuple(
                [
                    ststart + vstg * i
                    for i in range(int((stend - ststart) / vstg))
                ]
            ),
        ),
    ],
    filter_condition=task_filter_confition,
    metric_update_func=get_update_metric(is_train=False, task_name=task_name),
    log_prefix="Validation " + task_name,
    step_log_freq=-1,
)

(
    aidi_eval_callback,
    aidi_eval_page_label,
    aidi_eval_dataset_id,
    aidi_eval_type,
    old_prediction,
    new_prediction,
) = get_2d_detection_aidi_eval_callback(dataset_name, task_name)

aidi_eval_loader = get_2d_detection_aidi_eval_loaders(
    batch_size["val"], dataset_name
)
