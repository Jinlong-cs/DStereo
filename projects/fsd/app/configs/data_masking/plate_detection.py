import copy
import json

import torch
from common import (
    backbone,
    batch_size_per_gpu,
    drop_last,
    feat_channels,
    img_height,
    img_width,
    is_int_infer,
    log_freq,
    neck,
    neck_stride2channels,
    num_workers,
    out_strides,
    pin_memory,
    roi_region,
    vanishing_point,
)
from dataset import datapaths

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.data.collates.collates import collate_2d

# -------------------------------- Dataloader configuration ---------------------------------- """  # noqa
ds = datapaths.plate
train_rec_paths = [d["rec_path"] for d in ds["train_data_paths"]]

class_id = 2
transforms = [
    dict(
        type="Resize",
        img_scale=(img_height, img_width),
        keep_ratio=True,
        ratio_range=(1.0, 1.5),
    ),
    dict(
        type="RandomCrop",
        size=(img_height, img_width),
        truncate_gt=False,
        crop_around_gt=True,
    ),
    dict(type="RandomFlip", px=0.5),
    dict(type="Pad", size=(img_height, img_width), pad_val=0),
    dict(type="ToTensor", to_yuv=False),
]

dataset = dict(
    type="ConcatDataset",
    datasets=[
        dict(
            type="DenseboxDataset",
            data_path=data_path,
            anno_path=data_path.replace(".rec", ".anno.pb_rec"),
            class_id=class_id,
            category=0,
            use_ignore=False,
            ignore_hard=True,
            to_rgb=False,
            extend_ignore_region_into_gtbox=True,
            transforms=transforms,
        )
        for data_path in train_rec_paths
    ],
)

dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dataset,
    collate_fn=collate_2d,
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=drop_last,
    sampler=dict(
        type=torch.utils.data.distributed.DistributedSampler,
        shuffle=True,
    ),
)
# -------------------------------- Head configuration ---------------------------------- """  # noqa
task_type = "detection"
classnames = ["vehicle_plate"]
object_type = "_".join(classnames)
task_name = f"{object_type}_{task_type}"
num_classes = 1
norm_target_bbox = True
head = dict(
    type="FCOSHead",
    num_classes=num_classes,
    in_strides=out_strides,
    out_strides=out_strides,
    stride2channels=neck_stride2channels,
    feat_channels=feat_channels,
    stacked_convs=2,
    use_sigmoid=True,
    share_bn=False,
    upscale_bbox_pred=not norm_target_bbox,
    node_name=f"{object_type}_fcos_head",
)
# -------------------------------- Target configuration ---------------------------------- """  # noqa

INF = 1e8
default_regress_ranges = ((-1, INF),)
range_multiplier = 0.5
regress_ranges = tuple(
    (
        (x * range_multiplier, y * range_multiplier)
        for x, y in (default_regress_ranges)
    )
)
use_iou_replace_ctrness = True
targets = dict(
    type="FCOSTarget",
    strides=out_strides,
    regress_ranges=regress_ranges,
    cls_out_channels=num_classes,
    background_label=num_classes,
    use_iou_replace_ctrness=use_iou_replace_ctrness,
    norm_on_bbox=norm_target_bbox,
    node_name=f"{object_type}_fcos_target",
)

# -------------------------------- Loss configuration ---------------------------------- """  # noqa
task_loss_weight = 1.0
loss_cls = dict(
    type="FocalLoss",
    loss_name="loss_cls",
    num_classes=num_classes + 1,
    alpha=0.25,
    gamma=2.0,
    loss_weight=1.0 * task_loss_weight,
    node_name=f"{object_type}_fcos_loss_cls",
)
loss_reg = dict(
    type="GIoULoss",
    loss_name="loss_bbox",
    loss_weight=1.0 * task_loss_weight,
    node_name=f"{object_type}_fcos_loss_reg",
)
loss_centerness = dict(
    type="CrossEntropyLoss",
    use_sigmoid=True,
    loss_name="loss_centerness" if not use_iou_replace_ctrness else "loss_iou",
    loss_weight=1.0 * task_loss_weight,
    node_name=f"{object_type}_fcos_loss_centerness",
)
# -------------------------------- Model configuration ---------------------------------- """  # noqa

model = dict(
    type="FCOS",
    backbone=backbone,
    neck=dict(type="ExtSequential", modules=[neck]),
    head=head,
    targets=targets,
    loss_cls=loss_cls,
    loss_reg=loss_reg,
    loss_centerness=loss_centerness,
)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(name="nms", iou_threshold=0.2),
    max_per_img=100,
)
nms_sqrt = True
val_model = copy.deepcopy(model)
val_model["head"].update(dict(upscale_bbox_pred=True, dequant_output=True))
val_model["targets"] = None
val_model["loss_cls"] = None
val_model["loss_reg"] = None
val_model["loss_centerness"] = None
val_model["post_process"] = dict(
    type="FCOSDecoder",
    num_classes=num_classes,
    strides=out_strides,
    test_cfg=test_cfg,
    nms_sqrt=nms_sqrt,
    meta_data_bool=False,
    node_name=f"{object_type}_fcos_decoder",
)

deploy_model = copy.deepcopy(val_model)
deploy_model["targets"] = None
if is_int_infer:
    desc_task_name = "detection"
    desc_class_names = [task_name]
    desc_out_names = [
        "point_coordinate",
        "score",
        "bbox",
        "centerness",
    ]
    score_threshold = 0.5

    filter_module = dict(
        type="FCOSMultiStrideFilter",
        strides=out_strides,
        threshold=-4.59,
        node_name=f"{object_type}_fcos_multistride_filter",
    )
    add_desc_pp = dict(
        type="AddDesc",
        per_tensor_desc=[
            json.dumps(
                dict(
                    task=desc_task_name,
                    class_name=desc_class_names,
                    output_name=out_name,
                    stride=s,
                    score_threshold=score_threshold,
                    roi_regions=roi_region,
                    vanishing_point=vanishing_point,
                )
            )
            for s in out_strides
            for out_name in desc_out_names
        ],
        node_name=f"{object_type}_fcos_desc",
    )

    deploy_model["post_process"] = dict(
        type="MultiInputSequential",
        modules=[
            filter_module,
            add_desc_pp,
        ],  # should be the last module
    )
    deploy_model["head"]["upscale_bbox_pred"] = False
    deploy_model["head"]["dequant_output"] = False
else:
    deploy_model["post_process"] = dict(
        type="FCOSDecoder",
        __graph_model_name=f"{object_type}_fcos_decoder",
        num_classes=num_classes,
        strides=out_strides,
        test_cfg=test_cfg,
        nms_sqrt=nms_sqrt,
    )
    deploy_model["head"]["upscale_bbox_pred"] = True
    deploy_model["head"]["dequant_output"] = True

loss_names = [
    "loss_cls",
    "loss_bbox",
    "loss_centerness" if not use_iou_replace_ctrness else "loss_iou",
]
metric_updater = dict(
    type="MetricUpdater",
    metrics=[dict(type="LossShow", name=name) for name in loss_names],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[
            {"label_pattern": None, "pred_pattern": f"^.*{task_name}_{name}$"}
            for name in loss_names
        ]
    ),
    filter_condition=None,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
    reset_metrics_by="log",
)

# -------------------------------- Inputs ---------------------------------- """  # noqa
inputs = dict(
    train=dict(
        gt_bboxes=list(torch.ones([1, 1, 4], dtype=torch.float32)),
        gt_classes=list(torch.ones([1, 1], dtype=torch.int64)),
    ),
    val=dict(),
    deploy=dict(),
)
