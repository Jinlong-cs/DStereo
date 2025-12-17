import json
import os

import torch
from common import (
    backbone,
    crop_h,
    img_h,
    img_w,
    log_freq,
    neck,
    pipeline_test,
    task_type,
)
from datasets import datapaths, parse_dataset

from hat.callbacks.metric_updater import update_metric_using_regex

# -------------------------------------------------------
# task define
# ------------------------------------------------------
task_name = "tollgate_detection"
train_num_workers = 1
val_num_workers = 1
NUM_LDMK = 2
STRIDE_LDMK = 4
if task_type == "mono":
    SIGMA_LDMK = 1.0
else:  # SD
    SIGMA_LDMK = 2.0
ng_weights = 1.0
training_step = os.environ.get("HAT_TRAINING_STEP", "float")
classnames = ["tollgate_4pe"]
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
val_rec_paths = val_data_desc.rec_paths
val_anno_paths = val_data_desc.anno_paths
val_batch_size = val_data_desc.batch_size_per_ctx
sample_weights = train_data_desc.sample_weights


# # data loader
test_inputs = dict(img=torch.randn((1, 3, crop_h, img_w)))
train_datasets = [
    dict(
        type="TollgateDataset",
        rec_path=rec_path,
        anno_path=anno_path,
        transforms=[
            dict(
                type="RandomColorJitter",
            ),
            dict(
                type="Resize",
                img_scale=(img_h, img_w),
                ratio_range=(1.0, 1.0),
                keep_ratio=False,
            ),
            dict(
                type="FixedCrop",
                size=(0, 0, img_w, crop_h),
            ),
            dict(
                type="OvalHmTargetGenerator",
                img_scale=(crop_h, img_w),
                encode_lmks=NUM_LDMK,
                stride=STRIDE_LDMK,
                sigma=SIGMA_LDMK,
                ng_weights=ng_weights,
            ),
        ],
    )
    for rec_path, anno_path in zip(rec_paths, anno_paths)
]

val_datasets = [
    dict(
        type="TollgateDataset",
        rec_path=rec_path,
        anno_path=anno_path,
        transforms=[
            dict(
                type="Resize",
                img_scale=(img_h, img_w),
                ratio_range=(1.0, 1.0),
                keep_ratio=False,
            ),
            dict(
                type="FixedCrop",
                size=(0, 0, img_w, crop_h),
            ),
            dict(
                type="OvalHmTargetGenerator",
                img_scale=(crop_h, img_w),
                encode_lmks=NUM_LDMK,
                stride=STRIDE_LDMK,
                sigma=SIGMA_LDMK,
                ng_weights=ng_weights,
            ),
        ],
    )
    for rec_path, anno_path in zip(val_rec_paths, val_anno_paths)
]

inputs = dict(
    train=dict(
        gt_heatmap=torch.zeros(
            (1, NUM_LDMK, crop_h // STRIDE_LDMK, img_w // STRIDE_LDMK)
        ),
        gt_heatmap_weight=torch.ones(
            (1, NUM_LDMK, crop_h // STRIDE_LDMK, img_w // STRIDE_LDMK)
        ),
        gt_offset=torch.zeros(
            (1, NUM_LDMK, crop_h // STRIDE_LDMK, img_w // STRIDE_LDMK)
        ),
        gt_offset_weight=torch.ones(
            (1, 2, crop_h // STRIDE_LDMK, img_w // STRIDE_LDMK)
        ),
    ),
    val=dict(),
    test=dict(),
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type=torch.utils.data.ConcatDataset,
        datasets=train_datasets,
    ),
    batch_size=train_batch_size,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    shuffle=False,
    num_workers=train_num_workers,
    pin_memory=True,
    persistent_workers=train_num_workers > 0,
)

# -------------------------------- description ------------------------
global_desc = dict(
    image_size=[crop_h, img_w],
    sigma=[SIGMA_LDMK for _ in range(NUM_LDMK)],
    coefficient=[1.0 for _ in range(NUM_LDMK)],
    norm_method="max_width_height",
    conf_type="lmks_conf_max",
    linear_a_x=[4.0, 4.0],
    linear_b_x=[2.0, 2.0],
    linear_a_y=[4.0, 4.0],
    linear_b_y=[2.0, 2.0],
)


def tollgate_desc():
    per_tensor_desc = [
        {
            "task": "toll_gate_lmks3_heatmap",
            "class_name": "tollgate",
            "conf_thresh": 0.3,
            "conf_thresh_2": 0.35,
            "properties": [{"channel_labels": ["top", "down"]}],
            **global_desc,
        },
        {
            "task": "toll_gate_lmks3_offset",
            "class_name": "tollgate",
            "h_shift": crop_h // STRIDE_LDMK,
            "w_shift": img_w // STRIDE_LDMK,
            "properties": [{"channel_labels": ["w", "h"]}],
            **global_desc,
        },
    ]
    per_tensor_desc = [json.dumps(i) for i in per_tensor_desc]
    return per_tensor_desc


# -------------------------------- Head configuration ------------------------
head_in_channels = 16
head_hm_channels = 2
head_offset_channels = 2
head = dict(
    type="TollGateHead",
    in_channels=head_in_channels,
    hm_channels=head_hm_channels,
    offset_channels=head_offset_channels,
    use_bias=False,
    node_name=f"{task_name}_head",
)

loss_weights = dict(
    hm=1.0,
    offset=100.0,
)
loss = dict(
    type="TollGateLoss",
    task="tollgate",
    loss_weights=loss_weights,
    node_name=f"{task_name}_loss",
)

postprocess = dict(
    type="TollPostProcess",
    local_max_kernel=3,
    node_name=f"{task_name}_PostProcess",
)

# Train Model
model = dict(
    type="TollGageModel",
    backbone=backbone,
    neck=neck,
    head=head,
    loss=loss,
)

deploy_model = dict(
    type="TollGageModel",
    backbone=backbone,
    neck=neck,
    head=head,
    loss=None,
    desc=dict(
        type="AddDesc",
        per_tensor_desc=tollgate_desc(),
        node_name=f"{task_name}_desc",
    ),
    postprocess=None,
)

# ------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------
loss_names = [
    "tollgate_hm_loss",
    "tollgate_off_hm_l1_loss",
    "tollgate_off_smoothl1loss",
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
