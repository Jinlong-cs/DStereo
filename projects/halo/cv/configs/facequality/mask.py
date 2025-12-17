import torch
from common import (
    backbone,
    batch_size_per_gpu,
    bn_kwargs,
    expand_type,
    num_workers,
    rec_prefix,
    train_transforms,
)

task_type = "classification"
task_name = "mask"
num_classes = 2
# inputs
inputs = dict(
    train=dict(
        gt_mask=torch.zeros(2),
    ),
    val=dict(),
    test=dict(),
    deploy=dict(),
)


# ----------------- model ---------------------------
def get_model(mode):
    return dict(
        type="ImageClassifier",
        backbone=backbone,
        head=dict(
            type="FacequalityMultiHead",
            in_channels=256,
            feat_channels=[128, 128],
            output_dim=1,
            task_name=task_name,
            flat_output=False if mode == "deploy" else True,
            bn_kwargs=bn_kwargs,
            loss=dict(
                type=torch.nn.BCEWithLogitsLoss,
                reduction="mean",
            )
            if mode == "train"
            else None,
            node_name=f"{task_name}_classifier_head",
        ),
    )


# ----------------- metric ---------------------------
def update_metric(metrics, batch, model_outs):
    # batch is a tuple (MultitaskLoader)
    if task_name in model_outs:
        gt = batch[0][task_name][f"gt_{task_name}"]
        for metric in metrics:
            if "loss" in metric.name:
                metric.update(model_outs[task_name]["loss"])
            elif f"{task_name}_accuracy" in metric.name:
                preds = model_outs[task_name]["pred"]
                preds = preds > 0
                metric.update(gt, preds)
            else:
                raise ValueError(
                    "Can't find the metric, "
                    f"please cheek the name of {task_name} metric."
                )


train_metrics = [
    dict(type="LossShow", name=f"{task_name}_loss"),
    dict(type="Accuracy", axis=1, name=f"{task_name}_accuracy"),
]

metric_updater = dict(
    type="MetricUpdater",
    metrics=train_metrics,
    metric_update_func=update_metric,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix=task_name,
)


# ----------------- data ---------------------------
# num: 329153
train_rec_list = [
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/mask_ir_20220329_num175992.rec",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/mask_rgb_20220329_num153161.rec",  # noqa
]
train_idx_list = [
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/mask_ir_20220329_num175992.idx",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/mask_rgb_20220329_num153161.idx",  # noqa
]
train_label_list = [
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/mask_ir_20220329_num175992.label.npy",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/mask_rgb_20220329_num153161.label.npy",  # noqa
]
info_path = f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/label_idx_map_v1.0.yaml"  # noqa


train_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ConcatDataset",
        with_flag=True,
        record_index=False,
        datasets=[
            dict(
                type="RepeatDataset",
                times=1,
                dataset=dict(
                    type="FaceQualityDataset",
                    rec_path=train_rec_list[i],
                    idx_path=train_idx_list[i],
                    label_path=train_label_list[i],
                    info_path=info_path,
                    task_name=task_name,
                    need_flag=True,
                    transforms=train_transforms,
                ),
            )
            for i in range(len(train_rec_list))
        ],
    ),
    sampler=dict(
        type="DistributedProportionSampler",
        expect_distribution={"0": 0.5, "1": 0.5},
        task_name=task_name,
        num_reference=329153,
        # num_reference=865838,
    ),
    persistent_workers=True,
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
