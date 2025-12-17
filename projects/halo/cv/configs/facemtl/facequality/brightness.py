import torch
from common import (
    backbone,
    batch_size_per_gpu_facequality,
    bn_kwargs,
    expand_type,
    in_channels,
    loss_weights,
    num_workers,
    rec_prefix,
    step_log_freq,
    train_transforms,
)

task_type = "classification"
task_name = "brightness"
num_classes = 4

# inputs
inputs = dict(
    train=dict(
        gt_brightness=torch.zeros(2),
    ),
    val=dict(),
    test=dict(),
    deploy=dict(),
)


# ----------------- model ---------------------------
def get_model(mode):
    loss = dict(
        type=torch.nn.CrossEntropyLoss,
        reduction="mean",
    )
    return dict(
        type="ImageClassifier",
        backbone=backbone,
        head=dict(
            type="FaceMtlFacequalityHead",
            in_channels=in_channels,
            # feat_channels=[128, 128],  # vargnet
            feat_channels=[64, 64],  # mixvargenet0.75
            output_dim=num_classes,
            task_name=task_name,
            loss_weight=loss_weights[task_name],
            flat_output=False if mode == "deploy" else True,
            bn_kwargs=bn_kwargs,
            loss=loss if mode == "train" else None,
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
                preds = torch.argmax(preds, axis=1)
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
    step_log_freq=step_log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)


# ----------------- data ---------------------------
train_rec_list = [
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_J2DMS_allattr_ir_num151925.rec",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_J2DMS_allattr_rgb_num102190.rec",  # noqa
    # f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_J2DMS_allattr_ir_num151925.rec",  # noqa
    # f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_J2DMS_allattr_rgb_num102190.rec",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_x2_allattr_num331349.rec",  # noqa
]
train_idx_list = [
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_J2DMS_allattr_ir_num151925.idx",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_J2DMS_allattr_rgb_num102190.idx",  # noqa
    # f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_J2DMS_allattr_ir_num151925.idx",  # noqa
    # f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_J2DMS_allattr_rgb_num102190.idx",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_x2_allattr_num331349.idx",  # noqa
]
train_label_list = [
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_J2DMS_allattr_ir_num151925.label.npy",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_J2DMS_allattr_rgb_num102190.label.npy",  # noqa
    # f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_J2DMS_allattr_ir_num151925.label.npy",  # noqa
    # f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_J2DMS_allattr_rgb_num102190.label.npy",  # noqa
    f"{rec_prefix}/MultiMode_2/mm_algorithms_data/face-quality-attributes/mx-record/Trainset/{expand_type}/train_x2_allattr_num331349.label.npy",  # noqa
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
                    need_index=False,
                    transforms=train_transforms,
                ),
            )
            for i in range(len(train_rec_list))
        ],
    ),
    sampler=dict(
        type="DistributedProportionSampler",
        # expect_distribution={"0": 0.3, "1": 0.15, "2": 0.15, "3": 0.4},
        expect_distribution={
            "0": 0.45,
            "1": 0.15,
            "2": 0.15,
            "3": 0.25,
        },  # mixvargenet0.75  # noqa
        task_name=task_name,
        # num_reference=1119953,
        num_reference=585464,
        # num_reference=102190,
    ),
    persistent_workers=True,
    batch_size=batch_size_per_gpu_facequality,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
