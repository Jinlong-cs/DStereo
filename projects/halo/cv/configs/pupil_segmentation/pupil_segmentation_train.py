import os

import torch
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.models.losses.pupil_segmentation.utils import pupil_center_from_mask
from hat.models.task_modules.pupil_segmentation import get_sizes
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "leaky_relu"

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES

# parameters
is_local_train = not os.path.exists("/running_package")

step_log_freq = 500
float_num_epoch = 40
float_lr = 5e-4
qat_lr = 5e-5
qat_num_epoch = 10
stage = "qat"
if is_local_train:
    device_ids = [0]
    batch_size_per_gpu = 16
    num_workers = 8
    rec_prefix = "/horizon-bucket"
    # ckpt_dir = "/home/users/jiaqi.quan/project/gaze/pupil_segmentation/model/exp_resize/"  # noqa
    ckpt_dir = "/home/users/jiaqi.quan/model/deploy_model/gaze/pupil_segmentation/v0.0.1/"
else:
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    batch_size_per_gpu = 16
    num_workers = 8
    rec_prefix = "/bucket/output"
    ckpt_dir = "/job_data/models"
ckpt_dir = os.path.join(ckpt_dir, task_name)
bn_kwargs = dict(eps=1e-5, momentum=0.1)


# -------------------------- model ----------------------------
chz = 32
growth = 1.2
in_channel, mask_out_channel = 1, 1
sizes = get_sizes(chz, growth)
encoder_out_channels = int(sizes["enc"]["op"][3])
input_shape = (64, 64)
test_inputs = dict(img=torch.randn((1, 1, 64, 64)))

# --------------------------------------- models -----------------------------------
loss_weights = {
    "seg2pt": 1.0,
    "ellipse": 20.0,
    "seg": 20.0,
    "selfcorr": 10,
    "surface_loss_ratio": 0.1,
}
if stage == "float":
    loss_weights.update({"num_epochs": float_num_epoch})
else:
    loss_weights.update({"num_epochs": qat_num_epoch})

train_model = dict(
    type="PupilSegNet",
    encoder=dict(
        type="PupilSegEncoder",
        in_c=in_channel,
        chz=chz,
        growth=growth,
    ),
    decoder=dict(
        type="PupilSegDecoder",
        chz=chz,
        out_c=mask_out_channel,
        growth=growth,
    ),
    losses=dict(
        type="PupilSegLoss",
        loss_weights=loss_weights,
        do_self_corr=False,
        mask_shape=input_shape,
    ),
    encoder_out_channels=encoder_out_channels,
    mode="train",
)

deploy_model = dict(
    type="PupilSegNet",
    encoder=dict(
        type="PupilSegEncoder",
        in_c=in_channel,
        chz=chz,
        growth=growth,
    ),
    decoder=dict(
        type="PupilSegDecoder",
        chz=chz,
        out_c=mask_out_channel,
        growth=growth,
    ),
    encoder_out_channels=encoder_out_channels,
    mode="deploy",
)

# ---------------------------- data -----------------------------------
train_transforms = [
    # dict(
    #     type="RandomRotateCrop",
    #     net_input_size=input_shape,
    #     rot_prob=0.0,
    #     rot_angle_range=20,
    #     center_shift_prob=0.0,
    #     center_shift_range=0.01,
    #     norm_ratio=1.0,
    #     norm_jitter_range=0.0,
    #     interpolation="lanczos",
    # ),
    dict(
        type="RandomRotateCrop",
        net_input_size=input_shape,
        rot_prob=0.0,
        rot_angle_range=20,
        center_shift_prob=0.0,
        center_shift_range=0.01,
        norm_ratio=1.0,
        norm_jitter_range=0.0,
        interpolation="lanczos",
    ),
    # dict(
    #     type="CropRecROI",
    #     crop_type="center",
    #     target_shape=(64, 64, 3),
    #     base_roi=[32, 32, 96, 96],
    #     crop_jitter_range=0,
    # ),
    # dict(type="RandomFlip", px=0.2, py=0),
    # dict(
    #     type="GaussianBlur",
    #     p=0.2,
    #     # p=1,
    #     kernel_size_min=2,
    #     kernel_size_max=5,
    #     sigma_min=2,
    #     sigma_max=7,
    # ),
    # dict(
    #     type="GaussianNoise",
    #     prob=0.2,
    #     # prob=1,
    #     mean=0,
    #     sigma=2,
    # ),
    # dict(
    #     type="RandomShiftRotateScale",
    #     rotate_prob=0.2,
    #     # rotate_prob=1,
    #     bounded=False,
    #     max_rotate_angle=30,
    #     resize=True,
    #     border_value=4,
    # ),
    dict(type="GenerateEllipseMask"),
    dict(type="GenerateEdgeWeightMap"),
    dict(type="GenerateDistMap"),
    dict(type="NormEllipseParam"),
    dict(
        type="RandomGray",
        p=1,
        rgb_data=True,
        only_one_channel=True,
    ),
    dict(type="ToTensor"),
    dict(type="Normalize", mean=128.0, std=128.0),
]

val_transforms = [
    # dict(
    #     type="CropRecROI",
    #     crop_type="center",
    #     target_shape=(64, 64, 3),
    #     base_roi=[32, 32, 96, 96],
    #     crop_jitter_range=0,
    # ),
    dict(
        type="RandomRotateCrop",
        net_input_size=input_shape,
        rot_prob=0.0,
        rot_angle_range=20,
        center_shift_prob=0.0,
        center_shift_range=0.01,
        norm_ratio=1.0,
        norm_jitter_range=0.0,
        interpolation="lanczos",
    ),
    # dict(
    #     type="RandomRotateCrop",
    #     net_input_size=input_shape,
    #     rot_prob=0.0,
    #     rot_angle_range=20,
    #     center_shift_prob=0.0,
    #     center_shift_range=0.01,
    #     norm_ratio=1.0,
    #     norm_jitter_range=0.0,
    #     interpolation="lanczos",
    # ),
    dict(type="GenerateEllipseMask"),
    dict(type="GenerateEdgeWeightMap"),
    dict(type="GenerateDistMap"),
    dict(type="NormEllipseParam"),
    dict(
        type="RandomGray",
        p=1,
        rgb_data=True,
        only_one_channel=True,
    ),
    dict(type="ToTensor"),
    dict(type="Normalize", mean=128.0, std=128.0),
]

# # # # 每个lmdb只保存根目录
train_lmdb_list = [
    # "/home/users/jiaqi.quan/project/gaze/pupil_segmentation/pupil_seg/data/lmdb/train/",
    # resize
    "/home/users/jiaqi.quan/project/gaze/pupil_segmentation/data/lmdb/without_resize/train/"
]
val_lmdb_list = [
    # "/home/users/jiaqi.quan/project/gaze/pupil_segmentation/pupil_seg/data/lmdb/valid/",
    # resize
    "/home/users/jiaqi.quan/project/gaze/pupil_segmentation/data/lmdb/without_resize/valid/"
]
train_datasets, val_dataloaders = [], []
for lmdb_root in train_lmdb_list:
    train_datasets.append(
        dict(
            type="PupilSegDataset",
            image_path=os.path.join(lmdb_root, "image_lmdb"),  # noqa
            anno_path=os.path.join(lmdb_root, "anno_lmdb"),  # noqa
            data_type="train",
            transforms=train_transforms,
        )
    )

train_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type=torch.utils.data.ConcatDataset,
        datasets=train_datasets,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    # persistent_workers=True,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

for lmdb_root in val_lmdb_list:
    val_dataloaders.append(
        dict(
            type=torch.utils.data.DataLoader,
            dataset=dict(
                type="PupilSegDataset",
                image_path=os.path.join(lmdb_root, "image_lmdb"),  # noqa
                anno_path=os.path.join(lmdb_root, "anno_lmdb"),  # noqa
                data_type="val",
                transforms=val_transforms,
            ),
            sampler=dict(type=torch.utils.data.DistributedSampler),
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    )

# ----------------------------- metric ------------------------------
train_metrics = [
    dict(type="LossShow", name="seg2pt_loss"),
    dict(type="LossShow", name="ellipse_loss"),
    dict(type="LossShow", name="seg_loss"),
]


def train_update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs[metric.name])


train_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=train_update_metric,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix=task_name,
)

val_metrics = [
    dict(type="EllipseParamError", name="PupilEllipseParamError"),
    dict(type="MeanIOU", seg_class=["class1", "class2"], name="MeanIOU"),
]


def val_update_metric(metrics, batch, model_outs):
    gt_norm_ellipse_param = batch["gt_norm_pupil_ellipse_param"]
    gt_mask = batch["gt_pupil_mask"]
    mask_pred = model_outs["mask_pred"]
    mask_pred = mask_pred.squeeze(1)
    ellipse_param_pred = model_outs["ellipse_param_pred"]
    bc = ellipse_param_pred.shape[0]
    ellipse_param_pred = ellipse_param_pred.reshape(bc, -1)
    pup_c = torch.tanh(ellipse_param_pred[:, 0:2])
    pup_param = torch.sigmoid(ellipse_param_pred[:, 2:4])
    pup_angle = ellipse_param_pred[:, 4]
    ellipse_param_pred = torch.cat(
        [pup_c, pup_param, pup_angle.unsqueeze(1)], dim=1
    )
    pupil_mask_center = pupil_center_from_mask(mask_pred, temperature=4)
    el_pred = torch.cat([pupil_mask_center, ellipse_param_pred[:, 2:5]], dim=1)
    mask_pred = (mask_pred > 0.5).long()

    for metric in metrics:
        if "MeanIOU" in metric.name:
            metric.update(gt_mask, mask_pred)
        else:
            metric.update(gt_norm_ellipse_param, el_pred)


val_metric_updaters = []
for i in range(len(val_lmdb_list)):
    val_metric_updater = dict(
        type="MetricUpdater",
        metric_update_func=val_update_metric,
        step_log_freq=-1,
        epoch_log_freq=1,
        log_prefix=f"val_dataset_{i}",
    )
    val_metric_updaters.append([val_metric_updater])


# ------------------------- batch processor -----------------------
train_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


# ----------------------- callback ---------------------------------
loss_weight_callback = dict(type="LossWeightUpdater")

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    save_hash=False,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=test_inputs,
)

val_callback = dict(
    type="Validation",
    data_loader=val_dataloaders,
    batch_processor=val_batch_processor,
    callbacks=val_metric_updaters,
    val_interval=1,
    interval_by="epoch",
    share_callbacks=False,
)


# ------------------------------ trainer ------------------------------
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=train_model,
    data_loader=train_data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=5e-4)},
        lr=float_lr,
    ),
    batch_processor=train_batch_processor,
    num_epochs=float_num_epoch,
    device=None,
    callbacks=[
        loss_weight_callback,
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[15, 25, 35],
            lr_decay_factor=0.1,
            warmup_by="step",
            warmup_len=100,
            step_log_interval=50,
        ),
        train_metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=train_metrics,
    val_metrics=val_metrics,
)


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=train_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    # ckpt_dir, "float-checkpoint-last.pth.tar"
                    ckpt_dir,
                    "float-checkpoint-epoch-0024.pth.tar",
                ),
                check_hash=False,
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=train_data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=5e-4)},
        lr=qat_lr,
    ),
    batch_processor=train_batch_processor,
    num_epochs=qat_num_epoch,
    device=None,
    callbacks=[
        loss_weight_callback,
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=50,
            lr_decay_id=[5, 10],
            lr_decay_factor=0.1,
            warmup_by="step",
            warmup_len=100,
        ),
        train_metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=train_metrics,
    val_metrics=val_metrics,
)


int_infer_trainer = dict(
    type="Trainer",
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-epoch-0005.pth.tar"
                ),
                ignore_extra=True,
                verbose=True,
                check_hash=False,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[ckpt_callback, trace_callback],
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
)
