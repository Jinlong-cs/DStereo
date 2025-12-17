import os

import torch
from horizon_plugin_pytorch.quantization import March

from hat.models.losses.pupil_segmentation.utils import pupil_center_from_mask
from hat.models.task_modules.pupil_segmentation import get_sizes
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = ""

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES

# parameters
is_local_train = not os.path.exists("/running_package")

step_log_freq = 500
if is_local_train:
    device_ids = [3]
    batch_size_per_gpu = 16
    num_workers = 0
    rec_prefix = "/horizon-bucket"
    # ckpt_dir = "/home/users/jiaqi.quan/project/gaze/pupil_segmentation/pupil_seg/logs/ritnet_v2/"  # noqa
    ckpt_dir = "/home/users/jiaqi.quan/model/deploy_model/gaze/pupil_segmentation/v0.0.1/"
else:
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    batch_size_per_gpu = 16
    num_workers = 8
    rec_prefix = "/bucket/output"
    ckpt_dir = "/job_data/models"
ckpt_dir = os.path.join(ckpt_dir, task_name)


# -------------------------- model ----------------------------
chz = 32
growth = 1.2
in_channel, mask_out_channel = 1, 1
sizes = get_sizes(chz, growth)
encoder_out_channels = int(sizes["enc"]["op"][3])
input_shape = (64, 64)
test_inputs = dict(img=torch.randn((1, 1, 64, 64)))


# ------------------------------- model ----------------------------------
test_model = dict(
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
    mode="val",
)

# ---------------------------- data -----------------------------------
test_transforms = [
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
test_lmdb_list = [
    # "/home/users/jiaqi.quan/project/gaze/pupil_segmentation/pupil_seg/data/lmdb/test/",
    # resize
    "/home/users/jiaqi.quan/project/gaze/pupil_segmentation/data/lmdb/without_resize/test/"
]

test_dataloaders = []
for lmdb_root in test_lmdb_list:
    test_dataloaders.append(
        dict(
            type=torch.utils.data.DataLoader,
            dataset=dict(
                type="PupilSegDataset",
                image_path=os.path.join(lmdb_root, "image_lmdb"),  # noqa
                anno_path=os.path.join(lmdb_root, "anno_lmdb"),  # noqa
                data_type="val",
                transforms=test_transforms,
            ),
            sampler=dict(type=torch.utils.data.DistributedSampler),
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    )

# ----------------------------- metric ------------------------------
test_metrics = [
    dict(
        type="EllipseParamError", name="PupilEllipseParamError", do_norm=False
    ),
    dict(type="MeanIOU", seg_class=["class1", "class2"], name="MeanIOU"),
]


def test_update_metric(metrics, batch, model_outs):
    gt_norm_ellipse_param = batch["gt_norm_pupil_ellipse_param"]
    gt_pupil_center = batch["gt_pupil_center"]
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
    pupil_mask_center[:, 0] = (
        0.5 * input_shape[1] * (pupil_mask_center[:, 0] + 1)
    )
    pupil_mask_center[:, 1] = (
        0.5 * input_shape[0] * (pupil_mask_center[:, 1] + 1)
    )
    el_pred = torch.cat([pupil_mask_center, ellipse_param_pred[:, 2:5]], dim=1)
    mask_pred = (mask_pred > 0.5).long()
    el_gt = torch.cat([gt_pupil_center, gt_norm_ellipse_param[:, 2:5]], dim=1)

    for metric in metrics:
        if "MeanIOU" in metric.name:
            metric.update(gt_mask, mask_pred)
        else:
            # metric.update(gt_norm_ellipse_param, el_pred)
            metric.update(el_gt, el_pred)


test_metric_updaters = []
for i in range(len(test_lmdb_list)):
    val_metric_updater = dict(
        type="MetricUpdater",
        metrics=test_metrics,
        metric_update_func=test_update_metric,
        step_log_freq=-1,
        epoch_log_freq=1,
        log_prefix=f"test_rec{i}",
    )
    test_metric_updaters.append([val_metric_updater])


# ------------------------- batch processor -----------------------
test_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


# ------------------------------ predictor ------------------------------
float_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=False,
                ignore_extra=False,
                verbose=True,  # Show unexpect_key and miss_key info.  # noqa
                check_hash=False,
            ),
        ],
    ),
    data_loader=test_dataloaders,
    batch_processor=test_batch_processor,
    device=None,
    metrics=test_metrics,
    callbacks=test_metric_updaters,
    log_interval=0,
    share_callbacks=False,
)

qat_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir,
                    "qat-checkpoint-epoch-0005.pth.tar",
                ),
                allow_miss=False,
                ignore_extra=True,
                verbose=False,  # Show unexpect_key and miss_key info.  # noqa
                check_hash=False,
            ),
        ],
    ),
    data_loader=test_dataloaders,
    batch_processor=test_batch_processor,
    device=None,
    metrics=test_metrics,
    callbacks=test_metric_updaters,
    log_interval=0,
    share_callbacks=False,
)


int_infer_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=test_dataloaders,
    batch_processor=test_batch_processor,
    device=None,
    metrics=test_metrics,
    callbacks=test_metric_updaters,
    log_interval=0,
    share_callbacks=False,
)
