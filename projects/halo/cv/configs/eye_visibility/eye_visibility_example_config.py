import copy
import os

import torch
from hatbc.filestream.bucket.client import get_bucket_client
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.models.backbones.mixvargenet import MixVarGENetConfig
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
bucket_client = get_bucket_client()
training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "fatigue-eye-visibility"
local = True
save_bucket = True
if local:
    root_url = "dmpv2://MultiMode_3/weiyu.li/float_models/mix_vargenet/"
    root_dir = bucket_client.url_to_local(root_url)
    job_id = "mix_vargenet_data_aug_"
    num_workers = 5
    save_feat_dir = f"/home/users/weiyu.li/tmp/{job_id}"
else:
    root_url = "dmpv2://MultiMode_3/weiyu.li/float_models/mix_vargenet/"
    root_dir = bucket_client.url_to_local(root_url)
    job_id = os.environ.get("JOB_ID", "")
    job_id = f"mix_vargenet-{job_id}"
    num_workers = 12
    save_feat_dir = "/job_data/eval-feat/"

pretrain_url = "dmpv2://MultiMode_3/weiyu.li/float_models/mix_vargenet/"
batch_size_per_gpu = 512
device_ids = [2, 3]

ckpt_dir = os.path.join(root_dir, job_id)
pretrain_dir = bucket_client.url_to_local(pretrain_url)

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES
input_shape = (96, 160, 3)

bn_kwargs = dict(eps=1e-3, momentum=0.01)

in_channels = [32, 32, 32, 64, 128]
out_channels = [32, 32, 64, 128, 256]
alpha = 0.5
in_channels = [int(x * alpha) for x in in_channels]
out_channels = [int(x * alpha) for x in out_channels]

net_config = [
    [
        MixVarGENetConfig(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            head_op="mixvarge_f2",
            stack_ops=[],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 2
    [
        MixVarGENetConfig(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 4
    [
        MixVarGENetConfig(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            head_op="mixvarge_f4",
            stack_ops=["mixvarge_f4", "mixvarge_f4"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 8
    [
        MixVarGENetConfig(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            head_op="mixvarge_f2_gb16",
            stack_ops=[
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
                "mixvarge_f2_gb16",
            ],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 16
    [
        MixVarGENetConfig(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=2,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],  # stride 32
]

cls_net_config = [
    [
        MixVarGENetConfig(
            in_channels=out_channels[-1],
            out_channels=int(320 * alpha),
            head_op="mixvarge_f2_gb16",
            stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
            stack_factor=1,
            stride=1,
            fusion_strides=[],
            extra_downsample_num=0,
        ),  # noqa
    ],
]

mixvargenet_cls_head = dict(
    type="EyeMixVarGEClsHead",
    net_config=cls_net_config,
    num_classes=2,
    in_channels=int(320 * alpha),
    out_channels=1280,
    bn_kwargs=bn_kwargs,
    pool_size=(3, 5),
    include_top=True,
)

eye_vis_head = dict(
    type="EyeMultiBranchHead",
    l_cls_head=copy.deepcopy(mixvargenet_cls_head),
    r_cls_head=copy.deepcopy(mixvargenet_cls_head),
)

cls_head = eye_vis_head

# Train Model
model = dict(
    type="EyeStatusClassifier",
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
    ),
    cls_head=cls_head,
    left_cls_losses=dict(
        type="SoftTargetCrossEntropy",
    ),
    right_cls_losses=dict(
        type="SoftTargetCrossEntropy",
    ),
    num_classes=2,
)

# Val Model
val_model = dict(
    type="EyeStatusClassifier",
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
    ),
    cls_head=cls_head,
    left_cls_losses=dict(
        type="SoftTargetCrossEntropy",
    ),
    right_cls_losses=dict(
        type="SoftTargetCrossEntropy",
    ),
    num_classes=2,
)

deploy_model = dict(
    type="EyeStatusClassifier",
    backbone=dict(
        type="MixVarGENet",
        net_config=net_config,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
    ),
    cls_head=cls_head,
    deploy=True,
)

# Deploy Model
deploy_inputs = dict(
    img=torch.randn(1, 3, 96, 160),
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)

data_root = bucket_client.url_to_local(
    "dmpv2://MultiMode_3/weiyu.li/eye_visibility"
)

rec_names = [
    "DA_train_04.rec",
    "DA_train_occ_01.rec",
    "train02.rec",
    "DA_train_0902.rec",
    "train_0925.rec",
    "train_0925_occ.rec",
    "trainset_1119.rec",
    "trainset_1119_occ.rec",
    "CD569_train_01.rec",
    "CD569_train_01_occ.rec",
    "trainset_cd569_20210207.rec",
    "trainset_cd569_20210207_occ.rec",
    "trainset_cd569_20210527.rec",
    "trainset_cd569_20210527_occ.rec",
    "trainset_h9_20210609.rec",
    "trainset_h9_20210609_occ.rec",
]
rec_paths = [os.path.join(data_root, rec) for rec in rec_names]

val_rec_names = [
    "DA_test_03.rec",
]
val_rec_paths = [os.path.join(data_root, rec) for rec in val_rec_names]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="EyeStatusDataset",
        rec_paths=rec_paths,
        dataset_type="eye_vis",
        transforms=[
            dict(
                type="RandomFlip",
                px=0.5,
            ),
            dict(
                type="RandomShiftRotateScale",
                rotate_prob=0.4,
                max_rotate_angle=10.0,
                shift_prob=0.4,
                max_shift_range=(0.05, 0.05),
                resize=True,
            ),
            dict(
                type="CropRecROI",
                crop_type="scale",
                target_shape=(96, 160, 3),
                base_roi=[20, 12, 180, 108],
                crop_jitter_range=0.1,
                center_shift_range=0,
            ),
            dict(
                type="ToTensor",
            ),
        ],
    ),
    sampler=dict(
        type="DistributedEyeStatusBalanceSampler",
        drop_last=True,
        shuffle=True,
    ),
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    pin_memory=True,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="EyeStatusDataset",
        rec_paths=val_rec_paths,
        dataset_type="eye_vis",
        transforms=[
            dict(
                type="CropRecROI",
                crop_type="center",
                target_shape=(96, 160, 3),
                base_roi=[20, 12, 180, 108],
                crop_jitter_range=0.0,
            ),
            dict(
                type="ToTensor",
            ),
        ],
    ),
    sampler=dict(
        type=torch.utils.data.DistributedSampler,
        shuffle=False,
    ),
    batch_size=batch_size_per_gpu,
    num_workers=1,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(
            type="TorchVisionAdapter",
            interface="ColorJitter",
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        ),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_index(1),
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
)


def update_metric(metrics, batch, model_outs):
    _, losses = model_outs
    for metric in metrics:
        metric.update(losses)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix=task_name,
)

invis_cls_ind = 1


def update_eval_metric(metrics, batch, model_outs):
    cls_target = batch["gt_eye_cls_labels"]
    l_cls_target, r_cls_target = torch.split(cls_target, 2, dim=1)
    cls_target = torch.cat([l_cls_target, r_cls_target], dim=0)
    l_cls_invalid = torch.sum(l_cls_target, dim=1) - 1
    l_cls_label = torch.argmax(l_cls_target, dim=1) + l_cls_invalid
    r_cls_invalid = torch.sum(r_cls_target, dim=1) - 1
    r_cls_label = torch.argmax(r_cls_target, dim=1) + r_cls_invalid

    pred, loss = model_outs
    l_feat = pred["l_feat"]
    r_feat = pred["r_feat"]

    batch_size = l_feat.size()[0]
    pred = torch.cat([l_feat, r_feat], dim=0)
    pred = pred.view(batch_size * 2, -1)

    pred_ind = torch.argmax(pred, dim=1)
    target_ind = torch.argmax(cls_target, dim=1)
    for metric in metrics:
        if isinstance(metric.name, list):
            if metric.name[0].find("eye_visibility") == 0:
                metric.update(cls_target, pred)
        else:
            if metric.name.find("left_eye_acc") == 0:
                metric.update(
                    l_cls_label,
                    pred[
                        :batch_size,
                    ],
                )
            elif metric.name.find("right_eye_acc") == 0:
                metric.update(
                    r_cls_label,
                    pred[
                        batch_size:,
                    ],
                )
            elif metric.name.find("eye_visibility") == 0:
                metric.update(target_ind, pred_ind)
            else:
                metric.update(cls_target, pred)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_eval_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix="Validation" + task_name,
)
val_metric_updater["log_prefix"] = "Validation " + task_name

# predictor
float_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                ignore_extra=True,
                allow_miss=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="Accuracy",
            name="left_eye_acc",
        ),
        dict(
            type="Accuracy",
            name="right_eye_acc",
        ),
        dict(
            type="EyeStatusMetrics",
            num_classes=2,
            name="eye_visibility",
        ),
    ],
    callbacks=[
        val_metric_updater,
        # dict(
        #     type="SaveEyeStatusResult",
        #     output_dir=save_feat_dir,
        # ),
    ],
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-step-2999.pth.tar"
                ),
                ignore_extra=True,
                check_hash=False,
                # verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="Accuracy",
            name="left_eye_acc",
        ),
        dict(
            type="Accuracy",
            name="right_eye_acc",
        ),
        dict(
            type="EyeStatusMetrics",
            num_classes=2,
            name="eye_visibility",
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=100,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    interval_by="step",
    save_interval=500,
    name_prefix=training_step + "-",
    strict_match=True,
    save_hash=False,
)
trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

epoch_step = 202
decay_epoch = [20, 30, 45, 60, 70, 80]
lr_decay_id = [epoch_id * epoch_step for epoch_id in decay_epoch]

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    # model_convert_pipeline=dict(
    #     type="ModelConvertPipeline",
    #     converters=[
    #         dict(
    #             type="LoadCheckpoint",
    #             checkpoint_path=os.path.join(
    #                 pretrain_dir,
    #                 "float-checkpoint-last.pth.tar",
    #             ),
    #             verbose=True,
    #         ),
    #     ],
    # ),
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={
            "backbone": dict(weight_decay=2e-4),
            "cls_head": dict(weight_decay=2e-4),
        },
        lr=0.01,
    ),
    batch_processor=batch_processor,
    find_unused_parameters=False,
    num_epochs=100,
    stop_by="epoch",
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=500,
            update_by="step",
            warmup_by="step",
            warmup_len=epoch_step,
            lr_decay_id=lr_decay_id,
            lr_decay_factor=0.32,
        ),
        metric_updater,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="Loss"),
    ],
    val_metrics=[],
)

qat_decay_epoch = [10, 13, 15, 16, 17]
qat_lr_decay_id = [epoch_id * epoch_step for epoch_id in qat_decay_epoch]

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                verbose=True,
                ignore_extra=True,
            ),
            dict(type="Float2QAT"),
        ],
    ),
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0)},
        lr=5e-4,
    ),
    batch_processor=batch_processor,
    num_epochs=1,
    num_steps=1000,
    stop_by="epoch",
    device=None,
    find_unused_parameters=False,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=100,
            lr_decay_id=qat_lr_decay_id,
            lr_decay_factor=0.1,
        ),
        metric_updater,
        ckpt_callback,
        # trace_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="Loss"),
    ],
    val_metrics=[],
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
                    ckpt_dir, "qat-checkpoint-step-1499.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[
        ckpt_callback,
        trace_callback,
    ],
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

onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
)
