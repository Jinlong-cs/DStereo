import copy
import os

import torch
from horizon_plugin_pytorch.march import March

from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")
is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
DEBUG = True

task_name = "sndr_fasternet_lite_v1"
num_classes = 3
batch_size_per_gpu = 32
input_size = [128, 128]
device_ids = [0] if DEBUG else [0, 1, 2, 3]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES
num_classes = 3

model = dict(
    type="SmokeClsModel",
    backbone=dict(
        type="SNDRFasterNet",
        in_chans=3,
        mlp_ratio=2,
        embed_dim=32,
        depths=[1, 2, 8, 2],
        patch_size=4,
        patch_stride=4,
        patch_size2=2,
        patch_stride2=2,
        drop_path_rate=0.0,
        norm_layer="BN",
        act_layer="RELU",
        n_div=[1, 2, 4, 4],
    ),
    losses=dict(type="SoftmaxCELoss"),
    num_classes=num_classes,
)
deploy_model = dict(
    type="SmokeClsModel",
    backbone=dict(
        type="SNDRFasterNet",
        in_chans=3,
        mlp_ratio=2,
        embed_dim=32,
        depths=[1, 2, 8, 2],
        patch_size=4,
        patch_stride=4,
        patch_size2=2,
        patch_stride2=2,
        drop_path_rate=0.0,
        norm_layer="BN",
        act_layer="RELU",
        n_div=[1, 2, 4, 4],
    ),
    losses=dict(type="SoftmaxCELoss"),
    num_classes=num_classes,
)
deploy_inputs = dict(img=torch.randn((1, 3, 128, 128)))

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)

data_root = os.path.join(
    bucket_root, "HDLTAlgorithm/data/orig_data/action/video/data"
)
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="SmokeClsDataset",
        rec_list=[
            f"{data_root}/2020092122_small_image_train.rec",
        ],
        all_rec_info_list=[
            f"{data_root}/anno_phone_refresh_inter_newRule_recInfo_0605.json",
            f"{data_root}/anno_phone_recInfo_0530_newRule.json",
        ],
        all_rec_imgIdx_list=[
            f"{data_root}/anno_phone_refresh_inter_newRule_summary_0605.json",
            f"{data_root}/anno_phone_oldVal_summary_0524.json",
            f"{data_root}/anno_phone_imgIdx_0530_newRule.json",
        ],
        transforms=[
            dict(
                type="RandomRotateCrop",
                net_input_size=(128, 128),
                rot_prob=1.0,
                rot_angle_range=30,
                center_shift_prob=1.0,
                center_shift_range=0.01,
                norm_ratio=1.25,
                norm_method="longside_square",
                norm_jitter_range=0.25,
                net_target_size=(128, 128),
                base_len=1.0,
            ),
            dict(
                type="OneFromMultiple",
                transforms=[
                    dict(
                        type="RandomNoise",
                        prob=1,
                        min=-5,
                        max=5,
                    ),
                    dict(
                        type="GaussianNoise",
                        prob=1,
                        mean=0,
                        sigma=1,
                    ),
                    dict(
                        type="SaltPepperNoise",
                        prob=1,
                        s_ratio=0.05,
                        p_ratio=0.05,
                    ),
                ],
                probs=[1, 0.0, 0.0],
            ),
            dict(
                type="RandomFlip",
                px=1,
                py=0,
            ),
            dict(
                type="RandomBrightnessContrast",
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                brightness_by_max=True,
                p=0.5,
            ),
            dict(
                type="HueSaturationValue",
                hue_range=(-20, 20),
                sat_range=(-30, 30),
                val_range=(-20, 20),
                p=0.5,
            ),
            dict(type="ToTensor", to_yuv=False),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=0 if DEBUG else 12,
    pin_memory=True,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="SmokeClsDataset",
        rec_list=[
            f"{data_root}/2020092122_small_image_train.rec",
        ],
        all_rec_info_list=[
            f"{data_root}/anno_phone_refresh_inter_newRule_recInfo_0605.json",
            f"{data_root}/anno_phone_recInfo_0530_newRule.json",
        ],
        all_rec_imgIdx_list=[
            f"{data_root}/anno_phone_refresh_inter_newRule_summary_0605.json",
            f"{data_root}/anno_phone_oldVal_summary_0524.json",
            f"{data_root}/anno_phone_imgIdx_0530_newRule.json",
        ],
        transforms=[
            dict(
                type="RandomRotateCrop",
                net_input_size=(128, 128),
                rot_prob=0.0,
                center_shift_prob=0.0,
                norm_ratio=1.0,
                norm_jitter_range=0.0,
                net_target_size=(128, 128),
                base_len=1.0,
            ),
            dict(type="ToTensor", to_yuv=False),
        ],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=0 if DEBUG else 12,
    pin_memory=True,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
        dict(type="OneHot", num_classes=num_classes),
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
        dict(type="OneHot", num_classes=num_classes),
    ],
    loss_collector=None,
)


def update_metric(metrics, batch, model_outs):
    target = batch["labels"]
    preds, losses = model_outs
    for metric in metrics:
        if "loss" in metric.name.lower():
            metric.update(losses)
        else:
            metric.update(target, preds)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=1000,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Validation " + task_name

stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.4,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=5,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_by="epoch",
            warmup_len=0,
            step_log_interval=1000,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="SoftmaxCELoss"),
        dict(type="SmokeClsRecall", name="SmokeClsRecall"),
        dict(type="SmokeClsPrecision", name="SmokeClsPrecision"),
    ],
    val_metrics=[
        dict(type="SmokeClsRecall", name="SmokeClsRecall"),
        dict(type="SmokeClsPrecision", name="SmokeClsPrecision"),
    ],
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.0001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=5,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=[15, 25],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="SoftmaxCELoss"),
        dict(type="SmokeClsRecall", name="SmokeClsRecall"),
        dict(type="SmokeClsPrecision", name="SmokeClsPrecision"),
    ],
    val_metrics=[
        dict(type="SmokeClsRecall", name="SmokeClsRecall"),
        dict(type="SmokeClsPrecision", name="SmokeClsPrecision"),
    ],
)

# just for saving int_infer pth and pt
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
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),  # noqa
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

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path="tmp_models/sndr_fasternet_lite_v1/float-checkpoint-epoch-0004-15b2d8e5.pth.tar",  # noqa
        ),
    ],
)

onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=deploy_model_convert_pipeline,
    kwargs=dict(
        verbose=True,
        opset_version=11,
    ),
)
