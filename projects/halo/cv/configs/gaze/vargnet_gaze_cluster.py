import copy
import glob
import os

import torch
import torchvision
from horizon_plugin_pytorch.march import March

from hat.data.transforms.detection import (
    Normalize,
    RandomFlip,
    Resize,
    ToTensor,
)
from hat.data.transforms.gaze import (
    Clip,
    GazeRandomCropWoResize,
    GazeYUVTransform,
    RandomColorJitter,
)
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

DEBUG = False
CHIP = "J5"

task_name = "gaze_estimation_two_head"
num_classes = 1000
batch_size_per_gpu = 128
if DEBUG:
    device_ids = [0, 1]
else:
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES if CHIP == "J5" else March.BERNOULLI2

input_h = 192
input_w = 320
alpha = 0.75
group_base = 8
head_factor = 2
shared_feat_size = max(int(alpha * 1024), 1024)
last_channel_out = int(256 * alpha)
bn_kwargs = {
    "eps": 2e-05,
    "momentum": 0.1,
}
dropout_ratio = 0.2
# two_head
use_glass = True

model = dict(
    type="GazeModel",
    backbone=dict(
        type="VargNetV2",
        num_classes=1000,
        include_top=False,
        group_base=group_base,
        head_factor=head_factor,
        bn_kwargs=bn_kwargs,
        alpha=alpha,
    ),
    head=dict(
        type="GazeEyeldmkHead",
        input_size=[input_w, input_h],
        last_channel_out=last_channel_out,
        shared_feat_size=shared_feat_size,
        use_glass=use_glass,
        gaze_head_params=dict(
            use_pool=True,
            dropout_ratio=dropout_ratio,
            channels=(shared_feat_size, 128, 128, 4),
            output_add_bias=False,
            bn_kwargs=bn_kwargs,
        ),
        eyeldmk_head_params=dict(
            alpha=0.25,
            bn_kwargs=bn_kwargs,
            channels=[128, 128, 16],
        ),
        glass_head_params=dict(
            bn_kwargs=bn_kwargs,
            dropout_rate=None,
        ),
    ),
    losses=dict(
        type="GazeEyeldmkLoss",
        eye_ldmk_num=21,
        gaze_loss_name="wing",
        gaze_loss_weight=1,
        gaze_loss_params=dict(
            pitch_w=15,
            yaw_w=15,
            pitch_e=0.5,
            yaw_e=0.5,
            averaged_output=False,
            batch_axis=0,
        ),
        eye_ldmk_loss_name="wing",
        eye_ldmk_loss_weight=5,
        eye_ldmk_loss_params=dict(
            w=15,
            epsilon=0.1,
            averaged_output=False,
            batch_axis=0,
        ),
        use_glass=use_glass,
        cls_loss_weight=1.0,
    ),
)

deploy_model = dict(
    type="GazeModel",
    backbone=dict(
        type="VargNetV2",
        num_classes=1000,
        group_base=group_base,
        head_factor=head_factor,
        include_top=False,
        bn_kwargs=bn_kwargs,
        alpha=alpha,
    ),
    head=dict(
        type="GazeEyeldmkHead",
        input_size=[input_w, input_h],
        last_channel_out=last_channel_out,
        shared_feat_size=shared_feat_size,
        use_glass=True,
        gaze_head_params=dict(
            use_pool=True,
            dropout_ratio=dropout_ratio,
            channels=(shared_feat_size, 128, 128, 4),
            output_add_bias=False,
            bn_kwargs=bn_kwargs,
        ),
        eyeldmk_head_params=dict(
            alpha=0.25,
            bn_kwargs=bn_kwargs,
            channels=[128, 128, 16],
        ),
        glass_head_params=dict(
            bn_kwargs=bn_kwargs,
            dropout_rate=None,
        ),
    ),
    losses=None,
)
deploy_inputs = dict(img=torch.randn((1, 3, 192, 320)))

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)


rec_root = "./tmp_orig_data/gaze/newldmk_newgt_20220903"
rec_list = glob.glob(rec_root + "/recs/*.rec") + [
    os.path.join(_, "training/mega.rec")
    for _ in glob.glob(rec_root + "/gaze_mtl*")
]
data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="GazeDataset",
        rec_list=rec_list,
        input_size=(input_w, input_h),
        transforms=torchvision.transforms.Compose(
            [
                RandomFlip(0.5),
                RandomColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                    saturation=0.05,
                    hue=0.0,
                    prob=0.5,
                ),
                Clip(),
                GazeRandomCropWoResize(
                    size=(input_w, input_h),
                    prob=1.0,
                    area=(0.85, 1),
                    ratio=(1.25, 2),
                ),
                GazeYUVTransform(rgb_data=False, nc=3),
                Resize(img_scale=(192, 320), keep_ratio=False),
                ToTensor(),
                Normalize(mean=128.0, std=128.0),
            ]
        ),
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=0 if DEBUG else 16,
    pin_memory=True,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="GazeDataset",
        rec_list=[
            "./tmp_orig_data/gaze/ldmk_v205/recs_waug/fortesting/"
            + "testing/test_68pts_da04car_testing_clean_label_witheyelm.rec"
        ],
        input_size=(input_w, input_h),
        transforms=torchvision.transforms.Compose(
            [
                GazeRandomCropWoResize(
                    size=(input_w, input_h),
                    prob=1.0,
                    area=(0.85, 1),
                    ratio=(1.25, 2),
                    is_train=False,
                ),
                GazeYUVTransform(rgb_data=False, nc=3),
                Resize(img_scale=(192, 320), keep_ratio=False),
                ToTensor(),
                Normalize(mean=128.0, std=128.0),
            ]
        ),
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=0 if DEBUG else 16,
    pin_memory=True,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=None,
    loss_collector=collect_loss_by_index(1),
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=None,
    loss_collector=None,
)


def update_metric(metrics, batch, model_outs):
    gt_label = batch["label"]
    target_eye_ldmk = batch["label"]["gt_normed_eye_ldmk"]
    preds, losses = model_outs
    for metric in metrics:
        if metric._get_name() == "LossShow":
            metric.update(losses)
        elif metric._get_name() == "AngleDifferenceMetric":
            metric.update(gt_label, preds)
        elif metric._get_name() == "EyeLdmksDist":
            metric.update(target_eye_ldmk, preds)
        elif metric._get_name() == "Accuracy":
            metric.update(gt_label["gt_glass"], preds["glass_cls"])
        else:
            raise ValueError


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
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=0.0)},
        lr=5e-3,
    ),
    batch_processor=batch_processor,
    num_epochs=250,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=[150, 170, 190, 210, 230],
            lr_decay_factor=0.5,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(
            type="AngleDifferenceMetric", name="left_eye", use_glass=use_glass
        ),
        dict(
            type="AngleDifferenceMetric", name="right_eye", use_glass=use_glass
        ),
        dict(type="EyeLdmksDist", name="eye_ldmk", size=(input_w, input_h)),
        dict(type="Accuracy", name="glass_acc"),
    ],
    val_metrics=[
        dict(
            type="AngleDifferenceMetric", name="left_eye", use_glass=use_glass
        ),
        dict(
            type="AngleDifferenceMetric", name="right_eye", use_glass=use_glass
        ),
        dict(type="EyeLdmksDist", name="eye_ldmk", size=(input_w, input_h)),
        dict(type="Accuracy", name="glass_acc"),
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
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0)},
        lr=1e-4,
    ),
    batch_processor=batch_processor,
    num_epochs=30,
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
        dict(type="LossShow"),
        dict(type="AngleDifferenceMetric", name="left_eye"),
        dict(type="AngleDifferenceMetric", name="right_eye"),
        dict(type="EyeLdmksDist", name="eye_ldmk", size=(input_w, input_h)),
    ],
    val_metrics=[
        dict(type="AngleDifferenceMetric", name="left_eye"),
        dict(type="AngleDifferenceMetric", name="right_eye"),
        dict(type="EyeLdmksDist", name="eye_ldmk", size=(input_w, input_h)),
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
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
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

# predictor
float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="AngleDifferenceMetric", name="left_eye"),
        dict(type="AngleDifferenceMetric", name="right_eye"),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
    model=model,
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
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="AngleDifferenceMetric", name="left_eye"),
        dict(type="AngleDifferenceMetric", name="right_eye"),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

int_infer_predictor = dict(
    type="Predictor",
    model=model,
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
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="AngleDifferenceMetric", name="left_eye"),
        dict(type="AngleDifferenceMetric", name="right_eye"),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)
