import copy
import os

import torch
import torchvision
from horizon_plugin_pytorch.march import March

from hat.data.transforms.detection import ToTensor
from hat.data.transforms.faceid import SpatialVariantBrightness
from hat.data.transforms.iris import IrisMtlTrans
from hat.data.transforms.landmark import RandomShiftRotateScale
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.apply_func import _as_list
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

DEBUG = False

task_name = "iris_visibility"
num_classes = 1000
batch_size_per_gpu = 128
input_size = [192, 320]
data_root = "tmp_orig_data/iris/"
if DEBUG:
    device_ids = [0, 1, 2, 3]
else:
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES

model = dict(
    type="IrisClassifier",
    backbone=dict(
        type="VargNetV2",
        num_classes=1000,
        bn_kwargs={},
        alpha=0.25,
        include_top=False,
    ),
    head=dict(
        type="IrisMutilBranchHead",
        bn_kwargs={},
        alpha=0.25,
        classfier_num=2,
        use_pool=False,
        bias=True,
    ),
    losses=dict(type="CEWithLabelSmooth"),
)
deploy_model = dict(
    type="IrisClassifier",
    backbone=dict(
        type="VargNetV2",
        num_classes=1000,
        bn_kwargs={},
        alpha=0.25,
        include_top=False,
    ),
    head=dict(
        type="IrisMutilBranchHead",
        bn_kwargs={},
        alpha=0.25,
        classfier_num=2,
        use_pool=False,
        bias=True,
    ),
    losses=None,
)
deploy_inputs = dict(img=torch.randn((1, 3, 192, 360)))

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="IrisDataset",
        rec_list=[
            f"{data_root}/train_iris_cd569_20210527_cls_2.rec",
            f"{data_root}/train_iris_20211116.rec",
            f"{data_root}/train_iris_20211116_flip.rec",
            f"{data_root}/train_iris_20211116_occ.rec",
            f"{data_root}/train_iris_20220124.rec",
            f"{data_root}/train_iris_20220124_flip.rec",
            f"{data_root}/train_iris_20220124_occ.rec",
        ],
        transforms=torchvision.transforms.Compose(
            [
                RandomShiftRotateScale(
                    rotate_prob=0.5,
                    max_rotate_angle=15,
                    shift_prob=1,
                    max_shift_range=(0.1, 0.1),
                    scale_range=(0.72, 0.88),
                    out_shape=(192, 320),
                    bounded=False,
                ),
                SpatialVariantBrightness(
                    p=0.5,
                    brightness=0.6,
                    max_template_type=3,
                    online_template=False,
                ),
                IrisMtlTrans(
                    input_size=input_size,
                    cam_hw=[720, 1280],
                    standard_focal=600,
                    to_yuv420sp=False,
                ),
                ToTensor(),
            ]
        ),
        shuffle=False,
        visible_index=[0, 2],
        invisible_index=[1, 3, 4],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=0 if DEBUG else 24,
    pin_memory=True,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="IrisDataset",
        rec_list=[
            f"{data_root}/test_iris_20211116.rec",
        ],
        transforms=torchvision.transforms.Compose(
            [
                RandomShiftRotateScale(
                    out_shape=input_size,
                ),
                IrisMtlTrans(
                    input_size=input_size,
                    cam_hw=[720, 1280],
                    standard_focal=600,
                    to_yuv420sp=False,
                ),
                ToTensor(),
            ]
        ),
        shuffle=False,
        visible_index=[0, 2],
        invisible_index=[1, 3, 4],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=0 if DEBUG else 24,
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
    # two branch update metric
    if len(model_outs) == 4:
        preds, losses, l_loss, r_loss = model_outs
    else:
        preds, _ = model_outs
    preds = _as_list(preds)
    target = _as_list(batch["labels"])
    preds_l, preds_r = preds[0].reshape(-1, 2), preds[1].reshape(-1, 2)
    mask_l, mask_r = target[1], target[2]
    label_l, label_r = target[3], target[4]

    mask_preds_l = preds_l.index_select(
        dim=0, index=torch.where(mask_l == 1)[0]
    )
    mask_preds_r = preds_r.index_select(
        dim=0, index=torch.where(mask_r == 1)[0]
    )
    mask_label_l = label_l.masked_select(mask_l.type(torch.bool))
    mask_label_r = label_r.masked_select(mask_r.type(torch.bool))

    for metric in metrics:
        assert metric.name in [
            "l_acc",
            "r_acc",
            "l_loss",
            "r_loss",
        ], f"not support {metric.name} for metric!"
        if metric.name == "l_acc":
            metric.update(mask_label_l, mask_preds_l)
        elif metric.name == "r_acc":
            metric.update(mask_label_r, mask_preds_r)
        elif metric.name == "l_loss":
            metric.update(l_loss)
        elif metric.name == "r_loss":
            metric.update(r_loss)


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
    num_epochs=240,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_by="epoch",
            warmup_len=5,
            step_log_interval=1000,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow", name="l_loss"),
        dict(type="LossShow", name="r_loss"),
        dict(type="Accuracy", name="l_acc"),
        dict(type="Accuracy", name="r_acc"),
    ],
    val_metrics=[
        dict(type="Accuracy", name="l_acc"),
        dict(type="Accuracy", name="r_acc"),
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
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.0001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=40,
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
        dict(type="LossShow", name="l_loss"),
        dict(type="LossShow", name="r_loss"),
        dict(type="Accuracy", name="l_acc"),
        dict(type="Accuracy", name="r_acc"),
    ],
    val_metrics=[
        dict(type="Accuracy", name="l_acc"),
        dict(type="Accuracy", name="r_acc"),
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
        dict(type="Accuracy", name="l_acc"),
        dict(type="Accuracy", name="r_acc"),
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
        dict(type="Accuracy", name="l_acc"),
        dict(type="Accuracy", name="r_acc"),
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
        dict(type="Accuracy", name="l_acc"),
        dict(type="Accuracy", name="r_acc"),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)
