import copy
import os

import numpy as np
import torch
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion
from projects.halo.cv.configs.gesture.plugins.get_dyn_demo_config import (
    get_test_transforms,
    get_testdata_miniv2,
    get_train_transforms,
    get_traindata_miniv2,
)
from projects.halo.cv.configs.gesture.plugins.get_dyn_model import (
    get_dyn_vargnet,
)

# --------- global params -----------
VERSION = ConfigVersion.v2
DEBUG = True
SHIP = "J5"


# --------- task params -----------
training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "dyn_vargnet_demo"

if DEBUG:
    device_ids = [0, 1]
    batch_size_per_gpu = 512
    num_epochs_float = 1
    step_epochs_float = [1]
    step_log_freq = 10
    data_small = True
else:
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    batch_size_per_gpu = 128
    num_epochs_float = 60
    step_epochs_float = [36, 54]
    step_log_freq = 1000
    data_small = False


num_epochs_qat = 1
step_epochs_qat = [1]


# for trainer
lr_multiplier_list = [4.0, 2.0, 0.5, 0.25, 1.0]
lr_choice = 0


def get_lr(num_gpus=4, base_lr=0.00125, batch_size_per_gpu=128):
    return base_lr * batch_size_per_gpu * num_gpus


lr = lr_multiplier_list[lr_choice]
lr = get_lr(
    num_gpus=len(device_ids),
    base_lr=1e-12,  # 0.0125 * lr / batch_size_per_gpu,
    batch_size_per_gpu=batch_size_per_gpu,
)
qnn_lr = 0.01 * get_lr(
    num_gpus=len(device_ids),
    base_lr=1e-12,  # 0.04 * lr / batch_size_per_gpu,
    batch_size_per_gpu=batch_size_per_gpu,
)

# tmp
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES if SHIP == "J5" else March.BERNOULLI2

# for model
num_classes = 59

# for metric
score_groups = [
    [
        0,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
    ],
    [1],
    [2],
    [12],
    [13],
    [23],
    [24, 25],
    [58, 26],
]

class_vali = len(score_groups)

# for sampler
ref_num = 500000 if not DEBUG else 10000
train_valid_classes = list(range(num_classes))
# balance_proportion = [1.0] + [0] * 22 + [2.0, 2.0, 4.0] + [0] * 32 + [4.0]
balance_proportion = [1.0] + [0] * 22 + [2.0, 2.0] + [0] * 34

scale_ref = sum(balance_proportion)
balance_proportion = [_ / scale_ref for _ in balance_proportion]
ref_num = ref_num * scale_ref

expect_distribution = {}
for key, val in zip(train_valid_classes, balance_proportion):
    if val:
        expect_distribution.update({f"{key}": val})

assert (
    sum(list(expect_distribution.values())) == 1
), "The probabilities of each category must sum to one!"
# --------- models -----------
# channel 17, depend on aug params, gluon:multi_ratio_list[multiscale_choice]
input_channel = 17
model = get_dyn_vargnet(
    num_classes=num_classes, input_channels=input_channel, deploy=False
)
deploy_model = get_dyn_vargnet(
    num_classes=num_classes, input_channels=input_channel, deploy=True
)
deploy_inputs = dict(
    clip_keypoints=torch.randn(1, input_channel, 32, 21)  # * 2 - 1
)
deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)

# --------- dataloader -----------
(
    train_rec_list,
    train_roidb_list,
    train_roidb_seq_list,
    repeat_times,
) = get_traindata_miniv2(data_small)
train_transforms = get_train_transforms()

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ConcatDataset",
        with_flag=True,
        record_index=False,
        datasets=[
            dict(
                type="RepeatDataset",
                times=repeat_times[i],
                dataset=dict(
                    type="RoidbActDataset",
                    rec_path=train_rec_list[i],
                    roidb_path=train_roidb_list[i],
                    roidb_seq_path=train_roidb_seq_list[i],
                    transforms=train_transforms,
                    is_read_rec=False,
                ),
            )
            for i in range(len(train_rec_list))
        ],
    ),
    sampler=dict(
        type="DistributedProportionSampler",
        expect_distribution=expect_distribution,
        task_name="gesture",
        num_reference=ref_num,
    ),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=2 if DEBUG else 16,
    pin_memory=True,
)


(
    test_rec_list,
    test_roidb_list,
    test_roidb_seq_list,
    repeat_times,
) = get_testdata_miniv2(data_small)
test_transforms = get_test_transforms()

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ConcatDataset",
        with_flag=True,
        record_index=False,
        datasets=[
            dict(
                type="RepeatDataset",
                times=repeat_times[i],
                dataset=dict(
                    type="RoidbActDataset",
                    rec_path=test_rec_list[i],
                    roidb_path=test_roidb_list[i],
                    roidb_seq_path=test_roidb_seq_list[i],
                    transforms=test_transforms,
                    is_read_rec=False,
                ),
            )
            for i in range(len(test_rec_list))
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=2 if DEBUG else 16,
    pin_memory=True,
)

# --------- processor -----------
batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=None,
    loss_collector=collect_loss_by_index(1),
    # model return preds, losses
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=None,
    loss_collector=None,
)


# --------- metric -----------

convert_matrix = np.zeros(shape=(59, class_vali))
convert_map = {}
for idx, groups in enumerate(score_groups):
    convert_matrix[groups, idx] = 1
    for group_item in groups:
        convert_map[f"{group_item}"] = idx


def update_metric(metrics, batch, model_outs):
    # preprocess
    gt_label = batch["act_label"].squeeze()
    preds, losses = model_outs  # if test, losses is labeles

    # DEL -1
    mask = gt_label.ge(0)
    _gt_label = gt_label[mask].cpu()
    _preds = preds[mask, :].detach().cpu()

    # test score groups
    _gt_label = _gt_label.apply_(lambda x: convert_map[str(x)])
    _preds = _preds @ convert_matrix

    for metric in metrics:
        # for train
        if metric._get_name() == "LossShow":
            metric.update(losses)
        elif metric._get_name() == "Accuracy":
            metric.update(_gt_label, _preds)
        else:
            raise ValueError


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=step_log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Val- " + task_name


# --------- callback -----------
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
    save_hash=False,
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

# --------- trainer -----------

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},  # ？？
        lr=lr,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=num_epochs_float,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=step_epochs_float,
            lr_decay_factor=0.1,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(type="Accuracy", name="Dgest_acc"),
    ],
    val_metrics=[
        dict(type="Accuracy", name="Dgest_acc"),
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
                # ignore_extra=True,
                # allow_miss=True,
                check_hash=False,
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=0)},
        lr=qnn_lr,
    ),
    batch_processor=batch_processor,
    num_epochs=num_epochs_qat,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=step_epochs_qat,
            lr_decay_factor=0.1,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(type="Accuracy", name="Dgest_acc"),
    ],
    val_metrics=[
        dict(type="Accuracy", name="Dgest_acc"),
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
    input_source=["ddr"],  # pyramid, ddr
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
        dict(type="Accuracy", name="Dgest_acc"),
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
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="Accuracy", name="Dgest_acc"),
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
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="Accuracy", name="Dgest_acc"),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)
