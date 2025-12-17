import copy
import os

import numpy as np
import torch
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion
from projects.halo.cv.configs.gesture.plugins.get_sta_demo_config import (
    get_lmdb_dataset,
)
from projects.halo.cv.configs.gesture.plugins.get_sta_model import (
    get_sta_network,
)

# --------- global params -----------
VERSION = ConfigVersion.v2
SHIP = "J5"
use_rgb = True

run_mode = "white_box"  # ['local_debug', 'cluster_debug', 'cluster_train']

# --------- task params -----------
training_step = os.environ.get("HAT_TRAINING_STEP", "float")
task_name = "sta_ges_example_white_box"
num_machines = 1


if run_mode == "white_box":
    device_ids = [0, 1]
    batch_size_per_gpu = 128
    num_epochs_float = 25
    step_epochs_float = [7, 18, 22]
    step_log_freq = 10
    data_small = True  # white box data
    max_jobtime = 16000
    num_per_class = 100000
    num_workers = 0
    class_in_data = [3, 4, 5, 7, 17, 69, 70]
    val_interval = 1

elif run_mode == "cluster_debug":
    device_ids = [0, 1]
    batch_size_per_gpu = 64
    num_epochs_float = 25
    step_epochs_float = [7, 18, 22]
    step_log_freq = 1000
    data_small = False  # static gesture data
    max_jobtime = 1000
    num_per_class = 100000
    num_workers = 2
    val_interval = 1
    class_in_data = [
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        17,
        18,
        19,
        20,
        21,
        22,
        69,
        70,
    ]

elif run_mode == "cluster_train":
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    batch_size_per_gpu = 128
    num_epochs_float = 25
    step_epochs_float = [7, 18, 22]
    step_log_freq = 1000
    data_small = False  # static gesture data
    max_jobtime = 16000
    num_per_class = 100000
    num_workers = 6
    val_interval = num_epochs_float - 1
    class_in_data = [
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        17,
        18,
        19,
        20,
        21,
        22,
        69,
        70,
    ]
else:
    raise ValueError(run_mode)

num_gpus_per_machine = len(device_ids)


num_epochs_qat = 1
step_epochs_qat = [1]


# for trainer
lr_multiplier_list = [5.0, 4.0, 2.0, 0.5, 0.25, 1.0]
lr_choice = 1


def get_lr(num_gpus=4, base_lr=0.00125, batch_size_per_gpu=128):
    return base_lr * batch_size_per_gpu * num_gpus


lr = lr_multiplier_list[lr_choice]
lr = get_lr(
    num_gpus=len(device_ids),
    base_lr=0.0125 * lr / batch_size_per_gpu,
    batch_size_per_gpu=batch_size_per_gpu,
)
qnn_lr = 0.01 * get_lr(
    num_gpus=len(device_ids),
    base_lr=0.0125 * lr / batch_size_per_gpu,
    batch_size_per_gpu=batch_size_per_gpu,
)

# tmp
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES if SHIP == "J5" else March.BERNOULLI2

# for model
num_classes = 71


# for sampler
train_valid_classes = list(range(num_classes))

balance_proportion = (
    [4.0, 0.1, 0.1]
    + [1.0, 1.0, 1.0]
    + [0.1, 1.0]
    + [0.1] * 9
    + [1.0]
    + [0.1] * 9
    + [0.0] * 42
    + [1.0] * 2
)  # balance5
for i in range(len(balance_proportion)):
    if i not in class_in_data:
        balance_proportion[i] = 0
sum_prop = sum(balance_proportion)
ref_num = num_per_class * sum_prop
expect_distribution = {}
for i in range(len(balance_proportion)):
    if balance_proportion[i] <= 0:
        continue
    if i != (len(balance_proportion) - 1):
        expect_distribution[i] = round(balance_proportion[i] / sum_prop, 4)
    else:
        expect_distribution[i] = 1 - sum(expect_distribution.values())
assert sum(list(expect_distribution.values())) == 1, sum(
    list(expect_distribution.values())
)

#
model_size = "0.5"
pretrain_param_path = (
    "hdfs://hobot-bigdata/user/zining.xu/models/pretrain/hgr/Vargnet_v2_"
    + model_size
    + "_c1000_2stream.params"
)
_modality = ["rgb", "kps"]

# --------- models -----------
# channel 17, depend on aug params, gluon:multi_ratio_list[multiscale_choice]
input_channel = 3
model = get_sta_network(num_classes=num_classes, deploy=False)
deploy_model = get_sta_network(num_classes=num_classes, deploy=True)
deploy_inputs = dict(
    clip_keypoints=torch.randn(1, input_channel, 32, 21),  # * 2 - 1
    frames=torch.randn(1, 8, 3, 128, 128),  # * 2 - 1
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
train_dataset, val_dataset = get_lmdb_dataset(data_small)


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    sampler=dict(
        type="DistributedProportionSampler",
        expect_distribution=expect_distribution,
        task_name="gesture",
        num_reference=ref_num,
    ),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=num_workers,
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

train_valid_classes = np.arange(71).tolist()
test_score_groups = [list(np.arange(71))] + [
    [] for _ in range(num_classes - 1)
]
test_score_groups[0].remove(3)
test_score_groups[0].remove(4)
test_score_groups[0].remove(5)
test_score_groups[0].remove(7)
test_score_groups[0].remove(17)
test_score_groups[0].remove(69)
test_score_groups[0].remove(70)
test_score_groups[3] = [3]
test_score_groups[4] = [4]
test_score_groups[5] = [5]
test_score_groups[7] = [7]
test_score_groups[17] = [17]
test_score_groups[69] = [69]
test_score_groups[70] = [70]

class_vali = len(test_score_groups)
convert_matrix = np.zeros(shape=(71, class_vali))
convert_map = {}
for idx, groups in enumerate(test_score_groups):
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
    # _preds = preds[mask, :].detach().cpu()
    kps_pred_mask = preds[0][mask, :].detach().cpu()
    rgb_pred_mask = preds[1][mask, :].detach().cpu()

    # test score groups
    _gt_label = _gt_label.apply_(lambda x: convert_map[str(x)])

    _kps_pred_mask = kps_pred_mask @ convert_matrix
    _rgb_pred_mask = rgb_pred_mask @ convert_matrix

    for metric in metrics:
        # for train
        if metric._get_name() == "LossShow":
            metric.update(losses)
        elif metric.name == "kps_acc":
            metric.update(_gt_label, _kps_pred_mask)
        elif metric.name == "rgb_acc":
            metric.update(_gt_label, _rgb_pred_mask)
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
val_metric_updater["step_log_freq"] = 50


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
    val_interval=val_interval,
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
        val_callback,  # mark
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(type="Accuracy", name="kps_acc"),
        dict(type="Accuracy", name="rgb_acc"),
    ],
    val_metrics=[
        # dict(type="Accuracy", name="Dgest_acc"),
        dict(type="Accuracy", name="kps_acc"),
        dict(type="Accuracy", name="rgb_acc"),
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
                ignore_extra=True,
                allow_miss=True,
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
        # val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(type="Accuracy", name="kps_acc"),
        dict(type="Accuracy", name="rgb_acc"),
    ],
    val_metrics=[
        dict(type="Accuracy", name="kps_acc"),
        dict(type="Accuracy", name="rgb_acc"),
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
        # dict(type="Accuracy", name="Dgest_acc"),
        dict(type="Accuracy", name="kps_acc"),
        dict(type="Accuracy", name="rgb_acc"),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
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
        # dict(type="Accuracy", name="Dgest_acc"),
        dict(type="Accuracy", name="kps_acc"),
        dict(type="Accuracy", name="rgb_acc"),
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
        dict(type="Accuracy", name="kps_acc"),
        dict(type="Accuracy", name="rgb_acc"),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)


job_list = [
    # "sleep 100000",
    "pip3 install pytorch-crf==0.7.2 -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",
    "pip3 install --upgrade hatbc==0.9.0b202304251506+8b3da9d -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc",
    "python3 -W ignore tools/train.py --config hat/mr_files/static_gesture_config_example.py --stage float NCCL_DEBUG=WARN",  # noqa E501
    # "python3 -W ignore tools/predict.py --config hat/mr_files/static_gesture_config_example.py --stage float NCCL_DEBUG=WARN",  # noqa E501
    "python3 -W ignore tools/train.py --config hat/mr_files/static_gesture_config_example.py --stage qat",  # noqa E501
    "python3 -W ignore tools/train.py --config hat/mr_files/static_gesture_config_example.py --stage int_infer",  # noqa E501
]

k8s_config = dict(
    job_name=task_name,
    job_password="123666",
    num_machines=num_machines,
    num_gpus_per_machine=num_gpus_per_machine,
    framework="pytorch",
    task_label="halo_multitask",
    project_id="TD20220003",
    input_bucket="MultiMode_2",
    output_bucket="MultiMode",
    priority=5,
    docker_image="docker.hobot.cc/imagesys/hat:halo-runtime-py38-cu111-torch1-102",  # with ablu and trackeval  # noqa
    max_jobtime=max_jobtime,
    # launcher only for multi-machines
    launcher="mpi",
    # upload folder
    upload_folder_name="k8s_job",
    folder_list=[
        "hat",
        "plugins",
        "tools",
        "configs",
        "projects",
        "projects/halo/cv/",
        "plugins/k8s_submit/url2IP.py",
        "plugins/k8s_submit/ssh_launcher.py",
    ],
    job_list=job_list,
)
