# # isp configuration entry
#
# > Config file goes in, trained model comes out.
#
# This is isp training configuration entry. It is a little long, but don't
# worry, it's also well documented.
#
# This document tries to help you get familiar with isp config,
# and develop it easily and happily.

import os
import os.path as osp

import torch
from horizon_plugin_pytorch.quantization import March

from hat.utils import Config
from projects.isp.app.det.create_dataloader import (
    get_train_chunk_dataloader,
    get_train_dataloader,
    get_val_dataloader,
)
from projects.isp.app.det.data_transforms import (
    get_train_transform,
    get_val_transform,
)
from projects.isp.app.det.model_select_j3 import get_model_j3
from projects.isp.app.det.model_select_j5 import get_model_j5
from projects.isp.app.det.utils import (
    get_datasets,
    get_norm_mean_std,
    reformat_aidi_eval_out,
    tb_update,
    tb_update_sd,
)

cfg_dir = osp.dirname(osp.dirname(__file__))

cudnn_benchmark = True
march = March.BAYES
seed = None
log_rank_zero_only = True
enable_tensorboard = True

step = "float"  # float | qat | int_infer

############## job type settings ##############

preprocess_type = "demosaic"
assert preprocess_type in [
    "demosaic",
    "rawpack",
    "none",
]
me_type = "baseline"
assert me_type in [
    "genisp",
    "nisp",
    "baseline",
]

me_channels = 3  # me channels setting
job_type = "_".join([preprocess_type, me_type, str(me_channels)])
det_task = "person"
assert det_task in [
    "person",
    "vehicle",
    "traffic_light",
]

# j5 means using super dirve model config
base_model = "j5"  # j5 | j3
sensor = "ovx8b"  # ovx8b | ar0820
time = "night"  # day | night
car = "LX179"
data_type = "raw"  # raw | yuv
bit_depth = 12  # 8 | 12 | 16 | 24
norm = "11"  # 01 | 11
split_transform = True
split_trans_h = 256
split_trans_w = 256


############## task settings ##############
data_shape = (3, 512, 960)
lr = 1e-3
weight_decay = 4e-5
epoch = 100
bs = 16
img_scale = data_shape[1:]
mean, std = get_norm_mean_std(data_type, norm, bit_depth)

############# aidi job settings ##############
python_file_name = osp.basename(__file__).replace(".py", "")
job_name = "%s_%s_det_%s_%s_%s_%s_norm%s_bs%d_epoch%d_lr%s" % (
    base_model,
    det_task,
    sensor,
    time,
    job_type,
    data_type,
    norm,
    bs,
    epoch,
    str(lr).replace(".", "_"),
)

job_list = [
    "pip3 install hat_sim -i https://pypi.hobot.cc/simple \
        --extra-index-url=https://pypi.hobot.cc/hobot-local/simple \
              --trusted-host pypi.hobot.cc",
    "python3 tools/train.py --stage %s --config projects/isp/app/det/%s.py"
    % (step, python_file_name),  # your running command
]

local_train = not os.path.exists("/running_package")
if local_train:
    device_ids = [
        0,
    ]
    num_machines = 1
    num_gpus_per_machine = len(device_ids)
    num_gpus = num_machines * num_gpus_per_machine
    log_freq = 1
    batch_size_per_gpu = bs
    batch_size = batch_size_per_gpu * num_gpus
    val_interval = 1
    ckpt_dir = "checkpoints/det/%s/%s" % (
        job_type,
        job_name,
    )  # replace with your own ckpt dir.
    log_dir = "logs/det/%s/%s" % (
        job_type,
        osp.basename(__file__).replace(".py", ""),
    )
    train_num_workers, val_num_workers = 4, 2
    train_prefetch_factor, val_prefetch_factor = 4, 2
    json_save_prefix = "results/det/%s/%s" % (
        job_type,
        job_name,
    )  # replace with your own log dir.
    json_save_prefix_adas = "results_adas/det/%s/%s" % (
        job_type,
        job_name,
    )
    det_data_yaml = "projects/isp/app/det/dataset/det_4pe_%s.yaml" % sensor

# k8s setiings, for aidi platform
k8s_config_file = osp.join(cfg_dir, "k8s_config.py")
base_k8s_config = Config.fromfile(k8s_config_file)
k8s_config = dict()
k8s_config.update(base_k8s_config._cfg_dict)
k8s_config["job_name"] = job_name
k8s_config["num_machines"] = 1
k8s_config["num_gpus_per_machine"] = 8
k8s_config["job_list"] = job_list
k8s_config["max_jobtime"] = 14000

if not local_train:
    device_ids = [i for i in range(k8s_config["num_gpus_per_machine"])]
    num_machines = k8s_config["num_machines"]
    num_gpus_per_machine = k8s_config["num_gpus_per_machine"]
    val_interval = 2
    log_freq = 30
    batch_size_per_gpu = bs
    num_gpus = num_machines * num_gpus_per_machine
    batch_size = batch_size_per_gpu * num_gpus
    ckpt_dir = "/cluster_home/custom_data/models/det/%s/%s" % (
        job_type,
        job_name,
    )
    log_dir = "/job_tboard/"
    train_num_workers, val_num_workers = 4, 2
    train_prefetch_factor, val_prefetch_factor = 4, 2
    json_save_prefix = "/cluster_home/custom_data/results/det/%s/%s" % (
        job_type,
        job_name,
    )
    json_save_prefix_adas = (
        "/cluster_home/custom_data/results_adas/det/%s/%s"
        % (
            job_type,
            job_name,
        )
    )
    det_data_yaml = "projects/isp/app/det/dataset/det_4pe_%s.yaml" % sensor

######### model settings #########
get_model = get_model_j5 if base_model == "j5" else get_model_j3
model = get_model(
    det_task=det_task,
    preprocess_type=preprocess_type,
    me_type=me_type,
    me_channels=me_channels,
    split_transform=split_transform,
)
deploy_model = get_model(
    det_task=det_task,
    preprocess_type=preprocess_type,
    me_type=me_type,
    me_channels=me_channels,
    deploy=True,
    split_transform=split_transform,
)
deploy_inputs = dict(img=torch.randn((1, 3, 512, 960)))

######### datasets settings #########
dataset_info = get_datasets(
    dataset_yaml=det_data_yaml,
    det_task=det_task,
    time=time,
    data_type=data_type,
)

leaderboard_tags = [
    "resize",
    sensor,
    data_type,
    preprocess_type,
    me_type,
    base_model,
]

######### dataloader settings #########
train_transforms = get_train_transform(
    data_type=data_type,
    preprocess_type=preprocess_type,
    img_scale=img_scale,
    std=std,
    mean=mean,
    split_transform=split_transform,
    split_trans_h=split_trans_h,
    split_trans_w=split_trans_w,
)

val_transforms = get_val_transform(
    data_type=data_type,
    preprocess_type=preprocess_type,
    img_scale=img_scale,
    std=std,
    mean=mean,
    split_transform=split_transform,
    split_trans_h=split_trans_h,
    split_trans_w=split_trans_w,
)

train_dataloader = get_train_chunk_dataloader(
    lmdbs=dataset_info["train_lmdbs"],
    nums=dataset_info["train_nums"],
    transforms=train_transforms,
    batch_size_per_gpu=batch_size_per_gpu,
    num_workers=train_num_workers,
    prefetch_factor=train_prefetch_factor,
)

val_dataloader = get_val_dataloader(
    lmdbs=dataset_info["val_lmdbs"][:1],
    nums=dataset_info["val_nums"][:1],
    transforms=val_transforms,
    batch_size_per_gpu=batch_size_per_gpu,
    num_workers=val_num_workers,
    prefetch_factor=val_prefetch_factor,
)

eval_dataloader = get_val_dataloader(
    lmdbs=dataset_info["eval_lmdbs"],
    nums=dataset_info["eval_nums"],
    transforms=val_transforms,
    batch_size_per_gpu=batch_size_per_gpu,
    num_workers=val_num_workers,
    prefetch_factor=val_prefetch_factor,
)


############# callbacks settings #############
def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="loss_" + job_name,
)

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=5000,
    epoch_log_freq=val_interval,
    log_prefix="Validation " + job_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix="%s-" % step,
    save_interval=1,
    strict_match=True,
    mode="max",
    monitor_metric_key="mAP",
    save_hash=False,
)

val_callback = dict(
    type="Validation",
    data_loader=val_dataloader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    init_with_train_model=False,
    val_interval=val_interval,
    val_on_train_end=True,
)


tb_update_funcs = None
if enable_tensorboard and base_model == "j3":
    tb_update_funcs = [tb_update]
elif enable_tensorboard and base_model == "j5":
    tb_update_funcs = [tb_update_sd]

tensorboard_callback = dict(
    type="TensorBoard",
    save_dir=log_dir,  # noqa
    update_freq=log_freq,
    tb_update_funcs=tb_update_funcs,
)


aidi_eval_callback = dict(
    type="AIDIEval",
    output_root=json_save_prefix_adas,
    project_id="TD20230010",
    prediction_name=job_name,
    prediction_tags=leaderboard_tags,
    aidi_eval_dataset_id=dataset_info["leaderboard_id"],
    reformat_output_fn=reformat_aidi_eval_out,
    reformat_out_fn_kwargs={"task_name": "%s" % det_task},
)

############### training settings ###############

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    # model_convert_pipeline=model_convert_pipeline,
    data_loader=train_dataloader,
    optimizer=dict(type=torch.optim.Adam, lr=lr, weight_decay=weight_decay),
    batch_processor=batch_processor,
    num_epochs=epoch,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="StepDecayLrUpdater",
            warmup_begin_lr=lr,
            warmup_by="epoch",
            step_log_interval=log_freq,
            lr_decay_id=[int(8 / 12.0 * epoch), int(11 / 12.0 * epoch)],
        ),
        val_callback,
        ckpt_callback,
        tensorboard_callback,
    ],
    train_metrics=[
        dict(
            type="LossShow",
        ),
    ],
    sync_bn=True,
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file=dataset_info["val_coco_jsons"][0],
        save_prefix=json_save_prefix,
    ),
)

float_predictor = dict(
    type="Predictor",
    model=model,
    data_loader=eval_dataloader,
    batch_processor=val_batch_processor,
    num_epochs=1,
    device=None,
    callbacks=[
        stat_callback,
        aidi_eval_callback,
    ],
    share_callbacks=False,
)

############### evaluation settings ###############

ckpt_path = os.path.join(
    "/cluster_home/custom_data/models/det/%s/%s"
    % (
        job_type,
        job_name,
    ),
    "float-checkpoint-best.pth.tar",
)
k8s_config["job_list"].append(
    "python3 tools/predict.py --stage %s --config projects/isp/app/det/%s.py --ckpt %s"
    % (step, python_file_name, ckpt_path),
)
