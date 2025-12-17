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

import fsspec
import torch
import yaml
from horizon_plugin_pytorch.quantization import March

from hat.data.collates.collates import collate_2d
from hat.metrics.acc import AccuracySeg
from hat.metrics.loss_show import LossShow
from hat.metrics.mean_iou import MeanIOU
from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.utils import Config

cfg_dir = osp.dirname(osp.dirname(__file__))


cudnn_benchmark = True
march = March.BAYES
seed = None
log_rank_zero_only = True

step = "float"  # float | qat | int_infer | predict

if step not in ["float", "qat", "int_infer", "predict"]:
    raise NotImplementedError("step type %s is not supported." % step)

job_name = (
    osp.basename(__file__).replace(".py", "") + "_%s" % step
)  # your job name
job_password = "nopassword123"  # enter your password

if step == "predict":
    job_list = [
        "python3 tools/predict.py --config projects/isp/app/seg/%s"
        % (osp.basename(__file__)),
    ]
else:
    job_list = [
        "python3 tools/train.py --step %s --config projects/isp/app/seg/%s"
        % (step, osp.basename(__file__)),  # your running command
    ]

local_train = not os.path.exists("/running_package")
seg_data_yaml = "projects/isp/app/seg/dataset/isp_seg_mono.yaml"

if local_train:
    device_ids = [
        3,
    ]
    num_machines = 1
    num_gpus_per_machine = len(device_ids)
    num_gpus = num_machines * num_gpus_per_machine
    log_freq = 4
    batch_size_per_gpu = 5
    batch_size = batch_size_per_gpu * num_gpus
    val_interval = 1
    ckpt_dir = "checkpoints/seg/%s" % (
        osp.basename(__file__).replace(".py", "")
    )  # replace with your own ckpt dir.
    train_num_workers, val_num_workers = 0, 0
    json_save_prefix = (
        "results/seg/%s" % job_name
    )  # replace with your own log dir.
    json_save_prefix_adas = "results_adas/seg/%s" % job_name

# k8s setiings, for aidi platform
k8s_config_file = osp.join(cfg_dir, "k8s_config.py")
base_k8s_config = Config.fromfile(k8s_config_file)
k8s_config = dict()
k8s_config.update(base_k8s_config._cfg_dict)
k8s_config["job_name"] = job_name
k8s_config["job_password"] = job_password
k8s_config["num_machines"] = 1
k8s_config["num_gpus_per_machine"] = 4
k8s_config["job_list"] = job_list

if not local_train:
    if step == "predict":
        device_ids = [
            0,
        ]
        k8s_config["num_gpus_per_machine"] = 1
    else:
        device_ids = [i for i in range(k8s_config["num_gpus_per_machine"])]
    num_machines = k8s_config["num_machines"]
    num_gpus_per_machine = k8s_config["num_gpus_per_machine"]
    val_interval = 10
    log_freq = 10
    batch_size_per_gpu = 16
    num_gpus = num_machines * num_gpus_per_machine
    batch_size = batch_size_per_gpu * num_gpus
    ckpt_dir = "/cluster_home/custom_data/models/seg/%s" % (
        osp.basename(__file__).replace(".py", "")
    )
    train_num_workers, val_num_workers = 12, 8
    json_save_prefix = "/cluster_home/custom_data/results/seg/%s" % job_name
    json_save_prefix_adas = (
        "/cluster_home/custom_data/results_adas/seg/%s" % job_name
    )

# task settings
data_shape = (3, 512, 960)
weight_decay = 4e-5

if step == "qat":
    lr = 1e-3
    epoch = 50
else:
    lr = 2e-3
    epoch = 200


seg_class = [
    "background",
    "lane",
    "curb",
    "double_line",
    "wide_line",
]
class_weight = [1.0, 1.0, 1.0, 1.0, 1.0]
ignore_index = 255

# model
bn_kwargs = dict(eps=1e-5, momentum=0.1)
alpha = 0.5
feat_channels = 64
num_classes = 5
img_scale = data_shape[1:]
size_divisor = 64

model = dict(
    type="Segmentor",
    backbone=dict(
        type="VargNetV2ISP",
        num_classes=-1,
        input_channels=3,
        alpha=alpha,
        bn_kwargs=bn_kwargs,
        group_base=8,
        include_top=False,
        model_type="VargNetV2",
        factor=2,
        bias=True,
        extend_features=False,
        disable_quanti_input=False,
        flat_output=False,
        input_sequence_length=1,
        head_factor=1,
    ),
    neck=dict(
        type="BiFPN",
        fpn_name="bifpn_sum",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[4, 8, 16, 32, 64],
        stride2channels=get_vargnetv2_stride2channels(alpha),
        out_channels=feat_channels,
        stack=1,
        start_level=1,
        end_level=-1,
        num_outs=5,
    ),
    head=dict(
        type="SegHead",
        num_classes=num_classes,
        in_strides=(4,),
        out_strides=(4,),
        stride2channels={4: feat_channels},
        feat_channels=feat_channels,
        stacked_convs=6,
        int8_output=True,
        dequant_output=True,
        upsample_output_scale=4,
    ),
    losses=dict(
        type="CrossEntropyLoss",
        loss_name="loss_crossEntropy",
        use_sigmoid=False,
        ignore_index=ignore_index,
        class_weight=class_weight,
        loss_weight=5.0,
    ),
)


deploy_model = dict(
    type="Segmentor",
    backbone=dict(
        type="VargNetV2ISP",
        num_classes=-1,
        input_channels=3,
        alpha=alpha,
        bn_kwargs=bn_kwargs,
        group_base=8,
        include_top=False,
        model_type="VargNetV2",
        factor=2,
        bias=True,
        extend_features=False,
        disable_quanti_input=False,
        flat_output=False,
        input_sequence_length=1,
        head_factor=1,
    ),
    neck=dict(
        type="BiFPN",
        fpn_name="bifpn_sum",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[4, 8, 16, 32, 64],
        stride2channels=get_vargnetv2_stride2channels(alpha),
        out_channels=feat_channels,
        stack=1,
        start_level=1,
        end_level=-1,
        num_outs=5,
    ),
    head=dict(
        type="SegHead",
        num_classes=num_classes,
        in_strides=(4,),
        out_strides=(4,),
        stride2channels={4: feat_channels},
        feat_channels=feat_channels,
        stacked_convs=6,
        int8_output=True,
        dequant_output=True,
        upsample_output_scale=4,
    ),
)


def load_isp_data(filepath):
    train_lmdb_path = []
    val_lmdb_path = []

    with fsspec.open(filepath, "r") as fid:
        content = fid.read()
    data = yaml.safe_load(content)
    if local_train:
        data = data["person"]["local"]
    else:
        data = data["person"]["aidi"]
    for task, infos in data.items():
        for info in infos:
            train_lmdb_path.append(
                osp.join("/horizon-bucket", info["train_lmdb"])
            )
            val_lmdb_path.append(osp.join("/horizon-bucket", info["val_lmdb"]))
            train_num_samples = info["train_num_samples"]
            val_num_samples = info["val_num_samples"]

    return train_lmdb_path, val_lmdb_path, train_num_samples, val_num_samples


(
    isp_train_lmdb,
    isp_val_lmdb,
    train_num_samples,
    val_num_samples,
) = load_isp_data(seg_data_yaml)


train_dataset = [
    dict(
        type="Auto2dFromLMDB",
        data_path=train_data_path_i,
        num_samples=train_num_samples,
        to_rgb=True,
        transforms=[
            dict(
                type="Resize",
                img_scale=img_scale,
                keep_ratio=True,
                ratio_range=(0.8, 1.3),
            ),
            dict(
                type="SegRandomCrop",
                size=img_scale,
                cat_max_ratio=0.8,
                ignore_index=ignore_index,
            ),
            dict(
                type="RandomFlip",
                px=0.5,
                py=0,
            ),
            dict(type="Normalize", mean=0.0, std=255.0),
            dict(
                type="Pad",
                size=img_scale,
            ),
            dict(type="ToTensor", to_yuv=False),
        ],
    )
    for train_data_path_i in isp_train_lmdb
]

train_dataset = dict(
    type="ConcatDataset",
    datasets=train_dataset,
)

train_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    collate_fn=collate_2d,
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=train_num_workers,
    pin_memory=True,
)

val_dataset = [
    dict(
        type="Auto2dFromLMDB",
        data_path=val_data_path_i,
        num_samples=val_num_samples,
        to_rgb=True,
        transforms=[
            dict(type="Resize", img_scale=img_scale, keep_ratio=True),
            dict(
                type="RandomFlip",
                px=0.5,
                py=0,
            ),
            dict(type="Normalize", mean=0.0, std=255.0),
            dict(
                type="Pad",
                size=img_scale,
            ),
            dict(type="ToTensor", to_yuv=False),
        ],
    )
    for val_data_path_i in isp_val_lmdb
]

val_dataset = dict(
    type="ConcatDataset",
    datasets=val_dataset,
)

val_dataloader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    batch_size=batch_size_per_gpu,
    collate_fn=collate_2d,
    shuffle=False,
    num_workers=val_num_workers,
    pin_memory=True,
    drop_last=False,
)

# val_dataloader = dict(
#         type=torch.utils.data.DataLoader,
#         dataset=dict(
#             type="Auto2dFromLMDB",
#             data_path=val_data_path,
#             num_samples=val_num_samples,
#             to_rgb=True,
#             transforms=[
#                 dict(type="Resize", img_scale=(470, 912), keep_ratio=True),
#                 dict(type="FixedCrop", size=(912, 448)),
#                 dict(type="Normalize", mean=0.0, std=255.0),
#                 dict(type="Pad", size=(448, 960),),
#                 dict(type="ToTensor", to_yuv=False),
#             ],
#         ),
#         batch_size=batch_size_per_gpu,
#         collate_fn=collate_2d,
#         shuffle=False,
#         num_workers=val_num_workers,
#         pin_memory=True,
#         drop_last=False,
# )


def loss_collector(outputs: tuple):
    losses = []
    loss_dict = outputs[1]
    for _, loss in loss_dict.items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)


def update_train_metric(metrics, batch, model_outs):
    output = model_outs[0][0]
    loss_dict = model_outs[1]
    for metric in metrics:
        if isinstance(metric, LossShow):
            metric.update(loss_dict)
        elif isinstance(metric, AccuracySeg):
            metric.update({"gt_seg": batch["gt_seg"], "pred_seg": output})
        else:
            pass


train_metric_update = dict(
    type="MetricUpdater",
    metric_update_func=update_train_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="train_metric_" + job_name,
)

val_miou_metric = MeanIOU(seg_class=seg_class, ignore_index=ignore_index)


def update_val_metric(metrics, batch, model_outs):
    # Convert one hot to index
    preds = model_outs[0][0]
    preds = torch.argmax(preds, dim=1, keepdim=False)
    target = batch["gt_seg"]
    for metric in metrics:
        metric.update(target, preds)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_val_metric,
    step_log_freq=5000,
    epoch_log_freq=val_interval,
    log_prefix="Validation " + job_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

deploy_inputs = dict(img=torch.randn((1, 3, 512, 960)))
# ckpt_callback = dict(
#     type="Checkpoint",
#     save_dir=ckpt_dir,
#     name_prefix="%s-" % step,
#     save_interval=1,
#     deploy_model=deploy_model,
#     deploy_inputs=deploy_inputs,
#     strict_match=True,
#     mode="max",
#     best_refer_metric=val_miou_metric,
# )

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix="%s-" % step,
    save_interval=1,
    strict_match=True,
    mode="max",
    best_refer_metric=val_miou_metric,
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

# adas_eval_callback = dict(
#     type="AdasEval",
#     dataloaders=val_dataloader,
#     save_dir=json_save_prefix_adas,
#     task_type_list=['detection'],
#     task_name_list=['person'],
# )

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=train_dataloader,
    optimizer=dict(type=torch.optim.Adam, lr=lr, weight_decay=weight_decay),
    batch_processor=batch_processor,
    num_epochs=epoch,
    device=None,
    callbacks=[
        stat_callback,
        train_metric_update,
        dict(
            type="CosLrUpdater",
            warmup_begin_lr=lr,
            warmup_by="epoch",
            step_log_interval=log_freq,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(type="AccuracySeg"),
    ],
    sync_bn=True,
    val_metrics=[val_miou_metric],
)

float_solver = dict(
    trainer=float_trainer,
    allow_miss=True,
    ignore_extra=True,
    resume_optimizer=None,
    resume_epoch_or_step=False,
    quantize=False,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=train_dataloader,
    optimizer=dict(type=torch.optim.Adam, lr=lr, weight_decay=weight_decay),
    batch_processor=batch_processor,
    num_epochs=epoch,
    device=None,
    callbacks=[
        stat_callback,
        train_metric_update,
        dict(
            type="CosLrUpdater",
            warmup_begin_lr=lr,
            warmup_by="epoch",
            step_log_interval=log_freq,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(type="AccuracySeg"),
    ],
    val_metrics=[val_miou_metric],
)

qat_solver = dict(
    trainer=qat_trainer,
    quantize=True,
    check_quantize_model=True,
    pre_step="float",
    pre_step_checkpoint=os.path.join(
        ckpt_dir, "float-checkpoint-best.pth.tar"
    ),
    resume_optimizer=None,
    resume_epoch_or_step=False,
    allow_miss=True,
    strict_match=True,
    ignore_extra=True,
)


int_trainer = dict(
    type="Trainer",
    model=model,
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[val_callback],
    val_metrics=val_miou_metric,
)

int_solver = dict(
    trainer=int_trainer,
    quantize=True,
    check_quantize_model=True,
    pre_step="qat",
    pre_step_checkpoint=os.path.join(ckpt_dir, "qat-checkpoint-best.pth.tar"),
    strict_match=True,
    resume_optimizer=False,
    resume_epoch_or_step=False,
    allow_miss=True,
    ignore_extra=True,
)

step2solver = dict(float=float_solver, qat=qat_solver, int_infer=int_solver)


adas_eval_callback = dict(
    type="AdasEval",
    dataloaders=val_dataloader,
    save_dir=json_save_prefix_adas,
    task_type_list=["segmentation"],
    task_name_list=["vehicle"],
)


predictor = dict(
    type="Predictor",
    model=model,
    data_loader=val_dataloader,
    batch_processor=val_batch_processor,
    num_epochs=1,
    device=None,
    callbacks=[
        stat_callback,
        adas_eval_callback,
    ],
    share_callbacks=False,
)

predict_solver = dict(
    predictor=predictor,
    train_step="float",
    checkpoint=osp.join(ckpt_dir, "float-checkpoint-best.pth.tar"),
    ckpt_step="float",
    ignore_extra=True,
)
