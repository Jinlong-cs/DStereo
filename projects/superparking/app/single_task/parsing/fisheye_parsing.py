import copy
import json
import os

import torch
from horizon_plugin_pytorch.quantization import March
from torchvision.transforms import InterpolationMode

from hat.metrics.mean_iou import MeanIOU
from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from projects.superparking.tools.parsing.dataset_version import (
    get_dataset_list,
)

training_step = os.environ.get("HAT_TRAINING_STEP", "float")
pipeline_test = os.environ.get("HAT_PIPELINE_TEST", "0") == "1"

task_name = "fisheye_parsing"

# baseline: http://fm-yunfeng-zhang.ucloudtrain.hogpu.cc/plat_gpu/fisheye_parsing_0.8.0_with0.7pretrain-20220607-121502/log/hobot-job-37291-task-0.log # noqa
# task desc
labels = [
    dict(class_name=["road"], color_map=[128, 64, 128]),
    dict(class_name=["sidewalk"], color_map=[244, 35, 232]),
    dict(class_name=["sky"], color_map=[70, 130, 180]),
    dict(class_name=["terrain"], color_map=[107, 142, 35]),
    dict(class_name=["curb"], color_map=[34, 237, 242]),
    dict(class_name=["fence"], color_map=[190, 153, 153]),
    dict(class_name=["vegetation"], color_map=[152, 251, 152]),
    dict(class_name=["pole"], color_map=[153, 153, 153]),
    dict(class_name=["traffic_sign"], color_map=[220, 220, 0]),
    dict(class_name=["car"], color_map=[0, 0, 142]),
    dict(class_name=["tricycle"], color_map=[0, 0, 255]),
    dict(class_name=["bicycle"], color_map=[119, 11, 32]),
    dict(class_name=["person"], color_map=[220, 20, 60]),
    dict(class_name=["rider"], color_map=[237, 162, 13]),
    dict(class_name=["trolley"], color_map=[150, 50, 150]),
    dict(class_name=["traffic_cone"], color_map=[111, 74, 0]),
    dict(class_name=["bollard"], color_map=[70, 70, 70]),
    dict(class_name=["folding_warning_sign"], color_map=[220, 220, 220]),
    dict(class_name=["single_water_barrier"], color_map=[0, 80, 90]),
    dict(class_name=["untraversable"], color_map=[150, 120, 120]),
    dict(class_name=["parking_rod"], color_map=[100, 5, 180]),
    dict(class_name=["parking_lock"], color_map=[237, 159, 241]),
    dict(class_name=["column"], color_map=[0, 128, 128]),
    dict(class_name=["background"], color_map=[0, 0, 0]),
]

# train config
train_epochs = 40  # 40
qat_train_epochs = 12
float_lr = 0.1
weight_decay = 4.0e-5
momentum = 0.9
qat_lr = 1e-3
sync_bn = True
bn_kwargs = dict(eps=2.0e-5, momentum=0.1)
batch_size_per_gpu = 8
dataloader_workers = 4  # per gpu
num_gpus_per_machine = 8
march = March.BERNOULLI2
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
enable_amp = False
with_last_pretrain = True
warmup_len = 0 if with_last_pretrain else 1
train_by = "epoch"
train_steps = None
qat_train_steps = None
if pipeline_test:
    train_by = "step"
    warmup_len = 2
    train_steps = 5
    qat_train_steps = 5
# repeat dataset config
repeat_times = 3

local_train = not os.path.exists("/running_package")
if local_train:
    device_ids = [2]
    log_freq = 50
    bucket_root = "/horizon-bucket"
    ckpt_dir = "tmp_output/fish_parsing"
else:
    device_ids = list(range(num_gpus_per_machine))
    log_freq = 50
    bucket_root = "/bucket/input"
    ckpt_dir = "/job_data"

if with_last_pretrain:
    pretrain_checkpoint = os.path.join(
        bucket_root,
        "SuperParking/yunfeng.zhang/model_tmp/fisheye_parsing/v0.8.0/"
        "float-checkpoint-best-ce13776d.pth.tar",
    )
else:
    pretrain_checkpoint = os.path.join(
        bucket_root,
        "SuperParking/model/pretrained_backbone/"
        "vargnetv2_05_headfactor2.pth.tar",
    )

tensorboard_log_path = os.path.join(ckpt_dir, "tensorboard")
if local_train:
    tb_save_dir = os.path.join(tensorboard_log_path, training_step)
else:
    tb_save_dir = os.path.join(
        os.getenv("TENSORBOARD_LOG_PATH")
        or os.path.join(tensorboard_log_path, ".aidi"),
        training_step,
    )

# model config
num_classes = 24
width = 704
height = 576
data_shape = (3, height, width)
class_weight = None
feat_channels = 32
alpha = 0.5
head_factor = 2
bifpn_out_strides = [4, 8, 16, 32, 64]
ignore_index = 255

# head config
use_auxi_loss = False
num_auxi_layer = 3
start_level = 0
end_level = 3
assert (end_level - start_level) == num_auxi_layer
head_out_strides = [2, 8, 16, 32] if use_auxi_loss else [2]
stacked_convs = 3
has_project_layer = True
conv_method = "varg_conv"
share_conv = False
aggregation_method = "sum"

# loss config
auto_class_weight = False
weight_min = 0.0
weight_noobj = 0.75

losses = [
    dict(
        type="MixSegLoss",
        losses=[
            dict(
                type="CEWithWeightMap",  # ce_loss
                use_sigmoid=False,
                loss_weight=1,
                reduction="mean",
                loss_name="ce_loss2",
                ignore_index=ignore_index,
                class_weight=class_weight,
                num_class=num_classes,
                auto_class_weight=auto_class_weight,
                weight_min=weight_min,
                weight_noobj=weight_noobj,
            ),
            dict(
                type="LovaszSoftmaxLoss",  # lovasz_loss
                per_image=False,
                ignore_index=255,
                loss_name="lovasz_loss2",
            ),
        ],
        losses_weight=[1.0, 1.0],
    )
]

loss_auxi_weight = [0.5, 0.25, 0.125, 0.125]
losses_weight = [1.0]
if use_auxi_loss:
    for i in range(num_auxi_layer):
        losses.append(
            dict(
                type="CEWithWeightMap",
                use_sigmoid=False,
                class_weight=class_weight,
                loss_weight=1.0,
                ignore_index=ignore_index,
                loss_name="ce_loss" + str(pow(2, 3 + i)),
                num_class=num_classes,
                auto_class_weight=auto_class_weight,
                weight_min=weight_min,
                weight_noobj=weight_noobj,
            )
        )
        losses_weight.append(loss_auxi_weight[i])  # need improve

# train model structure
model = dict(
    type="BMSegmentor",
    backbone=dict(
        type="VargNetV2",
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        alpha=alpha,
        group_base=8,
        factor=2,
        bias=True,
        include_top=False,
        head_factor=head_factor,
    ),
    neck=dict(
        type="BiFPN",
        fpn_name="bifpn_sum",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=bifpn_out_strides,
        stride2channels=get_vargnetv2_stride2channels(alpha),
        out_channels=feat_channels,
        stack=3,
        start_level=1,
        end_level=-1,
        num_outs=5,
    ),
    head=dict(
        type="MaskcatFeatHead",
        has_project_layer=has_project_layer,
        num_classes=num_classes,
        in_strides=bifpn_out_strides,
        out_strides=head_out_strides,
        in_channels=feat_channels,
        out_channels=feat_channels,  # need to be improved
        stacked_convs=stacked_convs,
        start_level=start_level,
        end_level=end_level,
        bn_kwargs=bn_kwargs,
        group_base=8,
        conv_method=conv_method,
        share_conv=share_conv,
        use_auxi_loss=use_auxi_loss,
        auxi_use_bifpn=False,
        aggregation_method=aggregation_method,
        argmax_output=False,
        dequant_output=True,
        int8_output=True,
        upsample_output_scale=True,
    ),
    loss=dict(
        type="MixSegLossMultipreds",
        losses=losses,
        losses_weight=losses_weight,
        loss_name="semseg_loss",
    ),
)

# val model structure
val_model = copy.deepcopy(model)
val_model["loss"] = None
val_model["postprocess"] = dict(
    type="SemSegDecoder",
    output_name="semseg_pred",
)
# test model structure
add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[
        # num of desc == num of pred output tensors (one tensor per out stride)
        json.dumps(
            dict(
                task=task_name,
                num_classes=num_classes,
                labels=labels,
                size=[width // 2, height // 2],
            )
        )
    ],
)
deploy_model = copy.deepcopy(model)
deploy_model["loss"] = None
deploy_model["head"]["use_auxi_loss"] = False
deploy_model["head"]["upsample_output_scale"] = False
if training_step == "int_infer":
    deploy_model["postprocess"] = add_desc_pp
    deploy_model["head"]["argmax_output"] = True

deploy_inputs = {"img": [torch.randn((1,) + data_shape)]}

# data augmentation
to_rgb = True
train_transforms = [
    dict(
        type="Resize",
        keep_ratio=True,
        img_scale=(height * 2, width * 2),
        ratio_range=(0.75, 1.25),
    ),
    dict(
        type="SegRandomCrop",
        size=(height, width),
        cat_max_ratio=0.5,
        ignore_index=ignore_index,
    ),
    dict(type="Pad", size=(height, width)),
    dict(
        type="SegRandomCutOut",
        prob=0.5,
        n_holes=(2, 6),
        cutout_ratio=[
            (0.05, 0.05),
            (0.02, 0.02),
            (0.07, 0.07),
            (0.1, 0.1),
            (0.2, 0.2),
        ],
    ),
    dict(type="ToTensor", to_yuv=False),
    dict(
        type="ColorJitter",
        brightness=0.2,
        contrast=(0.5, 1.5),
        saturation=(0.5, 1.5),
        hue=0.1,
    ),
    dict(type="RandomFlip", px=0.5, py=0.0),
    dict(
        type="SegRandomAffine",
        degrees=15,
        label_fill_value=ignore_index,
        interpolation=InterpolationMode.BILINEAR,
        rotate_p=0.5,
        translate_p=0,
        scale_p=0,
    ),
    dict(type="ToTensor", to_yuv=True),
    dict(type="ImageNormalize", mean=128.0, std=128.0),
    dict(type="CopyKeys", keys=["gt_seg|labels"]),
]
val_transforms = [
    dict(type="Resize", keep_ratio=False, img_scale=(height, width)),
    dict(type="ToTensor", to_yuv=True),
    dict(type="ImageNormalize", mean=128.0, std=128.0),
]
transforms = {"train": train_transforms, "val": val_transforms}
# dataset config

data_paths = get_dataset_list(
    bucket_root=bucket_root,
    attribution="superparking",
    model_version="v0.9.0",
    version_yaml="parsing_dataset_version.yaml",
    dataset_yaml="parsing_all_datasets.yaml",
)
train_rec_paths, train_anno_paths = data_paths[:2]
val_rec_paths, val_anno_paths = data_paths[-2:]


def get_dataset(mode):
    rec_paths = eval(mode + "_rec_paths")
    anno_paths = eval(mode + "_anno_paths")

    concat_dataset = dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="DenseboxDataset",
                data_path=rec_path_i,
                anno_path=anno_path_i,
                task_type="segmentation",
                to_rgb=to_rgb,
                transforms=transforms[mode],
            )
            for rec_path_i, anno_path_i in zip(rec_paths, anno_paths)
        ],
    )
    return concat_dataset


train_dataset = dict(
    type="RepeatDataset",
    dataset=get_dataset("train"),
    times=repeat_times,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=True,
)

val_dataset = get_dataset("val")

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
)


# batch processor config
def loss_collector(model_outs):
    losses = model_outs["semseg_loss"]
    losses_val = []
    for _, value in losses.items():
        losses_val.append(value)
    return losses_val


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    enable_amp=enable_amp,
)

val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)


# callback config in train stage
def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        total_loss = 0.0
        for _, val in model_outs["semseg_loss"].items():
            total_loss += val
        model_outs["semseg_loss"]["all_loss"] = total_loss
        metric.update(model_outs["semseg_loss"])


metric_updater = dict(
    type="MetricUpdater",
    metrics=[dict(type="LossShow", name="loss_semseg")],
    metric_update_func=update_loss,
    step_log_freq=log_freq,
    epoch_log_freq=1,
)

tb_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tb_save_dir, "loss"),
    loss_name_reg="^.*_loss.*",
    update_freq=log_freq,
    update_by="step",
)

# callback config in validation stage
val_miou_metric = MeanIOU(
    seg_class=[labels[i]["class_name"][0] for i in range(num_classes)],
    ignore_index=ignore_index,
)


def update_val_metric(metrics, batch, model_outs):
    label = batch["gt_seg"]
    if label.dim() == 4:
        label = label.squeeze(dim=-1)
    preds_label = model_outs["semseg_pred"]

    for metric in metrics:
        metric.update(label, preds_label)


val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[val_miou_metric],
    metric_update_func=update_val_metric,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)


def tb_val_func(writer, epoch_id, **kwargs):
    name, value = val_miou_metric.get()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = [value.item()]
        else:
            value = value.cpu().numpy().tolist()
    elif not isinstance(value, list):
        value = [value]
    else:
        if isinstance(value[0], torch.Tensor):
            for i in range(len(value)):
                value[i] = value[i].item()
    if not isinstance(name, list):
        name = [name]
    for k, v in zip(name, value):
        writer.add_scalar(k, v, global_step=epoch_id)


tb_val_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tb_save_dir, "val"),
    update_freq=1,
    update_by="epoch",
    tb_update_funcs=[tb_val_func],
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater, tb_val_callback],
    val_model=val_model,
)

qat_val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater, tb_val_callback],
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
        ],
    ),
    val_model=val_model,
)

# common callback
stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=True,
    mode="max",
    best_refer_metric=val_miou_metric,
)

# fp32 trainer config
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=pretrain_checkpoint,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=weight_decay)},
        lr=float_lr,  # 0.1
        momentum=momentum,
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by=train_by,
    num_epochs=train_epochs,
    num_steps=train_steps,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",  # CosLrUpdater
            step_log_interval=log_freq,
            warmup_mode="linear",
            warmup_by=train_by,
            warmup_len=warmup_len,  # 0
        ),
        metric_updater,
        tb_loss_callback,
        val_callback,
        ckpt_callback,
    ],
    sync_bn=sync_bn,
)


# quantized trainer config
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
                allow_miss=True,
                ignore_extra=True,
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=weight_decay)},
        lr=qat_lr,
        momentum=momentum,
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by=train_by,
    num_steps=train_steps,
    num_epochs=qat_train_epochs,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",  # CosLrUpdater
            step_log_interval=log_freq,
            warmup_mode="linear",
            warmup_by="epoch",
            warmup_len=0,
        ),
        metric_updater,
        tb_loss_callback,
        qat_val_callback,
        ckpt_callback,
    ],
    sync_bn=sync_bn,
)

# int_infer config
trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
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
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
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
    callbacks=[trace_callback],
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name + "_model",
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
    opt=3,
)
