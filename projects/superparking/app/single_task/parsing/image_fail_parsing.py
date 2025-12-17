import copy
import json
import os

import torch
from hatbc.filestream.bucket.client import BucketClient
from horizon_plugin_pytorch.quantization import March
from torchvision.transforms import InterpolationMode

from hat.metrics.mean_iou import MeanIOU
from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from projects.superparking.tools.image_fail_parsing import (
    image_fail_parsing_train_path,
    image_fail_parsing_val_path,
)

training_step = os.environ.get("HAT_TRAINING_STEP", "float")
pipeline_test = os.environ.get("HAT_PIPELINE_TEST", "0") == "1"

task_name = "image_fail_parsing"
labels = [
    dict(class_name=["normal"], color_map=[64, 192, 0]),
    dict(class_name=["light_blur"], color_map=[255, 255, 0]),
    dict(class_name=["heavy_blur"], color_map=[250, 170, 30]),
    dict(class_name=["light_glare"], color_map=[128, 64, 192]),
    dict(class_name=["heavy_glare"], color_map=[160, 0, 0]),
    dict(class_name=["light_blockage"], color_map=[0, 0, 255]),
    dict(class_name=["heavy_blockage"], color_map=[255, 255, 255]),
    dict(class_name=["ignore"], color_map=[70, 70, 70]),
]
# train config
binary_loss = True
mcp_pretrain = False
train_by = "epoch"
train_epochs = 30  # 30
warmup_len = 1
qat_train_epochs = 15  # 15
train_steps = None
qat_train_steps = None
if pipeline_test:
    train_by = "step"
    train_steps = 5
    qat_train_steps = 5
    warmup_len = 2

float_lr = 0.002
weight_decay = 4.0e-5
momentum = 0.9
qat_lr = 2e-4
sync_bn = True
bn_kwargs = dict(eps=2.0e-5, momentum=0.1)
batch_size_per_gpu = 12
val_batch_size_per_gpu = 20
dataloader_workers = 4  # per gpu
num_gpus_per_machine = 8
march = March.BERNOULLI2
cudnn_benchmark = True
seed = 6787
log_rank_zero_only = True
enable_amp = False

local_train = not os.path.exists("/running_package")
if local_train:
    device_ids = [0]
    log_freq = 50
    ckpt_dir = "tmp_output/image_fail_parsing"
    # ckpt_dir = args.qat_best.split("/")[-3]
    print(ckpt_dir)
else:
    device_ids = list(range(num_gpus_per_machine))
    log_freq = 50
    ckpt_dir = "/job_data"

pretrain_checkpoint = (
    "dmpv2://SuperParking/model/pretrained_backbone/"
    + "vargnetv2_05_headfactor2.pth.tar"
)
bkt_clt = BucketClient()
pretrain_checkpoint = bkt_clt.url_to_local(pretrain_checkpoint)

tensorboard_log_path = os.path.join(ckpt_dir, "tensorboard")
if local_train:
    tb_save_dir = os.path.join(tensorboard_log_path, training_step)
else:
    tb_save_dir = os.path.join(
        os.getenv("TENSORBOARD_LOG_PATH")
        or os.path.join(tensorboard_log_path, ".aidi"),
        training_step,
    )

head_factor = 2
# model config
num_classes = 7
data_shape = (3, 576 // 2, 704 // 2)
resize_hw = [576 // 2, 704 // 2]
class_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
feat_channels = 32
alpha = 0.5
bifpn_out_strides = [2, 4, 8, 16, 32]
ignore_index = 255
# head config
use_auxi_loss = True
num_auxi_layer = 2
share_conv = False
head_out_strides = [2, 4, 8]
assert num_auxi_layer + 1 == len(head_out_strides)
stacked_convs = 6
start_level = 0
end_level = 2
assert (end_level - start_level) == num_auxi_layer
weight_min = 0.15
binary_weight = 5
weight_min_binary = 0.15
weight_min_level = 0.15
weight_noobj = 0.75
auto_class_weight = False

main_loss = dict(
    type="CEWithWeightMap",  # CrossEntropyLossV2
    use_sigmoid=False,
    loss_weight=1,
    reduction="mean",
    loss_name="ce_loss",
    ignore_index=255,
    class_weight=class_weight,
    num_class=num_classes,
    auto_class_weight=auto_class_weight,
    weight_min=weight_min,
    weight_noobj=weight_noobj,
)

main_loss_2 = dict(
    type="CEWithWeightMap",
    use_sigmoid=False,
    loss_weight=1,
    reduction="mean",
    loss_name="ce_loss_binary_Remap",
    ignore_index=255,
    class_weight=None,
    auto_class_weight=auto_class_weight,
    weight_min=weight_min_binary,
    weight_noobj=weight_noobj,
    num_class=2,
    remap_params={
        "raw_cls_num": 7,
        "old2newmap": {
            0: 0,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
        },
    },
)
losses = [
    dict(
        type="MixSegLoss",
        losses=[main_loss, main_loss_2],
        losses_weight=[1.0, binary_weight],
    )
]
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
        losses_weight.append(0.5)

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
        start_level=0,
        end_level=-1,
        num_outs=5,
    ),
    head=dict(
        type="MaskcatFeatHead",
        has_project_layer=True,
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
        conv_method="varg_conv",
        share_conv=share_conv,
        use_auxi_loss=use_auxi_loss,
        aggregation_method="concat",
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
    output_name="pred",
)
val_model["head"]["use_auxi_loss"] = False
# test model structure
deploy_model = copy.deepcopy(model)
deploy_model["loss"] = None
deploy_model["head"]["use_auxi_loss"] = False
deploy_model["head"]["upsample_output_scale"] = False

add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[
        json.dumps(
            dict(
                task=task_name,
                num_classes=num_classes,
                labels=labels,
                size=[resize_hw[1], resize_hw[0]],
            )
        )
    ],
)

if training_step == "int_infer":
    deploy_model["postprocess"] = add_desc_pp
    deploy_model["head"]["argmax_output"] = True

deploy_inputs = {"img": [torch.randn((1,) + data_shape)]}

# data augmentation
transforms = [
    dict(
        type="Resize",
        keep_ratio=True,
        img_scale=resize_hw,
        ratio_range=(0.75, 1.25),
    ),
    dict(
        type="SegRandomCrop",
        size=resize_hw,
        cat_max_ratio=0.5,
        ignore_index=ignore_index,
    ),
    dict(type="Pad", size=resize_hw),
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
    dict(type="RandomFlip", px=0.5, py=0.0),
    dict(type="ToTensor", to_yuv=False),
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

train_dataset_cfg_list = []
train_data_paths = []
train_data_paths = image_fail_parsing_train_path


for rec in train_data_paths:
    train_dataset_cfg = dict(
        type="DenseboxDataset",
        data_path=rec + ".rec",
        anno_path=rec + ".json",
        task_type="segmentation",
        rec_idx_file_path=rec + ".rec.idx",
        transforms=transforms,
        to_rgb=True,
    )
    train_dataset_cfg_list.append(train_dataset_cfg)

train_dataset = dict(
    type="ConcatDataset",
    datasets=train_dataset_cfg_list,
)
train_dataset = dict(
    type="RepeatDataset",
    dataset=train_dataset,
    times=2,
)

val_only_transforms = [
    dict(type="YUVTurboJPEGDecoder", to_string=True),
    dict(type="BPUPyramidResizer", scale_wh=[0.25, 0.25], pyramid_type="ips"),
    dict(type="ImgBufToYUV444"),
    dict(type="ToTensor", to_yuv=False),
    dict(type="Normalize", mean=128.0, std=128.0),
    dict(type="RenameKeys", keys=["imgs|img"]),
]

val_transforms = val_only_transforms
val_data_paths = image_fail_parsing_val_path
val_index = list(range(len(val_data_paths)))
val_dataset_cfg_list = []
for i in range(len(val_index)):
    val_dataset = dict(
        type="DenseboxDataset",
        data_path=val_data_paths[val_index[i]] + ".rec",
        anno_path=val_data_paths[val_index[i]] + ".json",
        task_type="segmentation",
        rec_idx_file_path=val_data_paths[val_index[i]] + ".rec.idx",
        transforms=val_transforms,
        with_img_buf=True,
        to_rgb=True,  # no influence
    )
    val_dataset_cfg_list.append(val_dataset)
val_dataset = dict(
    type="ConcatDataset",
    datasets=val_dataset_cfg_list,
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

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=val_batch_size_per_gpu,
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

val_miou_metric = MeanIOU(
    seg_class=[str(i) for i in range(num_classes)],
    ignore_index=ignore_index,
)
val_miou_metric_cls2 = MeanIOU(
    seg_class=[str(i) for i in range(2)], ignore_index=ignore_index
)
val_miou_metric_cls3 = MeanIOU(
    seg_class=[str(i) for i in range(3)], ignore_index=ignore_index
)


def update_val_metric(metrics, batch, model_outs):
    label = batch["gt_seg"]
    if label.dim() == 4:
        label = label.squeeze(dim=-1)
    preds_label = model_outs["pred"]
    for i, metric in enumerate(metrics):
        label_i = label.clone()
        preds_label_i = preds_label.clone()
        if i == 1:
            label_i[(label != 255) & (label != 0)] = 1
            preds_label_i[(preds_label != 255) & (preds_label != 0)] = 1
        if i == 2:
            old2newmap = {
                0: 0,
                1: 1,
                2: 2,
                3: 1,
                4: 2,
                5: 1,
                6: 2,
            }
            for k, v in old2newmap.items():
                label_i[label == k] = v
                preds_label_i[preds_label == k] = v
        metric.update(label_i, preds_label_i)


metrics = [val_miou_metric, val_miou_metric_cls2, val_miou_metric_cls3]
metrics_deploy = [val_miou_metric, val_miou_metric_cls2, val_miou_metric_cls3]

val_metric_updater = dict(
    type="MetricUpdater",
    metrics=metrics,
    metric_update_func=update_val_metric,
    step_log_freq=100,
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

deploy_val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater, tb_val_callback],
    val_model=None,
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
    num_steps=train_steps,
    num_epochs=train_epochs,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",  # CosLrUpdater
            step_log_interval=log_freq,
            warmup_mode="linear",
            warmup_by=train_by,
            warmup_len=warmup_len,
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
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=weight_decay)},
        lr=qat_lr,
        momentum=momentum,
    ),
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
    batch_processor=batch_processor,
    device=None,
    stop_by=train_by,
    num_steps=qat_train_steps,
    num_epochs=qat_train_epochs,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[25],
            lr_decay_factor=0.1,
        ),
        metric_updater,
        tb_loss_callback,
        qat_val_callback,
        ckpt_callback,
    ],
    sync_bn=sync_bn,
)


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
    callbacks=[
        trace_callback,
    ],
)
