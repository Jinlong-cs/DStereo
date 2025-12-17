import copy
import json
import os
import re
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from horizon_plugin_pytorch.quantization import March
from torchvision.transforms import InterpolationMode

from hat.metrics.metric_3dv import RMSE, AbsRel, ConfRMSE
from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from projects.superparking.tools.depth.dataset_version import get_dataset_list

training_step = os.environ.get("HAT_TRAINING_STEP", "float")
pipeline_test = os.environ.get("HAT_PIPELINE_TEST", "0") == "1"

task_name = "fisheye_depth"

# train config
train_epochs = 50  # 30
qat_train_epochs = 12
float_lr = 4e-3  # 4e-3
warmup_len = 2
warmup_begin_lr = 1e-4  # 1e-3
stop_lr = 1e-5
weight_decay = 1.0e-5
momentum = 0.9
qat_lr = 4e-4  # 4e-4
sync_bn = True
bn_kwargs = dict(eps=1e-5, momentum=0.1)
batch_size_per_gpu = 16
dataloader_workers = 8  # per gpu
num_gpus_per_machine = 2
march = March.BERNOULLI2
cudnn_benchmark = True
seed = 6787
log_rank_zero_only = True
enable_amp = False

train_by = "epoch"
train_steps = 5
qat_train_steps = 5
warmup_len = 2
if pipeline_test:
    train_by = "step"

local_train = not os.path.exists("/running_package")
if local_train:
    device_ids = [0]
    log_freq = 50
    bucket_root = "/horizon-bucket"
    ckpt_dir = "tmp_output/fisheye_depth"
else:
    device_ids = list(range(num_gpus_per_machine))
    log_freq = 50
    bucket_root = "/bucket/input"
    ckpt_dir = "/job_data"

pretrain_checkpoint = os.path.join(
    bucket_root,
    "SuperParking/model/pretrained_backbone/",
    "vargnetv2_05_headfactor2.pth.tar",
)

df_metrics = defaultdict(list)
df_excel_path = os.path.join(ckpt_dir, f"output_{training_step}.xlsx")
tensorboard_log_path = os.path.join(ckpt_dir, "tensorboard")
if local_train:
    tb_save_dir = os.path.join(tensorboard_log_path, training_step)
else:
    tb_save_dir = os.path.join(
        os.getenv("TENSORBOARD_LOG_PATH")
        or os.path.join(tensorboard_log_path, ".aidi"),
        training_step,
    )

# data config

width = 352
height = 288
data_shape = (3, height, width)
ignore_index = 0
low_depth = 0.1  # 0.01
high_depth = 40.0
depth_scale = 256.0
hfov = 170
gt_scale = 20

depth_type = "Cylindrical"
assert depth_type in [
    "Cartesian",
    "Cylindrical",
], "depth type \
            currently only supports Cartesian or Cylindrical."
virtual_cam_params = np.array(
    [[110, 0, 176], [0, 110, 144], [0, 0, 1]], dtype=np.float32
)

# model config
neck_feat_channels = 32
alpha = 0.5
head_factor = 2
bifpn_in_strides = [2, 4, 8, 16, 32]
bifpn_out_strides = [2, 4, 8, 16, 32]
bifpn_start_level = 0
bifpn_end_level = -1

# head config
head_in_channels = [neck_feat_channels] * len(bifpn_out_strides)
# head_in_channels = [16, 16, 32, 64, 128]
start_level = 0
end_level = 3
head_out_channels = [neck_feat_channels] * (end_level - start_level + 1)
use_auxi_loss = False
head_out_names = {"depth": "pred_depth"}
head_out_strides = [1, 4, 8] if use_auxi_loss else [1]
head_parser_strides = {head_out_names["depth"]: head_out_strides}
pred_out_channel = 2
stacked_convs = 1
conv_method = "conv2d"
with_refine = False
last_with_relu = False

# loss cfg
with_smooth = True if training_step != "int_infer" else False
l1_loss_beta = 1e-6

with_virtual_normal = True if training_step != "int_infer" else False
vnl_sample_ratio = 0.6
vnl_weight = 5.0

# loss config
losses = [
    dict(
        type="DepthConfidenceCombinedLoss",
        depth_loss=dict(
            type="DepthL1Loss",
            max_gt_depth=high_depth / gt_scale,
            loss_weight=1.0,
            loss_name="depth_l1_loss",
            use_weight_map=True,
            beta=l1_loss_beta,
        ),
        conf_loss=dict(
            type="ConfL1Loss",
            max_gt_depth=high_depth / gt_scale,
            loss_weight=1.0,
            loss_name="conf_l1_loss",
            use_weight_map=False,
        ),
        depth_loss_name="depth_loss_s1",
        conf_loss_name="conf_loss_s1",
        loss_weight=[20.0, 2.0],
    )
]

loss_auxi_weight = [0.5, 0.25]
losses_weight = [1.0]
if use_auxi_loss:
    for i in range(len(loss_auxi_weight)):
        loss_name_suffix = str(pow(2, i + 2))
        losses.append(
            dict(
                type="DepthConfidenceCombinedLoss",
                depth_loss=dict(
                    type="DepthL1Loss",
                    max_gt_depth=high_depth / gt_scale,
                    loss_weight=1.0,
                    loss_name="depth_l1_loss",
                    use_weight_map=True,
                    beta=l1_loss_beta,
                ),
                conf_loss=dict(
                    type="ConfL1Loss",
                    max_gt_depth=high_depth / gt_scale,
                    loss_weight=1.0,
                    loss_name="conf_l1_loss",
                    use_weight_map=False,
                ),
                depth_loss_name="depth_loss_s" + loss_name_suffix,
                conf_loss_name="conf_loss_s" + loss_name_suffix,
                loss_weight=[1.0, 10.0],
            )
        )
        losses_weight.append(loss_auxi_weight[i])  # need improve

# smooth_loss
if with_smooth:
    losses.append(
        dict(
            type="SmoothDepthLoss",
            do_normalization=True,
            loss_weight=1.0,
            loss_name="smooth_loss_s1",
        )
    )
    if training_step == "qat":
        losses_weight.extend([1])
    else:
        losses_weight.extend([5])

if with_virtual_normal:
    losses.append(
        dict(
            type="DepthVNLLoss",
            focal_x=int(virtual_cam_params[0, 0]),
            focal_y=int(virtual_cam_params[1, 1]),
            input_size=[height, width],
            delta_cos=0.867,
            delta_diff_x=0.01,
            delta_diff_y=0.01,
            delta_diff_z=0.01,
            delta_z=0.00001,
            sample_ratio=vnl_sample_ratio,
            gt_scale=gt_scale,
            scale_pred=True,
            depth_type=depth_type,
            loss_name="vnl_loss",
        )
    )
    losses_weight.extend([vnl_weight])

model = dict(
    type="DepthModel",
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
        in_strides=bifpn_in_strides,
        out_strides=bifpn_out_strides,
        stride2channels=get_vargnetv2_stride2channels(alpha),
        out_channels=neck_feat_channels,
        stack=3,
        start_level=bifpn_start_level,
        end_level=bifpn_end_level,
        num_outs=5,
    ),
    head=dict(
        type="DepthUnetHead",
        in_strides=bifpn_out_strides,
        out_strides=head_out_strides,
        in_channels=head_in_channels,
        out_channels=head_out_channels,  # need to be improved
        pred_out_channel=pred_out_channel,
        stacked_convs=stacked_convs,
        start_level=start_level,
        end_level=end_level,
        bn_kwargs=bn_kwargs,
        group_base=8,
        conv_method="conv2d",
        use_auxi_loss=use_auxi_loss,
        with_refine=with_refine,
        last_with_relu=last_with_relu,
        dequant_output=True,
        int8_output=True,  # True
        head_out_names=head_out_names,
    ),
    head_parser=dict(
        type="DepthHeadParserWithScale",
        out_strides=head_parser_strides,
    ),
    target_generator=dict(
        type="DepthMultiTargets",
        downsample_scale=2,
        depth_type=depth_type,
        low_depth=low_depth,
        high_depth=high_depth,
        gt_scale=gt_scale,
        label_name="gt_depth",
        pred_names=list(head_out_names.values()),
        target_modules=dict(
            type="DepthValueTarget",
            label_name="gt_depth",
            with_smooth=with_smooth,
            with_virtual_normal=with_virtual_normal,
        ),
        valid_hfov=None if hfov is None else np.deg2rad(hfov),
    ),
    decoder=None,
    losses=dict(
        type="SegLoss",
        loss=[
            dict(
                type="MixSegLossMultipreds",
                losses=losses,
                losses_weight=losses_weight,
            ),
        ],
    ),
    head_out_name=head_out_names["depth"],
)

# val model structure
val_model = copy.deepcopy(model)
val_model["losses"] = None
val_model["target_generator"]["gt_scale"] = 1.0
val_model["target_generator"]["target_modules"]["with_smooth"] = False
val_model["target_generator"]["target_modules"]["with_virtual_normal"] = False
val_model["target_generator"]["depth_type"] = "Cartesian"
val_model["head_parser"] = dict(
    type="DepthHeadParserWithScale",
    out_strides=head_parser_strides,
    scale=gt_scale,
)
val_model["decoder"] = dict(
    type="DepthMultiDecoder",
    pred_names=[
        head_out_names["depth"],
    ],
    decoder_modules=[
        dict(
            type="DepthDecoder",
            output_name="depth_preds",
            in_strides=head_out_strides,
            out_stride=head_out_strides[0],
            depth_type=depth_type,
        )
    ],
)
# test model structure
add_desc_pp = dict(
    type="AddDesc",
    per_tensor_desc=[
        # num of desc == num of pred output tensors (one tensor per out stride)
        json.dumps(
            dict(
                task=task_name,
                size=[width, height],
                min_depth_threshold=low_depth,
                max_depth_threshold=high_depth,
                virtual_cam_focalu=virtual_cam_params[0, 0],
                virtual_cam_cx=virtual_cam_params[0, 2],
                hfov=hfov,
                scale_number=gt_scale,
                proj_type=2,
                proj_shift=1,
            )
        )
    ],
)
deploy_model = copy.deepcopy(model)
deploy_model["head"]["use_auxi_loss"] = False
deploy_model["head_parser"] = None
deploy_model["losses"] = None
deploy_model["target_generator"] = None
if training_step == "int_infer":
    deploy_model["add_desc"] = add_desc_pp

deploy_inputs = {"img": [torch.randn((1,) + data_shape)]}

# data augmentation
to_rgb = True
transforms = []
# universal transforms
transforms.extend(
    [
        dict(
            type="Resize",
            keep_ratio=True,
            img_scale=(height, width),
        ),
        dict(type="ToTensor", to_yuv=False),
        dict(
            type="ColorJitter",
            brightness=0.1,
            contrast=(0.5, 1.5),
            saturation=(0.5, 1.5),
            hue=0.1,
        ),
        dict(type="RandomFlip", px=0.5, py=0.0),
        dict(
            type="SegRandomAffine",
            degrees=1,
            label_fill_value=ignore_index,
            interpolation=InterpolationMode.BILINEAR,
            rotate_p=0.5,
            translate_p=0,
            scale_p=0,
        ),
    ]
)
# model transform
transforms.extend(
    [
        dict(type="CopyKeys", keys=["img|origin_img"]),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(type="ImageNormalize", mean=128.0, std=128.0),
    ]
)

# dataset config
train_paths, val_paths = get_dataset_list(
    bucket_root=bucket_root,
    attribution="superparking",
    model_version="v0.5.0",
    version_yaml="depth_dataset_version.yaml",
    dataset_yaml="depth_all_datasets.yaml",
)


train_dataset = dict(
    type="DepthDatasetRec",
    paths=train_paths,
    to_rgb=to_rgb,
    depth_scale_factor=depth_scale,
    select_sample=False,
    with_parsing=True,
    virtual_cam_params=virtual_cam_params,
    mode="train",
    transforms=transforms,
)
train_dataset = dict(
    type="RepeatDataset",
    dataset=train_dataset,
    times=1,
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

val_transforms = [
    dict(
        type="Resize",
        keep_ratio=True,
        img_scale=(height, width),
    ),
    dict(type="CopyKeys", keys=["img|origin_img"]),
    dict(type="ToTensor", to_yuv=False),
    dict(type="BgrToYuv444", rgb_input=True),
    dict(type="ImageNormalize", mean=128.0, std=128.0),
]

val_dataset = dict(
    type="DepthDatasetRec",
    paths=val_paths,
    to_rgb=to_rgb,
    depth_scale_factor=depth_scale,
    select_sample=False,
    with_parsing=True,
    virtual_cam_params=virtual_cam_params,
    mode="val" if training_step != "int_infer" else "test",
    transforms=val_transforms,
)

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
    losses = model_outs["depth_loss"]
    losses_val = []
    for loss in losses:
        for _, value in loss["multipredsloss"].items():
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
        loss_show = dict()
        total_loss = 0.0
        losses = model_outs["depth_loss"]
        for loss in losses:
            loss = loss["multipredsloss"]
            loss_show.update(loss)
            for _, val in loss.items():
                total_loss += val
        loss_show["all_loss"] = total_loss
        metric.update(loss_show)


metric_updater = dict(
    type="MetricUpdater",
    metrics=[dict(type="LossShow", name="loss_depth")],
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

depth_pat = re.compile("^.*((depth_preds)|(gt_depth))")


def tb_train_vis(writer, model_outs, global_step_id, **kwargs):
    def apply_color(data, cmp=cv2.COLORMAP_JET):
        data = (data - data.min()) / (data.max() - data.min()) * 255
        data = data.astype("uint8")
        data = cv2.applyColorMap(data, cmp)[:, :, [2, 1, 0]]
        return data

    for k, v in model_outs.items():
        if depth_pat.match(k):
            depth_vis = v[0].detach().cpu().numpy()[0:1, :, :].squeeze()
            depth_vis = np.clip(depth_vis, 0, high_depth)
            vis = apply_color(depth_vis)
            writer.add_image(
                k, vis, global_step=global_step_id, dataformats="HWC"
            )
        if k == "ori_img":
            ori_img = v[0].detach().cpu().numpy()
            ori_img = ori_img.astype("uint8")
            vis = ori_img[:, :, [2, 1, 0]]
            writer.add_image(
                k, vis, global_step=global_step_id, dataformats="HWC"
            )


tb_train_vis_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tb_save_dir, "train_vis"),
    update_freq=log_freq,
    update_by="step",
    tb_update_funcs=[tb_train_vis],
)

# callback config in validation stage
range_border = [low_depth, 5, 20, 30, high_depth]
range_list = []
for i in range(1, len(range_border)):
    range_list.append((range_border[i - 1], range_border[i]))
range_list.extend([(low_depth, 30), (30, high_depth)])
val_absrel_metric = AbsRel(range_list=range_list)
val_rmse_metric = RMSE(range_list=range_list)
val_confrmse_metric = ConfRMSE(range_list=range_list, name="ConfRMSE")


def update_val_metric(metrics, batch, model_outs):
    label = batch["gt_depth"]
    preds_label = model_outs["depth_preds"]
    pred_depth_split = torch.split(preds_label, 1, dim=1)
    if preds_label.size()[1] == 2:
        pred_depth, pred_conf = pred_depth_split
    else:
        pred_depth = pred_depth_split[0]
        pred_conf = None
        gt_conf = None
    if pred_conf is not None:
        gt_conf = (pred_depth - label).abs()
        gt_conf = torch.exp(-gt_conf / (label + 1e-6))

    for i, metric in enumerate(metrics):
        if i == len(metrics) - 1 and pred_conf is not None:
            metric.update(label, gt_conf, pred_conf)
        else:
            metric.update(label, pred_depth)


val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[val_absrel_metric, val_rmse_metric, val_confrmse_metric],
    metric_update_func=update_val_metric,
    step_log_freq=50,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)


def tb_val_absrel_func(writer, epoch_id, **kwargs):
    name, value = val_absrel_metric.get(log=False)
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
    for i, (k, v) in enumerate(zip(name, value)):
        if "std" in k:
            continue
        df_metrics[k + f"_{i}"].append(v)
        writer.add_scalar(k, v, global_step=epoch_id)


def tb_val_rmse_func(writer, epoch_id, **kwargs):
    name, value = val_rmse_metric.get(log=False)
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
    for i, (k, v) in enumerate(zip(name, value)):
        if "std" in k:
            continue
        df_metrics[k + f"_{i}"].append(v)
        writer.add_scalar(k, v, global_step=epoch_id)


def tb_dataframe(writer, epoch_id, **kwargs):
    df_metrics["step"].append(training_step)
    df = pd.DataFrame.from_dict(df_metrics, orient="columns")
    df.to_excel(df_excel_path)


tb_val_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tb_save_dir, "val"),
    update_freq=1,
    update_by="epoch",
    tb_update_funcs=[tb_val_absrel_func, tb_val_rmse_func, tb_dataframe],
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater, tb_val_callback],
    val_model=val_model,
)

depoly_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater, tb_val_callback],
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
    mode=None,
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
        type=torch.optim.Adam,
        lr=float_lr,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=weight_decay,
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
            warmup_len=warmup_len,  # 0
            warmup_begin_lr=warmup_begin_lr,
            stop_lr=stop_lr,
        ),
        metric_updater,
        tb_loss_callback,
        tb_train_vis_callback,
        # val_callback,
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
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=qat_lr,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=weight_decay,
    ),
    batch_processor=batch_processor,
    device=None,
    stop_by=train_by,
    num_steps=qat_train_steps,
    num_epochs=qat_train_epochs,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",  # CosLrUpdater
            step_log_interval=log_freq,
        ),
        metric_updater,
        tb_loss_callback,
        tb_train_vis_callback,
        # val_callback,
        ckpt_callback,
    ],
    sync_bn=sync_bn,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

# int_infer config
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
