import copy
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from horizon_plugin_pytorch.march import March
from horizon_plugin_pytorch.qat_mode import ReLUMode, tricks
from skimage.io import imread
from torch import nn
from torchvision.transforms import InterpolationMode

from hat.data.collates.collates import collate_2d
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
tricks.relu6 = ReLUMode.FORCE_RELU6
tricks.fx_force_duplicate_shared_convbn = True
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "pwcnet_pwcnetneck_flyingchairs"
batch_size_per_gpu = 64
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
num_machines = 2
data_num_workers = 1
march = March.NASH
convert_mode = "fx"
num_gpus_per_machine = len(device_ids)

all_bs = batch_size_per_gpu * num_gpus_per_machine * num_machines
num_train_samples = 22232
lr_base = 0.0001
max_steps = 1200000
max_epoch = int(max_steps * 16 / num_train_samples)
lr = lr_base * all_bs / 16

ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True

loss_weights = [0.005, 0.01, 0.02, 0.08, 0.32]
out_channels = [16, 32, 64, 96, 128, 196]
flow_pred_lvl = 2
pyr_lvls = 6
train_scales = [2 ** x for x in range(flow_pred_lvl, pyr_lvls + 1)]
use_bn = True
bn_kwargs = {}
use_res = True
use_dense = True

model = dict(
    type="PwcNet",
    backbone=dict(
        type="PwcNetNeck",
        out_channels=out_channels,
        use_bn=use_bn,
        bn_kwargs=bn_kwargs,
        pyr_lvls=pyr_lvls,
        flow_pred_lvl=flow_pred_lvl,
        act_type=nn.ReLU(),
    ),
    head=dict(
        type="PwcNetHead",
        in_channels=out_channels,
        bn_kwargs=bn_kwargs,
        use_bn=use_bn,
        md=4,
        use_res=use_res,
        use_dense=use_dense,
        pyr_lvls=pyr_lvls,
        flow_pred_lvl=flow_pred_lvl,
        act_type=nn.ReLU(),
    ),
    loss=dict(type="LnNormLoss", norm_order=2, power=1, reduction="mean"),
    loss_weights=loss_weights,
)
deploy_model = dict(
    type="PwcNet",
    backbone=dict(
        type="PwcNetNeck",
        out_channels=out_channels,
        use_bn=use_bn,
        bn_kwargs=bn_kwargs,
        pyr_lvls=pyr_lvls,
        flow_pred_lvl=flow_pred_lvl,
        act_type=nn.ReLU(),
    ),
    head=dict(
        type="PwcNetHead",
        in_channels=out_channels,
        bn_kwargs=bn_kwargs,
        use_bn=use_bn,
        md=4,
        use_res=use_res,
        use_dense=use_dense,
        flow_pred_lvl=flow_pred_lvl,
        pyr_lvls=pyr_lvls,
        act_type=nn.ReLU(),
    ),
    loss=None,
)
deploy_inputs = dict(img=torch.randn((1, 6, 384, 512)))

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="FlyingChairs",
        data_path="./tmp_data/FlyingChairs/train_lmdb/",
        transforms=[
            dict(
                type="RandomCrop",
                size=(256, 448),
            ),
            dict(
                type="RandomFlip",
                px=0.5,
                py=0.5,
            ),
            dict(
                type="ToTensor",
                to_yuv=False,
            ),
            dict(
                type="SegRandomAffine",
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
                label_fill_value=0,
                translate_p=0.5,
                scale_p=0.0,
            ),
            dict(
                type="FlowRandomAffineScale",
                scale_p=0.5,
                scale_r=0.05,
            ),
        ],
        to_rgb=True,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    pin_memory=True,
    shuffle=True,
    num_workers=data_num_workers,
    collate_fn=collate_2d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="FlyingChairs",
        data_path="./tmp_data/FlyingChairs/val_lmdb/",
        transforms=[
            dict(
                type="ToTensor",
                to_yuv=False,
            ),
        ],
        to_rgb=True,
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=data_num_workers,
    pin_memory=True,
    collate_fn=collate_2d,
)


def loss_collector(outputs: dict):
    return outputs["losses"]


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
        dict(
            type="Scale",
            scales=tuple(1 / np.array(train_scales)),
            mode="bilinear",
            mul_scale=True,
        ),
    ],
    loss_collector=loss_collector,
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
    ],
    loss_collector=None,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)


def update_loss_metric(metrics, batch, model_outs):
    loss = sum(model_outs["losses"])
    metrics[0].update(loss)
    labels = batch["gt_ori_flow"]
    preds = model_outs["pred_flows"]
    preds = F.interpolate(preds.float(), scale_factor=4, mode="bilinear") * 4.0
    metrics[1].update(labels, preds)


loss_metirc_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss_metric,
    step_log_freq=10,
    epoch_log_freq=1,
    log_prefix="train_" + task_name,
)


def update_metric(metrics, batch, model_outs):
    labels = batch["gt_flow"]
    preds = model_outs
    preds = F.interpolate(preds.float(), scale_factor=4, mode="bilinear") * 4.0
    for metric in metrics:
        metric.update(labels, preds)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
        dict(type="QAT2Quantize", convert_mode=convert_mode),
    ],
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=False,
)
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="min",
    monitor_metric_key="EPE",
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=4e-4)},
        lr=lr,
    ),
    batch_processor=batch_processor,
    stop_by="epoch",
    num_epochs=max_epoch,
    device=None,
    callbacks=[
        stat_callback,
        loss_metirc_show_update,
        dict(
            type="CosLrUpdater",
            warmup_by="epoch",
            warmup_len=10,
            step_log_interval=1000,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(type="EndPointError"),
    ],
    val_metrics=[
        dict(type="EndPointError"),
    ],
    sync_bn=True,
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 10

calibration_trainer = dict(
    type="Calibrator",
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
            dict(type="Float2Calibration", convert_mode=convert_mode),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        stat_callback,
        val_callback,
        ckpt_callback,
    ],
    val_metrics=[
        dict(type="EndPointError"),
    ],
    log_interval=calibration_step / 10,
)


qat_data_loader = copy.deepcopy(data_loader)
qat_data_loader["dataset"]["transforms"][1]["px"] = 0.2
qat_data_loader["dataset"]["transforms"][1]["py"] = 0.2
qat_data_loader["dataset"]["transforms"][3]["translate_p"] = 0.2
qat_data_loader["dataset"]["transforms"][4]["scale_p"] = 0.2

qat_val_callback = copy.deepcopy(val_callback)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=qat_data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=4e-4)},
        lr=0.0001,
    ),
    batch_processor=batch_processor,
    num_epochs=120,
    device=None,
    callbacks=[
        stat_callback,
        loss_metirc_show_update,
        dict(
            type="CosLrUpdater",
            step_log_interval=1000,
        ),
        qat_val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(
            type="LossShow",
        ),
        dict(type="EndPointError"),
    ],
    val_metrics=[
        dict(type="EndPointError"),
    ],
)


compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["ddr"],
    opt="O3",
)

# predict
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
    callbacks=[
        val_metric_updater,
    ],
    metrics=[
        dict(type="EndPointError"),
    ],
    log_interval=50,
)

qat_val_data_loader = copy.deepcopy(val_data_loader)
qat_val_data_loader["batch_size"] = int(qat_val_data_loader["batch_size"] / 2)

calibration_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2Calibration", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=[qat_val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="EndPointError"),
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
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=[qat_val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[
        val_metric_updater,
    ],
    metrics=[
        dict(type="EndPointError"),
    ],
    log_interval=50,
)

hbir_infer_model = dict(
    type="PwcNetHbirInfer",
    model_path=os.path.join(ckpt_dir, "qat.bc"),
)
int_infer_data_loader = copy.deepcopy(qat_val_data_loader)
int_infer_data_loader["sampler"] = None
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False

int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=[int_infer_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[
        val_metric_updater,
    ],
    metrics=[
        dict(type="EndPointError"),
    ],
    log_interval=1,
)

infer_transforms = [
    dict(
        type="ToTensor",
        to_yuv=False,
    ),
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]


align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="FlyingChairs",
        data_path="./tmp_data/FlyingChairs/val_lmdb/",
        transforms=infer_transforms,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=align_bpu_data_loader,
    metrics=[
        dict(type="EndPointError"),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def process_inputs(infer_inputs, transforms):
    image1 = imread(infer_inputs["img1"])
    image2 = imread(infer_inputs["img2"])
    cv2.cvtColor(image1, cv2.COLOR_RGB2BGR, image1)
    cv2.cvtColor(image2, cv2.COLOR_RGB2BGR, image2)
    img = np.concatenate((image1, image2), axis=2)

    model_input = {
        "img": img,
        "ori_img": copy.deepcopy(img),
        "layout": "hwc",
        "color_space": "rgb",
        "img_shape": image1.shape,
    }

    model_input = transforms(model_input)
    model_input["img"] = model_input["img"].unsqueeze(0)

    return model_input, model_input["ori_img"]


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs
    preds = F.interpolate(preds.float(), scale_factor=4, mode="bilinear") * 4.0
    preds = preds.permute((0, 2, 3, 1))
    preds = viz_func(vis_inputs, preds)
    return None


infer_cfg = dict(
    model=hbir_infer_model,
    infer_inputs=dict(
        img1="./tmp_orig_data/FlyingChairs/FlyingChairs_release/data/00006_img1.ppm",
        img2="./tmp_orig_data/FlyingChairs/FlyingChairs_release/data/00006_img2.ppm",
    ),
    process_inputs=process_inputs,
    transforms=infer_transforms,
    viz_func=dict(type="FlowViz", is_plot=True),
    process_outputs=process_outputs,
)

onnx_cfg = dict(
    model=deploy_model,
    stage="qat",
    inputs=deploy_inputs,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
)
