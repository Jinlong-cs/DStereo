import copy
import os

import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from horizon_plugin_pytorch.qat_mode import tricks
from PIL import Image
from torch import nn

from hat.data.collates.collates import collate_2d
from hat.utils.config import ConfigVersion

tricks.fx_force_duplicate_shared_convbn = False
VERSION = ConfigVersion.v2

training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "stereonet_stereonetneck_sceneflow"
data_num_workers = 2
march = March.BAYES
ckpt_dir = "./tmp_models/%s" % task_name

train_batch_size_per_gpu = 16
test_batch_size_per_gpu = 4

device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
convert_mode = "fx"

loss_weights = [0.3, 0.3, 0.5, 0.5, 1.0]
maxdisp = 192
use_bn = True
bias = False
bn_kwargs = {}
refine_levels = 4
base_lr = 0.001
num_epochs = 200
out_channels = [32, 32, 64, 128, 128, 16]

model = dict(
    type="StereoNet",
    backbone=dict(
        type="StereoNetNeck",
        out_channels=out_channels,
        use_bn=use_bn,
        bias=bias,
        bn_kwargs=bn_kwargs,
        act_type=nn.ReLU(),
    ),
    head=dict(
        type="StereoNetHead",
        maxdisp=maxdisp,
        bn_kwargs=bn_kwargs,
        refine_levels=refine_levels,
    ),
    post_process=dict(
        type="StereoNetPostProcess",
        maxdisp=maxdisp,
    ),
    loss=dict(type="SmoothL1Loss"),
    loss_weights=loss_weights,
)

deploy_model = dict(
    type="StereoNet",
    backbone=dict(
        type="StereoNetNeck",
        out_channels=out_channels,
        use_bn=use_bn,
        bias=bias,
        bn_kwargs=bn_kwargs,
        act_type=nn.ReLU(),
    ),
    head=dict(
        type="StereoNetHead",
        maxdisp=maxdisp,
        bn_kwargs=bn_kwargs,
        refine_levels=refine_levels,
    ),
)

deploy_inputs = dict(img=torch.randn((1, 6, 540, 960)))


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="SceneFlow",
        data_path="./tmp_data/SceneFlow/train_lmdb",
        transforms=[
            dict(
                type="RandomCrop",
                size=(256, 512),
            ),
            dict(
                type="ToTensor",
                to_yuv=False,
                use_yuv_v2=False,
            ),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=train_batch_size_per_gpu,
    pin_memory=True,
    shuffle=False,
    num_workers=data_num_workers,
    collate_fn=collate_2d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="SceneFlow",
        data_path="./tmp_data/SceneFlow/test_lmdb",
        transforms=[
            dict(
                type="ToTensor",
                to_yuv=False,
                use_yuv_v2=False,
            ),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=test_batch_size_per_gpu,
    pin_memory=True,
    shuffle=False,
    num_workers=data_num_workers,
    collate_fn=collate_2d,
)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)


def loss_collector(outputs: dict):
    return outputs["losses"]


train_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


def update_loss_metric(metrics, batch, model_outs):
    loss = sum(model_outs["losses"])
    metrics[0].update(loss)
    labels = batch["gt_disp"]
    preds = model_outs["pred_disps"]
    masks = (labels > 0) & (labels < maxdisp)
    metrics[1].update(labels, preds, masks)


loss_show_callback = dict(
    type="MetricUpdater",
    metric_update_func=update_loss_metric,
    step_log_freq=50,
    epoch_log_freq=1,
    log_prefix="train_" + task_name,
)


def update_metric(metrics, batch, model_outs):
    labels = batch["gt_disp"]
    preds = model_outs
    masks = (labels > 0) & (labels < maxdisp)
    metrics[0].update(labels, preds, masks)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=50,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
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
        params={"weight": dict(weight_decay=4e-5)},
        lr=base_lr,
    ),
    batch_processor=train_batch_processor,
    stop_by="epoch",
    num_epochs=num_epochs,
    device=None,
    sync_bn=True,
    callbacks=[
        stat_callback,
        loss_show_callback,
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
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    val_metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 20

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_calibration_observer="percentile",
            activation_calibration_qkwargs=dict(
                percentile=99.985,
                bins=8192,
            ),
        ),
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
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    log_interval=calibration_step / 10,
)
qat_data = copy.deepcopy(data_loader)
qat_data["dataset"]["transforms"] = val_data_loader["dataset"]["transforms"]
qat_data["batch_size"] = int(8 / 2)


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=qat_data,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.00005,
    ),
    batch_processor=train_batch_processor,
    num_epochs=40,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_callback,
        dict(
            type="CosLrUpdater",
            step_log_interval=1000,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    val_metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
)
deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
        dict(type="QAT2Quantize", convert_mode=convert_mode),
    ],
)

int_infer_trainer = dict(
    type="Trainer",
    model=deploy_model,
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
            dict(type="QAT2Quantize", convert_mode=convert_mode),
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
    input_source=["ddr"],
)

float_predictor = dict(
    type="Predictor",
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
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        stat_callback,
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
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
            ),
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
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        stat_callback,
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
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
            ),
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
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        stat_callback,
        val_metric_updater,
    ],
    log_interval=1,
)


onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
)
infer_transforms = [
    dict(
        type="ToTensor",
        to_yuv=False,
        use_yuv_v2=False,
    ),
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

infer_ckpt = int_infer_trainer["model_convert_pipeline"]["converters"][1][
    "checkpoint_path"
]

align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="SceneFlowFromImage",
        data_path="./tmp_orig_data/SceneFlow/",
        data_list="./tmp_orig_data/SceneFlow/SceneFlow_finalpass_test.txt",
        transforms=infer_transforms,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

align_bpu_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=infer_ckpt,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=val_data_loader,
    metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)


def process_inputs(infer_inputs, transforms):
    ori_left_img = Image.open(infer_inputs["imgl"]).convert("RGB")
    img_l = np.array(ori_left_img)
    ori_right_img = Image.open(infer_inputs["imgr"]).convert("RGB")
    img_r = np.array(ori_right_img)
    img = np.concatenate((img_l, img_r), axis=2)

    model_input = {
        "img": img,
        "ori_img": copy.deepcopy(img),
        "layout": "hwc",
        "color_space": "rgb",
        "img_shape": img_l.shape,
    }

    model_input = transforms(model_input)
    model_input["img"] = model_input["img"].unsqueeze(0)

    vis_inputs = {}
    vis_inputs["f"] = infer_inputs["f"]
    vis_inputs["baseline"] = infer_inputs["baseline"]
    vis_inputs["img"] = copy.deepcopy(img)

    return model_input, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs.squeeze(0).cpu().numpy()
    f = float(vis_inputs["f"])
    baseline = float(vis_inputs["baseline"])
    img = vis_inputs["img"]
    depth = baseline * f / preds
    preds = viz_func(img, preds, depth)
    return None


infer_cfg = dict(
    model=model,
    infer_inputs=dict(
        imgl="./tmp_orig_data/SceneFlow/FlyingThings3D/frames_finalpass/TEST/A/0000/left/0006.png",
        imgr="./tmp_orig_data/SceneFlow/FlyingThings3D/frames_finalpass/TEST/A/0000/right/0006.png",
        baseline="0.54",
        f="1050",
    ),
    process_inputs=process_inputs,
    transforms=infer_transforms,
    viz_func=dict(type="DispViz", is_plot=True),
    process_outputs=process_outputs,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=infer_ckpt,
            ),
            dict(type="QAT2Quantize", convert_mode=convert_mode),
        ],
    ),
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
                checkpoint_path=infer_ckpt,
            ),
        ],
    ),
)
