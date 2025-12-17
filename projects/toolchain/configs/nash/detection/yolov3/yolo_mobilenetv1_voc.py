import copy
import os

import cv2
import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image

from hat.data.collates.collates import collate_2d
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "yolo_mobilenetv1_voc"
data_shape = (3, 416, 416)
batch_size_per_gpu = 1
device_ids = [1]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH
convert_mode = "fx"

num_classes = 20
bn_kwargs = {}
anchors = [
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)],
]
model = dict(
    type="YOLOV3",
    backbone=dict(
        type="MobileNetV1",
        alpha=1.0,
        bn_kwargs=bn_kwargs,
        num_classes=num_classes,
        include_top=False,
    ),
    neck=dict(
        type="YOLOV3Neck",
        backbone_idx=[-1, -2, -3],
        in_channels_list=[1024, 512, 256],
        out_channels_list=[512, 256, 128],
        bn_kwargs=bn_kwargs,
    ),
    head=dict(
        type="YOLOV3Head",
        feature_idx=[-3, -2, -1],
        in_channels_list=[1024, 512, 256],
        num_classes=num_classes,
        anchors=anchors,
        bn_kwargs=bn_kwargs,
    ),
    loss=dict(
        type="YOLOV3Loss",
        num_classes=num_classes,
        anchors=anchors,
        strides=[8, 16, 32],
        ignore_thresh=0.5,
        loss_xy=dict(type=torch.nn.BCELoss, reduce=False),
        loss_wh=dict(type=torch.nn.L1Loss, reduce=False),
        loss_conf=dict(type=torch.nn.BCELoss, reduction="sum"),
        loss_cls=dict(type=torch.nn.BCELoss, reduction="sum"),
        lambda_loss=[2.0, 2.0, 1.0, 1.0],
    ),
    postprocess=dict(
        type="YOLOV3PostProcess",
        anchors=anchors,
        strides=[8, 16, 32],
        num_classes=num_classes,
        score_thresh=0.01,
        nms_thresh=0.45,
        topK=200,
    ),
)

deploy_model = copy.deepcopy(model)
deploy_model["loss"] = None
deploy_model["postprocess"] = None
deploy_model["head"]["dequant_output"] = False
deploy_model["filter_module"] = dict(
    type="YOLOv3Filter",
    strides=[8, 16, 32],
    idx_range=[5, num_classes + 5],
    threshold=-2.944,
)
deploy_inputs = dict(
    img=torch.randn((1, 3, 416, 416)),
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
        dict(type="QAT2Quantize", convert_mode=convert_mode),
    ],
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="PascalVOC",
        data_path="./tmp_data/voc/trainval_lmdb",
        transforms=[
            dict(type="RandomExpand", ratio_range=(1, 4)),
            dict(
                type="MinIoURandomCrop",
                min_ious=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            ),
            dict(type="Resize", img_scale=[data_shape[1:]], keep_ratio=False),
            dict(type="RandomFlip"),
            dict(type="ToTensor", to_yuv=True, use_yuv_v2=False),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
    collate_fn=collate_2d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="PascalVOC",
        data_path="./tmp_data/voc/test_lmdb",
        transforms=[
            dict(type="Resize", img_scale=[data_shape[1:]], keep_ratio=False),
            dict(type="ToTensor", to_yuv=True, use_yuv_v2=False),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
    collate_fn=collate_2d,
)

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_index(0),
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=-1,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Validation " + task_name

stat_callback = dict(
    type="StatsMonitor",
    log_freq=100,
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
    val_on_train_end=False,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=(
                    "./tmp_pretrained_models/mobilenetv1_imagenet/float-checkpoint-best.pth.tar"  # noqa: E501
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=5e-4)},
        lr=0.001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=200,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=4,
            step_log_interval=100,
            lr_decay_id=[160, 180],
            lr_decay_factor=0.1,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=None,
    val_metrics=[
        dict(type="VOC07MApMetric", num_classes=num_classes),
    ],
)

# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 100

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_calibration_observer="mse",
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
        dict(type="VOC07MApMetric", num_classes=num_classes),
    ],
    log_interval=calibration_step / 10,
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
    output_layout="NHWC",
    opt="O3",
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
        dict(type="VOC07MApMetric", num_classes=num_classes),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

calibration_predictor = dict(
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
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="VOC07MApMetric", num_classes=num_classes),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)


hbir_infer_model = dict(
    type="YOLOHbirInfer",
    model_path=os.path.join(ckpt_dir, "quantized.bc"),
    postprocess=dict(
        type="YOLOV3HbirPostProcess",
        anchors=anchors,
        strides=[8, 16, 32],
        num_classes=num_classes,
        score_thresh=0.01,
        nms_thresh=0.45,
        topK=200,
    ),
)


int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["sampler"] = None
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False


int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=[int_infer_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="VOC07MApMetric", num_classes=num_classes),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

infer_transforms = [
    dict(
        type="TorchVisionAdapter",
        interface="Resize",
        size=(416, 416),
    ),
    dict(
        type="TorchVisionAdapter",
        interface="PILToTensor",
    ),
    dict(
        type="BgrToYuv444",
        rgb_input=True,
    ),
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
        type="VOCFromImage",
        root="./tmp_orig_data/voc/",
        year="2007",
        image_set="test",
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
        dict(type="VOC07MApMetric", num_classes=num_classes),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def process_inputs(infer_inputs, transforms):
    ori_img = Image.open(infer_inputs["imgs"])
    ori_img.convert("RGB")
    model_input = {
        "img": ori_img,
        "ori_img": np.array(ori_img),
        "layout": "hwc",
        "color_space": "rgb",
    }

    model_input = transforms(model_input)
    model_input["img"] = model_input["img"].unsqueeze(0)
    model_input["ori_img"] = [model_input["ori_img"]]
    ori_img = model_input["ori_img"][0]
    size = model_input["img"].size()[-2:]
    vis_img = cv2.resize(ori_img, size)

    return model_input, vis_img


def process_outputs(model_outs, viz_func, vis_inputs):
    preds = model_outs["pred_bboxes"]
    bboxes = preds[0][:, :4]
    labels = preds[0][:, 4]
    scores = preds[0][:, 5]
    res = torch.cat(
        (
            bboxes,
            labels.unsqueeze(-1),
            scores.unsqueeze(-1),
        ),
        -1,
    )
    preds = viz_func(vis_inputs, res)
    return None


infer_cfg = dict(
    model=hbir_infer_model,
    infer_inputs=dict(
        imgs="./tmp_orig_data/voc/VOCdevkit/VOC2007/JPEGImages/000001.jpg",
    ),
    process_inputs=process_inputs,
    transforms=infer_transforms,
    viz_func=dict(type="DetViz", is_plot=True),
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
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
)
