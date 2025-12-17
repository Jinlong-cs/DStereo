import copy
import os

import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from PIL import Image

from hat.data.collates.collates import collate_2d
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "yolo_varg_darknet53_mscoco"
data_shape = (3, 416, 416)
batch_size_per_gpu = 16
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH
convert_mode = "fx"

num_classes = 80
bn_kwargs = {}
anchors = [
    [(116, 90), (156, 198), (373, 326)],
    [(30, 61), (62, 45), (59, 119)],
    [(10, 13), (16, 30), (33, 23)],
]
model = dict(
    type="YOLOV3",
    backbone=dict(
        type="VarGDarkNet53",
        max_channels=1024,
        bn_kwargs=bn_kwargs,
        num_classes=num_classes,
        include_top=False,
        flat_output=True,
    ),
    neck=dict(
        type="YoloGroupNeck",
        head_group=True,
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
        reverse_feature=False,
        int16_output=False,
    ),
    loss=dict(
        type="YOLOV3Loss",
        num_classes=num_classes,
        anchors=anchors,
        strides=[32, 16, 8],
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
        strides=[32, 16, 8],
        num_classes=num_classes,
        score_thresh=0.01,
        nms_thresh=0.45,
        topK=200,
    ),
)

deploy_model = copy.deepcopy(model)
deploy_model["loss"] = None
deploy_model["postprocess"] = None
deploy_inputs = dict(
    img=torch.randn((1, 3, 416, 416)),
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="./tmp_data/mscoco/train_lmdb",
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

qat_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="./tmp_data/mscoco/train_lmdb",
        transforms=[
            dict(type="RandomExpand", ratio_range=(1, 4)),
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
        type="Coco",
        data_path="./tmp_data/mscoco/val_lmdb",
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
    step_log_freq=10000,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Validation " + task_name

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
                    "./tmp_pretrained_models/varg_darknet53_imagenet/float-checkpoint-best.pth.tar"  # noqa: E501
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
        lr=0.002,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=280,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=2,
            step_log_interval=100,
            lr_decay_id=[220, 250],
            lr_decay_factor=0.1,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=None,
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
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
            activation_calibration_observer="min_max",
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
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
    log_interval=calibration_step / 10,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0,
            ),
            weight_qat_qkwargs=dict(
                averaging_constant=1,
            ),
        ),
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
    data_loader=qat_data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=5e-4)},
        lr=0.00001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=3,
    device=None,
    callbacks=[
        stat_callback,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=None,
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
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
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
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
    metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
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
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

infer_transforms = [
    dict(type="Resize", img_scale=(416, 416), keep_ratio=True),
    dict(
        type="Pad",
        size=(416, 416),
    ),
    dict(type="ToTensor", to_yuv=True, use_yuv_v2=False),
    dict(type="Normalize", mean=128.0, std=128.0),
]

hbir_infer_model = dict(
    type="YOLOHbirInfer",
    model_path=os.path.join(ckpt_dir, "quantized.bc"),
    post_process=dict(
        type="YOLOV3PostProcess",
        anchors=anchors,
        strides=[32, 16, 8],
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
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

align_bpu_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="CocoFromImage",
        root="./tmp_orig_data/mscoco/val2017",
        annFile="./tmp_data/mscoco/instances_val2017.json",
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
    metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
    batch_processor=dict(type="BasicBatchProcessor", need_grad_update=False),
)


def process_inputs(infer_inputs, transforms):
    ori_img = Image.open(infer_inputs["imgs"])
    ori_img.convert("RGB")
    image = np.array(ori_img)
    model_input = {
        "img": image,
        "ori_img": image,
        "img_name": [os.path.basename(infer_inputs["imgs"])],
        "img_id": [0],
        "layout": "hwc",
        "color_space": "rgb",
        "img_shape": image.shape[0:2],
    }

    model_input = transforms(model_input)
    model_input["img"] = model_input["img"].unsqueeze(0)
    model_input["ori_img"] = [model_input["ori_img"]]
    model_input["layout"] = [model_input["layout"]]
    model_input["color_space"] = [model_input["color_space"]]
    model_input["img_shape"] = [model_input["img_shape"]]
    model_input["pad_shape"] = [model_input["pad_shape"]]

    return model_input, model_input["resized_ori_img"]


def process_outputs(model_outs, viz_func, vis_inputs):
    dets = model_outs["pred_bboxes"][0]
    viz_func(vis_inputs, dets)
    return None


infer_cfg = dict(
    model=hbir_infer_model,
    infer_inputs=dict(
        imgs="./tmp_orig_data/mscoco/val2017/000000000139.jpg",
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
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
)
