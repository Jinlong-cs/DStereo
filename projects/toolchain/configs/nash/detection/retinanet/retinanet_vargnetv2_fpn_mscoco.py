import copy
import os

import numpy as np
import torch
from horizon_plugin_pytorch.march import March
from horizon_plugin_pytorch.nn import AnchorGenerator
from PIL import Image

from hat.data.collates.collates import collate_2d
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "retinanet_vargnetv2_fpn_mscoco"
num_classes = 80
batch_size_per_gpu = 2
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH
convert_mode = "fx"

scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ratios = [0.5, 1, 2]
levels = [3, 4, 5, 6, 7]

ratios = (ratios,) * len(levels)
base_sizes = [2 ** (x + 2) for x in levels]
sizes = [[scale * base_size for scale in scales] for base_size in base_sizes]

model = dict(
    type="RetinaNet",
    backbone=dict(
        type="VargNetV2",
        num_classes=1000,
        bn_kwargs={},
        include_top=False,
    ),
    neck=dict(
        type="RetinaNetFPN",
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[32, 32, 64, 128, 256],
        out_strides=[8, 16, 32],
        out_channels=[256, 256, 256],
        fix_out_channel=256,
    ),
    head=dict(
        type="RetinaNetHead",
        num_classes=num_classes,
        num_anchors=len(ratios[0]) * len(sizes[0]),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        int16_output=True,
    ),
    filter_module=dict(
        type="RetinanetMultiStrideFilter",
        strides=[8, 16, 32, 64, 128],
        threshold=-2.944,
    ),
    anchors=dict(
        type=AnchorGenerator,
        feat_strides=[2 ** x for x in levels],
        scales=scales,
        ratios=[0.5, 1, 2],
        base_sizes=base_sizes,
        round_anchor=True,
    ),
    targets=dict(
        type="BBoxTargetGenerator",
        matcher=dict(
            type="MaxIoUMatcher",
            pos_iou=0.5,
            neg_iou=0.4,
        ),
        label_encoder=dict(
            type="MatchLabelSepEncoder",
            bbox_encoder=dict(
                type="XYWHBBoxEncoder",
                reg_mean=(0.0, 0.0, 0.0, 0.0),
                reg_std=(1.0, 1.0, 1.0, 1.0),
            ),
            class_encoder=dict(
                type="OneHotClassEncoder",
                num_classes=num_classes + 1,
                class_agnostic_neg=True,
                exclude_background=True,
            ),
        ),
    ),
    post_process=dict(
        type="RetinaNetPostProcess",
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        topk_candidates=1000,
    ),
    loss_cls=dict(
        type="FocalLossV2",
        alpha=0.25,
        gamma=2.0,
    ),
    loss_reg=dict(
        type="SmoothL1Loss",
        beta=1.0 / 9.0,
        reduction="mean",
    ),
)
deploy_model = dict(
    type="RetinaNet",
    backbone=dict(
        type="VargNetV2",
        num_classes=1000,
        bn_kwargs={},
        include_top=False,
    ),
    neck=dict(
        type="RetinaNetFPN",
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[32, 32, 64, 128, 256],
        out_strides=[8, 16, 32],
        out_channels=[256, 256, 256],
        fix_out_channel=256,
    ),
    head=dict(
        type="RetinaNetHead",
        num_classes=num_classes,
        num_anchors=len(ratios[0]) * len(sizes[0]),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        int16_output=True,
        dequant_output=False,
    ),
    filter_module=dict(
        type="RetinanetMultiStrideFilter",
        strides=[8, 16, 32, 64, 128],
        threshold=-2.944,
    ),
    anchors=dict(
        type=AnchorGenerator,
        feat_strides=[2 ** x for x in levels],
        scales=scales,
        ratios=[0.5, 1, 2],
        base_sizes=base_sizes,
    ),
    targets=dict(
        type="BBoxTargetGenerator",
        matcher=dict(
            type="MaxIoUMatcher",
            pos_iou=0.5,
            neg_iou=0.4,
        ),
        label_encoder=dict(
            type="MatchLabelSepEncoder",
            bbox_encoder=dict(
                type="XYWHBBoxEncoder",
                reg_mean=(0.0, 0.0, 0.0, 0.0),
                reg_std=(1.0, 1.0, 1.0, 1.0),
            ),
            class_encoder=dict(
                type="OneHotClassEncoder",
                num_classes=num_classes + 1,
                class_agnostic_neg=True,
                exclude_background=True,
            ),
        ),
    ),
)
deploy_inputs = dict(img=torch.randn((1, 3, 1024, 1024)))
deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
        dict(type="QAT2Quantize", convert_mode=convert_mode),
    ],
)
inputs = dict(
    img=torch.randn((1, 3, 1024, 1024)),
    gt_bboxes=[torch.tensor([[10, 10, 100, 100]])],
    gt_classes=[torch.tensor([10])],
    resized_shape=[(800, 1024)],
)
inputs_wo_bboxes = dict(
    img=torch.randn((1, 3, 1024, 1024)),
    gt_bboxes=[torch.zeros(0, 4)],
    gt_classes=[torch.zeros(0)],
    resized_shape=[(800, 1024)],
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="./tmp_data/mscoco/train_lmdb/",
        transforms=[
            dict(
                type="RandomFlip",
                px=0.5,
                py=0,
            ),
            dict(
                type="Resize",
                img_scale=(800, 1024),
                keep_ratio=True,
            ),
            dict(
                type="Pad",
                size=(1024, 1024),
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="./tmp_data/mscoco/val_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=(800, 1024),
                keep_ratio=True,
            ),
            dict(
                type="Pad",
                size=(1024, 1024),
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    batch_size=batch_size_per_gpu * 4,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
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


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=10000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)
loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=1000,
    epoch_log_freq=1,
    log_prefix="loss_ " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
    monitor_metric_key="mAP",
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
                    "./tmp_pretrained_models/vargnetv2_imagenet/float-checkpoint-best.pth.tar"  # noqa: E501
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=5e-5)},
        lr=0.01,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=12,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="StepDecayLrUpdater",
            warmup_len=0.3,
            lr_decay_id=[8, 11],
            step_log_interval=10,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
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
            activation_calibration_observer="mix",
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

hbir_infer_model = dict(
    type="RetinaNetHbirInfer",
    model_path=os.path.join(ckpt_dir, "quantized.bc"),
    anchors=dict(
        type=AnchorGenerator,
        feat_strides=[2 ** x for x in levels],
        scales=scales,
        ratios=[0.5, 1, 2],
        base_sizes=base_sizes,
        round_anchor=True,
    ),
    post_process=dict(
        type="RetinaNetPostProcess",
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        topk_candidates=1000,
    ),
    split_dim=[720, 36],
    featsizes=[
        [1, 256, 128, 128],
        [1, 256, 64, 64],
        [1, 256, 32, 32],
        [1, 256, 16, 16],
        [1, 256, 8, 8],
    ],
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
    metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)
infer_transforms = [
    dict(
        type="Resize",
        img_scale=(800, 1024),
        keep_ratio=True,
    ),
    dict(
        type="Pad",
        size=(1024, 1024),
    ),
    dict(
        type="ToTensor",
        to_yuv=True,
        use_yuv_v2=False,
    ),
    dict(
        type="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

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
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
)
