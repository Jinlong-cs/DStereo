import copy
import os

import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.march import March
from PIL import Image
from torchvision.transforms import Compose

from hat.data.collates.collates import collate_2d, collate_2d_pad
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")


task_name = "deform_detr_resnet50_mscoco"

num_classes = 80
batch_size_per_gpu = 2
device_ids = [i for i in range(8)]  # noqa [C416]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = False
seed = 1
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH
qat_mode = "fuse_bn"
img_shape = (800, 1333)
convert_mode = "fx"
num_queries = 900
with_box_refine = False
as_two_stage = False

model = dict(
    type="DeformableDETR",
    backbone=dict(
        type="ResNet50",
        bn_kwargs={},
        num_classes=num_classes,
        include_top=False,
    ),
    neck=dict(
        type="ChannelMapperNeck",
        in_channels=[512, 1024, 2048],
        out_indices=[2, 3, 4],
        out_channel=256,
        kernel_size=1,
        extra_convs=1,
        norm_layer=dict(
            type=nn.BatchNorm2d,
            num_features=256,
        ),
    ),
    position_embedding=dict(
        type="PositionEmbeddingSine",
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    transformer=dict(
        type="DeformableDetrTransformer",
        encoder=dict(
            type="DeformableDetrTransformerEncoder",
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            post_norm=False,
            num_feature_levels=4,
        ),
        decoder=dict(
            type="DeformableDetrTransformerDecoder",
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels=4,
            as_two_stage=as_two_stage,
            with_box_refine=with_box_refine,
        ),
        num_feature_levels=4,
        two_stage_num_proposals=num_queries,
        as_two_stage=as_two_stage,
    ),
    embed_dim=256,
    num_classes=num_classes,
    num_queries=num_queries,
    aux_loss=True,
    post_process=dict(
        type="DeformDetrPostProcess", select_box_nums_for_evaluation=300
    ),
    criterion=dict(
        type="DeformableCriterion",
        num_classes=num_classes,
        matcher=dict(
            type="HungarianMatcher",
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            use_focal=True,
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
    ),
    as_two_stage=as_two_stage,
    with_box_refine=with_box_refine,
)

deploy_model = dict(
    type="DeformableDETR",
    backbone=dict(
        type="ResNet50",
        bn_kwargs={},
        num_classes=num_classes,
        include_top=False,
    ),
    neck=dict(
        type="ChannelMapperNeck",
        in_channels=[512, 1024, 2048],
        out_indices=[2, 3, 4],
        out_channel=256,
        kernel_size=1,
        extra_convs=1,
        norm_layer=nn.BatchNorm2d(256),
    ),
    position_embedding=dict(
        type="PositionEmbeddingSine",
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    transformer=dict(
        type="DeformableDetrTransformer",
        encoder=dict(
            type="DeformableDetrTransformerEncoder",
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            post_norm=False,
            num_feature_levels=4,
        ),
        decoder=dict(
            type="DeformableDetrTransformerDecoder",
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels=4,
            as_two_stage=as_two_stage,
            with_box_refine=with_box_refine,
        ),
        num_feature_levels=4,
        two_stage_num_proposals=num_queries,
        as_two_stage=as_two_stage,
    ),
    embed_dim=256,
    num_classes=num_classes,
    num_queries=num_queries,
    as_two_stage=as_two_stage,
    with_box_refine=with_box_refine,
)


deploy_inputs = dict(
    img=torch.randn((1, 3, img_shape[0], img_shape[1])),
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
        type="Coco",
        data_path="./tmp_data/mscoco/train_lmdb/",
        transforms=[
            dict(
                type="RandomFlip",
                px=0.5,
                py=0,
            ),
            dict(
                type="RandomSelectOne",
                p=1.0,
                transforms=[
                    dict(
                        type="Resize",
                        img_scale=[
                            (480, 1333),
                            (512, 1333),
                            (544, 1333),
                            (576, 1333),
                            (608, 1333),
                            (640, 1333),
                            (672, 1333),
                            (704, 1333),
                            (736, 1333),
                            (768, 1333),
                            (800, 1333),
                        ],
                        multiscale_mode="value",
                        keep_ratio=True,
                    ),
                    dict(
                        type=Compose,
                        transforms=[
                            dict(
                                type="Resize",
                                img_scale=[
                                    (400, 9999999),
                                    (500, 9999999),
                                    (600, 9999999),
                                ],
                                multiscale_mode="value",
                                keep_ratio=True,
                            ),
                            dict(
                                type="RandomSizeCrop",
                                min_size=384,
                                max_size=600,
                                filter_area=False,
                            ),
                            dict(
                                type="Resize",
                                img_scale=[
                                    (480, 1333),
                                    (512, 1333),
                                    (544, 1333),
                                    (576, 1333),
                                    (608, 1333),
                                    (640, 1333),
                                    (672, 1333),
                                    (704, 1333),
                                    (736, 1333),
                                    (768, 1333),
                                    (800, 1333),
                                ],
                                multiscale_mode="value",
                                keep_ratio=True,
                            ),
                        ],
                    ),
                ],
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128,
                std=128,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d_pad,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="./tmp_data/mscoco/val_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=[(800, 1333)],
                multiscale_mode="value",
                pad_to_keep_ratio=True,
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128,
                std=128,
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
    step_log_freq=20,
    reset_metrics_by="log",
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    enable_amp=False,
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
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=100,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

bn_callback = dict(
    type="FreezeBNStatistics",
)

grad_callback = dict(
    type="GradClip",
    max_norm=0.1,
    norm_type=2,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    interval_by="epoch",
    strict_match=False,
    mode="max",
    monitor_metric_key="mAP",
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    init_with_train_model=False,
    val_interval=1,
    interval_by="epoch",
    val_on_train_end=False,
    log_interval=100,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=(
                    "./tmp_pretrained_models/resnet50_imagenet/float-checkpoint-best.pth.tar"  # noqa: E501
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    find_unused_parameters=False,
    data_loader=data_loader,
    optimizer=dict(
        type="custom_param_optimizer",
        optim_cls=torch.optim.AdamW,
        model=model,
        optim_cfgs=dict(lr=1e-4, weight_decay=1e-4),
        custom_param_mapper=dict(
            norm_types={"weight_decay": 0.0}, backbone={"lr": 1e-5}
        ),
    ),
    batch_processor=batch_processor,
    num_epochs=50,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[40],
            step_log_interval=100,
        ),
        bn_callback,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
)

calibration_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="./tmp_data/mscoco/train_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=[(800, 1333)],
                multiscale_mode="value",
                pad_to_keep_ratio=True,
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128,
                std=128,
            ),
        ],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d,
)

qat_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="./tmp_data/mscoco/train_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=[(800, 1333)],
                multiscale_mode="value",
                pad_to_keep_ratio=True,
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128,
                std=128,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d,
)


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
        type="custom_param_optimizer",
        optim_cls=torch.optim.AdamW,
        model=model,
        optim_cfgs=dict(lr=1e-5, weight_decay=0),
        custom_param_mapper=dict(
            norm_types={"weight_decay": 0.0},
        ),
    ),
    batch_processor=batch_processor,
    num_epochs=5,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[4],
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="./tmp_data/mscoco/instances_val2017.json",
    ),
)


compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name="deform_detr_resnet50_model",
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
    opt="O3",
)

# # predictor
float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir,
                    "float-checkpoint-best.pth.tar",
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
    dict(
        type="Resize",
        img_scale=[(800, 1333)],
        multiscale_mode="value",
        keep_ratio=False,
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

hbir_infer_model = dict(
    type="DetrHbirInfer",
    model_path=os.path.join(ckpt_dir, "quantized.bc"),
    post_process=dict(
        type="DeformDetrPostProcess",
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
    return model_input, image


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
                ignore_extra=True,
            ),
        ],
    ),
)
