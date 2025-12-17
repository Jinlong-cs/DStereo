import copy
import os

import torch
from horizon_plugin_pytorch.march import March

from hat.data.collates.nusc_collates import collate_nuscenes_sequencev2
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

enable_model_tracking = True

task_name = "bevformer_tiny_resnet50_detection_nuscenes"
num_classes = 1000
batch_size_per_gpu = 2
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
lossshow = 50
convert_mode = "fx"
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH
data_rootdir = "./tmp_data/nuscenes/v1.0-trainval"
meta_rootdir = "./tmp_data/nuscenes/meta"
map_size = (15, 30, 0.15)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
CLASSES = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)
bn_kwargs = {}
_dim_ = 256
_pos_dim_ = _dim_ // 2
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50
queue_length = 3  # each sequence contains `queue_length` frames.
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
num_classes = 10
model = dict(
    type="BevFormer",
    out_indices=(-1,),
    backbone=dict(
        type="ResNet50",
        num_classes=1000,
        bn_kwargs={},
        include_top=False,
    ),
    neck=dict(
        type="FPN",
        in_strides=[32],
        in_channels=[2048],
        out_strides=[32],
        out_channels=[_dim_],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="BevFormerViewTransformer",
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        embed_dims=_dim_,
        queue_length=3,
        in_indices=(-1,),
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        encoder=dict(
            type="BEVFormerEncoder",
            num_layers=3,
            return_intermediate=False,
            bev_h=bev_h_,
            bev_w=bev_w_,
            embed_dims=_dim_,
            encoder_layer=dict(
                type="BEVFormerEncoderLayer",
                selfattention=dict(
                    type="HorizonTemporalSelfAttention",
                    embed_dims=_dim_,
                    num_levels=1,
                    view_gird_in=1,
                    view_gird_out=100,
                    view_num=8,
                    feats_size=[[bev_w_, bev_h_]],
                ),
                crossattention=dict(
                    type="HorizonSpatialCrossAttention",
                    view_num=8,
                    deformable_attention=dict(
                        type="HorizonMSDeformableAttention3D",
                        embed_dims=_dim_,
                        num_points=8,
                        num_levels=_num_levels_,
                        view_gird_in=160,
                        view_gird_out=100,
                        feats_size=[[25, 15]],
                    ),
                    embed_dims=_dim_,
                ),
                dropout=0.1,
            ),
        ),
    ),
    bev_decoders=[
        dict(
            type="BEVFormerDetDecoder",
            bev_h=bev_h_,
            bev_w=bev_w_,
            num_query=900,
            embed_dims=_dim_,
            pc_range=point_cloud_range,
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                decoder_layer=dict(
                    type="DetrTransformerDecoderLayer",
                    crossattention=dict(
                        type="HorizonMSDeformableAttention",
                        embed_dims=_dim_,
                        num_levels=1,
                        view_gird_out=6,
                        view_gird_in=4,
                        feats_size=[[bev_w_, bev_h_]],
                    ),
                    dropout=0.1,
                ),
            ),
            criterion=dict(
                type="BevFormerCriterion",
                assigner=dict(
                    type="BevFormerHungarianAssigner3D",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                ),
                loss_cls=dict(
                    type="FocalLoss",
                    loss_name="cls",
                    num_classes=num_classes + 1,
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    reduction="mean",
                ),
                loss_bbox=dict(
                    type="L1Loss",
                    loss_weight=0.25,
                ),
                pc_range=point_cloud_range,
            ),
            post_process=dict(
                type="BevFormerProcess",
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=point_cloud_range,
                max_num=300,
                num_classes=10,
            ),
        ),
    ],
)
test_model = copy.deepcopy(model)

test_model["view_transformer"]["queue_length"] = 1

deploy_model = copy.deepcopy(model)

deploy_model["view_transformer"]["queue_length"] = 1
deploy_model["view_transformer"]["is_compile"] = True
# deploy_model["is_compile"] = True
deploy_model["bev_decoders"][0]["is_compile"] = True
deploy_model["bev_decoders"][0].pop("criterion")
deploy_model["bev_decoders"][0].pop("post_process")

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesBevSequenceDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        map_size=map_size,
        map_path=meta_rootdir,
        with_bev_bboxes=False,
        with_ego_bboxes=True,
        bev_range=bev_range,
        num_seq=3,
        transforms=[
            dict(type="MultiViewsImgResize", size=(450, 800)),
            dict(
                type="MultiViewsImgTransformWrapper",
                transforms=[
                    dict(
                        type="TorchVisionAdapter",
                        interface="ColorJitter",
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                    ),
                    dict(type="PILToTensor"),
                    dict(type="Pad", divisor=32),
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequencev2,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesBevSequenceDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        map_size=map_size,
        map_path=meta_rootdir,
        with_bev_bboxes=False,
        with_ego_bboxes=True,
        bev_range=bev_range,
        num_seq=1,
        transforms=[
            dict(type="MultiViewsImgResize", size=(450, 800)),
            dict(
                type="MultiViewsImgTransformWrapper",
                transforms=[
                    dict(type="PILToTensor"),
                    dict(type="Pad", divisor=32),
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
    ),
    sampler=None,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequencev2,
)


def loss_collector(outputs: dict):
    losses = []
    for output in outputs:
        for _, loss in output.items():
            losses.append(loss)
    return losses


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    enable_amp=True,
    enable_amp_dtype=torch.float16,
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


def update_loss(metrics, batch, model_outs):
    for model_out in model_outs:
        for metric in metrics:
            metric.update(model_out)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=lossshow,
    epoch_log_freq=1,
    log_prefix="",
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)


def update_val_metric(metrics, batch, model_outs):
    # Convert one hot to inde
    preds = model_outs
    metrci_gt = {}
    metrci_gt["meta"] = batch["seq_meta"][0]["meta"]
    metrics[0].update(metrci_gt, preds)


val_metric_updater = dict(
    type="MetricUpdater",
    # metrics=[val_nuscenes_metric],
    metric_update_func=update_val_metric,
    step_log_freq=1000000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

val_callback = dict(
    type="Validation",
    val_interval=4,
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=test_model,
    val_on_train_end=True,
    init_with_train_model=True,
)
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=4,
    strict_match=True,
    mode="max",
    # best_refer_metric=val_nuscenes_metric,
    monitor_metric_key="NDS",
)
grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=35,
    clip_norm_type=2,
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
                    "./tmp_pretrained_models/resnet50_imagenet/float-checkpoint-best.pth.tar"  # noqa: E501
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={
            "backbone": dict(lr_mult=0.1),
        },
        lr=4e-4,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    device=None,
    num_epochs=24,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=500,
            stop_lr=2e-4 * 1e-3,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=[
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_interval"] = 1
calibration_val_callback["val_on_train_end"] = False
calibration_step = 10
calibration_val_callback["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2Calibration", convert_mode=convert_mode),
    ],
)
calibration_ckpt_callback = copy.deepcopy(ckpt_callback)
calibration_ckpt_callback["save_interval"] = 1

online_ckpt = "/cluster_home/plat_gpu/hobot-dag-5325981_bevformer-r50-newattention-fp16-20231123-155634/output/models/bevformer_resnet_3d"
calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                # checkpoint_path=os.path.join(
                #     online_ckpt, "float-checkpoint-best.pth.tar"
                # ),
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
                verbose=True,
                allow_miss=True,
            ),
            dict(type="Float2Calibration", convert_mode=convert_mode),
            dict(
                type="FixWeightQScale",
            ),
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        stat_callback,
        calibration_val_callback,
        calibration_ckpt_callback,
    ],
    val_metrics=[
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
    log_interval=calibration_step / 10,
)
qat_val_callback = copy.deepcopy(val_callback)
qat_val_callback["val_interval"] = 1
qat_val_callback["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
    ],
)
qat_ckpt_callback = copy.deepcopy(ckpt_callback)
qat_ckpt_callback["save_interval"] = 1
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="FixWeightQScale",
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        lr=2e-5,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    device=None,
    num_epochs=10,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[5],
            step_log_interval=500,
        ),
        qat_val_callback,
        qat_ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=[
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
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
            dict(type="QAT2Quantize", convert_mode=convert_mode),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
)

float_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

calibration_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(type="Float2Calibration", convert_mode=convert_mode),
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
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)


qat_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)


int_infer_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(type="Float2QAT", convert_mode=convert_mode),
            # dict(
            #     type="LoadCheckpoint",
            #     checkpoint_path="./tmp_models/bevformer_resnet_3d/qat-checkpoint-best-be8937f9.pth.tar",
            #     ignore_extra=True,
            #     verbose=True,
            #     allow_miss=True,
            # ),
            dict(type="QAT2Quantize", convert_mode=convert_mode),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="NuscenesMetric",
            data_root=meta_rootdir,
            version="v1.0-trainval",
            use_lidar=False,
            classes=CLASSES,
            save_prefix="./WORKSPACE/results" + task_name,
            use_ddp=False,
        ),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

deploy_inputs = dict(
    img=torch.randn((6, 3, 480, 800)),
    prev_bev=torch.randn((1, 2500, 256)),
    hybird_ref_2d=torch.randn((2, 2500, 1, 2)),
    reference_points_cam=torch.randn((6, 1, 2500, 4, 2)),
    prev_bev_ref=torch.randn((1, 50, 50, 2)),
)
deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
        dict(type="QAT2Quantize", convert_mode=convert_mode),
    ],
)
compile_dir = os.path.join(ckpt_dir, "compile_view_trans")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
    opt="O0",
)

onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
)
