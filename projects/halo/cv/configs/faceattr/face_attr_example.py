import copy
import os

import torch
from horizon_plugin_pytorch.quantization import March

training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "face_attr_based_on_0406faceid_ldmkaug_loadlr0-01"

batch_size_per_gpu = 8
device_ids = [
    0,
]

ckpt_dir = "./tmp_models/%s" % task_name
log_dir = "./tmp_models/%s/log" % task_name
log_freq = 50
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.BAYES


age_bins = [6, 12, 18, 28, 35, 45, 55]


train_rec_list = [
    "train_test_recs/train_feitian_112_112_3.rec",
    "train_test_recs/train_new_student_sample_3k_112_112_3.rec",
    "train_test_recs/train_web_spider_sample_all_112_112_3.rec",
    "train_test_recs/train_shujutang_child_112_112_3.rec",
    "train_test_recs/train_from_emotion_zhikai_20211115_20ID_rgb_1k.rec",
    "train_test_recs/train_from_gaze_jiangxiao_1025_1109_RGB_50ID_11k.rec",
]

test_rec_list = [
    "train_test_recs/val_yft_all_112_112_3_wt_align.rec",
    "train_test_recs/test_from_shuxin.rec",
    "train_test_recs/nitao_dms_normal.rec",
    "train_test_recs/nitao_ims_normal.rec",
]

# faceid_checkpoint = 'faceid-0406.pth.tar'
faceid_checkpoint = (
    "tmp_models/face_attr_based_on_0406faceid/float-checkpoint-last.pth.tar"
)
# faceid_checkpoint = None


deploy_inputs = dict(img=torch.randn((1, 3, 112, 112)))

model = dict(
    type="FaceAttrClassifier",
    backbone=dict(
        type="FaceIDLargeVargNet",
        bn_kwargs=dict(eps=1e-3, momentum=0.01),
        bias=False,
        embedding_size=256,
        dropout=0.0,
        use_fp16=False,
        include_all=True,
        node_name="backbone",
    ),
    head=dict(
        type="FaceAttrHead",
        in_channels=768,
        age_classes=85,
        gender_classes=1,
        bn_kwargs={},
    ),
    loss_gender=dict(
        type="CrossEntropyLoss",
        reduction="sum",
        use_sigmoid=True,
    ),
    loss_age=dict(
        type="CostSensitiveLoss",
        num_classes=85,
        error_max=3,
        from_sigmoid=False,
    ),
)


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="FaceAttrRecDataset",
        imgrec_path_list=train_rec_list,
        age_classes=85,
        transforms=[
            dict(
                type="RandomShiftRotateScale",
                rotate_prob=0.3,
                max_rotate_angle=5,
                shift_prob=0.1,
                max_shift_range=(-0.01, 0.01),
                resize=True,
            ),
            dict(
                type="RandomColorJitter",
                brightness=0.1,
                contrast=0.02,
                saturation=0.02,
                hue=0.0,
                prob=0.2,
            ),
            dict(
                type="RandomGray",
                p=0.5,
                rgb_data=True,
            ),
            dict(
                type="TransformAgeLabel",
                age_classes=85,
            ),
            dict(
                type="RandomOcclusion",
                occ_type="forehead",
                prob=0.3,
            ),
            dict(
                type="ToTensor",
                to_yuv=False,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    # drop_last=True,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="FaceAttrRecDataset",
        imgrec_path_list=test_rec_list,
        age_classes=85,
        transforms=[
            dict(
                type="ToTensor",
                to_yuv=True,
            ),
            dict(
                type="TorchVisionAdapter",
                interface="CenterCrop",
                size=112,
            ),
            dict(
                type="TransformAgeLabel",
                age_classes=85,
            ),
            dict(
                type="TorchVisionAdapter",
                interface="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)


def loss_collector(outputs):
    return [outputs["losses"]]


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    batch_transforms=[
        dict(
            type="TorchVisionAdapter",
            interface="RandomHorizontalFlip",
            p=0.5,
        ),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)

train_metrics = [
    dict(type="LossShow", name="losses"),
    dict(type="LossShow", name="loss_age"),
    dict(type="LossShow", name="loss_gender"),
    dict(type="CumulativeAccuracy", offset=5, classes=85, name="gap5acc"),
    dict(type="SigmoidOrdinalMAE", name="age_mae"),
    dict(type="Accuracy", name="gender_acc"),
]


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        if "loss" in metric.name:
            if "age" in metric.name:
                metric.update(model_outs["loss_age"])
            elif "gender" in metric.name:
                metric.update(model_outs["loss_gender"])
            else:
                metric.update(model_outs["losses"])
        elif metric.name in ["gap5acc", "age_mae"]:
            metric.update(model_outs)
        elif "gender" in metric.name:
            labels = batch["gender"]
            preds = model_outs["pred_gender"] > 0
            assert preds.shape == labels.shape
            metric.update(labels, preds)
        else:
            assert 0


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix=task_name,
)

val_metrics = [
    dict(type="CumulativeAccuracy", offset=5, classes=85, name="gap5acc"),
    dict(type="SigmoidOrdinalMAE", name="age_mae"),
    dict(
        type="SigmoidOrdinalAccuracy",
        offset=0,
        age_bins=age_bins,
        name="age_seg_acc_0",
    ),
    dict(
        type="SigmoidOrdinalAccuracy",
        offset=1,
        age_bins=age_bins,
        name="age_seg_acc_1",
    ),
    dict(type="Accuracy", name="gender_acc"),
]


def val_update_metric(metrics, batch, model_outs):
    for metric in metrics:
        if metric.name in [
            "gap5acc",
            "age_mae",
            "age_seg_acc_0",
            "age_seg_acc_1",
        ]:
            metric.update(model_outs)
        elif "gender" in metric.name:
            labels = batch["gender"]
            preds = model_outs["pred_gender"] > 0
            assert preds.shape == labels.shape
            metric.update(labels, preds)
        else:
            assert 0


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=val_update_metric,
    step_log_freq=-1,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
    # save_hash=False,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_interval=10,
    val_model=None,
)

freezn_face_callback = dict(
    type="FreezeModule",
    modules=[["backbone"]],
    step_or_epoch=[0],
    update_by="epoch",
    only_batchnorm=False,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=1e-4)},
        lr=0.005,
    ),
    batch_processor=batch_processor,
    num_epochs=30,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[10, 20],
            lr_decay_factor=0.1,
            warmup_by="step",
            warmup_len=200,
            step_log_interval=500,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
        freezn_face_callback,
    ],
    train_metrics=train_metrics,
    val_metrics=val_metrics,
)
if faceid_checkpoint is not None:
    float_trainer.update(
        dict(
            model_convert_pipeline=dict(
                type="ModelConvertPipeline",
                converters=[
                    dict(
                        type="LoadCheckpoint",
                        checkpoint_path=faceid_checkpoint,
                        allow_miss=True,
                        ignore_extra=True,
                        verbose=1,
                    )
                ],
            )
        )
    )


deploy_model = copy.deepcopy(model)
deploy_model["deploy"] = True

float_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(deploy_model),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[val_metric_updater],
    metrics=val_metrics,
    log_interval=50,
)

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
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=1e-4)},
        lr=0.001,
    ),
    batch_processor=batch_processor,
    num_epochs=10,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=500,
            lr_decay_id=[5, 10],
            lr_decay_factor=0.1,
            warmup_by="step",
            warmup_len=2000,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=train_metrics,
    val_metrics=val_metrics,
)


int_infer_trainer = dict(
    type="Trainer",
    model=model,
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
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[ckpt_callback, trace_callback],
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
)
