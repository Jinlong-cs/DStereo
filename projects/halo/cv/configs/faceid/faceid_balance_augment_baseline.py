import copy
import os

import torch
from hatbc.filestream.bucket.client import get_bucket_client
from horizon_plugin_pytorch.quantization import March

from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "faceid_cls"
num_classes = 500000
batch_size_per_gpu = 80
embedding_size = 256
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
world_size = 8
cudnn_benchmark = True
enable_amp = False
seed = None
log_rank_zero_only = True
march = March.BAYES

local = False
backbone_type = "FaceIDLargeVargNet"
bucket_client = get_bucket_client()

train_data_url = "dmpv2://interaction/active/faceid/train/baseline_2030_V0.2/"
train_data_root = bucket_client.url_to_local(train_data_url)
test_data_url = "dmpv2://interaction/active/faceid/test/"
test_data_root = bucket_client.url_to_local(test_data_url)

if local:
    ckpt_url = ""
    ckpt_dir = bucket_client.url_to_local(ckpt_url)
    num_workers = 12
else:
    ckpt_url = ""
    ckpt_dir = bucket_client.url_to_local(ckpt_url)
    job_id = f"{backbone_type}_reproduce_baseline"
    ckpt_dir = os.path.join(ckpt_dir, job_id)
    num_workers = 5


# ckpt_dir = f"{root_dir}/{task_name}"

model = dict(
    type="Classifier",
    backbone=dict(
        type=backbone_type,
        bn_kwargs=dict(eps=1e-3, momentum=0.01),
        bias=False,
        embedding_size=embedding_size,
        dropout=0.0,
        use_fp16=enable_amp,
    ),
    losses=dict(
        type="DistFCCrossEntropyLoss",
        resume=False,
        margin_loss=dict(
            type="ArcFace",
            margin_arc=0.6,
            margin_am=0.0,
            scale=64.0,
        ),
        num_classes=num_classes,
        embedding_size=embedding_size,
        gpu_per_device=8,
    ),
)
deploy_model = dict(
    type="Classifier",
    backbone=dict(
        type=backbone_type,
        bn_kwargs=dict(eps=1e-3, momentum=0.01),
        bias=False,
        embedding_size=embedding_size,
        dropout=0.0,
    ),
)
deploy_inputs = dict(img=torch.randn((1, 3, 112, 112)))

val_model = dict(
    type="Classifier",
    backbone=dict(
        type=backbone_type,
        bn_kwargs=dict(eps=1e-3, momentum=0.01),
        bias=False,
        embedding_size=embedding_size,
        dropout=0.0,
    ),
)


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="DeepInsightRecordDataset",
        rec_path=os.path.join(train_data_root, "baseline_2030_V0.2.rec"),
        idx_path=os.path.join(train_data_root, "baseline_2030_V0.2.idx"),
        unpack64=True,
        transforms=[
            dict(
                type="RandomFlip",
            ),
            dict(
                type="RandomGray",
            ),
            dict(
                type="JPEGCompress",
                p=0.3,
            ),
            dict(
                type="SpatialVariantBrightness",
                p=0.3,
                brightness=0.5,
            ),
            dict(type="RandomDownSample", p=0.05),
            dict(
                type="MotionBlur",
                p=0.05,
            ),
            dict(
                type="GaussianBlur",
                p=0.15,
            ),
            dict(
                type="ToTensor",
            ),
        ],
    ),
    sampler=dict(
        type="DistributedFaceIDBalanceSampler",
        bounds=[8, 10, 8],
        replace=False,
        drop_last=True,
        shuffle=True,
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="DeepInsightRecordDataset",
        rec_path="",
        idx_path="",
        unpack64=False,
        transforms=[
            dict(
                type="ToTensor",
            ),
        ],
    ),
    sampler=dict(
        type=torch.utils.data.distributed.DistributedSampler,
        drop_last=False,
        shuffle=False,
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

valid_data_loader = copy.deepcopy(val_data_loader)
valid_data_loader["dataset"]["rec_path"] = os.path.join(
    test_data_root, "ValID/wanren_V0.2_indexed.rec"
)
valid_data_loader["dataset"]["idx_path"] = os.path.join(
    test_data_root, "ValID/wanren_V0.2_indexed.idx"
)

vallife_data_loader = copy.deepcopy(val_data_loader)
vallife_data_loader["dataset"]["rec_path"] = os.path.join(
    test_data_root, "ValLife/valLife_V0.2_indexed.rec"
)
vallife_data_loader["dataset"]["idx_path"] = os.path.join(
    test_data_root, "ValLife/valLife_V0.2_indexed.idx"
)


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    enable_amp=enable_amp,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_index(1),
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


def update_metric(metrics, batch, model_outs):
    _, losses = model_outs
    for metric in metrics:
        metric.update(losses)


def faceid_feature_update(metrics, batch, model_outs, norm=True):
    features, _ = model_outs

    for metric in metrics:
        if norm:
            features = torch.nn.functional.normalize(features)

        labels = batch["labels"]
        metric.update(labels, features)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=100,
    epoch_log_freq=1,
    log_prefix=task_name,
)


eval_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=faceid_feature_update,
    step_log_freq=0,
    epoch_log_freq=1,
    log_prefix=task_name,
)


stat_callback = dict(
    type="StatsMonitor",
    log_freq=100,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode=None,
    save_hash=False,
    only_save_ddp=True,
)
dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=[5.0],
),

model_convert_pipeline = (
    dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir,
                    # "float-checkpoint-last.pth.tar"
                    "float-checkpoint-last.pth.tar"
                    # "float-checkpoint-epoch-0002.pth.tar"
                ),
                ignore_extra=True,
                allow_miss=True,
                verbose=True,
            ),
        ],
    ),
)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={
            "backbone": dict(weight_decay=2e-4),
            "losses": dict(weight_decay=2e-4),
        },
        lr=0.15,
        momentum=0.9,
    ),
    convert_submodule_list=["backbone"],
    batch_processor=batch_processor,
    stop_by="epoch",
    num_epochs=70,
    device=None,
    find_unused_parameters=False,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            update_by="epoch",
            step_log_interval=100,
            warmup_by="step",
            warmup_len=2800,
            lr_decay_id=[30, 45, 50, 55, 60, 65],
            lr_decay_factor=0.25,
        ),
        dict(
            type="GradScale",
            module_and_scale=[],
            clip_grad_norm=5.0,
        ),
        metric_updater,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
    ],
    val_metrics=None,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=30,
    device=None,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            step_log_interval=1000,
            lr_decay_id=[15, 25],
            lr_decay_factor=0.1,
        ),
        ckpt_callback,
    ],
)

int_trainer = dict(
    type="Trainer",
    model=model,
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[ckpt_callback],
    val_metrics=[dict(type="Accuracy")],
)
dict(
    type="FaceIDGARMetrics",
    total_num=19897,
    save_dir="./tmp",
    name="valid",
),


# predictor
float_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir,
                    "float-checkpoint-last.pth.tar"
                    # "float-checkpoint-epoch-0048.pth.tar"
                    # "float-checkpoint-step-6499.pth.tar"
                ),
                ignore_extra=True,
                allow_miss=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[
        valid_data_loader,
        vallife_data_loader,
    ],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="FaceIDGARMetrics",
            total_num_list=[19897, 15498],
            save_dir="./tmp",
            name_list=["valid", "vallife"],
        ),
    ],
    callbacks=[
        eval_metric_updater,
    ],
    log_interval=50,
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

onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
    kwargs=dict(
        # verbose=True,
        opset_version=11,
    ),
)
