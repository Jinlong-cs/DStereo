import copy
import os
import random

import torch
from horizon_plugin_pytorch.march import March

from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.utils.config import ConfigVersion
from tests.data.toy_resnet18 import ToyResNet18Classifier  # noqa: F403,F401

enable_model_tracking = True

VERSION = ConfigVersion.v2
pipeline_test = True
training_step = os.environ.get("HAT_TRAINING_STEP", "float")
sensitive_op_qconfig_test = False

task_name = "resnet18_cls"
num_classes = 1000
batch_size_per_gpu = 64 if not pipeline_test else 2
device_ids = (
    list(range(torch.cuda.device_count())) if not pipeline_test else [0]
)  # noqa
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = os.getpid()
log_rank_zero_only = True  # to simplify logging info
march = March.BAYES
qat_mode = "fuse_bn"
convert_mode = "eager"

# see `horizon.quantization.perf_model`
compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name="cls_test_model",  # Name of the model, recorded in hbm
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],  # or ddr? custom by yourself
    opt="O1",
    save_qresults=True,
)

# -------------------------- model --------------------------
# model is train model, with loss, can not used for test, so we need test_model
model = dict(
    type="ToyResNet18Classifier",
    num_classes=num_classes,
    quanti_head=True,
    loss=dict(type="CEWithLabelSmooth"),
)
deploy_model = dict(
    type="ToyResNet18Classifier",
    num_classes=num_classes,
    quanti_head=True,
    loss=None,
)
deploy_inputs = dict(img=torch.randn((1, 3, 224, 224)))
deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT", convert_mode=convert_mode),
        dict(type="QAT2Quantize", convert_mode=convert_mode),
    ],
)

# -------------------------- data --------------------------
if not pipeline_test:
    data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="ImageNet",
            data_path="./tmp_data/imagenet/train_lmdb/",
            transforms=[
                dict(
                    type="TorchVisionAdapter",
                    interface="RandomResizedCrop",
                    size=224,
                    scale=(0.08, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                ),
                dict(
                    type="TorchVisionAdapter", interface="RandomHorizontalFlip"
                ),
                dict(
                    type="TorchVisionAdapter",
                    interface="ConvertImageDtype",
                    dtype=torch.float32,
                ),
                dict(
                    type="TorchVisionAdapter",
                    interface="Normalize",
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="ImageNet",
            data_path="./tmp_data/imagenet/val_lmdb/",
            transforms=[
                dict(type="TorchVisionAdapter", interface="Resize", size=256),
                dict(
                    type="TorchVisionAdapter", interface="CenterCrop", size=224
                ),
                dict(
                    type="TorchVisionAdapter",
                    interface="ConvertImageDtype",
                    dtype=torch.float32,
                ),
                dict(
                    type="TorchVisionAdapter",
                    interface="Normalize",
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
else:
    data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="RandDataset",
            length=batch_size_per_gpu * len(device_ids) * 2,
            example=dict(
                img=torch.randn((3, 224, 224)),
                labels=random.randint(0, num_classes - 1),
            ),
            clone=True,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_data_loader = dict(
        type=torch.utils.data.DataLoader,
        dataset=dict(
            type="RandDataset",
            length=batch_size_per_gpu * len(device_ids) * 2,
            example=dict(
                img=torch.randn((3, 224, 224)),
                labels=random.randint(0, num_classes - 1),
            ),
            clone=True,
        ),
        sampler=dict(type=torch.utils.data.DistributedSampler),
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

# -------------------------- callbacks --------------------------
profiler = dict(type="SimpleProfiler")
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_index(
        1
    ),  # model_outs=(preds, losses), refer `Classifier`  # noqa
    # or you can:
    # loss_collector=collect_loss_by_regex('1')
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=collect_loss_by_index(1),
    # or you can:
    # loss_collector=collect_loss_by_regex('1')
)


def update_metric(metrics, batch, model_outs):
    target = batch["labels"]
    preds, losses = model_outs
    for metric in metrics:
        metric.update(target, preds)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    # or you can:
    # metric_update_func=update_metric_using_index(
    #     per_metric_idxs=[
    #         # index of label in batch, index of prediction in model_outs
    #         dict(label_idx=1, pred_idx=0),
    #         dict(label_idx=1, pred_idx=0),
    #     ]
    # ),
    step_log_freq=10 if not pipeline_test else 1,
    epoch_log_freq=1,
    log_prefix=task_name,
)
val_metric_updater = copy.deepcopy(metric_updater)
val_metric_updater["log_prefix"] = "Validation " + task_name

stat_callback = dict(
    type="StatsMonitor",
    log_freq=10 if not pipeline_test else 1,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,  # model params match test_model params
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
    val_model=None,  # use train model as val model
    profiler=profiler,
    val_on_train_end=False,  # if True, profiler will raise exception
)


aidi_exp_model_callback = dict(
    type="AIDIExperimentManager",
    model_name="hat_integration_test-resnet18_cls",
    model_version="v0.0.1",
    save_model="last",
    upload_progressive_checkpoint=True,
    overwrite_file=True,
)

if enable_model_tracking or bool(enable_model_tracking):
    exp_callbacks = [aidi_exp_model_callback]
else:
    exp_callbacks = []

# -------------------------- solver --------------------------
# -------------------------- step float ----------------------
float_trainer = dict(
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
    num_epochs=30 if not pipeline_test else 1,
    device=None,  # set in train.py
    callbacks=[
        # the order of callbacks affects the logging order
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=0,
            lr_decay_id=[15, 25],
            step_log_interval=10,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ]
    + exp_callbacks,
    train_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    profiler=profiler,
    # convert_submodule_list=["backbone"],
)

calibration_data_loader = copy.deepcopy(data_loader)
if "sampler" in calibration_data_loader:
    del calibration_data_loader["sampler"]
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
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
    num_steps=1,
    device=None,
    callbacks=[
        val_callback,
        ckpt_callback,
    ],
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    log_interval=1,
)

# -------------------------- step quantize -------------------
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0.0,  # 0.0 ~ 1.0 之间的浮点数
            ),
            weight_qat_qkwargs=dict(
                averaging_constant=1.0,  # 0.0 ~ 1.0 之间的浮点数
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
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.0001,  # Custom by yourself #
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=10 if not pipeline_test else 1,
    device=None,  # set in train.py
    callbacks=[
        # the order of callbacks affects the logging order
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=0,
            lr_decay_id=[3, 6],
            step_log_interval=10,
        ),
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    # convert_submodule_list=["backbone"],
)

# just for saving int_infer pth and pt
int_infer_trainer = dict(
    type="Trainer",
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
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
    # export int model and test_model only, not need training, so others like
    # loader is unnecessary
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,  # will skip training, and run `loop.on_loop_begin/end()`
    device=None,
    callbacks=[
        # export only
        ckpt_callback,
        trace_callback,
    ],
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
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
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
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

int_infer_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
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
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

profile_dir = os.path.join(ckpt_dir, "profiler")
featuremap_similarity_tool = dict(
    type="FeaturemapSimilarity",
    similarity_func="Cosine",
    threshold=None,
    devices=torch.device("cpu"),
    out_dir=profile_dir,
)

profile_featuremap_tool = dict(
    type="ProfileFeaturemap",
    device=torch.device("cpu"),
    with_tensorboard=False,
    tensorboard_dir=None,
    print_per_channel_scale=False,
)

check_shared_tool = dict(
    type="CheckShared",
    check_leaf_module=None,
    print_tabulate=True,
)

check_fused_tool = dict(
    type="CheckFused",
    print_tabulate=True,
)

check_qconfig_tool = dict(
    type="CheckQConfig",
)

compare_weights_tool = dict(
    type="CompareWeights",
    similarity_func="L1",
)

deploy_device_tool = dict(
    type="CheckDeployDevice",
    print_tabulate=True,
)

model_profiler_tool = dict(
    type="ModelProfiler",
    mode="QvsQ",
    kwargs_dict=dict(
        FeaturemapSimilarity=dict(devices=torch.device("cpu")),
        ProfileFeaturemap=dict(
            device=torch.device("cpu"),
            print_per_channel_scale=True,
        ),
    ),
)

model_profiler_v2_tool = dict(type="ModelProfilerv2")

hbir_profiler_tool = dict(type="HbirModelProfiler")

similarity_convert_pipeline = [
    qat_predictor["model_convert_pipeline"],
    int_infer_predictor["model_convert_pipeline"],
]
other_convert_pipeline = float_predictor["model_convert_pipeline"]
fx_pipeline = qat_predictor["model_convert_pipeline"]
fx_pipeline["converters"][0]["convert_mode"] = "fx"

hbir_pipeline = (
    dict(
        type="ModelConvertPipeline",
        converters=[
            dict(type="LoadHbir", path=os.path.join(ckpt_dir, "qat.bc")),
        ],
    ),
)

model_profiler_solver = dict(
    model=deploy_model,
    inputs=deploy_inputs,
    model_convert_pipeline=similarity_convert_pipeline,
    tool=featuremap_similarity_tool,
)

quant_analysis_solver = dict(
    type="QuantAnalysis",
    model=deploy_model,
    device_id=0,
    dataloader=val_data_loader,
    num_steps=10,
    baseline_model_convert_pipeline=float_predictor["model_convert_pipeline"],
    analysis_model_convert_pipeline=qat_predictor["model_convert_pipeline"],
    analysis_model_type="fake_quant",
    out_dir=os.path.join(ckpt_dir, "float_qat_analysis"),
)

if sensitive_op_qconfig_test:
    from horizon_plugin_pytorch.quantization.qconfig_template import (
        sensitive_op_8bit_weight_16bit_act_calibration_setter,
    )

    sensitive_table = torch.load(
        os.path.join(ckpt_dir, "float_qat_analysis", "sensitive_ops.pt")
    )
    calibration_trainer = copy.deepcopy(calibration_trainer)
    calibration_trainer["model_convert_pipeline"]["converters"] = [
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
        ),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            qconfig_setter=sensitive_op_8bit_weight_16bit_act_calibration_setter(  # noqa E501
                sensitive_table, ratio=0.2, override=True
            ),
            example_inputs=deploy_inputs,
        ),
    ]

onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_predictor["model_convert_pipeline"],
)

data_loader_perf = dict(
    type="DataloaderSpeedPerf",
    dataloader=data_loader,
    iter_nums=20,
    frequent=5,
    profiler=dict(type="SimpleProfiler"),
)

model_training_perf = dict(
    type="ModelTrainingPerf",
    trainer=float_trainer,
    iter_nums=20,
    frequent=5,
    profiler=dict(type="SimpleProfiler"),
)
