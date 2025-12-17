import os
from importlib import import_module

import torch

# -------------------------- common --------------------------
from common import (
    batch_transforms,
    ckpt_dir,
    device_ids,
    input_hw,
    task_to_module,
    tasks,
    training_step,
)
from horizon_plugin_pytorch.quantization import March

from hat.core.task_sampler import TaskSampler
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.apply_func import _as_list
from hat.utils.config import ConfigVersion

# -------------------------- multitask --------------------------
seed = None
log_rank_zero_only = True
cudnn_benchmark = True
sync_bn = False
enable_amp = False
test_batch_size_per_gpu = 64
float_lr = 1e-3
float_num_epoch = 10
freeze_bn_lr = 1e-4
freeze_bn_num_epoch = 5
qat_lr = 1e-4
qat_num_epoch = 5


VERSION = ConfigVersion.v2
# march = March.BERNOULLI  # J2
# march = March.BERNOULLI2  # J3
march = March.BAYES  # J5
job_name = "facequality_mixvargenet0.75_size160_hat"
device_ids = device_ids

ckpt_dir = ckpt_dir + "/" + job_name

inputs = dict(img=torch.zeros((2, 3, *input_hw)))
test_inputs = dict(img=torch.randn((1, 3, *input_hw)))
deploy_inputs = test_inputs
opt_inputs = dict(
    train=dict(),
    val=dict(),
    test=dict(),
    deploy=dict(),
)

# -------------------------- pretrain model --------------------------
pretrain_model_path = ""

# -------------------------- train data --------------------------
task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(task_to_module[t]) for t in task_names]

train_task_sampler_config = {
    T.task_name: dict(sampling_factor=1) for T in TASK_CONFIGS
}

train_task_sampler = TaskSampler(
    task_config=train_task_sampler_config, method="sample_all"
)

train_loaders = {T.task_name: T.train_data_loader for T in TASK_CONFIGS}
train_data_loader = dict(
    type="MultitaskLoader",
    loaders=train_loaders,
    task_sampler=train_task_sampler,
    return_task=True,
    mode="max_size",
    __build_recursive=True,
)


# -------------------------- model --------------------------
def get_model(mode):
    return dict(
        type="MultitaskGraphModel",
        inputs=inputs,
        opt_inputs=opt_inputs[mode],
        task_inputs={T.task_name: T.inputs[mode] for T in TASK_CONFIGS},
        task_modules={T.task_name: T.get_model(mode) for T in TASK_CONFIGS},
        funnel_modules=None,
        flatten_outputs=True if mode == "deploy" else False,
        lazy_forward=False,
    )


train_model = get_model("train")
deploy_model = get_model("deploy")


# ----------------- callbacks ---------------------------
# train callbacks
stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)

lr_callback = dict(
    type="StepDecayLrUpdater",
    lr_decay_id=[8, 40],
    lr_decay_factor=0.1,
    warmup_by="step",
    warmup_len=1000,
    step_log_interval=500,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    save_hash=False,
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=test_inputs,
)

grad_callback = dict(
    type="GradCheck",
    param_name="facemtl_backbone.mod5.3.out_conv.1.1",
)

train_metric_updaters = [T.metric_updater for T in TASK_CONFIGS]
callbacks = [
    stat_callback,
    lr_callback,
    ckpt_callback,
]
callbacks += train_metric_updaters
# callbacks += [grad_callback]
train_metrics = []
for T in TASK_CONFIGS:
    train_metrics += _as_list(T.train_metrics)


# -------------------------- training ----------------------------
# -------------------------- step float --------------------------
train_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=batch_transforms,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
)


float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=train_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=pretrain_model_path,
                allow_miss=True,
                ignore_extra=False,
                verbose=True,  # Show unexpect_key and miss_key info.  # noqa
                check_hash=False,
            ),
        ],
    )
    if pretrain_model_path
    else None,
    data_loader=train_data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=5e-4)},
        lr=float_lr,
    ),
    # find_unused_parameters=False,
    batch_processor=train_batch_processor,
    stop_by="epoch",
    # num_steps=1,
    num_epochs=float_num_epoch,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=callbacks,
    train_metrics=train_metrics,
    # profiler=profiler,
    # resume_optimizer=True,
    # resume_epoch_or_step=True,
)

# ------------------------ step freeze bn ------------------
freeze_bn_1_modules = ["facemtl_backbone"]

freeze_bn_1_callback = dict(
    type="FreezeModule",
    modules=[freeze_bn_1_modules],
    step_or_epoch=[0],
    update_by="epoch",
    only_batchnorm=True,
)

freeze_bn_lr_callback = dict(
    type="StepDecayLrUpdater",
    lr_decay_id=[3, 5],
    lr_decay_factor=0.1,
    step_log_interval=500,
)

freeze_bn_callback = [
    stat_callback,
    freeze_bn_lr_callback,
    ckpt_callback,
]
freeze_bn_callback += train_metric_updaters
freeze_bn_callback = freeze_bn_callback + [freeze_bn_1_callback]

freeze_bn_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=train_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=False,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=train_data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=5e-4)},
        lr=freeze_bn_lr,
    ),
    # find_unused_parameters=False,
    batch_processor=train_batch_processor,
    stop_by="epoch",
    # num_steps=5,
    num_epochs=freeze_bn_num_epoch,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=freeze_bn_callback,
    train_metrics=train_metrics,
    # profiler=profiler,
    # resume_optimizer=True,
    # resume_epoch_or_step=True,
)

# ------------------------ step qat ------------------------
qat_lr_callback = dict(
    type="StepDecayLrUpdater",
    lr_decay_id=[3, 50],
    lr_decay_factor=0.1,
    step_log_interval=500,
)
qat_callbacks = [
    stat_callback,
    qat_lr_callback,
    ckpt_callback,
]
qat_callbacks += train_metric_updaters


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=train_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "freeze_bn-checkpoint-last.pth.tar"
                ),
            ),
            dict(type="Float2QAT"),
        ],
    ),
    # find_unused_parameters=False,
    data_loader=train_data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=5e-4)},
        lr=qat_lr,
    ),
    batch_processor=train_batch_processor,
    stop_by="epoch",
    num_epochs=qat_num_epoch,
    # stop_by="step",
    # num_steps=1,
    device=None,
    callbacks=qat_callbacks,
    train_metrics=train_metrics,
)


# -------------------------- step int_infer --------------------------
int_infer_trainer = dict(
    type="Trainer",
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir,
                    "qat-checkpoint-last.pth.tar"
                    # ckpt_dir, "qat-checkpoint-epoch-0003.pth.tar"
                ),
                ignore_extra=True,
                verbose=True,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[trace_callback],
)


compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=job_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    # layer_details=True,
    # output_layout="NHWC",
    input_source=["pyramid"],
    opt="O3",
)


onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=freeze_bn_trainer["model_convert_pipeline"],
    kwargs=dict(
        # verbose=True,
        opset_version=11,
    ),
)
