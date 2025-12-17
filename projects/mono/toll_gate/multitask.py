import copy
import os
from importlib import import_module

# -------------------------- common --------------------------
from common import (
    ckpt_dir,
    docker_image,
    enable_model_tracking,
    float_checkpoint,
    float_lr,
    float_resume_checkpoint,
    float_steps,
    folder_list,
    framework,
    input_bucket,
    interval_by,
    is_local_train,
    job_list,
    job_name,
    job_password,
    launcher,
    log_freq,
    march,
    max_jobtime,
    model_version,
    multitask_task_name,
    num_gpus_per_machine,
    num_machines,
    pretrain_checkpoint,
    priority,
    project_id,
    qat_lr,
    qat_resume_checkpoint,
    qat_steps,
    save_interval,
    task_label,
    tasks,
    training_step,
    upload_folder_name,
    warmup_steps,
)
from models import deploy_inputs, deploy_model, model

from hat.core.task_sampler import TaskSampler
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2
# step-specific args
# -------------------------- multitask --------------------------
log_dir = ckpt_dir / "logs"
redirect_config_logging_path = (
    log_dir / f"config_log_{training_step}.log"
).as_posix()
seed = None
log_rank_zero_only = True
cudnn_benchmark = True
sync_bn = True
enable_amp = False
device_ids = list(range(num_gpus_per_machine))
# -------------------------- task --------------------------
task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]
metric_updaters = [T.metric_updater for T in TASK_CONFIGS]


task_sampler_configs = {
    T.task_name: dict(sampling_factor=1) for T in TASK_CONFIGS
}

task_sampler = TaskSampler(
    task_config=task_sampler_configs,
    method="sample_all",
)

loaders = {T.task_name: T.data_loader for T in TASK_CONFIGS}

data_loader = dict(
    type="MultitaskInfLoader",
    loaders=loaders,
    task_sampler=task_sampler,
    return_task=True,
    __build_recursive=False,
)

# -------------------------- batch processor --------------------------
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    empty_cache=True,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=False),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
    delay_sync=True,
)

# -------------------------- callbacks --------------------------
stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

lr_callback = dict(
    type="PolyLrUpdater",
    max_update=(float_steps - warmup_steps) // num_machines,
    power=1.0,
    warmup_by="step",
    warmup_len=warmup_steps,
    warmup_begin_lr=0.00005,
    step_log_interval=log_freq,
)

checkpoint_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir.as_posix(),
    name_prefix=training_step + "-",
    strict_match=False,
    best_refer_metric=None,
    save_interval=save_interval,
    interval_by=interval_by,
    save_on_train_end=True,
)

tensorboard_callback = dict(
    type="TensorBoard",
    save_dir=log_dir.as_posix() if is_local_train else "/job_tboard/",  # noqa
    update_freq=log_freq,
    tb_update_funcs=[
        T.tb_update_func
        for T in TASK_CONFIGS
        if getattr(T, "tb_update_func", None) is not None
    ],
)

aidi_expmodel_callback = dict(
    type="AIDIExpModel",
    model_name=multitask_task_name,
    task_type="multitask",
    save_model="last",
    model_version=model_version,
    upload_progressive_checkpoint=True,
)

# the order of callbacks affects the logging order
float_callbacks = [
    stat_callback,
    lr_callback,
    checkpoint_callback,
    tensorboard_callback,
    aidi_expmodel_callback,
] + metric_updaters

if enable_model_tracking:
    float_callbacks.append(aidi_expmodel_callback)

profiler = dict(
    type="SimpleProfiler",
    # dirpath=log_dir.as_posix(),
    # filename='profile.log',
)

# -------------------------- trainer --------------------------
float_converters = [
    dict(
        type="LoadCheckpoint",
        checkpoint_path=pretrain_checkpoint,
        allow_miss=True,
        ignore_extra=True,
        check_hash=False,
    ),
    dict(
        type="LoadCheckpoint",
        checkpoint_path=float_resume_checkpoint,
        allow_miss=False,
        ignore_extra=False,
        check_hash=False,
    ),
]

float_trainer = dict(
    type="DistributedDataParallelTrainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=float_converters,
    ),
    data_loader=data_loader,
    optimizer=dict(
        type="LegacyNadamEx",
        params={"weight": dict(weight_decay=5e-4)},
        lr=float_lr * num_machines,
        rescale_grad=1,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=(float_steps + warmup_steps) // num_machines,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=float_callbacks,
    # assign_module_buffers=False,
    # profiler=profiler,
)

# ---------------------------------- qat trainer --------------------
qat_checkpoint_path = (
    float_checkpoint
    if float_checkpoint
    else os.path.join(ckpt_dir, "float-checkpoint-last.pth.tar")
)

qat_converters = [
    dict(
        type="LoadCheckpoint",
        checkpoint_path=qat_checkpoint_path,
        allow_miss=False,
        ignore_extra=True,
        check_hash=False,
        verbose=True,
    ),
    dict(type="Float2QAT"),
    dict(
        type="LoadCheckpoint",
        checkpoint_path=qat_resume_checkpoint,
        allow_miss=False,
        ignore_extra=False,
        check_hash=False,
    ),
]

qat_lr_callback = dict(
    type="PolyLrUpdater",
    max_update=(qat_steps - warmup_steps) // num_machines,
    power=1.0,
    warmup_by="step",
    warmup_len=warmup_steps,
    warmup_begin_lr=0.00005,
    step_log_interval=log_freq,
)

qat_callbacks = [
    stat_callback,
    qat_lr_callback,
    checkpoint_callback,
    tensorboard_callback,
    aidi_expmodel_callback,
] + metric_updaters

qat_trainer = dict(
    type="DistributedDataParallelTrainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=qat_converters,
    ),
    data_loader=data_loader,
    optimizer=dict(
        type="LegacyNadamEx",
        params={"weight": dict(weight_decay=5e-5)},
        lr=qat_lr * num_machines,
        rescale_grad=1,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=(qat_steps + warmup_steps) // num_machines,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=qat_callbacks,
)

# --------------------------- int infer trainer --------------------
trace_callback = dict(
    type="SaveTraced",
    trace_inputs=copy.deepcopy(deploy_inputs),
    save_dir=ckpt_dir,
    name_prefix="int_infer-",
)

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
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[
        checkpoint_callback,
        trace_callback,
    ],
)
# ------------------------------------ hbdk -------------------------
compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=multitask_task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "toll_pole_key_point_lmks.hbm"),
    layer_details=True,
    input_source=["pyramid"],
    output_layout="NCHW",
    opt=3,
)

k8s_config = dict(
    job_name=job_name,
    job_password=job_password,
    num_machines=num_machines,
    num_gpus_per_machine=num_gpus_per_machine,
    framework=framework,
    task_label=task_label,
    project_id=project_id,
    input_bucket=input_bucket,
    priority=priority,
    docker_image=docker_image,
    max_jobtime=max_jobtime,
    launcher=launcher,
    upload_folder_name=upload_folder_name,
    folder_list=folder_list,
    job_list=job_list,
)
