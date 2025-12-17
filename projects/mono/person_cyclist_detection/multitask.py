import copy
import os
from importlib import __import__

import torch
from common import device_ids  # noqa
from common import (
    buckets,
    ckpt_dir,
    deploy_input_hw,
    float_lr,
    float_steps,
    get_task_model,
    get_train_step,
    input_hw,
    interval_by,
    is_local_train,
    log_freq,
    march,
    multitask_task_name,
    num_machines,
    pretrain_checkpoint,
    qat_lr,
    qat_steps,
    save_interval,
    warmup_steps,
)

from hat.core.task_sampler import TaskSampler
from hat.engine.processors.loss_collector import collect_loss_by_regex

# import projects.mono.person_cyclist_det_2pe.v1_0_2_2.extra_data_transforms

# ---------------------------------- multitask ----------------------
job_name = "PersonCyclistDet2PE-v1.0.2.2"
task_name = multitask_task_name
training_step = get_train_step()
# step-specific args
log_dir = os.path.join(ckpt_dir, "logs")
log_rank_zero_only = True
cudnn_benchmark = True
delay_sync = False
enable_amp = False
seed = None
sync_bn = False

float_resume_checkpoint = None
qat_resume_checkpoint = None

# ------------------------------------ task -------------------------
tasks = [
    # person
    "person_cyclist_detection",
]


def get_task_configs():
    return [__import__(f"{t}") for t in tasks]


def get_per_task_flat_condition(mode, task_configs):
    condition_dict = dict()
    assert mode in ["train", "val", "test"]
    if mode == "train":
        condi_fn = "train_flat_condition"
    else:
        condi_fn = "val_flat_condition"
    for task_config in task_configs:
        if hasattr(task_config, condi_fn):
            assert callable(
                task_config.get(condi_fn)
            ), "Priveded condition function must be callable"
            condition_dict.update(
                {task_config.task_name: task_config.get(condi_fn)}
            )
    return condition_dict


TASK_CONFIGS = get_task_configs()

task_sampler_configs = {
    T.task_name: dict(sampling_factor=1) for T in TASK_CONFIGS
}

task_sampler = TaskSampler(
    task_config=task_sampler_configs,
    method="sample_all",
)

tb_update_funcs = [
    T.tb_update_func
    for T in TASK_CONFIGS
    if getattr(T, "tb_update_func", None) is not None
]
model = get_task_model("train", TASK_CONFIGS)
val_model = get_task_model("val", TASK_CONFIGS)
test_model = get_task_model("test", TASK_CONFIGS)


# ------------------------------------ data -------------------------
inputs = dict(img=torch.zeros((1, 3, *input_hw)))
deploy_inputs = dict(img=torch.zeros((1, 3, *deploy_input_hw)))


def build_dataloader(task_configs):
    loaders = {T.task_name: T.data_loader for T in task_configs}

    data_loader = dict(
        type="MultitaskInfLoader",
        loaders=loaders,
        task_sampler=task_sampler,
        return_task=True,
    )
    return data_loader


data_loader = build_dataloader(TASK_CONFIGS)

# ---------------------------------- trainer ------------------------
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    delay_sync=delay_sync,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

tensorboard_callback = dict(
    type="TensorBoard",
    save_dir=log_dir if is_local_train else "/job_tboard/",  # noqa
    update_freq=log_freq,
    tb_update_funcs=tb_update_funcs,
)
# freeze_bn_callback = dict(
#     type="FreezeModule",
#     modules=[freeze_bn_modules[training_step]],
#     step_or_epoch=[0],
#     update_by=interval_by,
#     strict_match=False,
#     only_batchnorm=True,
# ),
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=False,
    best_refer_metric=None,
    save_interval=save_interval,
    interval_by=interval_by,
    save_on_train_end=True,
)
metric_updaters = [T.metric_updater for T in TASK_CONFIGS]

# ---------------------------------- float trainer ------------------

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

float_lr_callback = dict(
    type="PolyLrUpdater",
    max_update=(float_steps - warmup_steps) // num_machines,
    power=1.0,
    warmup_by="step",
    warmup_len=warmup_steps,
    warmup_begin_lr=0.00005,
    step_log_interval=log_freq,
)

float_callbacks = [
    stat_callback,
    float_lr_callback,
    ckpt_callback,
    tensorboard_callback,
] + metric_updaters

float_trainer = dict(
    type="DistributedDataParallelTrainer",
    model=get_task_model("train", TASK_CONFIGS),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=float_converters,
    ),
    data_loader=build_dataloader(TASK_CONFIGS),
    optimizer=dict(
        type="LegacyNadamEx",
        params={"weight": dict(weight_decay=5e-5)},
        lr=float_lr * num_machines,
        rescale_grad=1,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=(float_steps + warmup_steps) // num_machines,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=float_callbacks,
)

# ---------------------------------- qat trainer --------------------

qat_converters = [
    dict(
        type="LoadCheckpoint",
        checkpoint_path=os.path.join(
            ckpt_dir, "float-checkpoint-last.pth.tar"
        ),
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
    ckpt_callback,
    tensorboard_callback,
] + metric_updaters

qat_trainer = dict(
    type="DistributedDataParallelTrainer",
    model=get_task_model("train", TASK_CONFIGS),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=qat_converters,
    ),
    data_loader=build_dataloader(TASK_CONFIGS),
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

# --------------------------- int inference trainer -----------------
int_infer_converters = [
    dict(type="Float2QAT"),
    dict(
        type="LoadCheckpoint",
        checkpoint_path=os.path.join(ckpt_dir, "qat-checkpoint-last.pth.tar"),
        allow_miss=True,
        ignore_extra=True,
    ),
    dict(type="QAT2Quantize"),
]

trace_callback = dict(
    type="SaveTraced",
    trace_inputs=copy.deepcopy(deploy_inputs),
    save_dir=ckpt_dir,
    save_hash=False,
    name_prefix="int_infer-",
)

int_infer_trainer = dict(
    type="Trainer",
    model=get_task_model("test", TASK_CONFIGS),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=int_infer_converters,
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[
        ckpt_callback,
        trace_callback,
    ],
)

# ------------------------------------ hbdk -------------------------
compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,  # Name of the model, recorded in hbm
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "person_cyclist_detection.hbm"),
    layer_details=True,
    input_source=["pyramid"],  # or ddr? custom by yourself
    opt="O0",  # change O3 for faster fps
)

# ------------------------------ k8s_submit config ------------------
curr_root = os.path.realpath(os.path.dirname(__file__))
work_dir = os.path.realpath(os.getcwd())
cfg_file = os.path.join(
    os.path.relpath(curr_root, work_dir),
    os.path.basename(os.path.basename(__file__)),
)
job_list = [
    f"python3 tools/train.py --config {cfg_file} --stage float",  # noqa
    f"python3 tools/train.py --config {cfg_file} --stage qat",  # noqa
    f"python3 tools/train.py --config {cfg_file} --stage int_infer",  # noqa
]


k8s_config = dict(
    job_name=job_name,
    job_password="6150",
    num_machines=num_machines,
    num_gpus_per_machine=8,
    framework="pytorch",
    task_label="mono_person_multitask",
    project_id="PDT20220004",
    input_bucket=",".join(buckets),
    priority=5,
    docker_image="docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-0d1b91b",  # noqa
    # default 7200 = 5days
    max_jobtime=20160,
    # max_jobtime=60,
    # launcher only for multi-machines
    launcher="mpi",
    # upload folder
    upload_folder_name="k8s_job",
    folder_list=[
        f"{curr_root}/../../../hat",
        f"{curr_root}/../../../tools",
        f"{curr_root}/../../../projects",
    ],
    job_list=job_list,
)
