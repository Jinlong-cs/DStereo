import os
import sys
from copy import deepcopy
from importlib import import_module
from pathlib import Path

# -------------------------- common --------------------------
from common import (
    enable_model_tracking,
    is_local_train,
    log_freq,
    model_checkpoint,
    model_name,
    model_type,
    model_version,
    num_machines,
    resume_training,
    save_prefix,
    tasks,
    training_step,
    view_num,
)
from data_loaders import data_loader
from models import (  # noqa
    march,
    model,
    test_model,
    traced_inputs,
    traced_inputs_split,
    val_model,
)
from schedule import (
    base_lr,
    interval_by,
    num_steps,
    save_interval,
    train_stages,
    warmup_steps,
)

from hat.core.task_sampler import TaskSampler
from hat.engine.processors.loss_collector import collect_loss_by_regex

# step-specific args
num_steps = num_steps[training_step]
base_lr = base_lr[training_step]

# -------------------------- multitask --------------------------
seed = None
log_rank_zero_only = True
cudnn_benchmark = True
sync_bn = True
enable_amp = False

do_validation = False

device_ids = [0]


ckpt_dir = Path(save_prefix) / model_type
log_dir = ckpt_dir / "logs"
redirect_config_logging_path = (
    log_dir / f"config_log_{training_step}.log"
).as_posix()

# -------------------------- task --------------------------
task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

# -------------------------- callbacks --------------------------
bgr2yuv = [dict(type="BgrToYuv444", affect_key="img", rgb_input=True)]
if view_num == 6:
    bgr2yuv += [
        dict(type="BgrToYuv444", affect_key="side_img", rgb_input=True)
    ]
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=bgr2yuv
    + [
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            affect_keys="img" if view_num != 6 else ["img", "side_img"],
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
    max_update=num_steps // num_machines,
    power=1.0,
    warmup_len=warmup_steps,
    step_log_interval=25,
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
    model_name=model_name,
    task_type="multitask",
    save_model="last",
    model_version=model_version,
)

metric_updaters = [T.metric_updater for T in TASK_CONFIGS]

# the order of callbacks affects the logging order
callbacks = [
    stat_callback,
    lr_callback,
    checkpoint_callback,
    # tensorboard_callback,
] + metric_updaters


if enable_model_tracking:
    callbacks.append(aidi_expmodel_callback)

freeze_bn_callback = dict(
    type="FreezeModule",
    modules=[
        [
            "backbone",
            "fpn_neck",
            "bev_stage1_head",
            "bev_stage2_back_bone",
            "bev_stage2_neck",
        ]
    ],
    step_or_epoch=[0],
    update_by="step",
    only_batchnorm=True,
)
if training_step == "float_freeze_bn":
    callbacks = callbacks + [freeze_bn_callback]
# ----------------------------- validation callback -----------------
val_batch_processor = dict(
    type="MultiBatchProcessor",
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
)


val_task_sampler_configs = {
    T.task_name: dict(sampling_factor=1) for T in TASK_CONFIGS
}
val_task_sampler = TaskSampler(
    task_config=val_task_sampler_configs,
    method="sample_all",
)

val_loaders = {
    T.task_name: T.val_data_loader
    for T in TASK_CONFIGS
    if hasattr(T, "val_data_loader")
}
# val_data_loader = dict(
#     type="MultitaskInfLoader",
#     loaders=val_loaders,
#     task_sampler=val_task_sampler,
#     return_task=False,
#     __build_recursive=False,
# )

val_data_loader = dict(
    type="MultitaskLoader",
    loaders=val_loaders,
    task_sampler=val_task_sampler,
    mode="max_size",
    return_task=True,
    custom_length=None,
    wrap_batch=True,
)

val_metric_updaters = [
    T.val_metric_updater
    for T in TASK_CONFIGS
    if hasattr(T, "val_metric_updater")
]
val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=val_metric_updaters,
    val_interval=80000,
    val_model=model,
    interval_by="step",
)

if do_validation:
    callbacks.append(val_callback)

profiler = dict(
    type="SimpleProfiler",
    # dirpath=log_dir.as_posix(),
    # filename="profile.log",
)

# -------------------------- trainer --------------------------
base_trainer = dict(
    type="DistributedDataParallelTrainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type="LegacyNadamEx",
        params={"weight": dict(weight_decay=5e-5)},
        lr=base_lr * num_machines,
        rescale_grad=1,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=(num_steps + warmup_steps) // num_machines,
    device=None,  # set when building
    sync_bn=sync_bn,
    sync_bn_by_host=True,
    callbacks=callbacks,
    assign_module_buffers=False,
    # profiler=profiler,
)

# if is_local_train and len(device_ids) == 1 and not do_validation:
if is_local_train and len(device_ids) == 1:
    base_trainer.pop("sync_bn")
    base_trainer.pop("sync_bn_by_host")
    base_trainer["type"] = "Trainer"

# -------------------------- training ----------------------------

# multitask v2.1 float_freeze_bn
# pretrain_checkpoint = "http://fm-yiding-liu.train.hogpu.cc/plat_gpu/pilot_multitask_resize2_bev_hc23_x3c_day_mcp5.0_hc23_v2.1.0_baseline_anon005_resumefloat-20221124-102240/output/models/pilot_multitask_resize2_bev/float_freeze_bn-checkpoint-last-6cb74d29.pth.tar"

# v3.0 float_freeze_bn
pretrain_checkpoint = "http://fm-yiding-liu.train.hogpu.cc/plat_gpu/pilot_multitask_resize2_bev_hc23_x3c_day_3.5.0-20221216-025216/output/models/pilot_multitask_resize2_bev/float_freeze_bn-checkpoint-last-2001bf3b.pth.tar"

stage_extend_params = dict(
    float=dict(
        checkpoint_path=pretrain_checkpoint,
        allow_miss=True,
        ignore_extra=True,
        # verbose=0,  # 1
        # checkpoint_mode="pre_stage",  # "resume"
        # resume_epoch_or_step=False,
        # resume_optimizer=False,
    ),
    float_freeze_bn=dict(
        checkpoint_path=os.path.join(
            ckpt_dir, "float-checkpoint-last.pth.tar"
        ),
        # checkpoint_path=pretrain_checkpoint,
    ),
    qat=dict(
        checkpoint_path=os.path.join(
            ckpt_dir, "float_freeze_bn-checkpoint-last.pth.tar"
        ),
    ),
)

for stage in train_stages:
    params = stage_extend_params.get(stage, {})
    checkpoint_mode = (
        "resume"
        if resume_training
        else params.get("checkpoint_mode", "pre_stage")
    )

    if model_checkpoint:
        checkpoint_path = model_checkpoint
        track_input_model = True
    else:
        checkpoint_path = params["checkpoint_path"]
        track_input_model = False

    track_input_model = track_input_model and enable_model_tracking

    trainer = deepcopy(base_trainer)
    trainer.update(
        model_convert_pipeline=dict(
            type="FloatQatConvertPipeline",
            qat_mode="fuse_bn",
            enable_qat=stage == "qat",
            checkpoint_mode=checkpoint_mode,
            checkpoint_configs=dict(
                checkpoint_path=checkpoint_path,
                enable_tracking=track_input_model,
                state_dict_update_func=params.get(
                    "state_dict_update_func", None
                ),
                allow_miss=params.get("allow_miss", False),
                ignore_extra=params.get("ignore_extra", False),
                verbose=params.get("verbose", 1),
            ),
            qconfig_params=None,
        )
    )
    if checkpoint_mode == "resume":
        trainer.update(
            resume_optimizer=params.get("resume_optimizer", False),
            resume_epoch_or_step=params.get("resume_epoch_or_step", False),
        )

    globals()[f"{stage}_trainer"] = trainer

# -------------------------- stage int_infer --------------------------

int_checkpoint = dict(
    type="SaveTraced",
    save_dir=ckpt_dir.as_posix(),
    trace_inputs=traced_inputs,
    name_prefix=training_step + "-",
)

int_infer_trainer = dict(
    type="Trainer",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir,
                    f"{train_stages[-1]}-checkpoint-last.pth.tar",
                ),
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    num_epochs=0,
    device=None,
    optimizer=None,
    data_loader=None,
    batch_processor=None,
    callbacks=[int_checkpoint],
)


# entry


for t in task_names + [
    "common",
    "data_loaders",
    "models",
    "schedule",
    "bev_common",
]:
    sys.modules.pop(t)


compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=model_type,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    output_layout="NHWC",
    input_source=["pyramid"],
)

# predictor
float_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=val_metric_updaters,
    log_interval=10,
)
