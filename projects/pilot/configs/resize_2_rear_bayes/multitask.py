import os
import sys
from copy import deepcopy
from importlib import import_module
from pathlib import Path

# -------------------------- common --------------------------
from common import (
    batch_size_factor,
    enable_model_tracking,
    is_with_pe,
    log_freq,
    model_checkpoint,
    model_name,
    model_setting,
    model_type,
    model_version,
    num_machines,
    resume_training,
    save_prefix,
    tasks,
    training_step,
)
from data_loaders import data_loader
from models import march, model, test_model, traced_inputs
from schedule import (
    base_lr,
    get_fuse_patterns_by_stage,
    interval_by,
    num_steps,
    save_interval,
    train_stages,
    warmup_steps,
)

from hat.engine.processors.loss_collector import collect_loss_by_regex

# step-specific args
num_steps = num_steps[training_step] // batch_size_factor
base_lr = base_lr[training_step] * batch_size_factor

# clipping: begin
from common import is_local_train

# clipping: end

# -------------------------- multitask --------------------------
seed = None
log_rank_zero_only = True
cudnn_benchmark = True
sync_bn = True
enable_amp = False

device_ids = [0]

ckpt_dir = Path(save_prefix) / model_type
log_dir = ckpt_dir / "logs"
redirect_config_logging_path = (
    log_dir / f"config_log_{training_step}.log"
).as_posix()

# -------------------------- task --------------------------
task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

# -------------------------- processor --------------------------
batch_processor = dict(
    type="MultiStageBatchProcessor"
    if "niofy" in model_setting
    else "MultiBatchProcessor",
    need_grad_update=True,
    empty_cache=True,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
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

metric_updaters = [T.metric_updater for T in TASK_CONFIGS]

# the order of callbacks affects the logging order
callbacks = [
    stat_callback,
    lr_callback,
    checkpoint_callback,
] + metric_updaters

profiler = dict(
    type="SimpleProfiler",
    # dirpath=log_dir.as_posix(),
    # filename='profile.log',
)
# clipping: begin
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
    type="AIDIExperimentManager",
    model_name=model_name,
    save_model="last",
    model_version=model_version,
    upload_progressive_checkpoint=True,
)

callbacks += [tensorboard_callback]
if enable_model_tracking:
    callbacks.append(aidi_expmodel_callback)
# clipping: end

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

# -------------------------- stages --------------------------
pretrain_checkpoint = None


def update_embedding_state_dict(state_dict):
    key_mapping = [
        (
            "ufpn_3d_neck.quant.activation_post_process",
            "ufpn_3d_neck.pos_embedding.quant.activation_post_process",
        ),
        (
            "ufpn_3d_neck.embedding",
            "ufpn_3d_neck.pos_embedding.embedding",
        ),
    ]
    state_dict_new = deepcopy(state_dict)
    for key, value in state_dict.items():
        for key_pair in key_mapping:
            if key_pair[0] in key:
                key = key.replace(key_pair[0], key_pair[1])
                break
        state_dict_new[key] = value

    return state_dict_new


stage_extend_params = dict(
    with_bn=dict(
        checkpoint_path=pretrain_checkpoint,
        allow_miss=True,
        ignore_extra=True,
        state_dict_update_func=update_embedding_state_dict,
        # verbose=0, # 1
        # checkpoint_mode="pre_stage", # "resume"
        # resume_epoch_or_step,
        # resume_optimizer,
    ),
    freeze_bn_1=dict(
        checkpoint_path=os.path.join(
            ckpt_dir, "with_bn-checkpoint-last.pth.tar"
        ),
        state_dict_update_func=update_embedding_state_dict,
    ),
    freeze_bn_2=dict(
        checkpoint_path=os.path.join(
            ckpt_dir, "freeze_bn_1-checkpoint-last.pth.tar"
        ),
        state_dict_update_func=update_embedding_state_dict,
    ),
    freeze_bn_3=dict(
        checkpoint_path=os.path.join(
            ckpt_dir, "freeze_bn_2-checkpoint-last.pth.tar"
        ),
        state_dict_update_func=update_embedding_state_dict,
    ),
    sparse_3d_freeze_bn_1=dict(
        checkpoint_path=os.path.join(
            ckpt_dir, "freeze_bn_3-checkpoint-last.pth.tar"
        ),
        allow_miss=True,
        ignore_extra=True,
        state_dict_update_func=update_embedding_state_dict,
    ),
    sparse_3d_freeze_bn_2=dict(
        checkpoint_path=os.path.join(
            ckpt_dir, "sparse_3d_freeze_bn_1-checkpoint-last.pth.tar"
        ),
        state_dict_update_func=update_embedding_state_dict,
    ),
)
# clipping: begin
pretrain_checkpoint = "http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/yilin.xiong/plat_gpu/hobot-dag-2823641_resize-2-rear-bayes-training-galaxy-0233-rear-lmdb/output/models/pilot5_multitask_resize2_rear_bayes/with_bn-checkpoint-last-a561a2c9.pth.tar"  # noqa
stage_extend_params["with_bn"]["checkpoint_path"] = pretrain_checkpoint
# clipping: end

for stage in train_stages:
    pre, cur = get_fuse_patterns_by_stage(stage)
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
            type="QATFuseBNConvertPipeline",
            qat_mode="with_bn_reverse_fold",
            pre_stage_fuse_patterns=pre,
            cur_stage_fuse_patterns=cur,
            fuse_part_configs=dict(
                fuse_method="fuse_norm",
                regex=True,
                strict=False,
            ),
            checkpoint_mode=checkpoint_mode,
            checkpoint_configs=dict(
                checkpoint_path=checkpoint_path,
                state_dict_update_func=params.get(
                    "state_dict_update_func", None
                ),
                enable_tracking=track_input_model,
                allow_miss=params.get("allow_miss", False),
                ignore_extra=params.get("ignore_extra", False),
                verbose=params.get("verbose", 1),
            ),
            qconfig_params=None,
        )
    )
    if checkpoint_mode == "resume":
        trainer.update(
            resume_optimizer=params.get("resume_optimizer", True),
            resume_epoch_or_step=params.get("resume_epoch_or_step", True),
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
                state_dict_update_func=update_embedding_state_dict,
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

for t in task_names + ["common", "data_loaders", "models", "schedule"]:
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


if is_with_pe:
    compile_cfg.update(
        input_layout="NHWC",
        input_source=["ddr", "pyramid"],
    )
