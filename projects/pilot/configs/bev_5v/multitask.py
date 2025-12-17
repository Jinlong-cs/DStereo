import os
import sys
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import torch

from hat.engine.processors.loss_collector import collect_loss_by_regex

# -------------------------- common --------------------------
from projects.pilot.configs.bev_5v.common import (
    deploy_stage1_inputs,
    deploy_stage2_inputs,
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
    split_flag,
    tasks,
    trace_inputs,
    training_step,
)
from projects.pilot.configs.bev_5v.data_loaders import data_loader
from projects.pilot.configs.bev_5v.models import deploy_model, model  # noqa
from projects.pilot.configs.bev_5v.schedule import (
    base_lr,
    interval_by,
    num_steps,
    save_interval,
    train_stages,
    warmup_steps,
)
from projects.pilot.configs.project_utils.enum import BEVModelName

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

# -------------------------- batch processor --------------------------
batch_transforms = [
    dict(type="ANCConvertToYuv"),
    dict(type="ANCNormalize3DV", mean=128, std=128),
]
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=batch_transforms,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
    # delay_sync=True,
)

# -------------------------- callbacks --------------------------
stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

lr_callback = dict(
    type="CosLrUpdater",
    max_steps=num_steps // num_machines - warmup_steps,
    warmup_by="step",
    warmup_len=warmup_steps,
    warmup_begin_lr=5e-5,
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
for T in TASK_CONFIGS:
    func_list = getattr(T, "tb_update_func_list", None)
    if func_list is not None:
        tensorboard_callback["tb_update_funcs"].extend(func_list)

aidi_expmodel_callback = dict(
    type="AIDIExperimentManager",
    model_name=model_name,
    save_model="last",
    model_version=model_version,
    upload_progressive_checkpoint=True,
)

metric_updaters = [T.metric_updater for T in TASK_CONFIGS]

# the order of callbacks affects the logging order
callbacks = [
    stat_callback,
    lr_callback,
    checkpoint_callback,
    tensorboard_callback,
] + metric_updaters


if enable_model_tracking:
    callbacks.append(aidi_expmodel_callback)

freeze_bn_callback = dict(
    type="FreezeModule",
    modules=[
        [
            "backbone",
            "round_backbone",
            "bifpn_neck",
            "round_bifpn_neck",
            "front_bev_stage1_head",
            "round_bev_stage1_head",
            "bev_fusion",
            "bev_stage2_backbone_small",
            "bev_stage2_neck_small",
        ]
    ],
    step_or_epoch=[0],
    update_by="non_global_step",
    only_batchnorm=True,
)
if training_step == "float_freeze_bn":
    callbacks = callbacks + [freeze_bn_callback]

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
        type=torch.optim.AdamW,
        params={},
        lr=base_lr * num_machines,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=num_steps // num_machines,
    device=None,  # set when building
    sync_bn=sync_bn,
    sync_bn_by_host=True,
    callbacks=callbacks,
    assign_module_buffers=False,
    # profiler=profiler,
)

# -------------------------- training ----------------------------
pretrain_checkpoint = "http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/jiaxi.wu/plat_gpu/hobot-dag-7200496_pilot-multitask-bev-5v-pilot5-1-master-3-0-5-fusionx2-keepneck-mtl-fromv3-fixrod-20240111-171459/output/models/pilot_multitask_bev_5v/float_freeze_bn-checkpoint-last-4244092e.pth.tar"

stage_extend_params = dict(
    float=dict(
        checkpoint_path=pretrain_checkpoint,
        allow_miss=True,
        ignore_extra=True,
        verbose=0,
        # checkpoint_mode="resume",
        # resume_epoch_or_step=True,
        # resume_optimizer=True,
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
            resume_optimizer=params.get("resume_optimizer", True),
            resume_epoch_or_step=params.get("resume_epoch_or_step", True),
        )

    globals()[f"{stage}_trainer"] = trainer


# -------------------------- stage int_infer --------------------------
int_checkpoint = dict(
    type="SaveTraced",
    save_dir=ckpt_dir.as_posix(),
    trace_inputs=trace_inputs,
    name_prefix=training_step + "-",
)

# split setting & meta info
# order split model
model_split_sequence = [
    BEVModelName.bev_multitask_small_stage1_fov120,
    BEVModelName.bev_multitask_small_stage1_fisheye,
    BEVModelName.bev_multitask_small_stage2,
]

model_split_trace_inputs = [
    {"camera_front": deploy_stage1_inputs["camera_front"]},
    {"fisheye_front": deploy_stage1_inputs["fisheye_front"]},
    deploy_stage2_inputs,
]

# split and pick one
# we split each view feature extract module,
# total num_view=5, so circulate split 5 times.
# And pick one of split models for compile.
if split_flag is not None:
    split_index = model_split_sequence.index(split_flag)
    graph_model_split = [
        dict(
            type="GraphModelSplit",
            split_nodes=[
                "OutputModule1",
                "OutputModule2",
                "OutputModule3",
                "OutputModule4",
                "OutputModule5",
            ],
            # each split base.
            # top or bottom,
            # top ref view feat extract,
            # bottom for split left (bev module and maybe some view feat extract).
            next_bases=[
                "bottom",
                "bottom",
                "bottom",
                "bottom",
                "bottom",
            ],
            # save split models
            # obtain 3 view feat extract: front, side (total 5x foward, we only keep one), narrow
            # and bev module.
            save_models=[
                "top",
                "top",
                "",
                "",
                "bottom",
            ],
            pick_models_index=int(split_index),
        )
    ]
    if split_flag == BEVModelName.bev_multitask_small_stage2:
        # compile DON'T support input_key with capitals
        input_key_mapping = {
            "OutputModule1": "pred_segs_frame0_0",
            "OutputModule2": "pred_segs_frame0_1",
            "OutputModule3": "pred_segs_frame0_2",
            "OutputModule4": "pred_segs_frame0_3",
            "OutputModule5": "pred_segs_frame0_4",
        }
        for k, v in input_key_mapping.items():
            _v = deploy_stage2_inputs.pop(k)
            deploy_stage2_inputs[v] = _v
        graph_model_split.append(
            dict(
                type="GraphModelInputKeyMapping",
                input_key_mapping=input_key_mapping,
            )
        )
    int_checkpoint["trace_inputs"] = model_split_trace_inputs[split_index]


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
                    f"{train_stages[-1]}-checkpoint-last.pth.tar",
                ),
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            dict(type="QAT2Quantize"),
        ]
        + graph_model_split
        if split_flag
        else [],
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
]:
    sys.modules.pop(t, None)

# -------------------------- compile --------------------------
compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march="bayes",
    name=model_type,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["ddr"],
)
