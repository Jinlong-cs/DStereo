import os
import sys
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import torch
from horizon_plugin_pytorch.qat_mode import QATMode

from hat.engine.processors.loss_collector import collect_loss_by_regex
from projects.pilot.configs.bev_7v_temporal.base import get_update_state_dict

# -------------------------- common --------------------------
from projects.pilot.configs.bev_7v_temporal.common import (
    camera_view_names,
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
    pipeline_test,
    resume_training,
    save_prefix,
    split_flag,
    tasks,
    trace_inputs,
    training_step,
)
from projects.pilot.configs.bev_7v_temporal.data_loaders import data_loader
from projects.pilot.configs.bev_7v_temporal.models import deploy_model, model
from projects.pilot.configs.bev_7v_temporal.schedule import (
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
warmup_steps = min(num_steps, warmup_steps)

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
    dict(
        type="ANCRpyParamAug",
        prob=1.0 if pipeline_test else 0.2,  # force test for pipelinetest
        aug_degree=1,
        camera_view_names=camera_view_names,
        filter_cameras=None,
        empty_cache=False,
    ),
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
    max_steps=(num_steps - warmup_steps) // num_machines,
    warmup_by="step",
    warmup_len=warmup_steps // num_machines,
    warmup_begin_lr=5e-8,
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
            "side_backbone",
            "narrow_backbone",
            "bifpn_neck",
            "side_bifpn_neck",
            "narrow_bifpn_neck",
            "front_bev_stage1_head",
            "side_bev_stage1_head",
            "narrow_bev_stage1_head",
            "bev_fusion",
            "bev_stage2_backbone",
            "bev_stage2_neck",
        ]
    ],
    step_or_epoch=[0],
    update_by="non_global_step",
    only_batchnorm=True,
)
if training_step in ["float_freeze_bn"]:
    callbacks = callbacks + [freeze_bn_callback]

grad_scale_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=0.1,
)
callbacks.append(grad_scale_callback)

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
        weight_decay=0.0001,
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
freeze_bn_checkpoint = "http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/wentao.xie/plat_gpu/hobot-dag-7669899_pilot-multitask-bev-7v-temporal-pilot5-1-master-v3-1-0-range-change-20240118-120408/output/models/pilot_multitask_bev_7v_temporal/float_freeze_bn-checkpoint-last-e829a02c.pth.tar"
pretrain_checkpoint = freeze_bn_checkpoint

stage_extend_params = dict(
    float=dict(
        checkpoint_path=pretrain_checkpoint,
        allow_miss=True,
        ignore_extra=True,
        verbose=0,
    ),
    float_freeze_bn=dict(
        checkpoint_path=freeze_bn_checkpoint
        if freeze_bn_checkpoint
        else os.path.join(ckpt_dir, "float-checkpoint-last.pth.tar"),
        state_dict_update_func=get_update_state_dict(),
        allow_miss=True,
        ignore_extra=True,
        verbose=1,
        # checkpoint_mode="resume",
        # resume_epoch_or_step=True,
        # resume_optimizer=True,
    ),
    # calibration=dict(  # del e2e
    #     checkpoint_path=os.path.join(
    #         ckpt_dir, "float_freeze_bn-checkpoint-last.pth.tar"
    #     ),
    #     allow_miss=False,
    #     ignore_extra=True,
    # ),
    qat=dict(
        checkpoint_path=os.path.join(
            ckpt_dir, "float_freeze_bn-checkpoint-last.pth.tar"
        ),
        allow_miss=False,
        ignore_extra=True,
        # concat_param=dict(  # del e2e
        #     checkpoint_path=os.path.join(
        #         ckpt_dir, "calibration-checkpoint-last.pth.tar"
        #     ),
        #     state_dict_update_func=get_update_state_dict(
        #         ["bev_stage2_e2e_dynamic_head"]
        #     ),
        #     allow_miss=True,
        #     ignore_extra=True,
        # ),
    ),
)

for stage in train_stages:
    params = stage_extend_params.get(stage, [])
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
            qat_mode=QATMode.FuseBN,
            enable_qat=stage == "qat",
            enable_calibraion=stage == "calibration",
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
        )
    )

    if params.get("concat_param") and checkpoint_mode != "resume":
        concat_param = params.get("concat_param")
        converts = [
            trainer["model_convert_pipeline"],
            dict(
                type="LoadCheckpoint",
                checkpoint_path=concat_param["checkpoint_path"],
                state_dict_update_func=concat_param.get(
                    "state_dict_update_func", None
                ),
                allow_miss=concat_param.get("allow_miss", False),
                ignore_extra=concat_param.get("ignore_extra", False),
                check_hash=concat_param.get("check_hash", False),
                verbose=concat_param.get("verbose", 1),
            ),
        ]
        trainer["model_convert_pipeline"] = dict(
            type="ModelConvertPipeline",
            converters=converts,
        )

    # if stage == "calibration":
    #     trainer.update(
    #         dict(
    #             type="Calibrator",
    #             callbacks=[stat_callback, checkpoint_callback],
    #             batch_processor=dict(
    #                 type="MultiBatchProcessor",
    #                 batch_transforms=batch_transforms,
    #                 need_grad_update=False,
    #             ),
    #             num_steps=num_steps,
    #         )
    #     )

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
    BEVModelName.bev_multitask_stage1_fov120,
    BEVModelName.bev_multitask_stage1_side,
    BEVModelName.bev_multitask_stage1_narrow,
    BEVModelName.bev_multitask_stage2_temporal,
]

model_split_trace_inputs = [
    {"camera_front": deploy_stage1_inputs["camera_front"]},
    {"camera_front_left": deploy_stage1_inputs["camera_front_left"]},
    {"camera_front_30fov": deploy_stage1_inputs["camera_front_30fov"]},
    deploy_stage2_inputs,
]

# split and pick one
# we split each view feature extract module,
# total num_view=7, so circulate split 7 times.
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
                "OutputModule6",
                "OutputModule7",
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
                "",
                "",
                "top,bottom",
            ],
            pick_models_index=int(split_index),
        )
    ]
    if split_flag == BEVModelName.bev_multitask_stage2_temporal:
        # compile DON'T support input_key with capitals
        input_key_mapping = {
            "OutputModule1": "pred_segs_frame0_0",
            "OutputModule2": "pred_segs_frame0_1",
            "OutputModule3": "pred_segs_frame0_2",
            "OutputModule4": "pred_segs_frame0_3",
            "OutputModule5": "pred_segs_frame0_4",
            "OutputModule6": "pred_segs_frame0_5",
            "OutputModule7": "pred_segs_frame0_6",
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


def get_om_cpts_update_state_dict():
    def update_om_crosspoint_state_dict(state_dict):
        state_dict_new = deepcopy(state_dict)
        copy_dict = [
            "cat_op",
            "sample_layers",
            "align_blocks",
            "enhance_block",
        ]
        # change cpts head params in om head into origin cpts head
        replace_dict = {
            "head.output_blocks.0.17": "head.output_blocks.0.0",
            "head.output_blocks.0.18": "head.output_blocks.0.1",
            "head.output_blocks.0.19": "head.output_blocks.0.2",
            "head.output_blocks.0.20": "head.output_blocks.0.3",
            "head.output_blocks.0.21": "head.output_blocks.0.4",
            "head.output_blocks.0.22": "head.output_blocks.0.5",
        }
        for key, value in state_dict.items():
            for copy_key in copy_dict:
                if copy_key in key:
                    if "bev_stage2_om_head" in key:
                        value = state_dict[key]
                        key = key.replace(
                            "bev_stage2_om_head", "bev_stage2_crosspoint_head"
                        )
                    state_dict_new[key] = value
                    break
            for replace_key in replace_dict:
                if replace_key in key:
                    if "bev_stage2_om_head" in key:
                        value = state_dict_new.pop(key)
                        key = key.replace(
                            "bev_stage2_om_head", "bev_stage2_crosspoint_head"
                        )
                        key = key.replace(
                            replace_key, replace_dict[replace_key]
                        )
                        state_dict_new[key] = value
                    break
        return state_dict_new

    return update_om_crosspoint_state_dict


int_infer_trainer = dict(
    type="Trainer",
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=QATMode.FuseBN,
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir,
                    f"{train_stages[-1]}-checkpoint-last.pth.tar",
                ),
                state_dict_update_func=get_om_cpts_update_state_dict(),
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            dict(type="QAT2Quantize"),
        ]
        + (graph_model_split if split_flag else []),
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
