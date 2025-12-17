"""
fisheye multitask entry
"""
import copy
import os
from functools import partial

import torch

# -------------------------- common --------------------------
from common import (  # noqa
    add_freeze_bn_step,
    aidi_eval,
    all_task_data_files,
    ctimestr,
    device_ids,
    do_freeze_bn,
    do_grad_scale,
    float_resume_checkpoint,
    float_steps,
    freeze_bn_steps,
    freezebn_step_ids,
    hbdk_input_path,
    infer_model_ckpt,
    input_size,
    int_infer_stage,
    interval_by,
    job_name,
    local_train,
    log_freq,
    lr,
    pipeline_test,
    pipeline_test_dump,
    pretrain_checkpoint,
    project_id,
    qat_lr,
    qat_num_steps,
    qat_resume_checkpoint,
    save_interval,
    save_prefix,
    task_common_transforms,
    train_gpu_transforms,
    training_step,
    use_step_lr,
    val_gpu_transforms,
    val_interval,
    val_interval_by,
    val_while_train,
    warmup_steps,
    wd,
)
from data_loaders import aidi_eval_dataloaders, data_loader, val_data_loader
from lib.utils import get_list
from models import (  # noqa
    march,
    model,
    test_model,
    traced_inputs,
    traced_inputs_stage_two,
    val_model,
)
from submit import k8s_config  # noqa
from task_schedule import TASK_CONFIGS

from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.checkpoint import load_state_dict

# -------------------------- multitask --------------------------
task_name = "fisheye_multitask"
seed = None
log_rank_zero_only = True
cudnn_benchmark = True
sync_bn = True
enable_amp = False
enable_model_tracking = False
ckpt_dir = os.path.join(save_prefix, task_name)
log_dir = os.path.join(ckpt_dir, "logs")

redirect_config_logging_path = os.path.join(
    log_dir, f"config_log_{training_step}.log"
)

# -------------------------- batch processor --------------------------
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    empty_cache=True,
    batch_transforms=train_gpu_transforms + task_common_transforms,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
)

val_batch_processor = copy.deepcopy(batch_processor)
val_batch_processor["need_grad_update"] = False
val_batch_processor["batch_transforms"] = (
    None if aidi_eval else val_gpu_transforms + task_common_transforms
)
val_batch_processor.pop("loss_collector")

# -------------------------- callbacks --------------------------

aidi_eval_callbacks = get_list(
    "aidi_eval_callback", TASK_CONFIGS, merge_list=True
)

page_label_list = get_list("aidi_eval_page_label", TASK_CONFIGS)
datasets_id_list = get_list("aidi_eval_dataset_id", TASK_CONFIGS)
type_list = get_list("aidi_eval_type", TASK_CONFIGS)
old_prediction_list = get_list("old_prediction", TASK_CONFIGS)
new_prediction_list = get_list("new_prediction", TASK_CONFIGS)

report_name = f"{job_name}_{ctimestr}"
aidireport_callback = dict(
    type="AIDIReport",
    project_id=project_id,
    report_name=report_name,
    page_label_list=page_label_list,
    type_list=type_list,
    old_prediction_list=old_prediction_list,
    new_prediction_list=new_prediction_list,
    datasets_id_list=datasets_id_list,
)
aidi_eval_callbacks[-1].append(aidireport_callback)

dump_callback = dict(
    type="DumpData",
    output_dir="./tmp_dump_data",  # 设置 dump 文件的保存路径
    dump_batch_data=True,  # dump 模型输入，默认 True，如不需要dump，设置 False
    dump_model_outs=True,  # dump 模型输出，默认 True，如不需要dump，设置 False
    dump_model_grad=True,  # dump 参数梯度，默认 True，如不需要dump，设置 False
    name_prefix="SP_dumpdata_allOneStage",  # dump 文件名称，会保存成 test_dump_data.pkl
    reformat_model_outs_fn=None,  # 自定义模型输出的转化函数，可选，默认 None
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

lr_callback = dict(
    type="CosLrUpdater",
    max_steps=float_steps - warmup_steps,
    warmup_by="step",
    warmup_len=warmup_steps,
    warmup_begin_lr=0.00005,
    step_log_interval=log_freq,
)
if add_freeze_bn_step and training_step == "float_freeze_bn":
    lr_callback["max_steps"] = freeze_bn_steps - warmup_steps

qat_lr_callback = copy.deepcopy(lr_callback)
qat_lr_callback["max_steps"] = qat_num_steps - warmup_steps

if use_step_lr:
    lr_callback = dict(
        type="StepDecayLrUpdater",
        warmup_by="step",
        warmup_begin_lr=0.00005,
        warmup_len=warmup_steps,
        step_log_interval=log_freq,
        update_by="step",
        lr_decay_id=[int(8 / 12.0 * float_steps), int(11 / 12.0 * float_steps)]
        if float_steps > 3
        else None,
    )
    if add_freeze_bn_step and training_step == "float_freeze_bn":
        lr_callback["lr_decay_id"] = (
            [int(8 / 12.0 * freeze_bn_steps), int(11 / 12.0 * freeze_bn_steps)]
            if freeze_bn_steps > 3
            else None
        )

    qat_lr_callback = copy.deepcopy(lr_callback)
    qat_lr_callback["lr_decay_id"] = (
        [
            int(8 / 12.0 * qat_num_steps),
            int(11 / 12.0 * qat_num_steps),
        ]
        if qat_num_steps > 3
        else None
    )

best_metric = None
ckpt_suffix = "last" if best_metric is None else "best"
checkpoint_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=False,
    best_refer_metric=None,
    save_interval=save_interval,
    interval_by=interval_by,
    save_on_train_end=True,
)

freeze_bn_callback = dict(
    type="FreezeModule",
    modules=[["backbone", "neck"]],
    step_or_epoch=freezebn_step_ids,
    update_by=interval_by,
    only_batchnorm=True,
)

tb_update_funcs = [
    T.tb_update_func
    for T in TASK_CONFIGS
    if getattr(T, "tb_update_func", None) is not None
]

tensorboard_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(log_dir, training_step, "loss")
    if local_train
    else "/job_tboard/",  # noqa
    update_freq=log_freq,
    update_by="step",
    tb_update_funcs=tb_update_funcs,
)

# because two dataset use same head,
# the shared head' val_metric_updaters use val_metric_updater_list
val_metric_updaters = []
for config in TASK_CONFIGS:
    if hasattr(config, "val_metric_updater_list"):
        for val_metric_updater in config.val_metric_updater_list:
            val_metric_updaters.append(val_metric_updater)
    else:
        val_metric_updaters.append(config.val_metric_updater)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=val_metric_updaters,
    val_interval=val_interval,
    val_model=val_model,
    interval_by=val_interval_by,
)

metric_updaters = [i.metric_updater for i in TASK_CONFIGS]
# the order of callbacks affects the logging order
callbacks = [
    stat_callback,
    lr_callback,
    checkpoint_callback,
    tensorboard_callback,
] + metric_updaters

qat_callbacks = [
    stat_callback,
    qat_lr_callback,
    checkpoint_callback,
    tensorboard_callback,
] + metric_updaters

if val_while_train:
    callbacks.append(val_callback)
    qat_callbacks.append(val_callback)

# Warning: not stable for publish job, and only for 2d3d job now.
grad_scale_callback = dict(
    type="GradScale",
    module_and_scale=[
        # 2D VRU
        # ("person_fcos_common_out_module", 2.0, 'person_detection'),
        # ("cyclist_fcos_common_out_module", 2.0, 'cyclist_detection'),
        # 3D
        # ("backbone", 1.5,'real3d_fisheye_vehicle'),
        # ("bifpn", 1.5,'real3d_fisheye_vehicle'),
        ("fcos2d3d_vehicle", 2.0, "vehicle_detection_3d"),
        # # SOD
        # ("fisheye_sod_all_head", 2.0, 'fisheye_sod_all'),
        # # Parsing
        # ("fisheye_parsing_head", 2.0, 'fisheye_parsing')
    ],
    clip_grad_norm=1,
)

if do_freeze_bn and training_step == "float":
    callbacks.append(freeze_bn_callback)

if do_grad_scale:  # do_grad_scale
    callbacks.append(grad_scale_callback)
    qat_callbacks.append(grad_scale_callback)

# -------------------------- trainer --------------------------
base_trainer = dict(
    type="DistributedDataParallelTrainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=lr,
        weight_decay=wd,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=float_steps,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=callbacks,
)

float_trainer = copy.deepcopy(base_trainer)

if float_resume_checkpoint or pretrain_checkpoint:
    checkpoint = (
        float_resume_checkpoint
        if float_resume_checkpoint
        else pretrain_checkpoint
    )
    float_trainer.update(
        model_convert_pipeline=dict(
            type="LoadCheckpoint",
            checkpoint_path=checkpoint,
            state_dict_update_func=None,
            allow_miss=True,
            ignore_extra=True,
        ),
    )
if float_resume_checkpoint and not pipeline_test_dump:
    float_trainer.update(
        resume_optimizer=True,
        resume_epoch_or_step=True,
    )

qat_trainer = copy.deepcopy(base_trainer)
qat_trainer.update(callbacks=qat_callbacks)
qat_trainer.update(num_steps=qat_num_steps)
qat_trainer.update(
    optimizer=dict(
        type=torch.optim.Adam,
        lr=qat_lr,
        weight_decay=wd,
    ),
)
qat_trainer.update(
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-%s.pth.tar" % ckpt_suffix
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
            dict(type="Float2QAT"),
        ],
    )
)
if qat_resume_checkpoint:
    qat_trainer.update(
        model_convert_pipeline=dict(
            type="LoadCheckpoint",
            checkpoint_path=qat_resume_checkpoint,
            allow_miss=True,
            ignore_extra=True,
        ),
    )
    qat_trainer.update(
        resume_optimizer=True,
        resume_epoch_or_step=True,
    )

# ----------------------- step int_inference (bpu deploy) ----------------
int_checkpoint = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=traced_inputs_stage_two
    if int_infer_stage == "stage_two"
    else traced_inputs,
    save_hash=not pipeline_test,
    name_prefix="pipelinetest-" if pipeline_test else training_step + "-",
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
                    "qat-checkpoint-last.pth.tar",
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

# -------------------------- inference --------------------------
load_pred_ckpt_func = partial(
    load_state_dict, allow_miss=True, ignore_extra=True
)

if aidi_eval:
    val_loaders = aidi_eval_dataloaders
    val_callbacks = aidi_eval_callbacks
else:
    val_loaders = [val_data_loader]
    val_callbacks = val_metric_updaters

# predictor
float_predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=val_loaders,
    batch_processor=val_batch_processor,
    device=None,
    callbacks=val_callbacks,
    share_callbacks=not aidi_eval,
    log_interval=log_freq,
)

qat_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=val_loaders,
    batch_processor=val_batch_processor,
    device=None,
    callbacks=val_callbacks,
    share_callbacks=not aidi_eval,
    log_interval=log_freq,
)

int_infer_predictor = dict(
    type="Predictor",
    model=val_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=infer_model_ckpt,
                allow_miss=False,
                ignore_extra=True,
                verbose=1,
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=val_loaders,
    batch_processor=val_batch_processor,
    device=None,
    callbacks=val_callbacks,
    share_callbacks=not aidi_eval,
    log_interval=log_freq,
)


data_loader_perf = dict(
    type="DataloaderSpeedPerf",
    dataloader=data_loader,
    iter_nums=100,
    frequent=10,
    profiler=dict(type="PythonProfiler"),
)


compile_shape = "1x3x{}".format("x".join(map(str, input_size)))
