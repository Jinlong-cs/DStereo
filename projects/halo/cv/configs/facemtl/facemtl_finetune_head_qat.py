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
qat_lr = 1e-4
qat_num_epoch = 5


VERSION = ConfigVersion.v2
# march = March.BERNOULLI  # J2
# march = March.BERNOULLI2  # J3
march = March.BAYES  # J5
job_name = "face_mtl_finetune_mouth_mixvargenet1.0_0.6"
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

select_tasks = [
    # "mouth",
    # "blur",
    # "glass",
    # "brightness",
    # "leye",
    # "reye",
    # "forehead",
    "hat",
    # "face_ldmk",
    # "face3d",
]
qat_freeze_modules = [
    "face_ldmk_backbone",
    "blur_classifier_head",
    "brightness_classifier_head",
    "leye_classifier_head",
    "reye_classifier_head",
    "forehead_classifier_head",
    "mouth_classifier_head",
    "mask_classifier_head",
    "glass_classifier_head",
    "hat_classifier_head",
    "face_ldmk_decoder",
    "face_ldmk_vector_head",
    "face3d_head",
    "face3d_post_module",
]

# -------------------------- pretrain model --------------------------
pretrain_model_path = "/home/users/jiaqi.quan/project/HAT/tmp_models/v1.0.0/deploy/dms_face_mtl.pth.tar"  # noqa

# -------------------------- train data --------------------------
task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(task_to_module[t]) for t in task_names]

train_task_sampler_config = {
    T.task_name: dict(sampling_factor=1)
    for T in TASK_CONFIGS
    if T.task_name in select_tasks  # noqa
}

train_task_sampler = TaskSampler(
    task_config=train_task_sampler_config, method="sample_all"
)

train_loaders = {
    T.task_name: T.train_data_loader
    for T in TASK_CONFIGS
    if T.task_name in select_tasks
}  # noqa
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

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    interval_by="step",
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
    param_name="face_ldmk_backbone.mod5.3.out_conv.1.1",
)

train_metric_updaters = [
    T.metric_updater for T in TASK_CONFIGS if T.task_name in select_tasks
]  # noqa
train_metrics = []
for T in TASK_CONFIGS:
    if T.task_name in select_tasks:
        train_metrics += _as_list(T.train_metrics)


# -------------------------- training ----------------------------
train_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=batch_transforms,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
)

qat_freeze_callback = dict(
    type="FreezeModule",
    modules=[qat_freeze_modules],
    step_or_epoch=[0],
    update_by="epoch",
    only_batchnorm=False,
)

qat_lr_callback = dict(
    type="StepDecayLrUpdater",
    lr_decay_id=[6, 10, 16],
    lr_decay_factor=0.1,
    step_log_interval=500,
)
qat_callbacks = [
    stat_callback,
    qat_lr_callback,
    ckpt_callback,
]
qat_callbacks += train_metric_updaters
qat_callbacks = qat_callbacks + [qat_freeze_callback]


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=train_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=pretrain_model_path,
                allow_miss=False,
                ignore_extra=False,
                verbose=True,
                check_hash=False,
            ),
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
    # num_steps=1,
    num_epochs=qat_num_epoch,
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
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
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
