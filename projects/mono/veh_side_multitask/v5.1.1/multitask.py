import copy
import os
from collections import ChainMap, OrderedDict
from functools import partial
from importlib import import_module

import torch
from common import (
    ckpt_dir,
    debug_mode,
    float_steps,
    freeze_bn_lr,
    freeze_bn_steps,
    input_hw,
    interval_by,
    local_train,
    log_dir,
    log_freq,
    lr,
    mono_unet_neck,
    pipeline_test,
    qat_lr,
    qat_steps,
    save_interval,
    training_step,
    vargnetv2_2631_backbone,
    warmup_begin_lr,
    warmup_steps,
    wd,
)
from horizon_plugin_pytorch.march import March

from hat.core.task_sampler import TaskSampler
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.checkpoint import load_state_dict
from hat.utils.config import ConfigVersion

model_name = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
model_version = os.path.basename(os.path.dirname(__file__))
job_name = model_name + "-" + model_version
task_names = [
    "vehicle_side",
    "face_plate",
    "real3d",
]
TASK_CONFIGS = [import_module(t) for t in task_names]

VERSION = ConfigVersion.v2

march = March.BAYES
num_machines = 1 if pipeline_test else 2
num_gpus_per_machine = 2 if pipeline_test else 8

# ------------------------ multitask ---------------------
sync_bn = True
enable_amp = True
save_hash = False
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
device_ids = list(range(num_gpus_per_machine))

# model
common_nodes = dict(
    backbone=copy.deepcopy(vargnetv2_2631_backbone),
    neck=copy.deepcopy(mono_unet_neck),
)
task_builders = [i.topo_builder for i in TASK_CONFIGS]

metric_updaters = [i.metric_updater for i in TASK_CONFIGS]

# val_metric_updaters = [i.val_metric_updater for i in TASK_CONFIGS]
task_config = dict()
for config in TASK_CONFIGS:
    task_config[config.task_name] = dict(sampling_factor=1)
task_sampler = TaskSampler(task_config=task_config, method="sample_all")


def get_data_loader(mode="train"):
    if mode == "train":
        _mode = ""
    elif mode == "test":
        _mode = "test_"
    elif mode == "val":
        _mode = "val_"
    else:
        raise NotImplementedError()
    return dict(
        type="MultitaskLoader",
        loaders={
            i.task_name: getattr(i, f"{_mode}data_loader")
            for i in TASK_CONFIGS
        },
        task_sampler=task_sampler,
        mode="max_size",
        return_task=True,
        custom_length=None,
    )


data_loader = get_data_loader("train")
val_data_loader = get_data_loader("val")
test_data_loader = get_data_loader("test")
data_loader_perf = dict(
    type="DataloaderSpeedPerf",
    dataloader=data_loader,
    iter_nums=100,
    frequent=10,
    profiler=dict(type="PythonProfiler"),
)


def get_topo_builder(mode, out_feat_idx=None):
    def _build_topo(nodes, inputs):
        name2out = OrderedDict()
        backbone_feats = nodes["backbone"](inputs["img"])
        neck_feats = nodes["neck"](backbone_feats)

        for builder in task_builders:
            name2out.update(
                builder(
                    nodes=nodes, _inputs=inputs, feats=neck_feats, mode=mode
                )
            )
        return name2out

    return _build_topo


def get_model(mode="train"):
    _mode = copy.deepcopy(mode)
    if mode == "train":
        mode = ""
    elif mode == "test":
        mode = "test_"
    elif mode == "val":
        mode = "val_"
    else:
        raise NotImplementedError()
    return dict(
        type="GraphModel",
        nodes=dict(
            **copy.deepcopy(common_nodes),
            **dict(
                ChainMap(*[getattr(i, f"{mode}nodes") for i in TASK_CONFIGS])
            ),
        ),
        inputs=dict(
            img=None,
            **dict(
                ChainMap(*[getattr(i, f"{mode}inputs") for i in TASK_CONFIGS])
            ),
        ),
        topology_builder=get_topo_builder(mode=_mode),
        lazy_forward=True,
    )


model = get_model(mode="train")
val_model = get_model(mode="val")
test_model = get_model(mode="test")

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
)
val_batch_processor = copy.deepcopy(batch_processor)
val_batch_processor["need_grad_update"] = False
val_batch_processor.pop("loss_collector")
test_batch_processor = copy.deepcopy(val_batch_processor)
pretrain_ckpt = "http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/xiufeng.zhou/plat_gpu/hobot-dag-3429671_veh-side-multitask-v1-8-1-20230717-102004/output/models/vehicle_side/float-checkpoint-last.pth.tar"
skip_float_stage = True
freeze_bn_pretrain_ckpt = "http://svcspawner.bcloud-2nd.hobot.cc/user/homespace/xiufeng.zhou/plat_gpu/hobot-dag-3429671_veh-side-multitask-v1-8-1-20230717-102004/output/models/vehicle_side/freeze_bn-checkpoint-last.pth.tar"
skip_freeze_bn_stage = True
qat_pretrain_ckpt = None
skip_qat_stage = False
# -------------------------- solver --------------------------
load_pred_ckpt_func = partial(
    load_state_dict, allow_miss=True, ignore_extra=True
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
    warmup_begin_lr=warmup_begin_lr,
    step_log_interval=log_freq,
)
qat_lr_callback = dict(
    type="CosLrUpdater",
    max_steps=qat_steps - warmup_steps,
    warmup_by="step",
    warmup_len=warmup_steps,
    warmup_begin_lr=warmup_begin_lr,
    step_log_interval=log_freq,
)
freeze_module_callback = dict(
    type="FreezeModule",
    modules=[["backbone", "neck"]],
    step_or_epoch=[
        0,
    ],
    update_by=interval_by,
    only_batchnorm=True,
)
checkpoint_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=False,
    save_interval=save_interval,
    interval_by=interval_by,
    save_on_train_end=True,
    save_hash=False,
)
tb_update_funcs = []
for config in TASK_CONFIGS:
    if hasattr(config, "tb_update_func"):
        tb_update_funcs.append(config.tb_update_func)

tensorboard_callback = dict(
    type="TensorBoard",
    save_dir=log_dir if local_train else "/job_tboard/",  # noqa
    update_freq=log_freq,
    tb_update_funcs=tb_update_funcs,
)
grad_scale_callback = dict(
    type="GradScale",
    module_and_scale=[
        ("backbone", 0.5, "real3d"),
        ("neck", 0.5, "real3d"),
    ],
    clip_grad_norm=None,
)

callbacks = [
    stat_callback,
    lr_callback,
    checkpoint_callback,
    tensorboard_callback,
    grad_scale_callback,
]
qat_callbacks = [
    stat_callback,
    qat_lr_callback,
    checkpoint_callback,
    tensorboard_callback,
    grad_scale_callback,
]
# -------------------------- step float --------------------------
float_converters = [
    dict(
        type="LoadCheckpoint",
        checkpoint_path=pretrain_ckpt,
        allow_miss=True,
        ignore_extra=True,
        check_hash=False,
        verbose=False,
    ),
]
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    find_unused_parameters=True,
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=float_converters,
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        lr=lr,
        weight_decay=wd,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=0 if skip_float_stage else float_steps,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=callbacks + metric_updaters,
)
if debug_mode:
    float_trainer["type"] = "Trainer"
    float_trainer.pop("sync_bn", None)
    float_trainer.pop("find_unused_parameters", None)
ckpt_suffix = "last"
# -------------------------- step freeze_bn  --------------------------
freeze_bn_pre_step_ckpt = os.path.join(
    ckpt_dir, "float-checkpoint-last.pth.tar"
)
freeze_bn_converters = [
    dict(
        type="LoadCheckpoint",
        checkpoint_path=freeze_bn_pretrain_ckpt
        if freeze_bn_pretrain_ckpt
        else freeze_bn_pre_step_ckpt,
        allow_miss=False,
        ignore_extra=False,
        check_hash=False,
    ),
]

freeze_bn_trainer = copy.deepcopy(float_trainer)
freeze_bn_trainer["optimizer"]["lr"] = freeze_bn_lr
freeze_bn_trainer["num_steps"] = 0 if skip_freeze_bn_stage else freeze_bn_steps
freeze_bn_trainer["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline", converters=freeze_bn_converters
)
freeze_bn_trainer["callbacks"] += [freeze_module_callback]


# -------------------------- step quantize-aware -------------------
qat_pre_step_ckpt = os.path.join(
    ckpt_dir, "freeze_bn-checkpoint-%s.pth.tar" % ckpt_suffix
)
qat_converters = [
    dict(
        type="LoadCheckpoint",
        checkpoint_path=qat_pretrain_ckpt
        if qat_pretrain_ckpt
        else qat_pre_step_ckpt,
        allow_miss=False,
        ignore_extra=True,
        check_hash=False,
    ),
    dict(type="Float2QAT"),
]
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=qat_converters,
    ),
    data_loader=data_loader,
    optimizer=dict(type=torch.optim.Adam, lr=qat_lr, weight_decay=wd),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=0 if skip_qat_stage else qat_steps,
    device=None,  # set when building
    callbacks=qat_callbacks + metric_updaters,
)

# ----------------------- step int_inference (bpu deploy) ----------------
int_load_ckpt = dict(
    type="LoadCheckpoint",
    checkpoint_path=os.path.join(
        ckpt_dir, "qat-checkpoint-%s.pth.tar" % ckpt_suffix
    ),
    allow_miss=False,
    ignore_extra=True,  # qat has more params than int, such as loss params
    check_hash=False,
    verbose=False,
)
int_converters = [
    dict(type="Float2QAT"),
    int_load_ckpt,
    dict(type="QAT2Quantize"),
]
traced_callback = dict(
    type="SaveTraced",
    trace_inputs=copy.deepcopy(dict(img=torch.ones(1, 3, *input_hw))),
    save_dir=ckpt_dir,
    save_hash=False,
    name_prefix="int_infer-",
)
int_infer_trainer = dict(
    type="Trainer",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=int_converters,
    ),
    # export int model and test_model only, not need training, so others like
    # loader is unnecessary
    data_loader=None,
    optimizer=None,
    batch_processor=batch_processor,
    num_epochs=0,  # will skip training, and run `trainer.on_loop_begin/end()`
    device=None,
    callbacks=[
        # export only
        traced_callback
    ],
)

# project_id = "PDT20220004"
project_id = "PDT20220001"
docker_image = "docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-1.3.3"
curr_root = os.path.realpath(os.path.dirname(__file__))
work_dir = os.path.realpath(os.getcwd())
config_file = os.path.join(
    os.path.relpath(curr_root, work_dir),
    os.path.basename(os.path.basename(__file__)),
)
folder_list = [
    f"{work_dir}/hat",
    f"{work_dir}/tools",
    f"{work_dir}/projects",
    f"{work_dir}/plugins/k8s_submit/url2IP.py",
    f"{work_dir}/plugins/k8s_submit/ssh_launcher.py",
    f"{work_dir}/plugins/k8s_submit/watch_dog.py",
    f"{work_dir}/plugins/k8s_submit/elastic_launcher.py",
    f"{work_dir}/plugins/k8s_submit/default_elastic_patterns.yaml",
]

job_list = [
    f"python3 tools/train.py --config {config_file} --stage float",
    f"python3 tools/train.py --config {config_file} --stage freeze_bn",
    f"python3 tools/train.py --config {config_file} --stage qat",
]
deploy_path = os.path.join(ckpt_dir, "int_infer-deploy-checkpoint-last.pt")
compile_dir = os.path.dirname(ckpt_dir) + "_compile"
suffix_cmds_on_master = [
    f"python3 tools/train.py --config {config_file} --stage int_infer",
    f"mkdir -p {compile_dir}",
    f"python3 tools/compile_standalone.py {deploy_path} --input-size 1x3x192x960 --output /job_data/models/compile/ --opt O3 --march {march} --name real3d_with_desensitization --extra-args '--dev-remove-extra-output-cpu-op --max-time-per-fc 1000 --balance 10'",
]

k8s_config = dict(
    job_name=job_name,
    job_password="newk8s666",
    num_machines=num_machines,
    num_gpus_per_machine=num_gpus_per_machine,
    framework="pytorch",
    task_label="hat",
    project_id=project_id,
    input_bucket="mono,matrix,mono_3d_data",
    priority=5,
    docker_image=docker_image,
    max_jobtime=60 if pipeline_test else 10000,
    launcher="mpi",
    upload_folder_name="k8s_job",
    folder_list=folder_list,
    job_list=job_list,
    suffix_cmds_on_master=suffix_cmds_on_master,
)
