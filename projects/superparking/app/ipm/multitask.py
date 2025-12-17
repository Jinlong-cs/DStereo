"""
fsd entry.
"""
import copy
import getpass
import os
from collections import ChainMap, OrderedDict
from functools import partial

import torch
from horizon_plugin_pytorch.quantization import March
from lib.utils import get_list, init_task_config
from schedule import get_fuse_patterns_by_stage, train_stages

from hat.core.task_sampler import TaskSampler
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils import Config
from hat.utils.checkpoint import load_state_dict

# -------------------------- common --------------------------
cfg_dir = os.path.dirname(__file__)
BASE_CONFIG = Config.fromfile(os.path.join(cfg_dir, "base.py"))
march = March.BERNOULLI2

bucket_root = BASE_CONFIG.bucket_root
job_name = BASE_CONFIG.job_name
project_id = BASE_CONFIG.project_id
pipeline_test = BASE_CONFIG.pipeline_test
pipeline_test_dump = BASE_CONFIG.pipeline_test_dump
training_step = BASE_CONFIG.training_step
val_only = BASE_CONFIG.val_only
aidi_eval = BASE_CONFIG.aidi_eval
infer_model_ckpt = BASE_CONFIG.infer_model_ckpt
local_train = BASE_CONFIG.local_train
num_machines = BASE_CONFIG.num_machines
num_gpus_per_machine = BASE_CONFIG.num_gpus_per_machine
device_ids = BASE_CONFIG.device_ids
log_freq = BASE_CONFIG.log_freq
save_prefix = BASE_CONFIG.save_prefix
alpha = BASE_CONFIG.alpha
head_factor = BASE_CONFIG.head_factor
feat_channels = BASE_CONFIG.feat_channels
bn_kwargs = BASE_CONFIG.bn_kwargs

input_size = BASE_CONFIG.input_size  # WxH
input_sequence_length = BASE_CONFIG.input_sequence_length
stride2channels = BASE_CONFIG.stride2channels
bifpn_out_strides = BASE_CONFIG.bifpn_out_strides
neck_stack = BASE_CONFIG.neck_stack
redirect_config_logging_path = BASE_CONFIG.redirect_config_logging_path
qat_train_only = BASE_CONFIG.qat_train_only
infer_model_ckpt = BASE_CONFIG.infer_model_ckpt

CONFIGS = init_task_config(BASE_CONFIG.CONFIGS)

# -------------------------- exp --------------------------
lr = BASE_CONFIG.start_lr
momentum = BASE_CONFIG.momentum
wd = BASE_CONFIG.weight_decay
freeze_bn_lr = BASE_CONFIG.freeze_bn_lr
qat_lr = BASE_CONFIG.qat_lr
use_step_lr = BASE_CONFIG.use_step_lr
float_epoches = BASE_CONFIG.num_epoch_float
float_steps = BASE_CONFIG.num_steps_float
qat_steps = BASE_CONFIG.num_steps_qat
freeze_bn_steps = BASE_CONFIG.num_epoch_freezebn
num_epoch_qat = BASE_CONFIG.num_epoch_qat
warmup_by = BASE_CONFIG.warmup_by
warmup_steps = BASE_CONFIG.warmup_steps
do_freeze_bn = BASE_CONFIG.do_freeze_bn
add_freeze_bn_step = BASE_CONFIG.add_freeze_bn_step
freezebn_step_ids = BASE_CONFIG.freezebn_step_ids
sync_bn = BASE_CONFIG.sync_bn
enable_amp = BASE_CONFIG.enable_amp
save_interval = BASE_CONFIG.save_interval
interval_by = BASE_CONFIG.interval_by
do_validation = BASE_CONFIG.do_validation
do_freeze_backbone_neck = BASE_CONFIG.do_freeze_backbone_neck
pretrain_checkpoint = BASE_CONFIG.pretrain_checkpoint
float_resume_checkpoint = BASE_CONFIG.float_resume_checkpoint
# used by inference
float_checkpoint_path = None
# used by inference or int_infer.
qat_checkpoint_path = None
qat_resume_checkpoint = None
check_quantize_model = BASE_CONFIG.get(
    "check_quantize_model", True
)  # noqa: E221,E501
resume_optimizer = False
resume_epoch_or_step = False

# -------------------------- multitask --------------------------
task_name = "ipm_multitask"
seed = None
log_rank_zero_only = True
cudnn_benchmark = True
ckpt_dir = os.path.join(save_prefix, task_name)
log_dir = os.path.join(ckpt_dir, "logs")
custom_loader_length = None  # no-op

if pipeline_test:
    # quick debug
    float_steps = 2
    freeze_bn_steps = 2
    num_epoch_qat = 2
    warmup_steps = 1
    warmup_by = "step"
    custom_loader_length = 2
    save_interval = 1
    interval_by = "step"

# -------------------------- data --------------------------
task_sampler = TaskSampler(
    task_config={i.task_name: dict(sampling_factor=1) for i in CONFIGS},
    method="sample_all",
)

data_loader = dict(
    type="MultitaskLoader",
    loaders={i.task_name: i.data_loader for i in CONFIGS},
    task_sampler=task_sampler,
    mode="max_size",
    return_task=True,
    custom_length=custom_loader_length,
)
# Because two dataset use same head, the shared head' val_dataloaders
# use task_name_list.
val_task_config = dict()
for config in CONFIGS:
    if hasattr(config, "task_name_list"):
        for task_name in config.task_name_list:
            val_task_config[task_name] = dict(sampling_factor=1)
    else:
        val_task_config[config.task_name] = dict(sampling_factor=1)
val_task_sampler = TaskSampler(
    task_config=val_task_config, method="sample_all"
)
val_loaders = dict()
for config in CONFIGS:
    if isinstance(config.val_data_loader, list):
        assert len(config.task_name_list) == len(config.val_data_loader)
        for i in range(len(config.val_data_loader)):
            val_loaders[config.task_name_list[i]] = config.val_data_loader[i]
    else:
        val_loaders[config.task_name] = config.val_data_loader
val_data_loader = dict(
    type="MultitaskLoader",
    loaders=val_loaders,
    task_sampler=val_task_sampler,
    mode="validation",
    return_task=True,
    custom_length=custom_loader_length,
)

test_data_loader = dict(
    type="MultitaskLoader",
    loaders={
        i.task_name: i.test_data_loader
        for i in CONFIGS
        if hasattr(i, "test_data_loader")
    },
    mode="validation",
    return_task=True,
)

# -------------------------- model --------------------------
task_builders = [i.topo_builder for i in CONFIGS]


def get_topo_builder(mode, out_feat_idx=None):
    def _build_topo(nodes, inputs):
        name2out = OrderedDict()
        backbone_feats = nodes["backbone"](inputs["img"])
        bifpn_feats = nodes["bifpn_neck"](backbone_feats)

        for builder in task_builders:
            name2out.update(
                builder(
                    nodes=nodes, _inputs=inputs, feats=bifpn_feats, mode=mode
                )
            )
        return name2out

    return _build_topo


backbone = dict(
    type="VargNetV2",
    input_channels=3,
    input_sequence_length=input_sequence_length,
    num_classes=1000,
    factor=2,
    alpha=alpha,
    bias=True,
    bn_kwargs=bn_kwargs,
    group_base=8,
    include_top=False,
    head_factor=head_factor,
)

bifpn_neck = dict(
    type="BiFPN",
    fpn_name="bifpn_sum",
    in_strides=[2, 4, 8, 16, 32],
    out_strides=bifpn_out_strides,
    stride2channels=stride2channels,
    out_channels=feat_channels,
    stack=neck_stack,
    start_level=1,
    end_level=-1,
    num_outs=5,
)


def get_per_task_flat_condition(mode):
    condition_dict = dict()
    assert mode in ["train", "val", "test"]
    if mode == "train":
        condi_fn = "train_flat_condition"
    elif mode == "val":
        condi_fn = "val_flat_condition"
    else:
        condi_fn = "test_flat_condition"
    for i in CONFIGS:
        if hasattr(i, condi_fn):
            assert callable(
                i.get(condi_fn)
            ), "Priveded condition function must be callable"
            condition_dict.update({i.task_name: i.get(condi_fn)})
    return condition_dict


model = dict(
    type="GraphModel",
    nodes=dict(
        backbone=backbone,
        bifpn_neck=bifpn_neck,
        **dict(ChainMap(*[i.nodes for i in CONFIGS])),
    ),
    inputs=dict(img=None, **dict(ChainMap(*[i.inputs for i in CONFIGS]))),
    topology_builder=get_topo_builder(mode="train"),
    lazy_forward=True,
    output_names2flat_conditions=get_per_task_flat_condition(mode="train"),
)

# val model for multitask is not supported yet
val_model = dict(
    type="GraphModel",
    nodes=dict(
        backbone=backbone,
        bifpn_neck=bifpn_neck,
        **dict(ChainMap(*[i.val_nodes for i in CONFIGS])),
    ),
    inputs=dict(img=None, **dict(ChainMap(*[i.val_inputs for i in CONFIGS]))),
    topology_builder=get_topo_builder(mode="val"),
    lazy_forward=True,
    output_names2flat_conditions=get_per_task_flat_condition(mode="val"),
)

deploy_model = dict(
    type="GraphModel",
    nodes=dict(
        backbone=backbone,
        bifpn_neck=bifpn_neck,
        **dict(ChainMap(*[i.test_nodes for i in CONFIGS])),
    ),
    inputs=dict(
        # sometimes, test img shape may be different from train img
        # `img` can be None when `lazy_forward` is True
        img=None,
        **dict(
            ChainMap(
                *[i.test_inputs for i in CONFIGS if hasattr(i, "test_inputs")]
            )
        ),
    ),
    topology_builder=get_topo_builder(mode="test"),
    lazy_forward=True,
    output_names2flat_conditions=get_per_task_flat_condition(mode="test"),
)


test_inputs = {"img": torch.randn((1, 3, input_size[1], input_size[0]))}

# -------------------------- callbacks --------------------------
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    enable_amp=enable_amp,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)

metric_updaters = [i.metric_updater for i in CONFIGS]
# because two dataset use same head,
# the shared head' val_metric_updaters use val_metric_updater_list
val_metric_updaters = []
for config in CONFIGS:
    if hasattr(config, "val_metric_updater_list"):
        for val_metric_updater in config.val_metric_updater_list:
            val_metric_updaters.append(val_metric_updater)
    else:
        val_metric_updaters.append(config.val_metric_updater)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)

lr_callback = dict(
    type="CosLrUpdater",
    warmup_by="step",
    warmup_len=warmup_steps,
    max_steps=float_steps - warmup_steps,
    step_log_interval=log_freq,
)

qat_lr_callback = copy.deepcopy(lr_callback)
qat_lr_callback["max_steps"] = qat_steps - warmup_steps

if use_step_lr:
    lr_callback = dict(
        type="StepDecayLrUpdater",
        warmup_mode="linear",
        warmup_by="epoch",
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

    qat_lr_callback = dict(
        type="StepDecayLrUpdater",
        warmup_mode="linear",
        warmup_by="epoch",
        warmup_len=warmup_steps,
        step_log_interval=log_freq,
        update_by="epoch",
        lr_decay_id=[
            int(8 / 12.0 * num_epoch_qat),
            int(11 / 12.0 * num_epoch_qat),
        ]
        if num_epoch_qat > 3
        else None,
    )

tb_update_funcs = []
for config in CONFIGS:
    if hasattr(config, "tb_update_func"):
        tb_update_funcs.append(config.tb_update_func)

tensorboard_log_path = os.path.join(ckpt_dir, "tensorboard")
tensorboard_val_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_log_path, training_step, "val")
    if local_train
    else os.path.join(
        os.getenv("TENSORBOARD_LOG_PATH")
        or os.path.join(tensorboard_log_path, ".aidi"),
        training_step,
        "val",
    ),
    update_by="step",
    update_freq=1,
    tb_update_funcs=tb_update_funcs,
)
tensorboard_loss_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_log_path, training_step, "loss")
    if local_train
    else os.path.join(
        os.getenv("TENSORBOARD_LOG_PATH")
        or os.path.join(tensorboard_log_path, ".aidi"),
        training_step,
        "loss",
    ),
    loss_name_reg="^.*_loss.*",
    update_freq=log_freq,
    update_by="step",
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=val_metric_updaters,
    val_model=None,
)
Validation = val_callback.copy()

best_metric = None
# load last or best ckpt?
ckpt_suffix = "last" if best_metric is None else "best"
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    # disable trace and dump in checkpoint callback when float/qat training
    # deploy_model=deploy_model if training_step == "int_infer" else None,
    # deploy_inputs=test_inputs,
    # model params match deploy_model params
    strict_match=False,
    best_refer_metric=best_metric,
    save_interval=save_interval,
    interval_by=interval_by,
    save_on_train_end=True,
)

user_name = getpass.getuser()
cluster = os.getenv("CLUSTER", None)
job_id = os.getenv("JOB_ID", None)
if job_id is not None:
    job_id = int(job_id)

aidi_expmodel_callback = dict(
    type="AIDIExpModel",
    model_name=task_name + "_" + user_name,
    task_type="multitask",
    save_model="last",
    desc=BASE_CONFIG.get("aidi_expmodel_desc", ""),
    tags=None,
    model_version=BASE_CONFIG.get("aidi_expmodel_model_version", None),
    basic_versions=None,
    job_id=job_id,
    cluster_info=cluster,
    commit_message=BASE_CONFIG.get("aidi_expmodel_commit_message", None),
    platforms=["J3"],
    config_paths=[BASE_CONFIG._filename],
    attachments=None,
)

freeze_bn_callback = dict(
    type="FreezeModule",
    modules=[["backbone", "bifpn_neck"]],
    step_or_epoch=freezebn_step_ids,
    update_by=interval_by,
    only_batchnorm=True,
)

freeze_backbone_neck_callback = dict(
    type="FreezeModule",
    modules=[["backbone", "bifpn_neck"]],
    step_or_epoch=[
        0,
    ],
    update_by=interval_by,
    only_batchnorm=False,
)

# If you want to use it, add to callbacks and qat_callbacks. Default not use.
freeze_module_callback = dict(
    type="FreezeModule",
    modules=[["backbone", "bifpn_neck"]],
    freeze_step_or_epoch=[0],
    update_by=interval_by,
)

# the order of callbacks affects the logging order
callbacks = [
    stat_callback,
    lr_callback,
    ckpt_callback,
    tensorboard_loss_callback,
]
callbacks.extend(metric_updaters)
qat_callbacks = [
    stat_callback,
    qat_lr_callback,
    ckpt_callback,
    tensorboard_loss_callback,
]
qat_callbacks.extend(metric_updaters)

if do_validation or val_only:
    if val_only:
        # only do val without training
        callbacks = [val_callback]
        qat_callbacks = [val_callback]
    else:
        # do val while training
        callbacks.append(val_callback)
        qat_callbacks.append(val_callback)

if do_freeze_bn and training_step == "float":
    callbacks.append(freeze_bn_callback)
elif add_freeze_bn_step and training_step == "float_freeze_bn":
    callbacks.append(freeze_bn_callback)

if do_freeze_backbone_neck:
    callbacks.append(freeze_backbone_neck_callback)


# -------------------------- trainer --------------------------
base_trainer = dict(
    type="DistributedDataParallelTrainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=wd)},
        lr=lr,
        momentum=momentum,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=float_steps,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=callbacks,
)
if qat_train_only:
    stage_extend_params = dict(
        with_bn=dict(
            checkpoint_path=pretrain_checkpoint,
            allow_miss=True,
            ignore_extra=True,
        ),
        freeze_bn_1=dict(
            checkpoint_path=os.path.join(
                ckpt_dir, "with_bn-checkpoint-last.pth.tar"
            ),
        ),
    )
    for stage in train_stages:
        pre, cur = get_fuse_patterns_by_stage(stage)
        params = stage_extend_params.get(stage, {})
        checkpoint_mode = params.get("checkpoint_mode", "pre_stage")

        trainer = copy.deepcopy(base_trainer)
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
                    checkpoint_path=params.get("checkpoint_path", None),
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
else:
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
            resume_optimizer=resume_optimizer,
            resume_epoch_or_step=resume_epoch_or_step,
        )

    # -------------------------- step quantize-aware -------------------
    checkpoint_path = os.path.join(
        ckpt_dir, "float-checkpoint-%s.pth.tar" % ckpt_suffix
    )
    if float_resume_checkpoint is not None:
        checkpoint_path = float_resume_checkpoint
    converters = [
        dict(
            type="LoadCheckpoint",
            checkpoint_path=checkpoint_path,
            allow_miss=True,
            ignore_extra=True,
        ),
        dict(type="Float2QAT"),
    ]
    qat_trainer = copy.deepcopy(base_trainer)
    qat_trainer.update(callbacks=qat_callbacks)
    qat_trainer.update(num_steps=qat_steps)
    qat_trainer.update(
        optimizer=dict(
            type=torch.optim.SGD,
            params={"weight": dict(weight_decay=wd)},
            lr=qat_lr,
            momentum=momentum,
        ),
    )
    qat_trainer.update(
        model_convert_pipeline=dict(
            type="ModelConvertPipeline",
            qat_mode="fuse_bn",
            converters=converters,
        )
    )
    if qat_resume_checkpoint:
        qat_trainer.update(
            model_convert_pipeline=dict(
                type="ModelConvertPipeline",
                qat_mode="fuse_bn",
                converters=[
                    dict(type="Float2QAT"),
                    dict(
                        type="LoadCheckpoint",
                        checkpoint_path=qat_resume_checkpoint,
                        allow_miss=True,
                        ignore_extra=True,
                    ),
                ],
            ),
        )
        qat_trainer.update(
            resume_optimizer=resume_optimizer,
            resume_epoch_or_step=resume_epoch_or_step,
        )
# ----------------------- step int_inference (bpu deploy) ----------------
int_checkpoint = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=test_inputs,
    name_prefix=training_step + "-",
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
                checkpoint_path=BASE_CONFIG.qat_pretrain_checkpoint
                or os.path.join(
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
def dataset_wrap_fn(task_name, data_loader):
    ts_configs = {
        task_name: dict(sampling_factor=1),
        "chosen_tasks": [[task_name]],
    }
    ts = TaskSampler(
        task_config=ts_configs,
        method="sample_repeat",
    )
    mt_loader = copy.deepcopy(val_data_loader)
    mt_loader["loaders"] = {task_name: data_loader}
    mt_loader["task_sampler"] = ts
    return mt_loader


aidi_eval_dataloaders = get_list(
    "aidi_eval_loaders",
    CONFIGS,
    merge_list=True,
    wrap_fn=dataset_wrap_fn,
)
aidi_eval_callbacks = get_list("aidi_eval_callbacks", CONFIGS, merge_list=True)
page_label_list = get_list("aidi_eval_page_label", CONFIGS)
datasets_id_list = get_list("aidi_eval_dataset_id", CONFIGS)
type_list = get_list("aidi_eval_type", CONFIGS)
old_prediction_list = get_list("old_prediction", CONFIGS)
new_prediction_list = get_list("new_prediction", CONFIGS)

report_name = job_name
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

if aidi_eval:
    val_loaders = aidi_eval_dataloaders
    val_callbacks = aidi_eval_callbacks
else:
    val_loaders = [val_data_loader]
    val_callbacks = val_metric_updaters

load_pred_ckpt_func = partial(
    load_state_dict, allow_miss=True, ignore_extra=True
)

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

# -------------------------- hbdk --------------------------
compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,  # Name of the model, recorded in hbm
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],  # or ddr? custom by yourself
    opt="O3",  # change O3 for faster fps
)

# -------------------------- k8s_submit config --------------------------
cfg_file = (
    "./projects/superparking/app/ipm"
    + os.path.abspath(__file__).split("/projects/superparking/app/ipm")[1]
)  # noqa
job_list = [
    f"python3 -W ignore tools/train.py --config {cfg_file} --stage float",
    f"python3 -W ignore tools/train.py --config {cfg_file} --stage qat",
    # f"python3 -W ignore tools/train.py --config {cfg_file} --stage int_infer --val-only",  # noqa
]
if add_freeze_bn_step:
    job_list.insert(
        1,
        f"python3 -W ignore tools/train.py --config {cfg_file}"
        f" --stage float_freeze_bn",
    )

use_docker = "cu116"  # for gpu >= 3090
dockers = dict(
    cu111="docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-1.2.1",
    cu116="docker.hobot.cc/imagesys/hat:runtime-py3.8-torch1.13.0-cu116-1.3.1-numpy1.20.3",
)
docker = dockers[use_docker]

k8s_config = dict(
    job_name=job_name,
    job_password="newk8s666",
    num_machines=num_machines,
    num_gpus_per_machine=num_gpus_per_machine,
    framework="pytorch",
    task_label="ipm_multitask",
    project_id=project_id,
    input_bucket="SuperParking",
    # priority description:
    # 3: common experiment job, default
    # 4: model publish job
    # 5: data-related job, used by model publish
    priority=5,
    docker_image=docker,
    # default 7200 = 5days
    max_jobtime=20000,
    # launcher only for multi-machines
    launcher="mpi",
    # upload folder
    upload_folder_name="k8s_job",
    folder_list=[
        "../../hat",
        "../../tools",
        "../../projects",
        "url2IP.py",
        # "../../Makefile",
    ],
    job_list=job_list,
)

# set True to do perf job
if vars().get("perf_dataloader", False):
    k8s_config["job_list"] = [
        f"python3 -W ignore tools/analyze/perf_dataloader.py --config {cfg_file} --step float --iter-nums 2000 --frequent 25",  # noqa
    ]
if vars().get("perf_model_training", False):
    k8s_config["job_list"] = [
        f"python3 -W ignore tools/analyze/perf_model_training.py --config {cfg_file} --step float --iter-nums 2000 --frequent 25",  # noqa
    ]
