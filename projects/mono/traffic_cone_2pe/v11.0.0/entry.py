import copy
import getpass
import os
from collections import ChainMap, OrderedDict

import torch
from horizon_plugin_pytorch.quantization import March

from hat.core.task_sampler import TaskSampler
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils import Config

cfg_dir = os.path.dirname(__file__)
BASE_CONFIG = Config.fromfile(os.path.join(cfg_dir, "auto2d_base.py"))
task_list = BASE_CONFIG.task_list
CONFIGS = [
    Config.fromfile(os.path.join(cfg_dir, f"{task}.py")) for task in task_list
]
pipeline_test = BASE_CONFIG.pipeline_test
training_step = BASE_CONFIG.training_step
val_only = BASE_CONFIG.val_only
local_train = BASE_CONFIG.local_train
device_ids = BASE_CONFIG.device_ids
log_freq = BASE_CONFIG.log_freq
save_prefix = BASE_CONFIG.save_prefix
val_interval = BASE_CONFIG.val_interval
val_interval_by = BASE_CONFIG.val_interval_by
input_size = BASE_CONFIG.input_size  # WxH
num_machines = BASE_CONFIG.num_machines
num_gpus_per_machine = BASE_CONFIG.num_gpus_per_machine
input_sequence_length = BASE_CONFIG.input_sequence_length
append_extra_node_func_list = [BASE_CONFIG.append_extra_node_func]
backbone = BASE_CONFIG.backbone
pretrain_checkpoint = BASE_CONFIG.pretrain_checkpoint
backbone_neck_modules = BASE_CONFIG.backbone_neck_modules

# -------------------------- exp --------------------------
job_name = "_".join(os.path.abspath(os.curdir).split("/")[-2:])

lr = 1e-2
freeze_bn_lr = 2e-4
qat_lr = 2e-4
wd = 1e-2
use_step_lr = False
float_steps = 40000
freeze_bn_steps = 10000
qat_num_steps = 10000
warmup_steps = 1000
save_interval = 1000
do_freeze_bn = False
freezebn_step_ids = [0]
do_fuse_bn = False
fusebn_step_ids = [0]
interval_by = "step"
do_validation = False
do_freeze_backbone_neck = False
do_freeze_backbone_neck_qat = False
check_quantize_model = True
enable_model_tracking = True
sync_bn = True
add_freeze_bn_step = False
do_float_step = True
do_task_similarity = False
enable_amp = False
do_grad_scale = False

enable_checkpoint = False
input_bucket = "mono,auto_eval"
float_resume_checkpoint = None
freezebn_resume_checkpoint = None
qat_resume_checkpoint = None

# -------------------------- multitask --------------------------
task_name = "traffic_cone_2pe"
seed = None
log_rank_zero_only = True
cudnn_benchmark = True
ckpt_dir = os.path.join(save_prefix, task_name)
log_dir = os.path.join(ckpt_dir, "logs")
custom_loader_length = None  # no-op
march = os.environ.get("HAT_MARCH", March.BAYES)
assert march in [March.BERNOULLI, March.BERNOULLI2, March.BAYES]

if pipeline_test:
    # quick debug
    float_steps = 2
    freeze_bn_steps = 2
    qat_num_steps = 2
    warmup_steps = 1
    custom_loader_length = 2
    save_interval = 1
    interval_by = "step"

if enable_checkpoint:
    raise AssertionError(
        "Bug in fsd_multitask, see http://wiki.hobot.cc/pages/viewpage.action?pageId=199355601"  # noqa: E501
    )
    if training_step in ["float", "qat"]:
        os.environ["HAT_USE_CHECKPOINT"] = "1"
    else:
        os.environ["HAT_USE_CHECKPOINT"] = "0"


# -------------------------- data --------------------------
task_sampler = TaskSampler(
    task_config={
        i.task_name: dict(sampling_factor=1) for i in CONFIGS
    },  # noqa
    method="sample_all",
    unions=None,
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
        for _task_name in config.task_name_list:
            val_task_config[_task_name] = dict(sampling_factor=1)
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

        for builder in task_builders:
            name2out.update(
                builder(
                    nodes=nodes,
                    _inputs=inputs,
                    feats=backbone_feats,
                    mode=mode,
                )
            )
        for extra_node_func in append_extra_node_func_list:
            extra_nodes = extra_node_func(
                nodes=nodes, feats=backbone_feats, mode=mode
            )
            name2out.update(extra_nodes)
        return name2out

    return _build_topo


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
        backbone=copy.deepcopy(backbone),
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
        backbone=copy.deepcopy(backbone),
        **dict(ChainMap(*[i.val_nodes for i in CONFIGS])),
    ),
    inputs=dict(img=None, **dict(ChainMap(*[i.val_inputs for i in CONFIGS]))),
    topology_builder=get_topo_builder(mode="val"),
    lazy_forward=True,
    output_names2flat_conditions=get_per_task_flat_condition(mode="val"),
)

test_model = dict(
    type="GraphModel",
    nodes=dict(
        backbone=copy.deepcopy(backbone),
        **dict(ChainMap(*[i.val_nodes for i in CONFIGS])),
    ),
    inputs=dict(
        img=None,
        im_hw=(input_size[1], input_size[0]),
        img_buf=None,
        affine_aug_param=None,
        obj_id=None,
        **dict(ChainMap(*[i.val_inputs for i in CONFIGS])),
    ),
    topology_builder=get_topo_builder(mode="test"),
    lazy_forward=True,
    output_names2flat_conditions=get_per_task_flat_condition(mode="test"),
)

# used for trace and compile
# TODO(min.du): add more key to dump gpu-inference used pt #
# test_inputs = {
#     "img": [
#         torch.randn((1, 3, input_size[1], input_size[0]))
#         for _ in range(input_sequence_length)
#     ],
# }
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

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)
lr_type = "cos"
if lr_type == "cos":
    lr_callback = dict(
        type="CosLrUpdater",
        max_steps=float_steps - warmup_steps,
        warmup_by="step",
        warmup_len=warmup_steps,
        warmup_begin_lr=0.00005,
        step_log_interval=log_freq,
    )
    qat_lr_callback = dict(
        type="CosLrUpdater",
        max_steps=qat_num_steps - warmup_steps,
        warmup_by="step",
        warmup_len=warmup_steps,
        warmup_begin_lr=0.00005,
        step_log_interval=log_freq,
    )
elif lr_type == "poly":
    lr_callback = dict(
        type="PolyLrUpdater",
        max_update=float_steps - warmup_steps,
        power=1.0,
        final_lr=1e-6,
        warmup_by="step",
        warmup_len=warmup_steps,
        warmup_begin_lr=0.00005,
        step_log_interval=log_freq,
    )
    qat_lr_callback = dict(
        type="PolyLrUpdater",
        max_update=qat_num_steps - warmup_steps,
        power=1.0,
        final_lr=1e-8,
        warmup_by="step",
        warmup_len=warmup_steps,
        warmup_begin_lr=0.00005,
        step_log_interval=log_freq,
    )
else:
    raise ValueError(f"{lr_type} is not support")

if add_freeze_bn_step and training_step == "float_freeze_bn":
    lr_callback["max_steps"] = freeze_bn_steps - warmup_steps

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

    qat_lr_callback = dict(
        type="StepDecayLrUpdater",
        warmup_by="step",
        warmup_begin_lr=0.00005,
        warmup_len=warmup_steps,
        step_log_interval=log_freq,
        update_by="step",
        lr_decay_id=[
            int(8 / 12.0 * qat_num_steps),
            int(11 / 12.0 * qat_num_steps),
        ]
        if qat_num_steps > 3
        else None,
    )

tb_update_funcs = []
for config in CONFIGS:
    if hasattr(config, "tb_update_func"):
        tb_update_funcs.append(config.tb_update_func)

tensorboard_callback = dict(
    type="TensorBoard",
    save_dir=log_dir if local_train else "/job_tboard/",  # noqa
    update_freq=log_freq,
    tb_update_funcs=tb_update_funcs,
)

val_metric_updaters = []
val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=val_metric_updaters,
    val_interval=val_interval,
    val_model=val_model,
    interval_by=val_interval_by,
)

best_metric = None
# load last or best ckpt?
ckpt_suffix = "last" if best_metric is None else "best"
checkpoint_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=False,
    best_refer_metric=best_metric,
    save_interval=save_interval,
    interval_by=interval_by,
    save_on_train_end=True,
    save_hash=False,
)

qat_checkpoint_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=f"{training_step}-{march}-",
    strict_match=False,
    best_refer_metric=best_metric,
    save_interval=save_interval,
    interval_by=interval_by,
    save_on_train_end=True,
    save_hash=False,
)

user_name = getpass.getuser()
aidi_expmodel_callback = dict(
    type="AIDIExpModel",
    model_name="Traffic_Cone_Exp_2pe_6cls",
    task_type="classification",
    save_model="last",
    desc="",
    tags=None,
    model_version="v11.0.0",
    platforms=["J3"],
    attachments=None,
)

freeze_bn_callback = dict(
    type="FreezeModule",
    modules=backbone_neck_modules,
    step_or_epoch=freezebn_step_ids,
    update_by=interval_by,
    only_batchnorm=True,
)

fuse_bn_callback = dict(
    type="FuseBN",
    modules=backbone_neck_modules,
    step_or_epoch=fusebn_step_ids,
    update_by=interval_by,
)

freeze_backbone_neck_callback = dict(
    type="FreezeModule",
    modules=backbone_neck_modules,
    step_or_epoch=[
        0,
    ],
    update_by=interval_by,
    only_batchnorm=False,
)

# If you want to use it, add to callbacks and qat_callbacks. Default not use.
freeze_module_callback = dict(
    type="FreezeModule",
    modules=backbone_neck_modules,
    freeze_step_or_epoch=[0],
    update_by=interval_by,
)

# Warning: not stable for publish job, and only for 2d3d job now.
grad_scale_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=0.1,
)

compactor_update_callback = dict(
    type="CompactorUpdater",
    before_mask_iters=200,
    mask_interval=200,
    pruned_epsilon=1e-5,
    modules=backbone_neck_modules,
)

# the order of callbacks affects the logging order
callbacks = [
    # aidi_expmodel_callback,
    stat_callback,
    lr_callback,
    checkpoint_callback,
    tensorboard_callback,
]
qat_callbacks = [
    # aidi_expmodel_callback,
    stat_callback,
    qat_lr_callback,
    qat_checkpoint_callback,
    tensorboard_callback,
]

if do_validation or val_only:
    if val_only:
        # only do val without training
        callbacks = [val_callback]
        qat_callbacks = [val_callback]
    else:
        # do val while training
        callbacks.append(val_callback)
        qat_callbacks.append(val_callback)

metric_updaters = [i.metric_updater for i in CONFIGS]
callbacks.extend(metric_updaters)
qat_callbacks.extend(metric_updaters)

if do_freeze_bn and training_step == "float":
    callbacks.append(freeze_bn_callback)
elif add_freeze_bn_step and training_step == "float_freeze_bn":
    callbacks.append(freeze_bn_callback)

if do_freeze_backbone_neck:
    callbacks.append(freeze_backbone_neck_callback)

if do_freeze_backbone_neck_qat:
    qat_callbacks.append(freeze_backbone_neck_callback)

if do_fuse_bn:
    callbacks.append(fuse_bn_callback)

if do_grad_scale:
    callbacks.append(grad_scale_callback)
    qat_callbacks.append(grad_scale_callback)

if do_task_similarity:
    callbacks.insert(1, compactor_update_callback)


# -------------------------- solver --------------------------
# -------------------------- step float --------------------------
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
    type="distributed_data_parallel_trainer",
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=float_converters,
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        lr=lr,
        weight_decay=wd,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=float_steps,
    device=None,  # set when building
    sync_bn=sync_bn,
    callbacks=callbacks,
    # profiler=dict(tyep='PythonProfiler', dirpath='./', filename='ped'),
    train_metrics=[
        dict(type="Accuracy"),
    ],
    val_metrics=[
        dict(type="Accuracy"),
    ],
)

# -------------------------- step float_freeze_bn  --------------------------
float_freeze_bn_trainer = copy.deepcopy(float_trainer)
float_freeze_bn_trainer["optimizer"]["lr"] = freeze_bn_lr
float_freeze_bn_trainer["num_steps"] = freeze_bn_steps

float_freeze_bn_solver = dict(
    trainer=float_freeze_bn_trainer,
    quantize=False,
    check_quantize_model=False,
    resume_checkpoint=freezebn_resume_checkpoint,
    resume_optimizer=False,
    resume_epoch_or_step=False,
    pretrain_checkpoint=None,
    state_dict_update_func=None,
    allow_miss=True if freezebn_resume_checkpoint else False,
    ignore_extra=True,
    verbose=1,
    check_hash=False,
)

# -------------------------- step quantize-aware -------------------
qat_converters = [
    dict(
        type="LoadCheckpoint",
        checkpoint_path=os.path.join(
            ckpt_dir, "float-checkpoint-%s.pth.tar" % ckpt_suffix
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

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=qat_converters,
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        lr=qat_lr,
        weight_decay=wd,
    ),
    batch_processor=batch_processor,
    stop_by="step",
    num_steps=qat_num_steps,
    device=None,  # set when building
    callbacks=qat_callbacks,
    train_metrics=[
        dict(type="Accuracy"),
    ],
    val_metrics=[
        dict(type="Accuracy"),
    ],
)

if add_freeze_bn_step:
    qat_converters[0]["checkpoint_path"] = os.path.join(
        ckpt_dir, "float_freeze_bn-checkpoint-%s.pth.tar" % ckpt_suffix
    )

# ----------------------- step int_inference (bpu deploy) ----------------
int_load_ckpt = dict(
    type="LoadCheckpoint",
    checkpoint_path=os.path.join(
        ckpt_dir, f"qat-{march}-checkpoint-%s.pth.tar" % ckpt_suffix
    ),
    check_hash=False,
    allow_miss=True,
    ignore_extra=True,  # qat has more params than int, such as loss params
    verbose=True,
)
int_converters = [
    dict(type="Float2QAT"),
    int_load_ckpt,
    dict(type="QAT2Quantize"),
]
traced_callback = dict(
    type="SaveTraced",
    trace_inputs=copy.deepcopy(test_inputs),
    save_dir=ckpt_dir,
    save_hash=False,
    name_prefix=f"{training_step}-{march}-",
)
int_infer_trainer = dict(
    type="Trainer",
    model=test_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=int_converters,
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[
        # export only
        traced_callback
    ],
)

deploy_model = copy.deepcopy(test_model)
deploy_inputs = copy.deepcopy(test_inputs)
inputs = copy.deepcopy(test_inputs)

# -------------------------- hbdk --------------------------
compile_dir = os.path.join(ckpt_dir, f"compile-{march}")
compile_cfg = dict(
    march=march,
    name=task_name,  # Name of the model, recorded in hbm
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "cone_bollard_classification.hbm"),
    layer_details=False,
    input_source=["resizer"],  # or ddr? custom by yourself
    opt="O3",  # change O3 for faster fps
)

# --------------------------- onnx --------------------------
onnx_cfg = dict(
    model=deploy_model,
    stage="float",
    inputs=deploy_inputs,
    model_convert_pipeline=float_trainer["model_convert_pipeline"],
)

# 4. predictor
predictor = dict(
    type="Predictor",
    model=val_model,
    data_loader=None,
    batch_processor=val_batch_processor,
    device=None,
    metrics=None,
    callbacks=None,
    log_interval=50,
    share_callbacks=False,
)

predict_solver = dict(
    predictor=predictor,
    train_step="qat",
    checkpoint="tmp_output/fsd_multitask/qat-checkpoint-last.pth.tar",
    ckpt_step="qat",
    ignore_extra=True,
)

# -------------------------- k8s_submit config --------------------------
cfg_file = (
    "./projects/" + os.path.abspath(__file__).split("/projects/")[1]
)  # noqa

job_list = [
    "ulimit -u 655360",
    f"python3 -W ignore tools/train.py --config {cfg_file} -s float"
    if do_float_step
    else "echo ''",  # noqa
    f"python3 -W ignore tools/train.py --config {cfg_file} -s float_freeze_bn"  # noqa
    if add_freeze_bn_step
    else "echo ''",  # noqa
    f"export HAT_MARCH=bernoulli && python3 -W ignore tools/train.py --config {cfg_file} -s qat",  # noqa
    f"export HAT_MARCH=bernoulli && python3 -W ignore tools/train.py --config {cfg_file} -s int_infer",  # noqa
    f"export HAT_MARCH=bernoulli2 && python3 -W ignore tools/train.py --config {cfg_file} -s qat",  # noqa
    f"export HAT_MARCH=bernoulli2 && python3 -W ignore tools/train.py --config {cfg_file} -s int_infer",  # noqa
    f"export HAT_MARCH=bayes && python3 -W ignore tools/train.py --config {cfg_file} -s qat",  # noqa
    f"export HAT_MARCH=bayes && python3 -W ignore tools/train.py --config {cfg_file} -s int_infer",  # noqa
]

dockers = dict(
    # cu_old="docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-aeb8dd9",
    # cu111="docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-0d1b91b",
    # cu111="docker.hobot.cc/dlp/hat:runtime-py3.8-torch1.10.2-cu111-1.3.3",
    cu111="docker.hobot.cc/imagesys/hat:fsd_multitask-cu11-20230823-white-box-v3.9",  # noqa
)


k8s_config = dict(
    job_name=job_name,
    job_password="6150",
    num_machines=num_machines,
    num_gpus_per_machine=num_gpus_per_machine,
    framework="pytorch",
    task_label="traffic_cone_2pe",
    project_id="PDT20220004",
    input_bucket=input_bucket,
    priority=5,
    docker_image=dockers["cu111"],
    # default 7200 = 5days
    max_jobtime=60 if num_gpus_per_machine <= 2 else 20080,
    # launcher only for multi-machines
    launcher="mpi",
    # upload folder
    upload_folder_name="k8s_job",
    folder_list=[
        "./hat",
        "./tools",
        "./projects",
        # "./turbojpeg.py",
        # "./libturbojpeg.so",
    ],
    job_list=job_list,
)

# set True to do perf job
if vars().get("perf_dataloader", False):
    k8s_config["job_list"] = [
        f"python3 -W ignore tools/perf_dataloader.py --config {cfg_file} --step float --iter-nums 2000 --frequent 25",  # noqa
    ]
if vars().get("perf_model_training", False):
    k8s_config["job_list"] = [
        f"python3 -W ignore tools/perf_model_training.py --config {cfg_file} --step float --iter-nums 2000 --frequent 25",  # noqa
    ]
