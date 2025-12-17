import json
import os
from collections import ChainMap, OrderedDict

import torch

from hat.core.task_sampler import TaskSampler
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils.apply_func import _as_list
from hat.utils.config import Config, ConfigVersion
from tests.data.toy_modules import *  # noqa: F403,F401

# -------------------------- global params --------------------------
cfg_dir = os.path.dirname(__file__)
BASE_CFG = Config.fromfile(os.path.join(cfg_dir, "base.py"))
pipeline_test = BASE_CFG.pipeline_test
training_step = BASE_CFG.training_step
global_desc = BASE_CFG.global_desc
VERSION = ConfigVersion.v2

task_name = "toy_multitask"
img_name = "img"

device_ids = (
    list(range(torch.cuda.device_count())) if not pipeline_test else [0]
)  # noqa
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = os.getpid()
log_rank_zero_only = True  # to simplify logging info
sync_bn = True

# see `horizon.quantization.perf_model`
compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march="bayes",
    name=task_name,  # Name of the model, recorded in hbm
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],  # or ddr? custom by yourself
)

# -------------------------- model --------------------------
cfg_dir = os.path.dirname(__file__)
CONFIGS = [
    # add or remove task hear
    Config.fromfile(os.path.join(cfg_dir, "cls_task1.py")),
    Config.fromfile(os.path.join(cfg_dir, "cls_task2.py")),
]
task_builders = [i.topo_builder for i in CONFIGS]


def get_topo_builder(is_train, out_feat_idx=None):
    def _build_topo(nodes, inputs):
        r"""
        1. model topology:
                                        (task1_label)
                                             |
                                            \|/
        (img) -> backbone -> neck -> task1_out_module -> (task1_out)
                                \ -> task2_out_module -> (task2_out)
                                            /|\
                                             |
                                        (task2_label)

        2. test_model topology (without label):

        (img) -> backbone -> neck -> task1_out_module -> (task1_out)
                                \ -> task2_out_module -> (task2_out)

        """  # noqa
        name2out = OrderedDict()
        img = inputs["img"]
        # IMPORTANT: if :attr:`GraphModel.lazy_forward` is True, then
        # `nodes['backbone'](img)` will not run the `forward` func of
        # `backbone`. If you want to forward to debug, turn off `lazy_forward`.
        feats = nodes["backbone"](img)
        for builder in task_builders:
            name2out.update(
                builder(
                    nodes=nodes, inputs=inputs, feats=feats, is_train=is_train
                )
            )

        # Test Case: output backbone feats as tracking feats, delete this for
        # formal config.
        if out_feat_idx is not None:
            for i in _as_list(out_feat_idx):
                add_desc_i = nodes[f"add_out_feat_desc_{i}"]
                name2out["backbone_feat_%d" % i] = add_desc_i(feats[i])

        return name2out

    return _build_topo


backbone = dict(
    type="ToyBackbone",
    strides=(1, 2, 4, 8, 16, 32),
    channels=(3, 8, 8, 16, 32, 64),
)
model = dict(
    type="GraphModel",
    nodes=dict(
        backbone=backbone, **dict(ChainMap(*[i.nodes for i in CONFIGS]))
    ),
    inputs=dict(
        img=None,  # can be None when `lazy_forward` is True
        **dict(ChainMap(*[i.inputs for i in CONFIGS])),
    ),
    topology_builder=get_topo_builder(is_train=True),
    # not forward below nodes among model's topology building, delay to model's
    # `forward` time, so that you don't have to specific input shapes
    # (DataLoader known that).
    lazy_forward=True,
)
# add desc for backbone feature, just a test, is unnecessary for common model
out_feat_idx = []
add_out_feat_desc = {
    f"add_out_feat_desc_{i}": dict(
        type="AddDesc",
        per_tensor_desc=[
            json.dumps(
                dict(
                    task="tracking_feature",
                    output_name="backbone_feat_%d" % i,  # custom by yourself
                    **global_desc,
                )
            )
        ],
    )
    for i in out_feat_idx
}
deploy_model = dict(
    type="GraphModel",
    nodes=dict(
        backbone=backbone,
        **add_out_feat_desc,
        **dict(ChainMap(*[i.test_nodes for i in CONFIGS])),
    ),
    inputs=dict(
        # sometimes, test img shape may be different from train img
        img=None,  # can be None when `lazy_forward` is True
        **dict(
            ChainMap(
                *[i.test_inputs for i in CONFIGS if hasattr(i, "test_inputs")]
            )
        ),
    ),
    topology_builder=get_topo_builder(
        is_train=False, out_feat_idx=out_feat_idx
    ),
    lazy_forward=True,
)
deploy_inputs = {img_name: torch.randn((1, 3, 224, 224))}
deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="Float2QAT"),
        dict(type="QAT2Quantize"),
    ],
)

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
)
val_data_loader = dict(
    type="MultitaskLoader",
    loaders={i.task_name: i.val_data_loader for i in CONFIGS},
    task_sampler=task_sampler,
    mode="validation",
    return_task=True,
)

# -------------------------- callbacks --------------------------
batch_processor = dict(
    type="MultiBatchProcessor",
    delay_sync=False,
    grad_accumulation_step=1,
    need_grad_update=True,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=collect_loss_by_regex("^.*loss.*"),
)

metric_updaters = [i.metric_updater for i in CONFIGS]
val_metric_updaters = [i.val_metric_updater for i in CONFIGS]

stat_callback = dict(
    type="StatsMonitor",
    log_freq=10 if not pipeline_test else 1,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,  # model params match test_model params
)

trace_callback = dict(
    type="SaveTraced",
    save_dir=ckpt_dir,
    trace_inputs=deploy_inputs,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=val_metric_updaters,
    val_model=None,  # use train model as val model
)

# -------------------------- solver --------------------------
# -------------------------- step float ----------------------
float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=2 if not pipeline_test else 1,
    device=None,  # set in train.py
    callbacks=[
        # the order of callbacks affects the logging order
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=0,
            lr_decay_id=[0, 1],
            step_log_interval=10,
        ),
        val_callback,
        ckpt_callback,
    ]
    + metric_updaters,
    sync_bn=sync_bn,
)

# -------------------------- step quantize-aware -------------------
qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
            ),
            dict(type="Float2QAT"),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.0001,  # Custom by yourself #
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=2 if not pipeline_test else 1,
    device=None,  # set in train.py
    callbacks=[
        # the order of callbacks affects the logging order
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            warmup_by="epoch",
            warmup_len=0,
            lr_decay_id=[0, 1],
            step_log_interval=10,
        ),
        val_callback,
        ckpt_callback,
    ]
    + metric_updaters,
    sync_bn=sync_bn,
)

# just for saving int_infer pth and pt
int_infer_trainer = dict(
    # can not use `distributed_data_parallel_trainer` in int_infer step, it
    # will trigger error: DistributedDataParallel is not needed when a module
    # doesn't have any parameter that requires a gradient.
    # So use `data_parallel_trainer` or `Trainer`.
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
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    # export int model and test_model only, not need training, so others like
    # loader is unnecessary
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,  # will skip training, and run `loop.on_loop_begin/end()`
    device=None,
    callbacks=[
        ckpt_callback,
        trace_callback,
    ],
)

# predictor
float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=val_metric_updaters,
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
    model=model,
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
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=val_metric_updaters,
    log_interval=50,
)

int_infer_predictor = dict(
    type="Predictor",
    model=model,
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
            ),
            dict(type="QAT2Quantize"),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=val_metric_updaters,
    log_interval=50,
)
