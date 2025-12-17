# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import os
from collections import OrderedDict

import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.utils.config import ConfigVersion

VERSION = ConfigVersion.v2

# --------------------- common ----------------------
cfg_dir = os.path.dirname(__file__)
training_step = os.environ.get("HAT_TRAINING_STEP", "float")
num_machines = 1
num_gpus_per_machine = 2
device_ids = list(range(num_gpus_per_machine))
warmup_epoch = 0
log_freq = 5
bn_kwargs = dict(eps=1e-3, momentum=0.01)

# --------------------- task -------------------------
task_name = os.environ.get("TASK_NAME", "test")
lr = 5e-4
qat_lr = 1e-5
wd = 1e-2
save_prefix = os.environ.get("OUT_DIR", "./trained_params")
num_epochs = 5
qat_num_epochs = 5
sync_bn = True
pretrain_checkpoint = None  # change this line if you want to use checkpoints.
ckpt_dir = os.path.join(save_prefix, task_name)
log_dir = os.path.join(ckpt_dir, "logs")
tensorboard_log_path = os.path.join(ckpt_dir, "tensorboard")
save_interval = 1
interval_by = "epoch"
float_resume_checkpoint = None
log_rank_zero_only = True
cudnn_benchmark = True
seed = None
batch_size_per_gpu = 8
workers_per_gpu = 8


# --------------------- topo builder ---------------------
def get_topo_builder(mode):
    def _build_topo(nodes, inputs):

        # forward
        # inputs shape [N, 98, 512, 512]
        raster_inputs = inputs["raster_inputs"]
        # multiscale features after encoder
        # [N, 16, 256, 256]
        # [N, 24, 128, 128]
        # [N, 40, 64, 64]
        # [N, 112, 32, 32]
        # [N, 320, 16, 16]
        raster_feats = nodes["raster_encoder"](raster_inputs)
        # multiscale feature after aggregator
        # [N, 128, 256, 256]
        # [N, 128, 128, 128]
        # [N, 128, 64, 64]
        # [N, 128, 32, 32]
        # [N, 128, 16, 16]
        aggregator_feats = nodes["spatial_aggregator"](raster_feats)
        # predictions after head
        # occupancy pred [N, 16, 256, 256]
        # flow pred [N, 16, 256, 256]
        out = nodes["out_module"](aggregator_feats, inputs["ground_truth"])

        name2out = OrderedDict()

        # pred related
        name2out["occ_preds"] = out["out_predict_occ_preds"]
        name2out["flow_preds"] = out["out_predict_flow_preds"]

        # target related
        # name2out['target_occupancy'] = out['out__target_occupancy']
        # name2out['target_flow'] = out['out__target_flow']

        # loss related
        name2out["occupancy_loss"] = out["out__occupancy_loss"]
        name2out["flow_loss"] = out["out__flow_loss"]
        name2out["traced_bce_loss"] = out["out__traced_bce_loss"]
        name2out["traced_focal_loss"] = out["out__traced_focal_loss"]
        name2out["combined_loss"] = out["out__combined_loss"]

        return name2out

    return _build_topo


# --------------------- ESSENTIAL MODEL PARAM FUNCTION-------------------------

# --------------------- DATASET hparams-------------------------
dataset_root = os.environ.get("DATA_DIR")

# --------------------- model hparams-------------------------
encoder_alpha = 2.0
# vargnet
encoder_stride2channels = get_vargnetv2_stride2channels(encoder_alpha)
# efficientnet_b0
encoder_stride2channels = dict({2: 16, 4: 24, 8: 40, 16: 112, 32: 320})


# --------------------- model -------------------------
def state_dict_update_function(state_dict):
    state_dict_new = dict()
    for key, val in state_dict.items():
        state_dict_new[key] = val
    return state_dict_new


inputs = dict(raster_inputs=None, scenario_id=None, ground_truth=None)
val_inputs = dict(raster_inputs=None, scenario_id=None, ground_truth=None)

raster_encoder = dict(
    type="efficientnet",
    model_type="b0",
    num_classes=1000,
    bn_kwargs=bn_kwargs,
    include_top=False,
    activation="relu",
    use_se_block=False,
    input_channels=98,
)
spatial_aggregator = dict(
    type="BiFPN",
    in_strides=[2, 4, 8, 16, 32],
    out_strides=[2, 4, 8, 16, 32],
    stride2channels=encoder_stride2channels,
    out_channels=128,
    num_outs=5,
    stack=4,
    start_level=0,
    end_level=-1,
    fpn_name="bifpn_sum",
)
out_module = dict(
    type="OutputModule",  # pls refer to OutputModule for detailed infos.
    head=dict(
        type="OccflowHead",
        num_class=1,
        in_channels=128,
    ),
    head_parser=None,
    postprocess=None,
    target=None,
    loss=dict(
        type="BaseOccflowLossWaymo",
        compute_loss_cls_index=[0],
        loss_weights=[500, 1, 500, 500],
    ),
    prefix="out",
    keep_name=False,
)
model = dict(
    type="GraphModel",
    nodes=dict(
        raster_encoder=raster_encoder,
        spatial_aggregator=spatial_aggregator,
        out_module=out_module,
    ),
    inputs=inputs,
    topology_builder=get_topo_builder(mode="train"),
    lazy_forward=True,
)
val_model = dict(
    type="GraphModel",
    nodes=dict(
        raster_encoder=raster_encoder,
        spatial_aggregator=spatial_aggregator,
        out_module=out_module,
    ),
    inputs=val_inputs,
    topology_builder=get_topo_builder(mode="val"),
    lazy_forward=True,
)

# --------------------- data -------------------------

train_dataset = dict(
    type="OccflowWaymoDataset",
    dataroot=dataset_root,
    token_list="training_pkl_list.txt",
    subsample=10,
)

val_dataset = dict(
    type="OccflowWaymoDataset",
    dataroot=dataset_root,
    token_list="validation_pkl_list_new.txt",
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    collate_fn=None,
    dataset=train_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=workers_per_gpu,
    pin_memory=False,
    drop_last=True,
    persistent_workers=workers_per_gpu > 0,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    collate_fn=None,
    dataset=val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=workers_per_gpu,
)


# --------------------- evaluation -------------------------
evaluator = dict(type="OccflowEval")

# --------------------- processors -------------------------
# Collect only the combined loss in batch processor.
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_regex("^.*combined_loss$"),
)

val_batch_processor = dict(type="MultiBatchProcessor", need_grad_update=False)

# --------------------- callbacks -------------------------
# Basic stats monitor, such as time, speed, etc.
stat_callback = dict(type="StatsMonitor", log_freq=log_freq)

# Checkpoint saver
checkpoint_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=save_interval,
    interval_by=interval_by,
    save_on_train_end=True,
)

# Metric logger
metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(type="LossShow", name="occupancy_loss"),
        dict(type="LossShow", name="flow_loss"),
        dict(type="LossShow", name="traced_bce_loss"),
        dict(type="LossShow", name="traced_focal_loss"),
        dict(type="LossShow", name="combined_loss"),
    ],
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[
            dict(label_pattern=None, pred_pattern="^.*occupancy_loss$"),
            dict(label_pattern=None, pred_pattern="^.*flow_loss$"),
            dict(label_pattern=None, pred_pattern="^.*traced_bce_loss$"),
            dict(label_pattern=None, pred_pattern="^.*traced_focal_loss$"),
            dict(label_pattern=None, pred_pattern="^.*combined_loss$"),
        ]
    ),
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)


def update_metric_pred(metrics, batch, model_outs):
    target = batch["ground_truth"]
    preds = dict()
    preds["occ_preds"] = model_outs[0].occ_preds.detach()
    preds["flow_preds"] = model_outs[1].flow_preds.detach()
    for metric in metrics:
        metric.update(preds, target)


val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[
        dict(
            type="OccFlowMetrics",
            quantize=False,
            cls_index=[0],
        ),
    ],
    metric_update_func=update_metric_pred,
    step_log_freq=200,
    epoch_log_freq=1,
    log_prefix=task_name,
)

val_saver = dict(
    type="SaveOccFlowResult",
    output_dir=os.path.join(ckpt_dir, "dump_predictions"),
)

lr_callback = dict(
    type="CosLrUpdater",
    warmup_by="epoch",
    warmup_len=warmup_epoch,
    step_log_interval=log_freq,
)

qat_lr_callback = dict(
    type="StepDecayLrUpdater",
    warmup_by="epoch",
    warmup_len=warmup_epoch,
    step_log_interval=log_freq,
    lr_decay_id=[2, 4],
    lr_decay_factor=0.1,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[
        val_metric_updater,
        val_saver,
    ],
    val_model=None,
    val_on_train_end=False,
)


def tb_update_func(writer, model_outs, epoch_id, **kwargs):
    for k, v in model_outs.items():
        if "loss" in k:
            writer.add_scalar(k, v, global_step=epoch_id)


tb_callback = dict(
    type="TensorBoard",
    save_dir=os.path.join(tensorboard_log_path, training_step, "train"),
    loss_name_reg="^.*loss.*",
    update_freq=10,
    update_by="step",
    # tb_update_funcs=[tb_update_func],
)

callbacks = [
    stat_callback,
    lr_callback,
    metric_updater,
    tb_callback,
    val_metric_updater,
    checkpoint_callback,
    val_callback,
]

qat_callbacks = [
    stat_callback,
    lr_callback,
    metric_updater,
    tb_callback,
    val_metric_updater,
    checkpoint_callback,
    val_callback,
]

# --------------------- Float -------------------------

float_model = copy.deepcopy(model)
float_trainer = dict(
    type="distributed_data_parallel_trainer",  # distributed training
    model=float_model,  # model definition
    data_loader=data_loader,  # data loader definition
    optimizer=dict(
        type=torch.optim.AdamW,
        lr=lr,
        weight_decay=wd,
    ),
    batch_processor=batch_processor,
    stop_by="epoch",
    num_epochs=num_epochs,
    device=None,
    callbacks=callbacks,
    sync_bn=sync_bn,
    find_unused_parameters=True,
)

float_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

# --------------------- QAT -------------------------
qat_model = copy.deepcopy(model)
qat_trainer = dict(
    type="distributed_data_parallel_trainer",  # distributed training
    model=qat_model,  # model definition
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        qconfig_params=dict(
            activation_qat_qkwargs=dict(
                averaging_constant=0.0,
            ),
            weight_qat_qkwargs=dict(
                averaging_constant=1.0,
            ),
        ),
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=data_loader,  # data loader definition
    optimizer=dict(
        type=torch.optim.AdamW,
        lr=qat_lr,
        weight_decay=wd,
    ),
    batch_processor=batch_processor,
    stop_by="epoch",
    num_epochs=qat_num_epochs,
    device=None,
    callbacks=qat_callbacks,
    sync_bn=sync_bn,
    find_unused_parameters=True,
)

qat_predictor = dict(
    type="Predictor",
    model=copy.deepcopy(model),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(type="Float2QAT"),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)
