import os
import sys

import torch
from data import eval_datasets, get_val_sparse_transforms
from horizon_plugin_pytorch.march import March
from mmcv.parallel import collate
from model import get_loss_names, get_sparse_model

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.engine.processors.loss_collector import collect_loss_by_regex
from hat.utils import Config
from hat.utils.config import ConfigVersion
from projects.cloudmodel.configs.cloudsparse4d.data import (
    build_eval_dataloaders_and_callbacks,
    build_train_dataloader,
    build_train_datasets,
)
from projects.cloudmodel.configs.cloudsparse4d.utils import (
    get_ckpt_dir,
    get_k8s_config,
)

cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), "config.py"))
data_cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), "data.py"))

# common
seed = 1280
cudnn_benchmark = False
log_rank_zero_only = True
march = March.BAYES
VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

k8s_config = get_k8s_config(cfg)
ckpt_dir = get_ckpt_dir(cfg)
model = get_sparse_model()
device_ids = k8s_config["num_gpus_per_machine"]
if os.environ.get("CLUSTER"):
    redirect_config_logging_path = f"/job_data/config_log_{training_step}.log"
else:
    redirect_config_logging_path = os.path.join(
        f"./tmp_models/config_log_{training_step}.log"
    )
os.system(f"rm -rf {os.getenv('HOME')}/.cache/torch/hub/checkpoints")

# train
train_datasets = build_train_datasets(data_cfg.train_datasets, cfg)
train_dataloader = build_train_dataloader(
    train_datasets,
    cfg.train_batch_size,
    cfg.train_num_workers,
    collate_fn=collate,
)

optimizer = dict(
    type=torch.optim.AdamW,
    params={"weight": dict(weight_decay=4e-5)},
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)

# train callbacks
loss_names = get_loss_names()
train_metrics = [dict(type="LossShow", name=name) for name in loss_names]
sparse4d_metric_updater = dict(
    type="MetricUpdater",
    metrics=train_metrics,
    metric_update_func=update_metric_using_regex(
        per_metric_patterns=[  # corresponding to metrics
            dict(
                label_pattern=None,
                pred_pattern=f"{name}",
            )
            for name in loss_names
        ]
    ),
    step_log_freq=50,
    epoch_log_freq=1,
    log_prefix=cfg.task_name,
    reset_metrics_by="log",
)
batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=[],
    loss_collector=collect_loss_by_regex("^.*loss.*"),
    grad_scaler=dict(
        type=torch.cuda.amp.GradScaler,
        init_scale=32.0,
        growth_interval=int(1e8),
    ),
    enable_amp=False,
)
stat_callback = dict(
    type="StatsMonitor",
    log_freq=cfg.stat_log_freq,
    batch_size=cfg.train_batch_size,
)
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    save_interval=cfg.save_interval,
    name_prefix=training_step + "-",
    mode="max",
    strict_match=False,
    interval_by=cfg.interval_by,
    save_on_train_end=True,
)
lr_callback = dict(
    type="CosLrUpdater",
    warmup_by="step",
    warmup_len=cfg.warmup_steps,
    step_log_interval=cfg.step_log_interval,
    max_steps=cfg.num_steps - cfg.warmup_steps,
)
grad_scale_callback = dict(
    type="GradScale",
    module_and_scale=[
        ("backbone", 0.5),
    ],
    clip_grad_norm=35.0,
    clip_norm_type=2,
)

callbacks = [
    sparse4d_metric_updater,
    stat_callback,
    lr_callback,
    ckpt_callback,
    grad_scale_callback,
]

float_trainer = dict(
    type=cfg.trainer_type,
    model=model,
    optimizer=optimizer,
    batch_processor=batch_processor,
    data_loader=train_dataloader,
    num_steps=cfg.num_steps,
    stop_by="step",
    device=None,
    callbacks=callbacks,
    train_metrics=train_metrics,
    val_metrics=train_metrics,
    model_convert_pipeline=dict(
        type="FloatQatConvertPipeline",
        qat_mode="fuse_bn",
        enable_qat=False,
        checkpoint_mode="pre_stage",  # "resume"
        checkpoint_configs=dict(
            checkpoint_path=cfg.pretrain,
            state_dict_update_func=None,
            allow_miss=True,
            ignore_extra=True,
            ignore_tensor_shape=True,
            verbose=True,
        ),
        qconfig_params=None,
    ),
    find_unused_parameters=False,
    sync_bn=True,
)


# eval
val_transforms = get_val_sparse_transforms()
all_data_loaders, all_callbacks = build_eval_dataloaders_and_callbacks(
    eval_datasets, cfg, val_transforms
)

val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    batch_transforms=[],
    loss_collector=None,
    enable_amp=False,
)

float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.getenv(
                    "HAT_CLOUD_MODEL_CHECKPOINT", cfg.eval_ckpt
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    batch_processor=val_batch_processor,
    data_loader=all_data_loaders,
    device=None,
    share_callbacks=False,
    callbacks=all_callbacks,
    log_interval=2,
)
