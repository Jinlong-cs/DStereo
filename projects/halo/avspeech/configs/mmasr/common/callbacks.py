# flake8: noqa
import os
from copy import deepcopy

import torch
from aidisdk.utils import running_in_cluster
from common.default import global_keys

from hat.callbacks.lr_updater import NoamLrUpdater
from hat.data.transforms.avspeech.mask import (
    DynamicChunkMask,
    ModalityDropout,
    PadMask,
    StaticChunkMask,
)
from hat.engine.processors.loss_collector import collect_loss_by_index
from hat.metrics.avspeech_metric import WER
from hat.metrics.loss_show import LossShow
from hat.models.task_modules.avspeech.subsampling import (
    CausalSubsampling2,
    Conv2dSubsampling8,
)
from projects.halo.avspeech.utils.global_config import get_config

_keys = [k for k in global_keys if get_config(k) is None]
assert len(_keys) == 0, f"Please init {_keys} before import callbacks!"

stat_callback = dict(
    type="StatsMonitor",
    log_freq=get_config("step_log_freq"),
    batch_size=get_config("batch_size"),
)

grad_scale_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=5,
)

train_batch_transform = [
    dict(type=ModalityDropout, p=0),
    dict(
        type=PadMask,
        length_key="audio_lens",
        feat_key="audio",
        subsample_func=Conv2dSubsampling8.subsample_mask,
    ),
    dict(type=DynamicChunkMask, dynamic_left_chunk=True),
]

train_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=train_batch_transform,
    loss_collector=collect_loss_by_index(0),
    enable_amp=False,
)

calib_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=train_batch_transform,
    loss_collector=collect_loss_by_index(0),
    enable_amp=False,
)


multibatch_train_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    batch_transforms=train_batch_transform,
    loss_collector=collect_loss_by_index(0),
    enable_amp=False,
    delay_sync=True,  # 梯度同步xxx
    norm_loss=True,
)


def update_metric(metrics, batch, model_outs):
    loss, loss_att, loss_ctc, att_acc, ctc_out = model_outs
    att_acc = torch.Tensor([att_acc]).to(loss.device)
    for metric in metrics:
        if isinstance(metric, LossShow):
            metric.update(
                {
                    "loss": loss,
                    "loss_att": loss_att,
                    "loss_ctc": loss_ctc,
                    "att_acc": att_acc,
                }
            )
        if isinstance(metric, WER):
            tokens = [" ".join(_tokens) for _tokens in batch["tokens"]]
            metric.update(ctc_out, tokens)


train_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=get_config("step_log_freq"),
    reset_metrics_by="log",
    epoch_log_freq=get_config("epoch_log_freq"),
    log_prefix=get_config("model_prefix"),
)


val_batchtransforms = [
    dict(type=ModalityDropout, p=0),
    dict(
        type=PadMask,
        length_key="audio_lens",
        feat_key="audio",
        subsample_func=Conv2dSubsampling8.subsample_mask,
    ),
    dict(type=StaticChunkMask, chunk_size=8, num_left_chunks=4),
]

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=val_batchtransforms,
    loss_collector=None,
    enable_amp=False,
)


vo_val_batchtransforms = [
    dict(
        type=PadMask,
        length_key="images_lens",
        feat_key="images",
        subsample_func=CausalSubsampling2.subsample_mask,
    ),
    dict(type=StaticChunkMask, chunk_size=8, num_left_chunks=4),
]

vo_val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=vo_val_batchtransforms,
    loss_collector=None,
    enable_amp=False,
)

val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=get_config("step_log_freq"),
    reset_metrics_by="epoch",
    epoch_log_freq=get_config("epoch_log_freq"),
    log_prefix=get_config("model_prefix"),
)

test_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=0,
    reset_metrics_by="epoch",
    epoch_log_freq=get_config("epoch_log_freq"),
    log_prefix=get_config("model_prefix"),
)

ckpt_dir = (
    f"/job_data/models/{get_config('task_name')}/"
    if running_in_cluster()
    else f"tmp_models/{get_config('task_name')}"
)

tb_log_path = os.path.join(ckpt_dir, "tensorboard")
tb_log_path = os.getenv("TENSORBOARD_LOG_PATH", tb_log_path)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=get_config("training_step") + "-",
    strict_match=True,
    interval_by="epoch",
    mode=None,
    monitor_metric_key="att_acc",
)

noam_lr_updater = dict(
    type=NoamLrUpdater,
    d_model=25000,
    warmup_step=25000,
    step_log_interval=100,
)

metrics = [dict(type=LossShow), dict(type=WER, name="wer")]

train_callbacks = [
    deepcopy(noam_lr_updater),
    deepcopy(stat_callback),
    deepcopy(train_metric_updater),
    deepcopy(ckpt_callback),
    deepcopy(grad_scale_callback),
]

calib_callbacks = [
    deepcopy(stat_callback),
    deepcopy(train_metric_updater),
    deepcopy(ckpt_callback),
    deepcopy(grad_scale_callback),
]

test_metric_updater = deepcopy(val_metric_updater)
test_batch_processor = deepcopy(val_batch_processor)
vo_test_batch_processor = deepcopy(vo_val_batch_processor)
test_metrics = [dict(type=LossShow), dict(type=WER, name="wer", detail=True)]
