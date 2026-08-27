"""Fixed scale=0.8 full-FOV training profile for DStereoPlus.

Canonical 640x352 samples are resized to 512x282 and padded by three rows on
the top and bottom, producing one static 512x288 training tensor shape.
Validation remains at the canonical 640x352 resolution.
"""

import copy
import os

import DStereo.DStereoPlus as _base_config
from DStereo.DStereoPlus import *  # noqa: F401,F403

import torch

from hat.data.datasets.multi_disp_dataset.resize_aware import (
    FIXED_RESIZE_SCALE,
    build_resize_aware_spec,
    resize_aware_config,
)

task_name = "DStereoV23_DiscoverStereo_ResizeAware_S080"
ckpt_dir = os.environ.get(
    "DSTEREO_RUN_DIR",
    os.path.join("work_dirs", "discover_experiments", task_name),
)

resize_aware_scale = FIXED_RESIZE_SCALE
resize_aware_args = resize_aware_config(max_disp=_base_config.maxdisp)
resize_aware_spec = build_resize_aware_spec()
resize_content_shape = resize_aware_spec.content_shape
resize_tensor_shape = resize_aware_spec.tensor_shape
resize_padding = resize_aware_spec.padding

data_loader = copy.deepcopy(_base_config.data_loader)
data_loader["dataset"].update(
    res_args=[352, 640, False],
    resize_aware_args=resize_aware_args,
)

val_data_loader = copy.deepcopy(_base_config.val_data_loader)


def _channel_less_disparity(value, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dim() == 4 and value.size(1) == 1:
        value = value[:, 0]
    if value.dim() != 3:
        raise ValueError(
            f"{name} must have shape [N,H,W] or [N,1,H,W], "
            f"got {tuple(value.shape)}"
        )
    return value


def update_s080_loss_metric(metrics, batch, model_outs):
    """Update loss/EPE while normalizing the disparity channel dimension."""

    preds = None
    if isinstance(model_outs, dict):
        losses = model_outs.get("losses")
        if isinstance(losses, torch.Tensor):
            metrics[0].update(losses)
        elif isinstance(losses, (list, tuple)) and losses:
            metrics[0].update(sum(losses))
        preds = model_outs.get("pred_disps")
    elif isinstance(model_outs, (list, tuple)) and model_outs:
        preds = model_outs[0]
    if preds is None:
        return

    labels = _channel_less_disparity(batch["gt_disp"], "labels")
    preds = _channel_less_disparity(preds, "predictions")
    if labels.shape != preds.shape:
        raise ValueError(
            "label/prediction geometry mismatch: "
            f"{tuple(labels.shape)} != {tuple(preds.shape)}"
        )
    masks = (labels > 0) & (labels < _base_config.maxdisp)
    metrics[1].update(labels, preds, masks)


loss_show_callback = copy.deepcopy(_base_config.loss_show_callback)
loss_show_callback.update(
    metric_update_func=update_s080_loss_metric,
    log_prefix="train_" + task_name,
)

val_metric_updater = copy.deepcopy(_base_config.val_metric_updater)
val_metric_updater.update(
    metric_update_func=update_s080_loss_metric,
    log_prefix="val_" + task_name,
)

val_callback = copy.deepcopy(_base_config.val_callback)
val_callback["data_loader"] = val_data_loader
val_callback["callbacks"] = [val_metric_updater]

ckpt_callback = copy.deepcopy(_base_config.ckpt_callback)
ckpt_callback["save_dir"] = ckpt_dir

wandb_name = f"{task_name}-{_base_config.training_stage}"
wandb_tags = (
    "discover,v2,resize-aware,fixed-scale,scale0.80,"
    "content512x282,tensor512x288,maxdisp96,200k"
)
wandb_callback = copy.deepcopy(_base_config.wandb_callback)
wandb_callback.update(
    name=wandb_name,
    tags=wandb_tags.split(","),
    ckpt_dir=ckpt_dir,
)
wandb_callback["config"].update(
    task=task_name,
    resize_aware_scale=resize_aware_scale,
    resize_base_shape=[352, 640],
    resize_content_shape=list(resize_content_shape),
    resize_tensor_shape=list(resize_tensor_shape),
    resize_padding=list(resize_padding),
    resize_size_divisor=resize_aware_spec.size_divisor,
)

train_callbacks = [
    _base_config.stat_callback,
    loss_show_callback,
    {
        "type": "CosLrUpdater",
        "max_steps": _base_config.num_steps,
        "warmup_by": "step",
        "warmup_len": 2000,
        "step_log_interval": 1000,
    },
    val_callback,
    ckpt_callback,
]
if _base_config.enable_freeze_bn:
    train_callbacks.insert(
        0,
        {
            "type": "FreezeBN",
            "freeze_affine": _base_config.freeze_bn_affine,
            "unfreeze_step": _base_config.freeze_bn_until_step,
        },
    )
train_callbacks.append(wandb_callback)

float_trainer = copy.deepcopy(_base_config.float_trainer)
float_trainer.update(
    data_loader=data_loader,
    callbacks=train_callbacks,
)
initialization = float_trainer["model_convert_pipeline"]["converters"][0]
# Despite its historical directory name, this tracked artifact is a complete
# 533-tensor DStereo checkpoint.  Fail closed if that exact warm-start contract
# ever drifts.
initialization.update(
    checkpoint_path=_base_config.checkpoint_path,
    allow_miss=False,
    ignore_extra=False,
    ignore_tensor_shape=False,
)
