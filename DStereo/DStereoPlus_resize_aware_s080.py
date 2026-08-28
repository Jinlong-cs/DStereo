"""Batch-homogeneous s1.0/s0.8 training profile for DStereoPlus.

Every two training batches contain one canonical 640x352 batch and one
full-FOV 512x282 batch padded to 512x288. Validation traverses the complete
split once per scale and reports both EPE values in canonical disparity-pixel
units. The historical module name is retained so PR #4 keeps one config path.
"""

import copy
import os

import DStereo.DStereoPlus as _base_config
from DStereo.DStereoPlus import *  # noqa: F401,F403

import torch
import torch.nn.functional as F

from hat.data.datasets.multi_disp_dataset.resize_aware import (
    DEFAULT_RESIZE_SCALES,
    build_resize_aware_specs,
    resize_aware_config,
)
from hat.data.samplers.stereo_scale_sampler import StereoScaleSampler

task_name = "DStereoV23_DiscoverStereo_ResizeAware_S100_S080"
ckpt_dir = os.environ.get(
    "DSTEREO_RUN_DIR",
    os.path.join("work_dirs", "discover_experiments", task_name),
)
cudnn_benchmark = False

resize_aware_scales = DEFAULT_RESIZE_SCALES
resize_aware_args = resize_aware_config(
    resize_aware_scales,
    max_disp=_base_config.maxdisp,
)
resize_aware_specs = build_resize_aware_specs(resize_aware_scales)
resize_aware_spec_by_scale = {spec.scale: spec for spec in resize_aware_specs}
s100_spec = resize_aware_spec_by_scale[1.0]
s080_spec = resize_aware_spec_by_scale[0.8]

data_loader = copy.deepcopy(_base_config.data_loader)
data_loader["dataset"].update(
    res_args=[352, 640, False],
    resize_aware_args=resize_aware_args,
)
data_loader["sampler"] = {
    "type": StereoScaleSampler,
    "batch_size": _base_config.train_batch_size_per_gpu,
    "scales": resize_aware_scales,
    "shuffle": True,
    "seed": _base_config.seed,
    "drop_last": True,
}
data_loader["drop_last"] = True


def _build_val_loader(scale):
    loader = copy.deepcopy(_base_config.val_data_loader)
    loader["dataset"].update(
        res_args=[352, 640, False],
        resize_aware_args=resize_aware_config(
            (scale,),
            max_disp=_base_config.maxdisp,
        ),
    )
    return loader


val_data_loader_s100 = _build_val_loader(1.0)
val_data_loader_s080 = _build_val_loader(0.8)
val_data_loader = [val_data_loader_s100, val_data_loader_s080]


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


def _extract_prediction(model_outs):
    if isinstance(model_outs, dict):
        return model_outs.get("pred_disps")
    if isinstance(model_outs, (list, tuple)) and model_outs:
        return model_outs[0]
    return None


def _valid_mask(labels):
    return (
        torch.isfinite(labels) & (labels > 0) & (labels < _base_config.maxdisp)
    )


def update_resize_aware_train_metric(metrics, batch, model_outs):
    """Update native-scale training loss and diagnostic EPE."""

    if isinstance(model_outs, dict):
        losses = model_outs.get("losses")
        if isinstance(losses, torch.Tensor):
            metrics[0].update(losses)
        elif isinstance(losses, (list, tuple)) and losses:
            metrics[0].update(sum(losses))

    preds = _extract_prediction(model_outs)
    if preds is None:
        return
    labels = _channel_less_disparity(batch["gt_disp"], "labels")
    preds = _channel_less_disparity(preds, "predictions")
    if labels.shape != preds.shape:
        raise ValueError(
            "label/prediction geometry mismatch: "
            f"{tuple(labels.shape)} != {tuple(preds.shape)}"
        )
    metrics[1].update(labels, preds, _valid_mask(labels))


def _restore_prediction_to_canonical(predictions, spec):
    predictions = _channel_less_disparity(predictions, "predictions")
    if predictions.shape[-2:] != spec.tensor_shape:
        raise ValueError(
            "prediction geometry mismatch for scale "
            f"{spec.scale}: {tuple(predictions.shape[-2:])} "
            f"!= {spec.tensor_shape}"
        )

    height_end = spec.tensor_height - spec.pad_bottom
    width_end = spec.tensor_width - spec.pad_right
    content = predictions[
        :,
        spec.pad_top : height_end,
        spec.pad_left : width_end,
    ]
    if content.shape[-2:] != spec.content_shape:
        raise AssertionError((content.shape, spec.content_shape))
    if spec.content_shape != (spec.base_height, spec.base_width):
        content = F.interpolate(
            content.unsqueeze(1),
            size=(spec.base_height, spec.base_width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return content / spec.horizontal_scale


def _update_canonical_val_metric(metrics, batch, model_outs, spec):
    preds = _extract_prediction(model_outs)
    if preds is None:
        return
    labels = _channel_less_disparity(
        batch["metric_gt_disp"], "canonical labels"
    )
    preds = _restore_prediction_to_canonical(preds, spec)
    if labels.shape != preds.shape:
        raise ValueError(
            "canonical label/prediction geometry mismatch: "
            f"{tuple(labels.shape)} != {tuple(preds.shape)}"
        )
    metrics[0].update(labels, preds, _valid_mask(labels))


def update_s100_val_metric(metrics, batch, model_outs):
    _update_canonical_val_metric(metrics, batch, model_outs, s100_spec)


def update_s080_val_metric(metrics, batch, model_outs):
    _update_canonical_val_metric(metrics, batch, model_outs, s080_spec)


loss_show_callback = copy.deepcopy(_base_config.loss_show_callback)
loss_show_callback.update(
    metric_update_func=update_resize_aware_train_metric,
    log_prefix="train_" + task_name,
)

val_metric_s100 = {
    "type": "EndPointError",
    "name": "EPE_s100",
    "use_mask": True,
}
val_metric_s080 = {
    "type": "EndPointError",
    "name": "EPE_s080",
    "use_mask": True,
}
val_metrics = [val_metric_s100, val_metric_s080]

val_metric_updater_s100 = copy.deepcopy(_base_config.val_metric_updater)
val_metric_updater_s100.update(
    metric_update_func=update_s100_val_metric,
    metrics=[val_metric_s100],
    log_prefix="val_s100_" + task_name,
)
val_metric_updater_s080 = copy.deepcopy(_base_config.val_metric_updater)
val_metric_updater_s080.update(
    metric_update_func=update_s080_val_metric,
    metrics=[val_metric_s080],
    log_prefix="val_s080_" + task_name,
)

val_callback = copy.deepcopy(_base_config.val_callback)
val_callback.update(
    data_loader=val_data_loader,
    callbacks=[
        [val_metric_updater_s100],
        [val_metric_updater_s080],
    ],
    share_callbacks=False,
)

ckpt_callback = copy.deepcopy(_base_config.ckpt_callback)
ckpt_callback.update(
    save_dir=ckpt_dir,
    monitor_metric_key="EPE_s100",
)

wandb_name = f"{task_name}-{_base_config.training_stage}"
wandb_tags = (
    "discover,v2,resize-aware,dual-scale,scale1.00,scale0.80,"
    "canonical640x352,tensor512x288,maxdisp96,200k"
)
wandb_callback = copy.deepcopy(_base_config.wandb_callback)
wandb_callback.update(
    name=wandb_name,
    tags=wandb_tags.split(","),
    ckpt_dir=ckpt_dir,
)
wandb_callback["config"].update(
    task=task_name,
    resize_aware_scales=list(resize_aware_scales),
    resize_base_shape=[352, 640],
    resize_content_shapes={
        "s100": list(s100_spec.content_shape),
        "s080": list(s080_spec.content_shape),
    },
    resize_tensor_shapes={
        "s100": list(s100_spec.tensor_shape),
        "s080": list(s080_spec.tensor_shape),
    },
    validation_metrics=["val/epe_s100", "val/epe_s080"],
    checkpoint_monitor="EPE_s100",
    resize_size_divisor=s100_spec.size_divisor,
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
    val_metrics=val_metrics,
)
initialization = float_trainer["model_convert_pipeline"]["converters"][0]
# Despite its historical directory name, this tracked artifact is a complete
# 533-tensor DStereo checkpoint. Fail closed if that exact warm-start contract
# ever drifts.
initialization.update(
    checkpoint_path=_base_config.checkpoint_path,
    allow_miss=False,
    ignore_extra=False,
    ignore_tensor_shape=False,
)
