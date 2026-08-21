"""Resize-aware multi-scale float-training experiment for DStereoPlus.

The baseline preprocessing first creates its canonical 640x352 crop.  This
experiment then preserves that crop's full field of view at one batch-wide
scale, pads the result to a multiple of 32, and trains all configured shapes
with one shared checkpoint.  Validation remains at the canonical 640x352
resolution so its EPE is directly comparable with the baseline experiment.
"""

import copy
import os

from DStereo.DStereoPlus import *  # noqa: F401,F403

from hat.data.datasets.multi_disp_dataset.resize_aware import (
    resize_aware_config,
)
from hat.data.samplers.stereo_scale_sampler import StereoScaleSampler

task_name = "DStereoV23_DiscoverStereo_ResizeAware"
ckpt_dir = os.path.join("work_dirs", "discover_experiments", task_name)
cudnn_benchmark = False

resize_aware_scales = tuple(
    round(1.00 - 0.05 * index, 2) for index in range(15)
)
resize_aware_args = resize_aware_config(
    resize_aware_scales,
    base_height=352,
    base_width=640,
    size_divisor=32,
    max_disp=maxdisp,
)

wandb_name = f"{task_name}-{training_stage}"
wandb_tags = "discover,resize-aware,multiscale,scale1.00-0.30,maxdisp96"

data_loader = copy.deepcopy(data_loader)
data_loader["dataset"].update(
    # Disable the baseline's per-sample random spatial resize.  Resizor first
    # makes a deterministic 640-wide view and Cropper creates the canonical
    # 640x352 crop consumed by the resize-aware transform.
    res_args=[352, 640, False],
    resize_aware_args=resize_aware_args,
)
data_loader["sampler"] = dict(
    type=StereoScaleSampler,
    batch_size=train_batch_size_per_gpu,
    scales=resize_aware_scales,
    shuffle=True,
    seed=seed,
    drop_last=True,
)
data_loader["drop_last"] = True

loss_show_callback = copy.deepcopy(loss_show_callback)
loss_show_callback["log_prefix"] = "train_" + task_name

val_metric_updater = copy.deepcopy(val_metric_updater)
val_metric_updater["log_prefix"] = "val_" + task_name

val_callback = copy.deepcopy(val_callback)
val_callback["callbacks"] = [val_metric_updater]

ckpt_callback = copy.deepcopy(ckpt_callback)
ckpt_callback["save_dir"] = ckpt_dir

wandb_callback = copy.deepcopy(wandb_callback)
wandb_callback.update(
    name=wandb_name,
    tags=wandb_tags.split(","),
    ckpt_dir=ckpt_dir,
)
wandb_callback["config"].update(
    task=task_name,
    resize_aware_scales=list(resize_aware_scales),
    resize_base_shape=[352, 640],
    resize_size_divisor=32,
)

train_callbacks = [
    stat_callback,
    loss_show_callback,
    dict(
        type="CosLrUpdater",
        max_steps=num_steps,
        warmup_by="step",
        warmup_len=2000,
        step_log_interval=1000,
    ),
    val_callback,
    ckpt_callback,
]
if enable_freeze_bn:
    train_callbacks.insert(
        0,
        dict(
            type="FreezeBN",
            freeze_affine=freeze_bn_affine,
            unfreeze_step=freeze_bn_until_step,
        ),
    )
train_callbacks.append(wandb_callback)

float_trainer = copy.deepcopy(float_trainer)
float_trainer.update(
    data_loader=data_loader,
    callbacks=train_callbacks,
)
