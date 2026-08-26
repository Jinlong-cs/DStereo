"""Fixed-scale 0.8 resize-aware experiment for DStereoPlus.

Training and calibration preserve the canonical 640x352 field of view by
resizing it to 512x282 and padding three black rows at the top and bottom to
obtain a 512x288 tensor.  Validation stays at 640x352 for comparison with the
baseline.
"""

import copy
import os

import DStereo.DStereoPlus as _base_config
from DStereo.DStereoPlus import *  # noqa: F401,F403

import torch

from hat.data.datasets.multi_disp_dataset.resize_aware import (
    build_resize_aware_specs,
    resize_aware_config,
)
from hat.data.samplers.stereo_scale_sampler import StereoScaleSampler

task_name = "DStereoV23_DiscoverStereo_ResizeAware_S080"
ckpt_dir = os.path.join("work_dirs", "discover_experiments", task_name)
export_dir = os.path.join(ckpt_dir, "export_s080_512x288")
float_checkpoint_path = os.path.join(ckpt_dir, "float-checkpoint-best.pth.tar")
cudnn_benchmark = False

resize_aware_scales = (0.8,)
resize_aware_args = resize_aware_config(
    resize_aware_scales,
    base_height=352,
    base_width=640,
    size_divisor=32,
    max_disp=_base_config.maxdisp,
)
resize_aware_spec = build_resize_aware_specs(
    resize_aware_scales,
    base_height=352,
    base_width=640,
    size_divisor=32,
)[0]
resize_content_shape = resize_aware_spec.content_shape
resize_tensor_shape = resize_aware_spec.tensor_shape
resize_padding = resize_aware_spec.padding

wandb_name = f"{task_name}-{_base_config.training_stage}"
wandb_tags = (
    "discover,resize-aware,fixed-scale,scale0.80,tensor512x288,maxdisp96"
)

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

val_data_loader = copy.deepcopy(_base_config.val_data_loader)

loss_show_callback = copy.deepcopy(_base_config.loss_show_callback)
loss_show_callback["log_prefix"] = "train_" + task_name

val_metric_updater = copy.deepcopy(_base_config.val_metric_updater)
val_metric_updater["log_prefix"] = "val_" + task_name

val_callback = copy.deepcopy(_base_config.val_callback)
val_callback["data_loader"] = val_data_loader
val_callback["callbacks"] = [val_metric_updater]

ckpt_callback = copy.deepcopy(_base_config.ckpt_callback)
ckpt_callback["save_dir"] = ckpt_dir

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
    resize_content_shape=list(resize_content_shape),
    resize_tensor_shape=list(resize_tensor_shape),
    resize_padding=list(resize_padding),
    resize_size_divisor=32,
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

calib_data_loader = copy.deepcopy(_base_config.calib_data_loader)
calib_data_loader["dataset"].update(
    res_args=[352, 640, False],
    resize_aware_args=resize_aware_args,
)
calib_data_loader["sampler"] = {
    "type": StereoScaleSampler,
    "batch_size": _base_config.test_batch_size_per_gpu,
    "scales": resize_aware_scales,
    "shuffle": True,
    "seed": _base_config.seed,
    "sampler_len": 50,
    "drop_last": True,
}
calib_data_loader["drop_last"] = True

predict_callbacks = copy.deepcopy(_base_config.predict_callbacks)
predict_callbacks[0]["output_dir"] = os.path.join(export_dir, "calib_data")
predict_callbacks[2] = val_metric_updater

float_predictor = copy.deepcopy(_base_config.float_predictor)
float_predictor["model_convert_pipeline"]["converters"][0][
    "checkpoint_path"
] = float_checkpoint_path
float_predictor["data_loader"] = [calib_data_loader]
float_predictor["callbacks"] = predict_callbacks

deploy_inputs = {
    "data": {
        "infra1": torch.randn((1, 3, 288, 512)),
        "infra2": torch.randn((1, 3, 288, 512)),
    }
}

onnx_metric_updater = copy.deepcopy(_base_config.onnx_metric_updater)
onnx_metric_updater["log_prefix"] = "onnx_" + task_name

onnx_cfg = copy.deepcopy(_base_config.onnx_cfg)
onnx_cfg["inputs"] = deploy_inputs
onnx_cfg["out_dir"] = export_dir
onnx_cfg["model_convert_pipeline"]["converters"][0][
    "checkpoint_path"
] = float_checkpoint_path

floatonnx_predictor = copy.deepcopy(_base_config.floatonnx_predictor)
floatonnx_predictor["model"]["onnx_path"] = os.path.join(
    export_dir, "float.onnx"
)
floatonnx_predictor["data_loader"] = [calib_data_loader]
floatonnx_predictor["callbacks"][0] = onnx_metric_updater
floatonnx_predictor["callbacks"][2]["output_dir"] = os.path.join(
    export_dir, "vis", "float"
)

quantonnx_predictor = copy.deepcopy(_base_config.quantonnx_predictor)
quantonnx_predictor["model"]["onnx_path"] = os.path.join(
    export_dir, "Bin_model", "DStereo_quantized_model.onnx"
)
quantonnx_predictor["data_loader"] = [calib_data_loader]
quantonnx_predictor["callbacks"][0] = onnx_metric_updater
quantonnx_predictor["callbacks"][2]["output_dir"] = os.path.join(
    export_dir, "vis", "quant"
)
