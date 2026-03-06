import copy
import os

import torch
import torch.nn.functional as F

from DStereo.DStereoPlus import *  # noqa: F401,F403

from horizon_plugin_pytorch.quantization.qconfig import (
    default_calib_8bit_fake_quant_qconfig,
    default_calib_8bit_weight_16bit_act_fake_quant_qconfig,
    default_calib_8bit_weight_32bit_out_fake_quant_qconfig,
    default_qat_8bit_fake_quant_qconfig,
    default_qat_8bit_weight_16bit_act_fake_quant_qconfig,
    default_qat_8bit_weight_32bit_out_fake_quant_qconfig,
)

training_stage = "calibration"

qat_task_name = "DStereoV23"
qat_work_root = os.path.join("work_dirs", "tmp_models_save_best", qat_task_name)
float_stage_dir = os.path.join(qat_work_root, "float")
calibration_stage_dir = os.path.join(qat_work_root, "calibration")
qat_stage_dir = os.path.join(qat_work_root, "qat")
int_infer_stage_dir = os.path.join(qat_work_root, "int_infer")
compile_stage_dir = os.path.join(qat_work_root, "compile")

model = copy.deepcopy(model)
model["training_stage"] = training_stage

calibration_data_loader = copy.deepcopy(calib_data_loader)
calibration_data_loader["dataset"]["test_mode"] = False
calibration_data_loader["sampler"] = dict(
    type="InterleaveConcatSampler",
    shuffle=True,
    seed=666,
    sampler_len=256,
)

float_best_ckpt = os.path.join(float_stage_dir, "float-checkpoint-best.pth.tar")
calibration_last_ckpt = os.path.join(
    calibration_stage_dir,
    "calibration-checkpoint-last.pth.tar",
)
qat_best_ckpt = os.path.join(qat_stage_dir, "qat-checkpoint-best.pth.tar")


# Keep two-stream deploy inputs (infra1/infra2) aligned with PTQ runtime protocol.
deploy_inputs = dict(
    # hbdk pyramid inputs are integer typed; keep deploy sample dtype aligned.
    infra1=torch.clamp(
        (copy.deepcopy(deploy_inputs["data"]["infra1"]) * 128.0).round(),
        -128,
        127,
    ).to(torch.int8),
    infra2=torch.clamp(
        (copy.deepcopy(deploy_inputs["data"]["infra2"]) * 128.0).round(),
        -128,
        127,
    ).to(torch.int8),
)




def _dequantize_qat_tensor(value):
    return value.dequantize() if hasattr(value, "dequantize") else value


INT8_FALLBACK_MODULE_NAMES = (
    # Keep backbone/aggregation path on int8 to avoid int16-only branches in deploy.
    "backbone",
    # feature deconv path can hit unsupported int16 dtype in compile.
    "feature",
    # cost volume/front-end path: stabilize dtype before hourglass deconv.
    "before_costvolum",
    # cost_agg hourglass has deconv blocks that may receive unsupported int16.
    "cost_agg",
    "get_costvolum",
    "get_initdisp",
    # refinement prepare path: force int8 to match BPU-supported input types.
    "prepare_forrefinement",
    "prepare_forrefinement.context_zqr_conv",
    # "spx",
    # "spx_2",
    # "spx_4",
    # refinement update path: force int8 to avoid int16 Sumin checks in conv/add.
    "refinement",
    "refinement.update_block",
    "refinement.update_block.encoder",
    # "refinement.update_block.gru",
    "refinement.update_block.mask_feat_4",
    # "refinement.spx_2_gru",
    # "refinement.spx_gru",
    # SegmentLUT(tanh/sigmoid) nodes: keep q8->q8 and avoid q8->q16 LUT assert.
    "prepare_forrefinement_generated_tanh_0",
    "refinement.update_block.gru_generated_sigmoid_0",
    "refinement.update_block.gru_generated_sigmoid_1",
    "refinement.update_block.gru_generated_sigmoid_2",
    "refinement.update_block.gru_generated_sigmoid_3",
    "refinement.update_block.gru_generated_tanh_0",
    "refinement.update_block.gru_generated_tanh_1",
    # refinement/init-disp glue adds: avoid int16 Sumin on ConvAdd2d in compile.
    "get_initdisp_generated_add_22",
    "get_initdisp_generated_add_23",
    "refinement_generated_add_0",
    "refinement_generated_add_1",
)

INT8_EVAL_COMPAT_EXTRA_MODULE_NAMES = (
    # int_infer_eval runs training-style dataset batches (not deploy inputs).
    # Only add modules that are not already covered by INT8_FALLBACK_MODULE_NAMES.
    "backbone",
    "feature",
    "before_costvolum",
    "cost_agg",
    "refinement",
    "refinement.update_block",
)


def _compute_qat_sequence_loss(agg_pred, iter_preds, disp_gt, loss_gamma=0.9):
    agg_pred = _dequantize_qat_tensor(agg_pred)
    disp_gt = _dequantize_qat_tensor(disp_gt)
    iter_preds = [_dequantize_qat_tensor(pred) for pred in iter_preds]
    # Keep loss operands in [N, H, W] to match the valid mask shape.
    if isinstance(agg_pred, torch.Tensor) and agg_pred.dim() == 4 and agg_pred.size(1) == 1:
        agg_pred = agg_pred[:, 0]
    if isinstance(disp_gt, torch.Tensor) and disp_gt.dim() == 4 and disp_gt.size(1) == 1:
        disp_gt = disp_gt[:, 0]
    iter_preds = [
        pred[:, 0]
        if isinstance(pred, torch.Tensor) and pred.dim() == 4 and pred.size(1) == 1
        else pred
        for pred in iter_preds
    ]

    n_predictions = len(iter_preds)
    valid = (disp_gt > 0.0) & (disp_gt < maxdisp)
    losses = [
        F.smooth_l1_loss(
            agg_pred[valid.bool()],
            disp_gt[valid.bool()],
            reduction="mean",
        )
    ]
    adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
    for i in range(n_predictions):
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (iter_preds[i] - disp_gt).abs()
        losses.append(i_weight * i_loss[valid.bool()].mean())
    return losses


def qat_update_loss_metric(*args):
    # One entry for two hooks:
    # - loss_collector(outputs) -> return losses for backward
    # - metric_update_func(metrics, batch, model_outs) -> update logs/metrics
    if len(args) == 1:
        outputs = args[0]
        return _compute_qat_sequence_loss(*outputs["loss_inputs"])

    metrics, batch, model_outs = args
    labels = _dequantize_qat_tensor(batch["gt_disp"])
    preds = _dequantize_qat_tensor(model_outs["pred_disps"])
    if isinstance(labels, torch.Tensor) and labels.dim() == 4 and labels.size(1) == 1:
        labels = labels[:, 0]
    if isinstance(preds, torch.Tensor) and preds.dim() == 4 and preds.size(1) == 1:
        preds = preds[:, 0]
    masks = (labels > 0) & (labels < maxdisp)
    losses = _compute_qat_sequence_loss(*model_outs["loss_inputs"])
    metrics[0].update(sum(losses))
    metrics[1].update(labels, preds, masks)


def qat_update_metric(metrics, batch, model_outs):
    # Normalize model outputs (dict/tuple/tensor) to prediction tensor.
    if isinstance(model_outs, dict):
        preds = model_outs["pred_disps"]
    elif isinstance(model_outs, (tuple, list)):
        preds = model_outs[0]
    else:
        preds = model_outs

    labels = batch["gt_disp"]
    # Dequantize QTensor values before EPE computation.
    labels = _dequantize_qat_tensor(labels)
    preds = _dequantize_qat_tensor(preds)
    # Squeeze single-channel disparity to match metric input shape.
    if isinstance(labels, torch.Tensor) and labels.dim() == 4 and labels.size(1) == 1:
        labels = labels[:, 0]
    if isinstance(preds, torch.Tensor) and preds.dim() == 4 and preds.size(1) == 1:
        preds = preds[:, 0]
    masks = (labels > 0) & (labels < maxdisp)
    metrics[0].update(labels, preds, masks)


int_infer_update_metric = qat_update_metric


qat_train_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    enable_amp=False,
    loss_collector=qat_update_loss_metric,
)

calibration_converter = dict(
    type="Float2Calibration",
    convert_mode="fx",
    qconfig_dict={
        "__build_recursive": False,
        "": default_calib_8bit_weight_16bit_act_fake_quant_qconfig,
        "module_name": {
            **{
                name: default_calib_8bit_fake_quant_qconfig
                for name in INT8_FALLBACK_MODULE_NAMES
            },
            # Keep compile OUT0 (disp_unfold) on int32 so board postprocess
            # matches the PTQ S32xS16 accumulation path.
            "refinement.unfold_conv.unflod_conv": (
                default_calib_8bit_weight_32bit_out_fake_quant_qconfig
            ),
            "refinement.update_block.disp_head.conv2": (
                default_calib_8bit_weight_32bit_out_fake_quant_qconfig
            ),
        },
    },
)

calibration_ckpt_callback = dict(
    type="Checkpoint",
    interval_by="step",
    save_interval=50,
    save_dir=calibration_stage_dir,
    name_prefix="calibration-",
    strict_match=True,
)

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=float_best_ckpt,
                allow_miss=False,
                ignore_extra=False,
                ignore_tensor_shape=False,
                verbose=True,
            ),
            calibration_converter,
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=val_batch_processor,
    num_steps=256,
    device=None,
    callbacks=[stat_callback, calibration_ckpt_callback],
    val_metrics=[],
    log_interval=10,
)

qat_model = copy.deepcopy(model)
qat_model["training_stage"] = "qat"

qat_converter = dict(
    type="Float2QAT",
    convert_mode="fx",
    qconfig_dict={
        "__build_recursive": False,
        "": default_qat_8bit_weight_16bit_act_fake_quant_qconfig,
        "module_name": {
            **{
                name: default_qat_8bit_fake_quant_qconfig
                for name in INT8_FALLBACK_MODULE_NAMES
            },
            # Keep input concat in float domain: quantized cat requires QTensor inputs.
            "_generated_cat_0": None,
            # Keep compile OUT0 (disp_unfold) on int32 so board postprocess
            # matches the PTQ S32xS16 accumulation path.
            "refinement.unfold_conv.unflod_conv": (
                default_qat_8bit_weight_32bit_out_fake_quant_qconfig
            ),
            "refinement.update_block.disp_head.conv2": (
                default_qat_8bit_weight_32bit_out_fake_quant_qconfig
            ),
        },
    },
)

int_infer_eval_qat_converter = dict(
    type="Float2QAT",
    convert_mode="fx",
    qconfig_dict={
        "__build_recursive": False,
        "": default_qat_8bit_weight_16bit_act_fake_quant_qconfig,
        "module_name": {
            **{
                name: default_qat_8bit_fake_quant_qconfig
                for name in INT8_FALLBACK_MODULE_NAMES
            },
            **{
                name: default_qat_8bit_fake_quant_qconfig
                for name in INT8_EVAL_COMPAT_EXTRA_MODULE_NAMES
            },
            # Keep input concat in float domain: quantized cat requires QTensor inputs.
            "_generated_cat_0": None,
            # Avoid quantized interpolate remap assertion in eval path.
            "refinement.unfold_conv.unflod_conv": (
                default_qat_8bit_fake_quant_qconfig
            ),
            # Keep compile OUT0 (disp_unfold) on int32 so board postprocess
            # matches the PTQ S32xS16 accumulation path.
            "refinement.update_block.disp_head.conv2": (
                default_qat_8bit_weight_32bit_out_fake_quant_qconfig
            ),
        },
    },
)

qat_train_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=qat_update_loss_metric,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="train_qat_" + task_name,
)

qat_val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=qat_update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=log_freq,
    log_prefix="val_qat_" + task_name,
)

qat_val_callback = dict(
    type="Validation",
    interval_by="step",
    val_interval=1000,
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[qat_val_metric_updater],
    val_model=None,
    val_on_train_end=True,
)

qat_ckpt_callback = dict(
    type="Checkpoint",
    interval_by="step",
    save_interval=1000,
    save_dir=qat_stage_dir,
    name_prefix="qat-",
    strict_match=True,
    mode="min",
    monitor_metric_key="EPE",
)

qat_num_steps = 50000
qat_wandb_callback = copy.deepcopy(wandb_callback)
qat_wandb_callback.update(
    name=f"{task_name}-qat",
    ckpt_name_prefix="qat-",
    config={
        **qat_wandb_callback.get("config", {}),
        "num_steps": qat_num_steps,
    },
)

qat_callbacks = [
    stat_callback,
    qat_train_metric_updater,
    dict(
        type="CosLrUpdater",
        max_steps=qat_num_steps,
        warmup_by="step",
        warmup_len=500,
        step_log_interval=1000,
    ),
    qat_ckpt_callback,
    qat_val_callback,
    qat_wandb_callback,
]

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=qat_model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=4e-5)},
        lr=2e-5,
    ),
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=float_best_ckpt,
                allow_miss=False,
                ignore_extra=False,
                ignore_tensor_shape=False,
                verbose=True,
            ),
            qat_converter,
            dict(
                type="LoadCheckpoint",
                checkpoint_path=calibration_last_ckpt,
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=False,
                verbose=True,
            ),
        ],
    ),
    resume_optimizer=False,
    resume_epoch_or_step=False,
    resume_dataloader=False,
    batch_processor=qat_train_batch_processor,
    stop_by="step",
    num_steps=qat_num_steps,
    device=None,
    sync_bn=sync_bn,
    callbacks=qat_callbacks,
    train_metrics=[
        dict(type="LossShow"),
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    val_metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
)

int_infer_model = copy.deepcopy(qat_model)
int_infer_model["training_stage"] = "int_infer"
compile_model = copy.deepcopy(qat_model)
compile_model["training_stage"] = "compile"

int_infer_ckpt_callback = dict(
    type="Checkpoint",
    interval_by="epoch",
    save_interval=1,
    save_dir=int_infer_stage_dir,
    name_prefix="int_infer-",
    strict_match=True,
)

int_infer_trainer = dict(
    type="Trainer",
    model=int_infer_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            copy.deepcopy(qat_converter),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=qat_best_ckpt,
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=False,
                verbose=True,
            ),
            dict(
                type="QAT2Quantize",
                convert_mode="fx",
            ),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[stat_callback, int_infer_ckpt_callback],
)

compile_trainer = dict(
    type="Trainer",
    model=compile_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            copy.deepcopy(qat_converter),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=qat_best_ckpt,
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=False,
                verbose=True,
            ),
            dict(
                type="QAT2Quantize",
                convert_mode="fx",
            ),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[stat_callback],
)

int_infer_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=int_infer_update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=log_freq,
    log_prefix="int_infer_" + task_name,
)

int_infer_eval_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=int_infer_update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=log_freq,
    log_prefix="int_infer_eval_" + task_name,
)

int_infer_predictor = dict(
    type="Predictor",
    model=int_infer_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            copy.deepcopy(qat_converter),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=qat_best_ckpt,
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=False,
                verbose=True,
            ),
            dict(
                type="QAT2Quantize",
                convert_mode="fx",
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[int_infer_metric_updater, stat_callback],
    log_interval=log_freq,
)

int_infer_eval_predictor = copy.deepcopy(int_infer_predictor)
# Dataset-based int_infer eval follows training-style batch keys instead of
# deploy-style infra1/infra2 keys, so use qat forward contract here.
int_infer_eval_predictor["model"] = copy.deepcopy(qat_model)
int_infer_eval_predictor["model"]["training_stage"] = "qat"
int_infer_eval_predictor["model_convert_pipeline"]["converters"][0] = (
    int_infer_eval_qat_converter
)
int_infer_eval_predictor["callbacks"] = [
    int_infer_eval_metric_updater,
    stat_callback,
]

float_eval_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=qat_update_metric,
    step_log_freq=log_freq,
    epoch_log_freq=log_freq,
    log_prefix="float_eval_" + task_name,
)

float_eval_model = copy.deepcopy(model)
float_eval_model["training_stage"] = "float"

float_eval_predictor = dict(
    type="Predictor",
    model=float_eval_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=float_best_ckpt,
                allow_miss=False,
                ignore_extra=False,
                ignore_tensor_shape=False,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[
        dict(
            type="EndPointError",
            use_mask=True,
        ),
    ],
    callbacks=[
        float_eval_metric_updater,
        stat_callback,
    ],
    log_interval=log_freq,
)

compile_cfg = dict(
    march=march,
    name=qat_task_name,
    out_dir=compile_stage_dir,
    hbm=os.path.join(compile_stage_dir, "model.hbm"),
    layer_details=True,
    # Match PTQ's dual NV12 runtime path via two pyramid inputs.
    input_source=["pyramid", "pyramid"],
    # Keep HBM outputs in NCHW so PTQ/QAT share the same deploy postprocess.
    output_layout="NCHW",
)
