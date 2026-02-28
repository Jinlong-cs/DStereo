import copy
import os

import torch
import torch.nn.functional as F

from DStereo.DStereoPlus import *  # noqa: F401,F403

from horizon_plugin_pytorch.quantization.qconfig import (
    default_calib_8bit_fake_quant_qconfig,
    default_calib_8bit_weight_16bit_act_fake_quant_qconfig,
    default_calib_8bit_weight_32bit_out_fake_quant_qconfig,
    default_qat_8bit_weight_16bit_act_fake_quant_qconfig,
    default_qat_8bit_weight_32bit_out_fake_quant_qconfig,
)

training_stage = "calibration"

qat_task_name = "DStereoV23"
qat_work_root = os.path.join("work_dirs", "tmp_models_save_best", qat_task_name)
float_stage_dir = os.path.join(qat_work_root, "float")
calibration_stage_dir = os.path.join(qat_work_root, "calibration")
qat_stage_dir = os.path.join(qat_work_root, "qat")

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


def _dequantize_qat_tensor(value):
    return value.dequantize() if hasattr(value, "dequantize") else value


def _compute_qat_sequence_loss(agg_pred, iter_preds, disp_gt, loss_gamma=0.9):
    agg_pred = _dequantize_qat_tensor(agg_pred)
    disp_gt = _dequantize_qat_tensor(disp_gt)
    iter_preds = [_dequantize_qat_tensor(pred) for pred in iter_preds]

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
    labels = batch["gt_disp"]
    masks = (labels > 0) & (labels < maxdisp)
    losses = _compute_qat_sequence_loss(*model_outs["loss_inputs"])
    metrics[0].update(sum(losses))
    metrics[1].update(labels, model_outs["pred_disps"], masks)


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
                for name in (
                    # Keep backbone/aggregation path on int8 to avoid int16-only branches.
                    "backbone",
                    "feature",
                    "before_costvolum",
                    "cost_agg",
                    "get_costvolum",
                    "get_initdisp",
                    # refinement prepare path: force int8 to match BPU-supported input types.
                    "prepare_forrefinement",
                    "prepare_forrefinement.context_zqr_conv",
                    "spx",
                    "spx_2",
                    "spx_4",
                    # refinement update path: force int8 to avoid int16 Sumin checks in conv/add.
                    "refinement",
                    "refinement.update_block",
                    "refinement.update_block.encoder",
                    "refinement.update_block.gru",
                    "refinement.update_block.mask_feat_4",
                    "refinement.spx_2_gru",
                    "refinement.spx_gru",
                    # SegmentLUT(tanh/sigmoid): keep q8->q8 and avoid q8->q16 LUT assert.
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
            },
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
            # Keep final disparity delta output in higher precision during QAT.
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
    metric_update_func=update_metric,
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

qat_num_steps = 20000
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
