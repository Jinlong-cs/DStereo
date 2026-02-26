import copy
import os

from DStereo.DStereoPlus import *  # noqa: F401,F403

from horizon_plugin_pytorch.quantization.qconfig import (
    default_calib_8bit_fake_quant_qconfig,
    default_calib_8bit_weight_16bit_act_fake_quant_qconfig,
    default_calib_8bit_weight_32bit_out_fake_quant_qconfig,
)

training_stage = "calibration"

qat_task_name = "DStereoV23"
qat_work_root = os.path.join("work_dirs", "tmp_models_save_best", qat_task_name)
calibration_stage_dir = os.path.join(qat_work_root, "calibration")

model = copy.deepcopy(model)
model["training_stage"] = "calibration"

calibration_data_loader = copy.deepcopy(calib_data_loader)
calibration_data_loader["dataset"]["test_mode"] = False
calibration_data_loader["sampler"] = dict(
    type="InterleaveConcatSampler",
    shuffle=True,
    seed=666,
    sampler_len=256,
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
                checkpoint_path=os.path.join(
                    qat_work_root, "float", "float-checkpoint-best.pth.tar"
                ),
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
