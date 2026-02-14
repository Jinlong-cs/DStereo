import copy
import os

from DStereo.DStereoPlus import *  # noqa: F401,F403

from horizon_plugin_pytorch.quantization.qconfig import (
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
            # Keep final disparity delta output in high precision during
            # calibration; other modules follow global w8a16 policy.
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
