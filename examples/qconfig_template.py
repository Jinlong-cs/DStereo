# Copyright (c) Horizon Robotics. All rights reserved.
"""
This file is an example of qconfig template and sensitive ops qconfig setter.

You can run in following steps:

1. run with qconfig template setter

    use_sensitive_qconfig_setter = False
    python3 tools/train.py -c examples/qconfig_template.py -s calibration
    python3 tools/predict.py -c examples/qconfig_template.py -s calibration

2. run quant analysis to get sensitive_ops.pt

    python3 tools/analyze/quant_analysis.py -c examples/qconfig_template.py

3. run with sensitive ops qconfig setter

    use_sensitive_qconfig_setter = True
    python3 tools/train.py -c examples/qconfig_template.py -s calibration
    python3 tools/predict.py -c examples/qconfig_template.py -s calibration

"""

from examples.classification.resnet18 import *

from horizon_plugin_pytorch.quantization.qconfig_template import (
    default_calibration_qconfig_setter,
    default_qat_qconfig_setter,
    sensitive_op_8bit_weight_16bit_act_calibration_setter,
    sensitive_op_8bit_weight_16bit_act_qat_setter,
)

use_sensitive_qconfig_setter = False

calibration_trainer["model_convert_pipeline"]["converters"] = [
    dict(
        type="LoadCheckpoint",
        checkpoint_path=os.path.join(
            ckpt_dir, "float-checkpoint-best.pth.tar"
        ),
    ),
    dict(
        type="Float2Calibration",
        convert_mode=convert_mode,
        qconfig_setter=default_calibration_qconfig_setter,
        example_inputs=deploy_inputs,
    ),
]

calibration_predictor["model_convert_pipeline"]["converters"] = [
    dict(
        type="Float2QAT",
        convert_mode=convert_mode,
        qconfig_setter=default_qat_qconfig_setter,
        example_inputs=deploy_inputs,
    ),
    dict(
        type="LoadCheckpoint",
        checkpoint_path=os.path.join(
            ckpt_dir, "calibration-checkpoint-last.pth.tar"
        ),
        verbose=True,
    ),
]

quant_analysis_solver = dict(
    type="QuantAnalysis",
    model=deploy_model,
    device_id=3,
    dataloader=val_data_loader,
    batch_transforms=val_batch_processor["batch_transforms"],
    num_steps=10,
    baseline_model_convert_pipeline=float_predictor["model_convert_pipeline"],
    analysis_model_convert_pipeline=calibration_predictor[
        "model_convert_pipeline"
    ],
    analysis_model_type="fake_quant",
)

if use_sensitive_qconfig_setter:
    sensitive_path = os.path.join(ckpt_dir, "quant_analysis/sensitive_ops.pt")
    if not os.path.exists(sensitive_path):
        raise RuntimeError(
            "Please run quant analysis before use sensitive qconfig setter."
        )

    sensitive_table = torch.load(sensitive_path)
    calibration_trainer = copy.deepcopy(calibration_trainer)
    calibration_trainer["model_convert_pipeline"]["converters"] = [
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
        ),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            qconfig_setter=(
                default_calibration_qconfig_setter,
                sensitive_op_8bit_weight_16bit_act_calibration_setter(
                    sensitive_table, ratio=0.2
                ),
            ),
            example_inputs=deploy_inputs,
        ),
    ]

    calibration_predictor = copy.deepcopy(calibration_predictor)
    calibration_predictor["model_convert_pipeline"]["converters"] = [
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=(
                default_calibration_qconfig_setter,
                sensitive_op_8bit_weight_16bit_act_qat_setter(
                    sensitive_table, ratio=0.2
                )[0],
            ),
            example_inputs=deploy_inputs,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-last.pth.tar"
            ),
            verbose=True,
        ),
    ]
