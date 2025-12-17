import copy

import horizon_plugin_pytorch as horizon
import pytest

from hat.utils import qconfig_manager
from tests.data.toy_modules import ToyBackbone, ToyHead, ToyLoss, ToyModel


@pytest.mark.parametrize(
    [
        "activation_fake_quant",
        "weight_fake_quant",
        "activation_qat_observer",
        "weight_qat_observer",
        "activation_calibration_observer",
        "weight_calibration_observer",
        "activation_qat_qkwargs",
        "weight_qat_qkwargs",
        "activation_calibration_qkwargs",
        "weight_calibration_qkwargs",
    ],
    [
        pytest.param(
            "fake_quant",
            "fake_quant",
            "min_max",
            "min_max",
            "percentile",
            "min_max",
            {"averaging_constant": 0.0},
            None,
            None,
            None,
        ),
        pytest.param(
            "lsq",
            "fake_quant",
            "min_max",
            "min_max",
            "min_max",
            "min_max",
            {"averaging_constant": 0.0},
            None,
            None,
            {"averaging_constant": 1.0},
        ),
        pytest.param(
            "pact",
            "lsq",
            "min_max",
            "min_max",
            "clip_std",
            "min_max",
            {"averaging_constant": 0.0},
            None,
            None,
            {"averaging_constant": 1.0},
        ),
    ],
)
def test_set_qconfig(
    activation_fake_quant,
    weight_fake_quant,
    activation_qat_observer,
    weight_qat_observer,
    activation_calibration_observer,
    weight_calibration_observer,
    activation_qat_qkwargs,
    weight_qat_qkwargs,
    activation_calibration_qkwargs,
    weight_calibration_qkwargs,
):

    qconfig_params = {
        "activation_fake_quant": activation_fake_quant,
        "weight_fake_quant": weight_fake_quant,
        "activation_qat_observer": activation_qat_observer,
        "weight_qat_observer": weight_qat_observer,
        "activation_calibration_observer": activation_calibration_observer,
        "weight_calibration_observer": weight_calibration_observer,
        "activation_qat_qkwargs": activation_qat_qkwargs,
        "weight_qat_qkwargs": weight_qat_qkwargs,
        "activation_calibration_qkwargs": activation_calibration_qkwargs,
        "weight_calibration_qkwargs": weight_calibration_qkwargs,
    }

    model = ToyModel(
        backbone=ToyBackbone(strides=(1, 2), channels=(3, 8)),
        head=ToyHead(
            in_channels=8,
            fc_filter=16,
            num_classes=10,
            with_dequant=True,
        ),
        loss=ToyLoss(),
    )

    qconfig_manager.set_default_qconfig(**qconfig_params)

    model.fuse_model()
    qat_model = copy.deepcopy(model)

    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.CALIBRATION)
    default_calibration_qconfig = qconfig_manager.get_default_qconfig()
    default_calibration_out_qconfig = qconfig_manager.get_default_out_qconfig()
    model.set_qconfig()
    horizon.quantization.prepare_qat(model, inplace=True)

    for name, module in model.named_modules():
        if name == "loss":
            assert getattr(module, "qconfig", None) is None
        elif name == "head.fc":
            assert module.qconfig == default_calibration_out_qconfig
        else:
            if hasattr("module", "qconfig"):
                assert module.qconfig == default_calibration_qconfig

    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
    default_qat_qconfig = qconfig_manager.get_default_qconfig()
    default_qat_out_qconfig = qconfig_manager.get_default_out_qconfig()
    qat_model.set_qconfig()
    horizon.quantization.prepare_qat(qat_model, inplace=True)

    for name, module in qat_model.named_modules():
        if name == "loss":
            assert getattr(module, "qconfig", None) is None
        elif name == "head.fc":
            assert module.qconfig == default_qat_out_qconfig
        else:
            if hasattr("module", "qconfig"):
                assert module.qconfig == default_qat_qconfig

    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.COMPATIBLE)
