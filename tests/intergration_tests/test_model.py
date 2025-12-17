import copy

import horizon_plugin_pytorch as horizon
import pytest
import torch
from horizon_plugin_pytorch.quantization import check_model

from hat.registry import build_from_registry
from hat.utils.apply_func import to_cuda
from hat.utils.config import Config
from hat.utils.qconfig_manager import QconfigMode, set_qconfig_mode
from hat.utils.statistics import cal_ops


@pytest.mark.parametrize(
    ["fname"],
    [
        pytest.param("tests/data/retinanet_vargnetv2_fpn_mscoco.py"),
    ],
)
def test_model(fname):
    config = Config.fromfile(fname)

    # Test float
    model = build_from_registry(copy.deepcopy(config.model))

    # Test forward train with non-empty truth batch
    inputs = copy.deepcopy(config.inputs)
    model.train()
    outputs = model.forward(inputs)
    assert outputs is not None

    # Test forward train with an empty truth batch
    inputs_wo_bboxes = copy.deepcopy(config.inputs_wo_bboxes)
    outputs = model.forward(inputs_wo_bboxes)
    assert outputs is not None

    # Test forward test
    inputs = copy.deepcopy(config.inputs)
    model.eval()
    with torch.no_grad():
        outputs = model.forward(inputs)
        assert outputs is not None

    # Test calops
    inputs = copy.deepcopy(config.deploy_inputs)
    deploy_model = build_from_registry(copy.deepcopy(config.deploy_model))
    deploy_model.eval()
    total_ops, total_params = cal_ops(deploy_model, inputs)
    assert total_ops is not None
    assert total_params is not None

    # Test qat
    horizon.march.set_march(horizon.march.March.BAYES)
    inputs = copy.deepcopy(config.inputs)
    model = build_from_registry(copy.deepcopy(config.model))
    model.train()
    model.fuse_model()
    set_qconfig_mode(QconfigMode.QAT)
    model.set_qconfig()
    qat_model = horizon.quantization.prepare_qat(model, inplace=False)

    # Test forward train with non-empty truth batch
    outputs = qat_model.forward(inputs)
    assert outputs is not None

    # Test forward test
    inputs = copy.deepcopy(config.inputs)
    qat_model.eval()
    with torch.no_grad():
        outputs = qat_model.forward(inputs)
        assert outputs is not None

    # Test quantize
    inputs = copy.deepcopy(config.inputs)
    quantized_model = horizon.quantization.convert(
        qat_model.eval(), inplace=False
    )
    with torch.no_grad():
        outputs = quantized_model.forward(inputs)
        assert outputs is not None

    # Test model-checker
    horizon.march.set_march(horizon.march.March.BAYES)
    model = build_from_registry(copy.deepcopy(config.deploy_model))
    trace_inputs = copy.deepcopy(config.deploy_inputs)
    model.fuse_model()
    model.set_qconfig()
    horizon.quantization.prepare_qat(model, inplace=True)
    int_model = horizon.quantization.convert(model.eval(), inplace=False)

    flag = check_model(int_model, trace_inputs, advice=10)
    if flag != 0:
        raise AssertionError(f"{fname} Failed to pass hbdk checker")

    # Test gpu
    if not torch.cuda.is_available():
        pytest.skip("test requires GPU and torch+cuda")
    inputs = copy.deepcopy(config.inputs)
    model = build_from_registry(copy.deepcopy(config.model))
    model.train()
    model = model.cuda()
    inputs = to_cuda(inputs)

    # Test forward train
    outputs = model(inputs)
    assert outputs is not None

    # Test forward test
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        assert outputs is not None
