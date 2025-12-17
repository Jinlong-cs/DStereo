import copy
from abc import abstractmethod

import horizon_plugin_pytorch as horizon
import torch
from horizon_plugin_pytorch.qtensor import QTensor
from torch.nn import ReLU
from torch.nn.modules.batchnorm import _BatchNorm

from hat.utils import qconfig_manager
from hat.utils.statistics import cal_ops


def profile_test(model, input):
    total_ops, total_params = cal_ops(model, input)
    assert total_ops is not None
    assert total_params is not None
    return total_ops, total_params


def qat_test(model, input, with_quantized=True):
    horizon.march.set_march(horizon.march.March.BAYES)
    model.fuse_model()
    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
    model.qconfig = qconfig_manager.get_default_qat_qconfig()
    if hasattr(model, "set_qconfig"):
        model.set_qconfig()
    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.COMPATIBLE)
    qat_model = horizon.quantization.prepare_qat(model, inplace=False)

    qat_preds = qat_model(input)
    assert qat_preds is not None

    quantized_preds = None
    if with_quantized:
        quantized_model = horizon.quantization.convert(
            qat_model.eval(), inplace=False
        )
        quantized_preds = quantized_model(input)
        assert quantized_preds is not None

    return qat_preds, quantized_preds


def qat_test_fx(model, input, with_quantized=True):
    horizon.march.set_march(horizon.march.March.BAYES)
    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
    model.qconfig = qconfig_manager.get_default_qat_qconfig()
    if hasattr(model, "set_qconfig"):
        model.set_qconfig()
    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.COMPATIBLE)
    qat_model = horizon.quantization.prepare_qat_fx(model)

    qat_preds = qat_model(input)
    assert qat_preds is not None

    quantized_preds = None
    if with_quantized:
        quantized_model = horizon.quantization.convert_fx(qat_model.eval())
        quantized_preds = quantized_model(input)
        assert quantized_preds is not None

    return qat_preds, quantized_preds


def qat_test_with_multi_inputs(model, inputs, with_quantized=True):
    horizon.march.set_march(horizon.march.March.BAYES)
    model.fuse_model()
    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
    model.qconfig = qconfig_manager.get_default_qat_qconfig()
    if hasattr(model, "set_qconfig"):
        model.set_qconfig()
    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.COMPATIBLE)
    qat_model = horizon.quantization.prepare_qat(model, inplace=False)

    qat_preds = qat_model(*inputs)
    assert qat_preds is not None

    quantized_preds = None
    if with_quantized:
        quantized_model = horizon.quantization.convert(
            qat_model.eval(), inplace=False
        )
        quantized_preds = quantized_model(*inputs)
        assert quantized_preds is not None

    return qat_preds, quantized_preds


def qtensor_test(x, dtype="int8"):
    if isinstance(x, (list, tuple)):
        return [qtensor_test(data, dtype) for data in x]
    else:
        if dtype == "int8":
            x = QTensor(x, torch.tensor([1.0]), horizon.dtype.qint8)
        elif dtype == "int16":
            x = QTensor(x, torch.tensor([1.0]), horizon.dtype.qint16)
        else:
            raise ValueError(f"Unspport quant to dtype: {dtype}")
        assert isinstance(x, QTensor)
        return x


class ModelTemplate(object):
    @abstractmethod
    def setup(self):
        """The setup function will be executed before the start of each test case.
        It is used to define the network, inputs and some other variables.
        """
        self.model = None
        self.inputs = None

    def build_model(self, use_fx=False):
        # float model
        self.float_model = copy.deepcopy(self.model)
        self.float_model.eval()

        # qat model
        horizon.march.set_march(horizon.march.March.BAYES)
        self.fuse_model = copy.deepcopy(self.float_model)
        if not use_fx:
            if hasattr(self.fuse_model, "fuse_model"):
                self.fuse_model.fuse_model()
        qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
        self.fuse_model.qconfig = qconfig_manager.get_default_qat_qconfig()
        if hasattr(self.fuse_model, "set_qconfig"):
            self.fuse_model.set_qconfig()
        qconfig_manager.set_qconfig_mode(
            qconfig_manager.QconfigMode.COMPATIBLE
        )
        if not use_fx:
            self.qat_model = horizon.quantization.prepare_qat(
                self.fuse_model, inplace=False
            )
        else:
            self.qat_model = horizon.quantization.prepare_qat_fx(
                copy.deepcopy(self.fuse_model)
            )

        # quantize model
        self.quantized_model = horizon.quantization.convert(
            self.qat_model.eval(), inplace=False
        )

    def test_float_model(self):
        pass

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        if isinstance(outputs, (tuple, list)):
            assert all(
                torch.allclose(output, fuse_output, atol=1e-5)
                for output, fuse_output in zip(outputs, fuse_outputs)
            )
        else:
            assert torch.allclose(outputs, fuse_outputs, atol=1e-6)
        for module in self.qat_model.modules():
            assert not (
                isinstance(module, _BatchNorm) or isinstance(module, ReLU)
            )

    def test_qat_model(self):
        pass

    def test_quantize_model(self):
        pass
