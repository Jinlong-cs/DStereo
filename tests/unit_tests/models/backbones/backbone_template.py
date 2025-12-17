from abc import abstractmethod

import torch
from horizon_plugin_pytorch.qtensor import QTensor

from tests.unit_tests.models.base import ModelTemplate


class ClassifierTemplate(ModelTemplate):
    @abstractmethod
    def setup(self):
        self.input_size = 224
        self.num_classes = 2
        self.inputs = torch.randn(1, 3, self.input_size, self.input_size)

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, torch.Tensor)
        assert len(outputs.shape) == 2
        assert outputs.shape[-1] == self.num_classes

    def test_qat_model(self):
        qat_outputs = self.qat_model(self.inputs)
        assert isinstance(qat_outputs, torch.Tensor)
        assert len(qat_outputs.shape) == 2
        assert qat_outputs.shape[-1] == self.num_classes

    def test_quantize_model(self):
        quantized_outputs = self.quantized_model(self.inputs)
        assert isinstance(quantized_outputs, torch.Tensor)
        assert len(quantized_outputs.shape) == 2
        assert quantized_outputs.shape[-1] == self.num_classes


class BackboneTemplate(ModelTemplate):
    @abstractmethod
    def setup(self):
        self.input_size = 224
        self.num_classes = 2
        self.inputs = torch.randn(1, 3, self.input_size, self.input_size)

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, list) and len(outputs) == len(
            self.in_strides
        )
        for (i, output) in enumerate(outputs):
            assert isinstance(output, torch.Tensor)
            assert len(output.shape) == 4
            assert output.shape[-1] == self.input_size // self.in_strides[i]

    def test_qat_model(self):
        qat_outputs = self.qat_model(self.inputs)
        assert isinstance(qat_outputs, list) and len(qat_outputs) == len(
            self.in_strides
        )
        for (i, output) in enumerate(qat_outputs):
            assert isinstance(output, QTensor)
            assert len(output.shape) == 4
            assert output.shape[-1] == self.input_size // self.in_strides[i]

    def test_quantize_model(self):
        quantized_outputs = self.quantized_model(self.inputs)
        assert isinstance(quantized_outputs, list) and len(
            quantized_outputs
        ) == len(self.in_strides)
        for (i, output) in enumerate(quantized_outputs):
            assert isinstance(output, QTensor)
            assert len(output.shape) == 4
            assert output.shape[-1] == self.input_size // self.in_strides[i]
