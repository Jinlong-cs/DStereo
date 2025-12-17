from abc import abstractmethod

import torch
from horizon_plugin_pytorch.qtensor import QTensor

from tests.unit_tests.models.base import ModelTemplate, qtensor_test


class NeckTemplate(ModelTemplate):
    @abstractmethod
    def setup(self):
        self.input_size = 256
        self.in_strides = [8, 16, 32]
        self.in_channels = [16, 32, 64]
        self.out_strides = [8, 16, 32]
        self.out_channels = [64, 64, 64]
        self.inputs = [
            torch.rand(
                1,
                in_ch,
                self.input_size // in_stride,
                self.input_size // in_stride,
            )
            for (in_ch, in_stride) in zip(self.in_channels, self.in_strides)
        ]
        self.quantized_inputs = [
            torch.ones(
                1,
                in_ch,
                self.input_size // in_stride,
                self.input_size // in_stride,
                dtype=torch.int8,
            )
            for (in_ch, in_stride) in zip(self.in_channels, self.in_strides)
        ]

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, list) and len(outputs) == len(
            self.out_strides
        )
        for (i, output) in enumerate(outputs):
            assert isinstance(output, torch.Tensor)
            assert len(output.shape) == 4
            assert output.shape[-1] == self.input_size // self.out_strides[i]

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs, list) and len(qat_outputs) == len(
            self.out_strides
        )
        for (i, output) in enumerate(qat_outputs):
            assert isinstance(output, QTensor)
            assert len(output.shape) == 4
            assert output.shape[-1] == self.input_size // self.out_strides[i]

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs, list) and len(
            quantized_outputs
        ) == len(self.out_strides)
        for (i, output) in enumerate(quantized_outputs):
            assert isinstance(output, QTensor)
            assert len(output.shape) == 4
            assert output.shape[-1] == self.input_size // self.out_strides[i]
