import copy

import pytest
import torch
from torch.nn import ReLU
from torch.nn.modules.batchnorm import _BatchNorm

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.necks.neck_template import NeckTemplate

in_channels = [64, 128, 256]


class TestGaNetNeck(NeckTemplate):
    def setup(self):
        super(TestGaNetNeck, self).setup()
        self.inputs = []
        self.quantized_inputs = []
        self.input_h, self.input_w = 40, 100
        ratio = 1
        for i in range(3):
            x1 = torch.rand(
                1,
                in_channels[i],
                int(self.input_h / ratio),
                int(self.input_w / ratio),
            )
            x1_int = torch.ones(
                1,
                in_channels[i],
                int(self.input_h / ratio),
                int(self.input_w / ratio),
                dtype=torch.int8,
            )
            ratio *= 2
            self.inputs.append(x1)
            self.quantized_inputs.append(x1_int)

        neck_config = dict(
            type="GaNetNeck",
            fpn_module=dict(
                type="FPN",
                in_strides=[8, 16, 32],
                in_channels=[64, 128, 64],
                out_strides=[8, 16, 32],
                out_channels=[64, 64, 64],
            ),
            attn_in_channels=[256, 64],
            attn_out_channels=[64, 64],
            attn_ratios=[4, 4],
            pos_shape=(1, 10, 25),
        )

        self.model = build_from_registry(neck_config)

        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        stride = 1
        for output in outputs:
            assert isinstance(output, torch.Tensor)
            assert len(output.shape) == 4
            assert output.shape[1] == 64
            assert output.shape[2] == self.input_h // stride
            assert output.shape[3] == self.input_w // stride
            stride = stride * 2

    def test_fuse_model(self):
        fuse_input = copy.deepcopy(self.inputs)
        outputs = self.float_model(self.inputs)

        fuse_outputs = self.fuse_model(fuse_input)
        for output, fuse_output in zip(outputs, fuse_outputs):
            assert all(
                torch.allclose(output_tmp, fuse_output_tmp, atol=0.001)
                for output_tmp, fuse_output_tmp in zip(output, fuse_output)
            )
        # print(self.qat_model)
        for module in self.qat_model.modules():
            assert not (
                isinstance(module, _BatchNorm) or isinstance(module, ReLU)
            )

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        stride = 1
        for qat_output in qat_outputs:
            assert isinstance(qat_output, torch.Tensor)
            assert len(qat_output.shape) == 4
            assert qat_output.shape[1] == 64
            assert qat_output.shape[2] == self.input_h // stride
            assert qat_output.shape[3] == self.input_w // stride
            stride = stride * 2

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        stride = 1
        for quantized_output in quantized_outputs:
            assert isinstance(quantized_output, torch.Tensor)
            assert len(quantized_output.shape) == 4
            assert quantized_output.shape[1] == 64
            assert quantized_output.shape[2] == self.input_h // stride
            assert quantized_output.shape[3] == self.input_w // stride
            stride = stride * 2


if __name__ == "__main__":
    pytest.main(["-s", __file__])
