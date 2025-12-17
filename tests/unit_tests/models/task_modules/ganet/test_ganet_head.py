import pytest
import torch

from hat.models.task_modules.ganet.head import GaNetHead
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate


class TestGaNetHead(HeadTemplate):
    def setup(self):
        super(TestGaNetHead, self).setup()
        self.input_size = (320, 800)
        in_channel = 64
        h_tmp = int(self.input_size[0] / 8)
        w_tmp = int(self.input_size[1] / 8)

        self.inputs = torch.rand(1, 64, h_tmp, w_tmp)
        self.quantized_inputs = torch.ones(
            1, 64, h_tmp, w_tmp, dtype=torch.int8
        )
        self.model = GaNetHead(in_channel=in_channel)
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert outputs[0].shape == (1, 1, 40, 100)
        assert outputs[1].shape == (1, 2, 40, 100)
        assert outputs[2].shape == (1, 2, 40, 100)

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert qat_outputs[0].shape == (1, 1, 40, 100)
        assert qat_outputs[1].shape == (1, 2, 40, 100)
        assert qat_outputs[2].shape == (1, 2, 40, 100)

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert quantized_outputs[0].shape == (1, 1, 40, 100)
        assert quantized_outputs[1].shape == (1, 2, 40, 100)
        assert quantized_outputs[2].shape == (1, 2, 40, 100)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
