import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate


class TestQIM(HeadTemplate):
    def setup(self):
        super(TestQIM, self).setup()
        x1_input = torch.randn((1, 1, 256, 256))
        x2_input = torch.randn((1, 1, 256, 256))
        x3_input = torch.ones((1, 1, 1, 256))
        x1_input_int = torch.ones((1, 1, 256, 256), dtype=torch.int16)
        x2_input_int = torch.ones((1, 1, 256, 256), dtype=torch.int16)
        x3_input_int = torch.ones((1, 1, 1, 256), dtype=torch.int8)

        self.inputs = [x1_input, x2_input, x3_input]
        self.quantized_inputs = [x1_input_int, x2_input_int, x3_input_int]

        model_config = dict(
            type="QueryInteractionModule",
            dim_in=256,
            hidden_dim=1024,
        )
        self.model = build_from_registry(model_config)
        self.build_model()

    def test_float_model(self):

        outputs = self.float_model(
            self.inputs[0],
            self.inputs[1],
            self.inputs[2],
        )
        assert len(outputs) == 1
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (1, 256, 1, 256)

    def test_fuse_model(self):
        outputs = self.float_model(
            self.inputs[0],
            self.inputs[1],
            self.inputs[2],
        )
        fuse_outputs = self.fuse_model(
            self.inputs[0],
            self.inputs[1],
            self.inputs[2],
        )
        for i in range(len(outputs)):
            assert all(
                torch.allclose(output, fuse_output, atol=1e-5)
                for output, fuse_output in zip(outputs[i], fuse_outputs[i])
            )

    def test_qat_model(self):
        qat_inputs1 = qtensor_test(self.inputs[0], dtype="int16")
        qat_inputs2 = qtensor_test(self.inputs[1], dtype="int16")
        qat_inputs3 = qtensor_test(self.inputs[2])
        qat_outputs = self.qat_model(
            qat_inputs1,
            qat_inputs2,
            qat_inputs3,
        )
        assert len(qat_outputs) == 1
        assert isinstance(qat_outputs, torch.Tensor)
        assert qat_outputs.shape == (1, 256, 1, 256)

    def test_quantize_model(self):
        quantized_inputs1 = qtensor_test(
            self.quantized_inputs[0], dtype="int16"
        )
        quantized_inputs2 = qtensor_test(
            self.quantized_inputs[1], dtype="int16"
        )
        quantized_inputs3 = qtensor_test(self.quantized_inputs[2])
        quantized_outputs = self.quantized_model(
            quantized_inputs1,
            quantized_inputs2,
            quantized_inputs3,
        )
        assert len(quantized_outputs) == 1
        assert isinstance(quantized_outputs, torch.Tensor)
        assert quantized_outputs.shape == (1, 256, 1, 256)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
