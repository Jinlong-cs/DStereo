import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate

num_queries = 256


class TestMotrDeformableTransformer(HeadTemplate):
    def setup(self):
        super(TestMotrDeformableTransformer, self).setup()
        x1_input = [torch.randn((1, 256, 25, 45))]
        x2_input = torch.randn((1, 256, 4, 128))
        x3_input = torch.randn((1, 4, 4, 128))
        x4_input = torch.ones((512, 512))
        x5_input = torch.ones((256, 256))
        x1_input_int = [torch.ones((1, 256, 25, 45), dtype=torch.int8)]
        x2_input_int = torch.ones((1, 256, 4, 128), dtype=torch.int16)
        x3_input_int = torch.ones((1, 4, 4, 128), dtype=torch.int16)
        x4_input_int = torch.ones((512, 512), dtype=torch.int8)
        x5_input_int = torch.ones((256, 256), dtype=torch.int8)

        self.inputs = [x1_input, x2_input, x3_input, x4_input, x5_input]
        self.quantized_inputs = [
            x1_input_int,
            x2_input_int,
            x3_input_int,
            x4_input_int,
            x5_input_int,
        ]

        model_config = dict(
            type="MotrDeformableTransformer",
            pos_embed=dict(
                type="PositionEmbeddingSine",
                num_pos_feats=128,
                normalize=True,
                temperature=20,
            ),
            d_model=256,
            num_queries=num_queries,
            dim_feedforward=1024,
            dropout=0.0,
            return_intermediate_dec=True,
            extra_track_attn=True,
            enc_n_points=1,
            dec_n_points=1,
        )
        self.model = build_from_registry(model_config)
        self.build_model()

    def test_float_model(self):

        outputs = self.float_model(
            self.inputs[0],
            self.inputs[1],
            self.inputs[2],
            self.inputs[3],
            self.inputs[4],
        )
        assert len(outputs) == 2
        assert isinstance(outputs[0][0], torch.Tensor)
        assert isinstance(outputs[1][0], torch.Tensor)

        assert outputs[0][0].shape == (1, 256, 4, 128)
        assert outputs[1][0].shape == (1, 4, 4, 128)

    def test_fuse_model(self):
        outputs = self.float_model(
            self.inputs[0],
            self.inputs[1],
            self.inputs[2],
            self.inputs[3],
            self.inputs[4],
        )
        fuse_outputs = self.fuse_model(
            self.inputs[0],
            self.inputs[1],
            self.inputs[2],
            self.inputs[3],
            self.inputs[4],
        )
        for i in range(len(outputs)):
            assert all(
                torch.allclose(output, fuse_output, atol=1e-5)
                for output, fuse_output in zip(outputs[i], fuse_outputs[i])
            )

    def test_qat_model(self):
        qat_inputs1 = qtensor_test(self.inputs[0])
        qat_inputs2 = qtensor_test(self.inputs[1], dtype="int16")
        qat_inputs3 = qtensor_test(self.inputs[2], dtype="int16")
        qat_inputs4 = qtensor_test(self.inputs[3])
        qat_inputs5 = qtensor_test(self.inputs[4])
        qat_outputs = self.qat_model(
            qat_inputs1,
            qat_inputs2,
            qat_inputs3,
            qat_inputs4,
            qat_inputs5,
        )
        assert len(qat_outputs) == 2
        assert isinstance(qat_outputs[0][0], torch.Tensor)
        assert isinstance(qat_outputs[1][0], torch.Tensor)

        assert qat_outputs[0][0].shape == (1, 256, 4, 128)
        assert qat_outputs[1][0].shape == (1, 4, 4, 128)

    def test_quantize_model(self):
        quantized_inputs1 = qtensor_test(self.quantized_inputs[0])
        quantized_inputs2 = qtensor_test(
            self.quantized_inputs[1], dtype="int16"
        )
        quantized_inputs3 = qtensor_test(
            self.quantized_inputs[2], dtype="int16"
        )
        quantized_inputs4 = qtensor_test(self.quantized_inputs[3])
        quantized_inputs5 = qtensor_test(self.quantized_inputs[4])
        quantized_outputs = self.quantized_model(
            quantized_inputs1,
            quantized_inputs2,
            quantized_inputs3,
            quantized_inputs4,
            quantized_inputs5,
        )
        assert len(quantized_outputs) == 2
        assert isinstance(quantized_outputs[0][0], torch.Tensor)
        assert isinstance(quantized_outputs[1][0], torch.Tensor)

        assert quantized_outputs[0][0].shape == (1, 256, 4, 128)
        assert quantized_outputs[1][0].shape == (1, 4, 4, 128)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
