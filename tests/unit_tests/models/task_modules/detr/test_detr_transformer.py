import torch

from hat.models.task_modules.detr.transformer import Transformer
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate


class TestDetrTransformer(HeadTemplate):
    def setup(self):
        super(TestDetrTransformer, self).setup()
        self.embed_dims = 64
        self.num_heads = 8
        self.num_decoder_layers = 6
        self.ff_channels = 64
        self.return_intermediate_dec = True
        self.model = Transformer(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            num_decoder_layers=self.num_decoder_layers,
            feedforward_channels=self.ff_channels,
            return_intermediate_dec=self.return_intermediate_dec,
        )

        self.input = torch.rand(1, self.embed_dims, 8, 8)
        self.mask = torch.randint(0, 2, size=(1, 8, 8)).to(torch.bool)
        self.query_embed = torch.rand(10, self.embed_dims)
        self.pos_embed = torch.rand(1, self.embed_dims, 8, 8)
        self.quantized_input = torch.ones(
            1, self.embed_dims, 8, 8, dtype=torch.int8
        )
        self.quantized_query_embed = torch.ones(
            10, self.embed_dims, dtype=torch.int8
        )
        self.quantized_pos_embed = torch.ones(
            1, self.embed_dims, 8, 8, dtype=torch.int8
        )

        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(
            self.input, self.mask, self.query_embed, self.pos_embed
        )

        assert outputs[0].shape[0] == self.num_decoder_layers
        assert outputs[0].shape[1] == 1
        assert outputs[0].shape[2] == 10
        assert outputs[0].shape[3] == self.embed_dims

        assert outputs[1].shape[0] == 1
        assert outputs[1].shape[1] == self.embed_dims
        assert outputs[1].shape[2] == 8
        assert outputs[1].shape[3] == 8

    def test_qat_model(self):
        qat_input = qtensor_test(self.input)
        qat_query_embed = qtensor_test(self.query_embed)
        qat_pos_embed = qtensor_test(self.pos_embed)
        qat_outputs = self.qat_model(
            qat_input, self.mask, qat_query_embed, qat_pos_embed
        )

        assert qat_outputs[0].shape[0] == self.num_decoder_layers
        assert qat_outputs[0].shape[1] == 1
        assert qat_outputs[0].shape[2] == 10
        assert qat_outputs[0].shape[3] == self.embed_dims

        assert qat_outputs[1].shape[0] == 1
        assert qat_outputs[1].shape[1] == self.embed_dims
        assert qat_outputs[1].shape[2] == 8
        assert qat_outputs[1].shape[3] == 8

    def test_quantize_model(self):
        quantized_input = qtensor_test(self.quantized_input)
        quantized_query_embed = qtensor_test(self.quantized_query_embed)
        quantized_pos_embed = qtensor_test(self.quantized_pos_embed)
        quantized_outputs = self.quantized_model(
            quantized_input,
            self.mask,
            quantized_query_embed,
            quantized_pos_embed,
        )

        assert quantized_outputs[0].shape[0] == self.num_decoder_layers
        assert quantized_outputs[0].shape[1] == 1
        assert quantized_outputs[0].shape[2] == 10
        assert quantized_outputs[0].shape[3] == self.embed_dims

        assert quantized_outputs[1].shape[0] == 1
        assert quantized_outputs[1].shape[1] == self.embed_dims
        assert quantized_outputs[1].shape[2] == 8
        assert quantized_outputs[1].shape[3] == 8

    def test_fuse_model(self):
        outputs = self.float_model(
            self.input, self.mask, self.query_embed, self.pos_embed
        )
        fuse_outputs = self.fuse_model(
            self.input, self.mask, self.query_embed, self.pos_embed
        )
        for i in range(len(outputs)):
            assert all(
                torch.allclose(output, fuse_output, atol=1e-5)
                for output, fuse_output in zip(outputs[i], fuse_outputs[i])
            )
