import numpy as np
import pytest
import torch

from hat.models.task_modules.detr3d.head import (
    Detr3dDecoder,
    Detr3dHead,
    Detr3dTransformer,
)
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate

bev_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)


class TestDetr3dHead(HeadTemplate):
    def setup(self):
        super(TestDetr3dHead, self).setup()
        self.input_size = (512, 960)
        h_tmp = self.input_size[0]
        w_tmp = self.input_size[1]

        imgs = torch.from_numpy(np.random.randn(6, 3, 512, 960)).float()

        self.inputs = [
            torch.rand(6, 256, h_tmp // 4, w_tmp // 4),
            torch.rand(6, 256, h_tmp // 8, w_tmp // 8),
            torch.rand(6, 256, h_tmp // 16, w_tmp // 16),
            torch.rand(6, 256, h_tmp // 32, w_tmp // 32),
        ]

        ego2imgs = torch.from_numpy(np.random.randn(6, 4, 4)).float()
        self.meta = {
            "img": imgs,
            "ego2img": ego2imgs,
        }

        self.quantized_inputs = [
            torch.ones(6, 256, h_tmp // 4, w_tmp // 4, dtype=torch.int8),
            torch.ones(6, 256, h_tmp // 8, w_tmp // 8, dtype=torch.int8),
            torch.ones(6, 256, h_tmp // 16, w_tmp // 16, dtype=torch.int8),
            torch.ones(6, 256, h_tmp // 32, w_tmp // 32, dtype=torch.int8),
        ]

        decoder = Detr3dDecoder(
            num_layer=6,
            num_heads=8,
            embed_dims=256,
            dropout=0.1,
            num_levels=4,
            num_views=6,
            num_points=1,
        )

        transformer = Detr3dTransformer(
            decoder=decoder,
            embed_dims=256,
            num_views=6,
            grid_quant_scales=[1 / 128, 1 / 128, 1 / 128, 1 / 128],
        )
        self.model = Detr3dHead(
            num_query=900,
            query_align=256,
            embed_dims=256,
            num_levels=4,
            num_cls_fcs=2,
            num_reg_fcs=2,
            reg_out_channels=10,
            cls_out_channels=10,
            bev_range=bev_range,
            transformer=transformer,
            int8_output=False,
            dequant_output=True,
        )

        self.build_model()

    def test_float_model(self):
        outputs, ref_p = self.float_model(self.inputs, self.meta)
        for out in outputs[0]:
            assert out.shape == (1, 10, 4, 256)

        for out in outputs[1]:
            assert out.shape == (1, 10, 4, 256)
        assert ref_p.shape == (1, 4, 256, 3)

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs, ref_p = self.qat_model(qat_inputs, self.meta)
        for out in qat_outputs[0]:
            assert out.shape == (1, 10, 4, 256)

        for out in qat_outputs[1]:
            assert out.shape == (1, 10, 4, 256)

        assert ref_p.shape == (1, 4, 256, 3)

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs, ref_p = self.quantized_model(
            quantized_inputs, self.meta
        )
        for out in quantized_outputs[0]:
            assert out.shape == (1, 10, 4, 256)
        for out in quantized_outputs[1]:
            assert out.shape == (1, 10, 4, 256)
        assert ref_p.shape == (1, 4, 256, 3)

    def test_fuse_model(self):
        outputs, _ = self.float_model(self.inputs, self.meta)
        fuse_outputs, _ = self.fuse_model(self.inputs, self.meta)
        for output, fuse_output in zip(outputs, fuse_outputs):
            for out, fout in zip(output, fuse_output):
                assert torch.allclose(out, fout, atol=1e-5)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
