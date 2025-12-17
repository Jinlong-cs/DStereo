import copy

import horizon_plugin_pytorch as horizon
import torch
from horizon_plugin_pytorch.qtensor import QTensor

from hat.models.task_modules.hand3d.head.h3d_mano_head import (
    H3DHeatmapHead,
    H3DLinear,
    H3DManoMultiFcHead,
)
from tests.unit_tests.models.task_modules.head_template import HeadTemplate


class TestH3DLinear(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 16
        self.inputs = torch.rand(self.batch_size, 256, 1, 1)
        self.quantized_inputs = torch.rand(self.batch_size, 256, 1, 1)

    def setup(self):
        super().setup()
        self.get_inputs()
        self.model = H3DLinear(
            input_channels=256,
            output_channels=10,
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
            disable_act=False,
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert outputs.shape == (self.batch_size, 10, 1, 1)

    def test_qat_model(self):
        qat_inputs = copy.deepcopy(self.inputs)
        qat_inputs = QTensor(
            qat_inputs, torch.tensor([1.0]), horizon.dtype.qint8
        )
        qat_outputs = self.qat_model(qat_inputs)
        assert qat_outputs.shape == (self.batch_size, 10, 1, 1)

    def test_quantize_model(self):
        quantized_inputs = copy.deepcopy(self.quantized_inputs)
        quantized_inputs = QTensor(
            quantized_inputs, torch.tensor([1.0]), horizon.dtype.qint8
        )
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert quantized_outputs.shape == (self.batch_size, 10, 1, 1)

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        assert torch.allclose(outputs, fuse_outputs, atol=1e-5)


class TestH3DHeatmapHead(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 16
        self.inputs = torch.rand(self.batch_size, 21, 32, 32)
        self.quantized_inputs = torch.rand(self.batch_size, 21, 32, 32)

    def setup(self):
        super().setup()
        self.get_inputs()
        self.model = H3DHeatmapHead(
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
            num_joints=21,
            stride=1,
            bias=True,
            has_conv=False,
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert outputs.shape == (self.batch_size, 21, 32, 32)

    def test_qat_model(self):
        qat_inputs = copy.deepcopy(self.inputs)
        qat_inputs = QTensor(
            qat_inputs, torch.tensor([1.0]), horizon.dtype.qint8
        )
        qat_outputs = self.qat_model(qat_inputs)
        assert qat_outputs.shape == (self.batch_size, 21, 32, 32)

    def test_quantize_model(self):
        quantized_inputs = copy.deepcopy(self.quantized_inputs)
        quantized_inputs = QTensor(
            quantized_inputs, torch.tensor([1.0]), horizon.dtype.qint8
        )
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert quantized_outputs.shape == (self.batch_size, 21, 32, 32)

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        assert torch.allclose(outputs, fuse_outputs, atol=1e-5)


class TestH3DManoMultiFcHead(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 16
        self.inputs = [
            torch.rand(self.batch_size, 256, 1, 1),
            torch.rand(self.batch_size, 256, 1, 1),
            torch.rand(self.batch_size, 21, 32, 32),
            torch.rand(self.batch_size, 256, 32, 32),
        ]
        self.quantized_inputs = [
            torch.rand(self.batch_size, 256, 1, 1),
            torch.rand(self.batch_size, 256, 1, 1),
            torch.rand(self.batch_size, 21, 32, 32),
            torch.rand(self.batch_size, 256, 32, 32),
        ]

    @property
    def is_train(self):
        return True

    def setup(self):
        super().setup()
        self.get_inputs()
        self.model = H3DManoMultiFcHead(
            encoding_channel=256,
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
            scale_neurons=(256, 128, 1),
            pose_neurons=(256, 128, 90),
            shape_neurons=(256, 10),
            camera_base_neurons=(256,),
            rot_neurons=(256, 6),
            trans_neurons=(256, 3),
            text_fc_neurons=(256, 256, 128),
            text_reg_neurons=(1024, 778, 778),
            head_heatmap=None,
            head_heatmap_latent=True,
            enable_render_head=False,
            enable_handscale_head=False,
            num_joints=21,
            deploy=False,
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert len(outputs) == 5
        if self.is_train:
            assert "shape_mano" in outputs
            assert "pose_mano" in outputs
            assert "trans" in outputs
            assert "rot" in outputs
            assert "heatmap_latents" in outputs

            assert outputs["shape_mano"].shape == (self.batch_size, 10)
            assert outputs["pose_mano"].shape == (self.batch_size, 90)
            assert outputs["trans"].shape == (self.batch_size, 3)
            assert outputs["rot"].shape == (self.batch_size, 6)
            assert outputs["heatmap_latents"].shape == (
                self.batch_size,
                21,
                32,
                32,
            )

    def test_qat_model(self):
        qat_inputs = copy.deepcopy(self.inputs)
        qat_inputs = [
            QTensor(qat_input, torch.tensor([1.0]), horizon.dtype.qint8)
            for qat_input in qat_inputs
        ]
        qat_outputs = self.qat_model(qat_inputs)
        assert len(qat_outputs) == 5
        if self.is_train:
            assert "shape_mano" in qat_outputs
            assert "pose_mano" in qat_outputs
            assert "trans" in qat_outputs
            assert "rot" in qat_outputs
            assert "heatmap_latents" in qat_outputs

            assert qat_outputs["shape_mano"].shape == (self.batch_size, 10)
            assert qat_outputs["pose_mano"].shape == (self.batch_size, 90)
            assert qat_outputs["trans"].shape == (self.batch_size, 3)
            assert qat_outputs["rot"].shape == (self.batch_size, 6)
            assert qat_outputs["heatmap_latents"].shape == (
                self.batch_size,
                21,
                32,
                32,
            )

    def test_quantize_model(self):
        quantized_inputs = copy.deepcopy(self.quantized_inputs)
        quantized_inputs = [
            QTensor(quantized_input, torch.tensor([1.0]), horizon.dtype.qint8)
            for quantized_input in quantized_inputs
        ]
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert len(quantized_outputs) == 5
        if self.is_train:
            assert "shape_mano" in quantized_outputs
            assert "pose_mano" in quantized_outputs
            assert "trans" in quantized_outputs
            assert "rot" in quantized_outputs
            assert "heatmap_latents" in quantized_outputs

            assert quantized_outputs["shape_mano"].shape == (
                self.batch_size,
                10,
            )
            assert quantized_outputs["pose_mano"].shape == (
                self.batch_size,
                90,
            )
            assert quantized_outputs["trans"].shape == (self.batch_size, 3)
            assert quantized_outputs["rot"].shape == (self.batch_size, 6)
            assert quantized_outputs["heatmap_latents"].shape == (
                self.batch_size,
                21,
                32,
                32,
            )

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for key, value in outputs.items():
            assert torch.allclose(value, fuse_outputs[key], atol=1e-5)
