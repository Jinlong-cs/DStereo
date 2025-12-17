import copy

import horizon_plugin_pytorch as horizon
import torch
from horizon_plugin_pytorch.qtensor import QTensor

from hat.models.task_modules.hand3d.encoder.fuse_high_low_encoder import (
    H3DFuseHighLowEncoder,
    H3DHeatMapResEncoder,
)
from tests.unit_tests.models.task_modules.head_template import HeadTemplate


class TestH3DHeatMapResEncoder(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 16
        self.inputs = torch.rand(self.batch_size, 64, 32, 32)
        self.quantized_inputs = torch.rand(self.batch_size, 64, 32, 32)

    @property
    def is_train(self):
        return True

    def setup(self):
        super().setup()
        self.get_inputs()
        self.model = H3DHeatMapResEncoder(
            in_channels=64,
            out_channels=256,
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert len(outputs) == 2
        if self.is_train:
            assert outputs[0].shape == (self.batch_size, 21, 32, 32)
            assert outputs[1].shape == (self.batch_size, 256, 32, 32)

    def test_qat_model(self):
        qat_inputs = copy.deepcopy(self.inputs)
        qat_inputs = QTensor(
            qat_inputs, torch.tensor([1.0]), horizon.dtype.qint8
        )
        qat_outputs = self.qat_model(qat_inputs)
        assert len(qat_outputs) == 2
        if self.is_train:
            assert qat_outputs[0].shape == (self.batch_size, 21, 32, 32)
            assert qat_outputs[1].shape == (self.batch_size, 256, 32, 32)

    def test_quantize_model(self):
        quantized_inputs = copy.deepcopy(self.quantized_inputs)
        quantized_inputs = QTensor(
            quantized_inputs,
            torch.tensor([1.0]),
            horizon.dtype.qint8,
        )
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert len(quantized_outputs) == 2
        if self.is_train:
            assert quantized_outputs[0].shape == (self.batch_size, 21, 32, 32)
            assert quantized_outputs[1].shape == (self.batch_size, 256, 32, 32)

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for i, value in enumerate(outputs):
            assert torch.allclose(
                value,
                fuse_outputs[i],
                atol=1e-5,
            )


class TestH3DFuseHighLowEncoder(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 16
        self.inputs = [
            torch.rand(self.batch_size, 64, 32, 32),
            torch.rand(self.batch_size, 96, 16, 16),
            torch.rand(self.batch_size, 128, 8, 8),
        ]
        self.quantized_inputs = [
            torch.rand(self.batch_size, 64, 32, 32),
            torch.rand(self.batch_size, 96, 16, 16),
            torch.rand(self.batch_size, 128, 8, 8),
        ]

    @property
    def is_train(self):
        return True

    def setup(self):
        super().setup()
        self.get_inputs()
        self.model = H3DFuseHighLowEncoder(
            in_channels=128,
            bifpn_channels=64,
            encoding_channels=256,
            bn_kwargs=dict(eps=1e-5, momentum=0.01),
            heatmap_encoder=H3DHeatMapResEncoder(
                in_channels=64,
                out_channels=256,
                bn_kwargs=dict(eps=1e-5, momentum=0.01),
            ),
            global_avg_size=4,
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert len(outputs) == 4
        if self.is_train:
            assert outputs[0].shape == (self.batch_size, 256, 1, 1)
            assert outputs[1].shape == (self.batch_size, 256, 1, 1)
            assert outputs[2].shape == (self.batch_size, 21, 32, 32)
            assert outputs[3].shape == (self.batch_size, 256, 32, 32)

    def test_qat_model(self):
        qat_inputs = copy.deepcopy(self.inputs)
        qat_inputs = [
            QTensor(
                qat_input,
                torch.tensor([1.0]),
                horizon.dtype.qint8,
            )
            for qat_input in qat_inputs
        ]
        qat_outputs = self.qat_model(qat_inputs)
        assert len(qat_outputs) == 4
        if self.is_train:
            assert qat_outputs[0].shape == (self.batch_size, 256, 1, 1)
            assert qat_outputs[1].shape == (self.batch_size, 256, 1, 1)
            assert qat_outputs[2].shape == (self.batch_size, 21, 32, 32)
            assert qat_outputs[3].shape == (self.batch_size, 256, 32, 32)

    def test_quantize_model(self):
        quantized_inputs = copy.deepcopy(self.quantized_inputs)
        quantized_inputs = [
            QTensor(
                quantized_input,
                torch.tensor([1.0]),
                horizon.dtype.qint8,
            )
            for quantized_input in quantized_inputs
        ]
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert len(quantized_outputs) == 4
        if self.is_train:
            assert quantized_outputs[0].shape == (self.batch_size, 256, 1, 1)
            assert quantized_outputs[1].shape == (self.batch_size, 256, 1, 1)
            assert quantized_outputs[2].shape == (self.batch_size, 21, 32, 32)
            assert quantized_outputs[3].shape == (self.batch_size, 256, 32, 32)

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for i, value in enumerate(outputs):
            assert torch.allclose(
                value,
                fuse_outputs[i],
                atol=1e-5,
            )
