import copy
from collections import OrderedDict

import horizon_plugin_pytorch as horizon
import torch
from horizon_plugin_pytorch.qtensor import QTensor

from hat.models.losses.landmark import LdmkLoss
from hat.models.task_modules.landmark.ldmk_head import (
    BandConvModule,
    BandPoolModule,
    LdmkCoordsHead,
    LdmkDecoder,
    LdmkHeatmapHead,
    LdmkVectorHead,
)
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate

NUM_LDMK = 68


class TestLdmkCoordsHead(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 2
        self.inputs = {
            "feat": torch.rand(self.batch_size, 256, 4, 4),
            "loss_weight": 1.0,
            "gt_ldmk": torch.randint(0, 128, (self.batch_size, NUM_LDMK, 2)),
            "gt_ldmk_weight": torch.ones((self.batch_size, NUM_LDMK, 2)),
        }
        self.quantized_inputs = {
            "feat": torch.ones(self.batch_size, 256, 4, 4, dtype=torch.int8),
            "loss_weight": 1.0,
            "gt_ldmk": torch.randint(0, 128, (self.batch_size, NUM_LDMK, 2)),
            "gt_ldmk_weight": torch.ones((self.batch_size, NUM_LDMK, 2)),
        }

    def setup(self):
        super().setup()
        self.get_inputs()
        self.model = LdmkCoordsHead(
            in_channels=256,
            kernel_size=4,
            num_ldmk=NUM_LDMK,
            loss_func=LdmkLoss("l2"),
        )
        self.build_model()

    def test_float_model(self):
        self.float_model.training = True
        outputs = self.float_model(self.inputs)
        assert "ldmk_loss" in outputs
        assert isinstance(outputs, OrderedDict)
        assert len(outputs) == 1

        self.float_model.training = False
        outputs = self.float_model(self.inputs)
        assert "pr_ldmk" in outputs
        assert isinstance(outputs, OrderedDict)
        assert len(outputs) == 1
        assert outputs["pr_ldmk"].shape == (
            self.batch_size,
            NUM_LDMK * 2,
            1,
            1,
        )

    def test_qat_model(self):
        qat_inputs = copy.deepcopy(self.inputs)
        qat_inputs["feat"] = QTensor(
            qat_inputs["feat"], torch.tensor([1.0]), horizon.dtype.qint8
        )
        self.qat_model.training = True
        qat_outputs = self.qat_model(qat_inputs)
        assert "ldmk_loss" in qat_outputs
        assert isinstance(qat_outputs, OrderedDict)
        assert len(qat_outputs) == 1

        self.qat_model.training = False
        qat_outputs = self.qat_model(qat_inputs)
        assert "pr_ldmk" in qat_outputs
        assert isinstance(qat_outputs, OrderedDict)
        assert len(qat_outputs) == 1
        assert qat_outputs["pr_ldmk"].shape == (
            self.batch_size,
            NUM_LDMK * 2,
            1,
            1,
        )

    def test_quantize_model(self):
        quantized_inputs = copy.deepcopy(self.quantized_inputs)
        quantized_inputs["feat"] = QTensor(
            quantized_inputs["feat"], torch.tensor([1.0]), horizon.dtype.qint8
        )
        self.quantized_model.training = True
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert "ldmk_loss" in quantized_outputs
        assert isinstance(quantized_outputs, OrderedDict)
        assert len(quantized_outputs) == 1

        self.quantized_model.training = False
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert "pr_ldmk" in quantized_outputs
        assert isinstance(quantized_outputs, OrderedDict)
        assert len(quantized_outputs) == 1
        assert quantized_outputs["pr_ldmk"].shape == (
            self.batch_size,
            NUM_LDMK * 2,
            1,
            1,
        )

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for key, value in outputs.items():
            assert torch.allclose(value, fuse_outputs[key], atol=1e-5)


class TestLdmkHeatmapHead(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 2
        self.height = 32
        self.width = 32
        heatmap = torch.randint(
            0, 128, (self.batch_size, NUM_LDMK, self.height, self.width)
        )
        heatmap_weight = torch.zeros_like(heatmap)
        self.inputs = {
            "feat": torch.rand(self.batch_size, 128, self.height, self.width),
            "gt_heatmap_weight": heatmap_weight,
            "gt_heatmap": heatmap,
            "loss_weight": 1.0,
        }
        self.quantized_inputs = {
            "feat": torch.ones(
                self.batch_size, 128, self.height, self.width, dtype=torch.int8
            ),
            "gt_heatmap_weight": heatmap_weight,
            "gt_heatmap": heatmap,
            "loss_weight": 1.0,
        }

    def setup(self):
        super().setup()
        self.get_inputs()
        self.model = LdmkHeatmapHead(
            in_channels=128,
            num_ldmk=NUM_LDMK,
            loss_func=LdmkLoss("l2"),
        )
        self.build_model()

    def test_float_model(self):
        self.float_model.training = True
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, OrderedDict)
        assert len(outputs) == 1
        assert "heatmap_loss" in outputs

        self.float_model.training = False
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, OrderedDict)
        assert len(outputs) == 1
        assert "pr_heatmap" in outputs
        assert outputs["pr_heatmap"].shape == (
            self.batch_size,
            NUM_LDMK,
            self.height,
            self.width,
        )

    def test_qat_model(self):
        qat_inputs = copy.deepcopy(self.inputs)
        qat_inputs["feat"] = QTensor(
            qat_inputs["feat"], torch.tensor([1.0]), horizon.dtype.qint8
        )
        self.qat_model.training = True
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs, OrderedDict)
        assert len(qat_outputs) == 1
        assert "heatmap_loss" in qat_outputs

        self.qat_model.training = False
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs, OrderedDict)
        assert len(qat_outputs) == 1
        assert "pr_heatmap" in qat_outputs
        assert qat_outputs["pr_heatmap"].shape == (
            self.batch_size,
            NUM_LDMK,
            self.height,
            self.width,
        )

    def test_quantize_model(self):
        quantized_inputs = copy.deepcopy(self.quantized_inputs)
        quantized_inputs["feat"] = QTensor(
            quantized_inputs["feat"], torch.tensor([1.0]), horizon.dtype.qint8
        )
        self.quantized_model.training = True
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs, OrderedDict)
        assert len(quantized_outputs) == 1
        assert "heatmap_loss" in quantized_outputs

        self.quantized_model.training = False
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs, OrderedDict)
        assert len(quantized_outputs) == 1
        assert "pr_heatmap" in quantized_outputs
        assert quantized_outputs["pr_heatmap"].shape == (
            self.batch_size,
            NUM_LDMK,
            self.height,
            self.width,
        )

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for key, value in outputs.items():
            assert torch.allclose(value, fuse_outputs[key], atol=1e-5)


class TestLdmkVectorHead(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 2
        self.height = 32
        self.width = 32
        vector_x = torch.ones((self.batch_size, NUM_LDMK, self.width))
        vector_y = torch.ones((self.batch_size, NUM_LDMK, self.height))
        vector_weight_x = torch.ones_like(vector_x)
        vector_weight_y = torch.ones_like(vector_y)
        self.inputs = {
            "feat": torch.rand(self.batch_size, 128, self.height, self.width),
            "gt_vector_weight_x": vector_weight_x,
            "gt_vector_weight_y": vector_weight_y,
            "gt_vector_x": vector_x,
            "gt_vector_y": vector_y,
            "loss_weight": 1.0,
        }
        self.quantized_inputs = {
            "feat": torch.ones(
                self.batch_size, 128, self.height, self.width, dtype=torch.int8
            ),
            "gt_vector_weight_x": vector_weight_x,
            "gt_vector_weight_y": vector_weight_y,
            "gt_vector_x": vector_x,
            "gt_vector_y": vector_y,
            "loss_weight": 1.0,
        }

    def setup(self):
        super().setup()
        self.get_inputs()
        self.model = LdmkVectorHead(
            in_channels=128,
            num_ldmk=NUM_LDMK,
            band_width=1,
            vector_size=(self.height, self.width),
            loss_func=LdmkLoss("l2"),
        )
        self.build_model()

    def test_float_model(self):
        self.float_model.training = True
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, OrderedDict)
        assert len(outputs) == 1
        assert "vector_loss" in outputs

        self.float_model.training = False
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, OrderedDict)
        assert len(outputs) == 2
        assert outputs["pr_vector_x"].shape == (
            self.batch_size,
            NUM_LDMK,
            1,
            self.width,
        )
        assert outputs["pr_vector_y"].shape == (
            self.batch_size,
            NUM_LDMK,
            self.height,
            1,
        )

    def test_qat_model(self):
        qat_inputs = copy.deepcopy(self.inputs)
        qat_inputs["feat"] = QTensor(
            qat_inputs["feat"], torch.tensor([1.0]), horizon.dtype.qint8
        )

        self.qat_model.training = True
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs, OrderedDict)
        assert len(qat_outputs) == 1
        assert "vector_loss" in qat_outputs

        self.qat_model.training = False
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs, OrderedDict)
        assert len(qat_outputs) == 2
        assert qat_outputs["pr_vector_x"].shape == (
            self.batch_size,
            NUM_LDMK,
            1,
            self.width,
        )
        assert qat_outputs["pr_vector_y"].shape == (
            self.batch_size,
            NUM_LDMK,
            self.height,
            1,
        )

    def test_quantize_model(self):
        quantized_inputs = copy.deepcopy(self.quantized_inputs)
        quantized_inputs["feat"] = QTensor(
            quantized_inputs["feat"], torch.tensor([1.0]), horizon.dtype.qint8
        )

        self.quantized_model.training = True
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs, OrderedDict)
        assert len(quantized_outputs) == 1
        assert "vector_loss" in quantized_outputs

        self.quantized_model.training = False
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs, OrderedDict)
        assert len(quantized_outputs) == 2
        assert quantized_outputs["pr_vector_x"].shape == (
            self.batch_size,
            NUM_LDMK,
            1,
            self.width,
        )
        assert quantized_outputs["pr_vector_y"].shape == (
            self.batch_size,
            NUM_LDMK,
            self.height,
            1,
        )

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for key, value in outputs.items():
            assert torch.allclose(value, fuse_outputs[key], atol=1e-5)


class TestLdmkDecoder(HeadTemplate):
    def setup(self):
        super().setup()
        self.inputs = [
            torch.randn((2, 32, 32, 32)),
            torch.randn((2, 64, 16, 16)),
            torch.randn((2, 128, 8, 8)),
            torch.randn((2, 256, 4, 4)),
        ]
        self.quantized_inputs = [
            torch.randint(-127, 127, (2, 32, 32, 32), dtype=torch.int8),
            torch.randint(-127, 127, (2, 64, 16, 16), dtype=torch.int8),
            torch.randint(-127, 127, (2, 128, 8, 8), dtype=torch.int8),
            torch.randint(-127, 127, (2, 256, 4, 4), dtype=torch.int8),
        ]
        self.model = LdmkDecoder(
            in_channels=256, out_channels=128, in_stride=32, out_stride=4
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)[0]
        assert outputs.shape == (2, 128, 32, 32)

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)[0]
        assert qat_outputs.shape == (2, 128, 32, 32)

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)[0]
        assert quantized_outputs.shape == (2, 128, 32, 32)


class TestBandConvModule(HeadTemplate):
    def setup(self):
        super().setup()
        self.inputs = torch.randn((2, NUM_LDMK, 32, 32))
        self.quantized_inputs = torch.randint(
            -127, 127, (2, NUM_LDMK, 32, 32), dtype=torch.int8
        )
        self.model = BandConvModule(
            vector_size=32,
            band_width=1,
            num_channels=NUM_LDMK,
            horizontal=True,
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert outputs.shape == (2, NUM_LDMK, 1, 32)

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert qat_outputs.shape == (2, NUM_LDMK, 1, 32)

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert quantized_outputs.shape == (2, NUM_LDMK, 1, 32)


class TestBandPoolModule(HeadTemplate):
    def setup(self):
        super().setup()
        self.inputs = torch.randn((2, NUM_LDMK, 32, 32))
        self.quantized_inputs = torch.randint(
            -127, 127, (2, NUM_LDMK, 32, 32), dtype=torch.int8
        )
        self.model = BandPoolModule(
            vector_size=32, band_width=1, horizontal=True
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert outputs.shape == (2, NUM_LDMK, 1, 32)

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert qat_outputs.shape == (2, NUM_LDMK, 1, 32)

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert quantized_outputs.shape == (2, NUM_LDMK, 1, 32)
