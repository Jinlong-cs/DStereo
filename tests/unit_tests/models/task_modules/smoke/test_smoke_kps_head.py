import copy
from collections import OrderedDict

import horizon_plugin_pytorch as horizon
import torch
from horizon_plugin_pytorch.qtensor import QTensor

from hat.models.losses.landmark import LdmkLoss
from hat.models.task_modules.smoke import (
    SmokeKpsClsHead,
    SmokeKpsHeatmapHead,
    SmokeKpsVectorHead,
)
from tests.unit_tests.models.task_modules.head_template import HeadTemplate

NUM_LDMK = 4


class TestSmokeKpsClsHead(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 16
        self.inputs = {
            "feat": torch.rand(self.batch_size, 256, 4, 4),
            "loss_weight": 1.0,
            "gt_classes": torch.randint(0, 128, (self.batch_size, 2)),
            "gt_cls_weight": torch.ones((self.batch_size, 2)),
            "gt_visable": torch.randint(0, 128, (self.batch_size, 1)),
            "gt_vis_weight": torch.ones((self.batch_size, 1)),
        }
        self.quantized_inputs = {
            "feat": torch.ones(self.batch_size, 256, 4, 4, dtype=torch.int8),
            "loss_weight": 1.0,
            "gt_classes": torch.randint(0, 128, (self.batch_size, 2)),
            "gt_cls_weight": torch.ones((self.batch_size, 2)),
            "gt_visable": torch.randint(0, 128, (self.batch_size, 1)),
            "gt_vis_weight": torch.ones((self.batch_size, 1)),
        }

    @property
    def is_train(self):
        return True

    def setup(self):
        super().setup()
        self.get_inputs()
        self.model = SmokeKpsClsHead(
            in_channels=256,
            middle_dim=32,
            output_dim=3,
            is_train=self.is_train,
            loss_func=LdmkLoss("l2"),
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, OrderedDict)
        assert len(outputs) == 2
        if self.is_train:
            assert "cls_loss" in outputs
        else:
            assert "pr_visable" in outputs
            assert outputs["pr_visable"].shape == (
                self.batch_size,
                1,
            )

    def test_qat_model(self):
        qat_inputs = copy.deepcopy(self.inputs)
        qat_inputs["feat"] = QTensor(
            qat_inputs["feat"], torch.tensor([1.0]), horizon.dtype.qint8
        )
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs, OrderedDict)
        assert len(qat_outputs) == 2
        if self.is_train:
            assert "cls_loss" in qat_outputs
        else:
            assert "pr_visable" in qat_outputs
            assert qat_outputs["pr_visable"].shape == (
                self.batch_size,
                1,
            )

    def test_quantize_model(self):
        quantized_inputs = copy.deepcopy(self.quantized_inputs)
        quantized_inputs["feat"] = QTensor(
            quantized_inputs["feat"], torch.tensor([1.0]), horizon.dtype.qint8
        )
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs, OrderedDict)
        assert len(quantized_outputs) == 2
        if self.is_train:
            assert "cls_loss" in quantized_outputs
        else:
            assert "pr_visable" in quantized_outputs
            assert quantized_outputs["pr_visable"].shape == (
                self.batch_size,
                1,
            )

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for key, value in outputs.items():
            assert torch.allclose(value, fuse_outputs[key], atol=1e-5)


class TestSmokeKpsClsHeadValMode(TestSmokeKpsClsHead):
    @property
    def is_train(self):
        return False


class TestSmokeKpsHeatmapHead(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 2
        self.height = 32
        self.width = 32
        heatmap = torch.randint(
            0, 128, (self.batch_size, NUM_LDMK + 1, self.height, self.width)
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

    @property
    def is_train(self):
        return True

    def setup(self):
        super().setup()
        self.get_inputs()
        self.model = SmokeKpsHeatmapHead(
            in_channels=128,
            num_ldmk=NUM_LDMK,
            is_train=self.is_train,
            loss_func=LdmkLoss("l2"),
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, OrderedDict)
        assert len(outputs) == 1
        if self.is_train:
            assert "heatmap_loss" in outputs
        else:
            assert "pr_heatmap" in outputs
            assert outputs["pr_heatmap"].shape == (
                self.batch_size,
                5,
                self.height,
                self.width,
            )

    def test_qat_model(self):
        qat_inputs = copy.deepcopy(self.inputs)
        qat_inputs["feat"] = QTensor(
            qat_inputs["feat"], torch.tensor([1.0]), horizon.dtype.qint8
        )
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs, OrderedDict)
        assert len(qat_outputs) == 1
        if self.is_train:
            assert "heatmap_loss" in qat_outputs
        else:
            assert "pr_heatmap" in qat_outputs
            assert qat_outputs["pr_heatmap"].shape == (
                self.batch_size,
                5,
                self.height,
                self.width,
            )

    def test_quantize_model(self):
        quantized_inputs = copy.deepcopy(self.quantized_inputs)
        quantized_inputs["feat"] = QTensor(
            quantized_inputs["feat"], torch.tensor([1.0]), horizon.dtype.qint8
        )
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs, OrderedDict)
        assert len(quantized_outputs) == 1
        if self.is_train:
            assert "heatmap_loss" in quantized_outputs
        else:
            assert "pr_heatmap" in quantized_outputs
            assert quantized_outputs["pr_heatmap"].shape == (
                self.batch_size,
                5,
                self.height,
                self.width,
            )

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for key, value in outputs.items():
            assert torch.allclose(value, fuse_outputs[key], atol=1e-5)


class TestSmokeKpsHeatmapHeadValMode(TestSmokeKpsHeatmapHead):
    @property
    def is_train(self):
        return False


class TestSmokeKpsVectorHead(HeadTemplate):
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

    @property
    def is_train(self):
        return True

    def setup(self):
        super().setup()
        self.get_inputs()
        self.model = SmokeKpsVectorHead(
            in_channels=128,
            num_ldmk=NUM_LDMK,
            band_width=1,
            vector_size=(self.height, self.width),
            is_train=self.is_train,
            loss_func=LdmkLoss("l2"),
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, OrderedDict)
        if self.is_train:
            assert len(outputs) == 1
            assert "vector_loss" in outputs
        else:
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
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs, OrderedDict)
        if self.is_train:
            assert len(qat_outputs) == 1
            assert "vector_loss" in qat_outputs
        else:
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
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs, OrderedDict)
        if self.is_train:
            assert len(quantized_outputs) == 1
            assert "vector_loss" in quantized_outputs
        else:
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


class TestSmokeKpsVectorHeadValMode(TestSmokeKpsVectorHead):
    @property
    def is_train(self):
        return False
