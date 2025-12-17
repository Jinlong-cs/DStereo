import copy
from collections import OrderedDict

import horizon_plugin_pytorch as horizon
import torch
from horizon_plugin_pytorch.qtensor import QTensor

from hat.models.task_modules.landmark.human_pose_head import HumanPoseHead
from tests.unit_tests.models.task_modules.head_template import HeadTemplate

NUM_KPS = 15


class TestHumanPoseHead(HeadTemplate):
    def get_inputs(self):
        self.batch_size = 2
        self.height = 32
        self.width = 32
        self.target_h_out = 64
        self.target_w_out = 64

        self.inputs = torch.rand(self.batch_size, 128, self.height, self.width)

        self.quantized_inputs = torch.ones(
            self.batch_size, 128, self.height, self.width, dtype=torch.int8
        )

    def setup(self):
        super().setup()
        self.get_inputs()
        self.model = HumanPoseHead(
            num_ldmk=NUM_KPS,
            num_conv=3,
            group_base=8,
            head_conv_method="BasicVarGBlock",
            bn_kwargs={"eps": 2e-5, "momentum": 0.9},
            num_filter=128,
            in_num_filter=128,
            target_shape=(self.target_h_out, self.target_w_out),
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, OrderedDict)
        assert len(outputs) == 1
        assert "ldmk_pred" in outputs
        assert outputs["ldmk_pred"].shape == (
            self.batch_size,
            NUM_KPS * 3,
            self.target_h_out,
            self.target_w_out,
        )

    def test_qat_model(self):
        qat_inputs = copy.deepcopy(self.inputs)
        qat_inputs = QTensor(
            qat_inputs, torch.tensor([1.0]), horizon.dtype.qint8
        )
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs, OrderedDict)
        assert len(qat_outputs) == 1
        assert "ldmk_pred" in qat_outputs
        assert qat_outputs["ldmk_pred"].shape == (
            self.batch_size,
            NUM_KPS * 3,
            self.target_h_out,
            self.target_w_out,
        )

    def test_quantize_model(self):
        quantized_inputs = copy.deepcopy(self.quantized_inputs)
        quantized_inputs = QTensor(
            quantized_inputs, torch.tensor([1.0]), horizon.dtype.qint8
        )
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs, OrderedDict)
        assert len(quantized_outputs) == 1
        assert "ldmk_pred" in quantized_outputs
        assert quantized_outputs["ldmk_pred"].shape == (
            self.batch_size,
            NUM_KPS * 3,
            self.target_h_out,
            self.target_w_out,
        )

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for key, value in outputs.items():
            assert torch.allclose(value, fuse_outputs[key], atol=1e-5)
