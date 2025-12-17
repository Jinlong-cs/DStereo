import copy

import horizon_plugin_pytorch as horizon
import pytest
import torch

from hat.models.task_modules.motion_forecasting.encoders import Vectornet
from hat.utils import qconfig_manager
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate


class TestVectorNet(HeadTemplate):
    def setup(self):
        super(TestVectorNet, self).setup()

        self.inputs = [
            torch.randn((1, 9, 19, 32)),
            torch.randn((1, 11, 9, 64)),
            torch.randn((1, 1, 96)),
        ]
        self.quantized_inputs = [
            torch.randn((1, 9, 19, 32)).to(dtype=torch.int16),
            torch.randn((1, 11, 9, 64)).to(dtype=torch.int16),
            torch.randn((1, 1, 96)).to(dtype=torch.int8),
        ]
        self.model = Vectornet(
            depth=3,
            traj_in_channels=9,
            traj_num_vec=9,
            lane_in_channels=11,
            lane_num_vec=19,
            hidden_size=128,
        )

        self.build_model()

    def build_model(self):
        # float model
        self.float_model = copy.deepcopy(self.model)
        self.float_model.eval()

        # qat model
        horizon.march.set_march(horizon.march.March.BAYES)
        self.fuse_model = copy.deepcopy(self.float_model)
        qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
        self.fuse_model.qconfig = qconfig_manager.get_default_qat_qconfig()
        if hasattr(self.fuse_model, "set_qconfig"):
            self.fuse_model.set_qconfig()
        qconfig_manager.set_qconfig_mode(
            qconfig_manager.QconfigMode.COMPATIBLE
        )
        self.qat_model = horizon.quantization.prepare_qat_fx(self.fuse_model)

        # quantize model
        self.quantized_model = horizon.quantization.convert(
            self.qat_model.eval(), inplace=False
        )

    def test_float_model(self):
        self.float_model(*self.inputs)

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs, dtype="int16")
        self.qat_model(*qat_inputs)

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        self.quantized_model(*quantized_inputs)

    def test_fuse_model(self):
        pass


if __name__ == "__main__":
    pytest.main(["-s", __file__])
