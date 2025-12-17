import pytest
import torch
from torch import nn
from torch.nn import ReLU
from torch.nn.modules.batchnorm import _BatchNorm

from hat.models.task_modules.pwcnet.neck import PwcNetNeck
from tests.unit_tests.models.necks.neck_template import NeckTemplate

out_channels = [16, 32, 64, 96, 128, 196]
flow_pred_lvl = 2
pyr_lvls = 6
use_bn = True
bn_kwargs = {}
use_bias = True


class TestPwcNetNeck(NeckTemplate):
    def setup(self):
        super(TestPwcNetNeck, self).setup()
        self.input_h = 256
        self.input_w = 448
        self.inputs = torch.randn(1, 6, self.input_h, self.input_w)
        self.model = PwcNetNeck(
            out_channels=out_channels,
            use_bn=use_bn,
            bn_kwargs=bn_kwargs,
            bias=use_bias,
            pyr_lvls=pyr_lvls,
            flow_pred_lvl=flow_pred_lvl,
            act_type=nn.ReLU(),
        )
        self.build_model()

    def test_float_model(self):
        outputs1, outputs2 = self.float_model(self.inputs)
        stride = 2
        for i, (output1, output2) in enumerate(zip(outputs1, outputs2)):
            assert isinstance(output1, torch.Tensor)
            assert isinstance(output2, torch.Tensor)
            assert output1.shape == output2.shape
            assert len(output1.shape) == 4
            assert output1.shape[1] == out_channels[i]
            assert output1.shape[2] == self.input_h // stride
            assert output1.shape[3] == self.input_w // stride
            stride = stride * 2

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for output, fuse_output in zip(outputs, fuse_outputs):
            assert all(
                torch.allclose(output_tmp, fuse_output_tmp, atol=0.001)
                for output_tmp, fuse_output_tmp in zip(output, fuse_output)
            )
        print(self.qat_model)
        for module in self.qat_model.modules():
            assert not (
                isinstance(module, _BatchNorm) or isinstance(module, ReLU)
            )

    def test_qat_model(self):
        qat_outputs1, qat_outputs2 = self.qat_model(self.inputs)
        stride = 2
        for i, (qat_output1, qat_output2) in enumerate(
            zip(qat_outputs1, qat_outputs2)
        ):
            assert isinstance(qat_output1, torch.Tensor)
            assert isinstance(qat_output2, torch.Tensor)
            assert qat_output1.shape == qat_output2.shape
            assert len(qat_output1.shape) == 4
            assert qat_output1.shape[1] == out_channels[i]
            assert qat_output1.shape[2] == self.input_h // stride
            assert qat_output1.shape[3] == self.input_w // stride
            stride = stride * 2

    def test_quantize_model(self):
        quantized_outputs1, quantized_outputs2 = self.quantized_model(
            self.inputs
        )
        stride = 2
        for i, (quantized_output1, quantized_output2) in enumerate(
            zip(quantized_outputs1, quantized_outputs2)
        ):
            assert isinstance(quantized_output1, torch.Tensor)
            assert isinstance(quantized_output2, torch.Tensor)
            assert quantized_output1.shape == quantized_output2.shape
            assert len(quantized_output1.shape) == 4
            assert quantized_output1.shape[1] == out_channels[i]
            assert quantized_output1.shape[2] == self.input_h // stride
            assert quantized_output1.shape[3] == self.input_w // stride
            stride = stride * 2


if __name__ == "__main__":
    pytest.main(["-s", __file__])
