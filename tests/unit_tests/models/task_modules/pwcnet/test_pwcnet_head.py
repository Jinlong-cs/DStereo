import pytest
import torch
from torch import nn

from hat.models.task_modules.pwcnet.head import PwcNetHead
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate

out_channels = [16, 32, 64, 96, 128, 196]
flow_pred_lvl = 2
pyr_lvls = 6
use_bn = True
bn_kwargs = {}
use_res = True
use_dense = True


class TestPwcNetHead(HeadTemplate):
    def setup(self):
        super(TestPwcNetHead, self).setup()
        self.input_size = (256, 448)
        x1_input = []
        x2_input = []
        x1_int_input = []
        x2_int_input = []
        for idx in range(pyr_lvls):
            h_tmp = int(self.input_size[0] / (2 ** (idx + 1)))
            w_tmp = int(self.input_size[1] / (2 ** (idx + 1)))
            x1 = torch.rand(1, out_channels[idx], h_tmp, w_tmp)
            x2 = torch.rand(1, out_channels[idx], h_tmp, w_tmp)
            x1_int = torch.ones(
                1, out_channels[idx], h_tmp, w_tmp, dtype=torch.int8
            )
            x2_int = torch.ones(
                1, out_channels[idx], h_tmp, w_tmp, dtype=torch.int8
            )
            x1_input.append(x1)
            x2_input.append(x2)
            x1_int_input.append(x1_int)
            x2_int_input.append(x2_int)
        self.inputs = [x1_input, x2_input]
        self.quantized_inputs = [x1_int_input, x2_int_input]
        self.model = PwcNetHead(
            in_channels=out_channels,
            bn_kwargs=bn_kwargs,
            use_bn=use_bn,
            md=4,
            use_res=use_res,
            use_dense=use_dense,
            pyr_lvls=pyr_lvls,
            flow_pred_lvl=flow_pred_lvl,
            act_type=nn.ReLU(),
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, torch.Tensor)
        assert outputs.ndim == 4
        assert outputs.shape[1] == 2
        assert outputs.shape[2] == self.input_size[0] / (2 ** flow_pred_lvl)
        assert outputs.shape[3] == self.input_size[1] / (2 ** flow_pred_lvl)

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs, torch.Tensor)
        assert qat_outputs.ndim == 4
        assert qat_outputs.shape[1] == 2
        assert qat_outputs.shape[2] == self.input_size[0] / (
            2 ** flow_pred_lvl
        )
        assert qat_outputs.shape[3] == self.input_size[1] / (
            2 ** flow_pred_lvl
        )

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs, torch.Tensor)
        assert quantized_outputs.ndim == 4
        assert quantized_outputs.shape[1] == 2
        assert quantized_outputs.shape[2] == self.input_size[0] / (
            2 ** flow_pred_lvl
        )
        assert quantized_outputs.shape[3] == self.input_size[1] / (
            2 ** flow_pred_lvl
        )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
