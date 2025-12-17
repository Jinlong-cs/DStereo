import pytest
import torch
from torch import nn

from hat.models.task_modules.stereonet.neck import StereoNetNeck
from tests.unit_tests.models.necks.neck_template import NeckTemplate

use_bn = True
bias = False
bn_kwargs = {}
out_channels = [32, 32, 64, 128, 128, 16]


class TestStereoNetNeck(NeckTemplate):
    def setup(self):
        super(TestStereoNetNeck, self).setup()
        self.input_h = 256
        self.input_w = 512
        self.inputs = torch.randn(1, 6, self.input_h, self.input_w)
        self.model = StereoNetNeck(
            out_channels=out_channels,
            use_bn=use_bn,
            bias=bias,
            bn_kwargs=bn_kwargs,
            act_type=nn.ReLU(),
        )
        self.build_model()

        self.out_ch1 = out_channels[-2] + out_channels[-3]
        self.out_ch2 = out_channels[-1]
        self.h_tmp = self.input_h // 16
        self.w_tmp = self.input_w // 16

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert len(outputs) == 5
        assert outputs[0].shape == (1, self.out_ch1, self.h_tmp, self.w_tmp)
        assert outputs[0].shape == (1, self.out_ch1, self.h_tmp, self.w_tmp)
        assert outputs[2].shape == (1, self.out_ch2, self.h_tmp, self.w_tmp)
        assert outputs[3].shape == (1, self.out_ch2, self.h_tmp, self.w_tmp)
        assert outputs[4].shape == (1, 3, self.input_h, self.input_w)

    def test_fuse_model(self):
        pass

    def test_qat_model(self):
        qat_outputs = self.qat_model(self.inputs)

        assert qat_outputs[0].shape == (
            1,
            self.out_ch1,
            self.h_tmp,
            self.w_tmp,
        )
        assert qat_outputs[0].shape == (
            1,
            self.out_ch1,
            self.h_tmp,
            self.w_tmp,
        )
        assert qat_outputs[2].shape == (
            1,
            self.out_ch2,
            self.h_tmp,
            self.w_tmp,
        )
        assert qat_outputs[3].shape == (
            1,
            self.out_ch2,
            self.h_tmp,
            self.w_tmp,
        )
        assert qat_outputs[4].shape == (1, 3, self.input_h, self.input_w)

    def test_quantize_model(self):

        quantized_outputs = self.quantized_model(self.inputs)

        assert quantized_outputs[0].shape == (
            1,
            self.out_ch1,
            self.h_tmp,
            self.w_tmp,
        )
        assert quantized_outputs[0].shape == (
            1,
            self.out_ch1,
            self.h_tmp,
            self.w_tmp,
        )
        assert quantized_outputs[2].shape == (
            1,
            self.out_ch2,
            self.h_tmp,
            self.w_tmp,
        )
        assert quantized_outputs[3].shape == (
            1,
            self.out_ch2,
            self.h_tmp,
            self.w_tmp,
        )
        assert quantized_outputs[4].shape == (1, 3, self.input_h, self.input_w)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
