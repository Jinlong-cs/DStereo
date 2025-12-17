import pytest
import torch

from hat.models.task_modules.stereonet.head import StereoNetHead
from hat.models.task_modules.stereonet.headplus import StereoNetHeadPlus
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate

maxdisp = 192
bn_kwargs = {}
refine_levels = 4

out_channels = [32, 32, 64, 128, 128, 16]


@pytest.mark.serial_task
class TestStereoNetHead(HeadTemplate):
    def setup(self):
        super(TestStereoNetHead, self).setup()
        self.input_h = 256
        self.input_w = 512
        self.out_ch1 = out_channels[-2] + out_channels[-3]
        self.out_ch2 = out_channels[-1]
        self.h_tmp = self.input_h // 16
        self.w_tmp = self.input_w // 16

        input1 = torch.rand(1, self.out_ch1, self.h_tmp, self.w_tmp)
        input2 = torch.rand(1, self.out_ch1, self.h_tmp, self.w_tmp)
        input3 = torch.rand(1, self.out_ch2, self.h_tmp, self.w_tmp)
        input4 = torch.rand(1, self.out_ch2, self.h_tmp, self.w_tmp)
        input5 = torch.rand(1, 3, self.input_h, self.input_w)

        input1_int = torch.ones(
            1, self.out_ch1, self.h_tmp, self.w_tmp, dtype=torch.int8
        )

        input2_int = torch.ones(
            1, self.out_ch1, self.h_tmp, self.w_tmp, dtype=torch.int8
        )
        input3_int = torch.ones(
            1, self.out_ch2, self.h_tmp, self.w_tmp, dtype=torch.int8
        )
        input4_int = torch.ones(
            1, self.out_ch2, self.h_tmp, self.w_tmp, dtype=torch.int8
        )
        input5_int = torch.ones(
            1, 3, self.input_h, self.input_w, dtype=torch.int8
        )

        self.inputs = [input1, input2, input3, input4, input5]
        self.quantized_inputs = [
            input1_int,
            input2_int,
            input3_int,
            input4_int,
            input5_int,
        ]

        self.model = StereoNetHead(
            maxdisp=maxdisp,
            bn_kwargs=bn_kwargs,
            refine_levels=refine_levels,
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs[-1], torch.Tensor)
        assert outputs[-1].ndim == 4
        assert outputs[-1].shape[2] == self.input_h
        assert outputs[-1].shape[3] == self.input_w

    def test_fuse_model(self):
        pass

    def test_qat_model(self):

        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs[-1], torch.Tensor)
        assert qat_outputs[-1].ndim == 4
        assert qat_outputs[-1].shape[2] == self.input_h
        assert qat_outputs[-1].shape[3] == self.input_w

    def test_quantize_model(self):

        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs[-1], torch.Tensor)
        assert quantized_outputs[-1].ndim == 4
        assert quantized_outputs[-1].shape[2] == self.input_h
        assert quantized_outputs[-1].shape[3] == self.input_w


@pytest.mark.serial_task
class TestStereoNetHeadPlus(HeadTemplate):
    def setup(self):
        super(TestStereoNetHeadPlus, self).setup()
        self.input_h = 256
        self.input_w = 512
        self.h_tmp = self.input_h // 8
        self.w_tmp = self.input_w // 8
        self.inchannels = [32, 32, 16, 16, 16]

        input1 = torch.rand(
            2, self.inchannels[0], self.input_h // 2, self.input_w // 2
        )
        input2 = torch.rand(
            2, self.inchannels[1], self.input_h // 4, self.input_w // 4
        )
        input3 = torch.rand(
            2, self.inchannels[2], self.input_h // 8, self.input_w // 8
        )
        input4 = torch.rand(
            2, self.inchannels[3], self.input_h // 16, self.input_w // 16
        )
        input5 = torch.rand(
            2, self.inchannels[4], self.input_h // 32, self.input_w // 32
        )

        input1_int = torch.ones(
            2,
            self.inchannels[0],
            self.input_h // 2,
            self.input_w // 2,
            dtype=torch.int8,
        )
        input2_int = torch.ones(
            2,
            self.inchannels[1],
            self.input_h // 4,
            self.input_w // 4,
            dtype=torch.int8,
        )
        input3_int = torch.ones(
            2,
            self.inchannels[2],
            self.input_h // 8,
            self.input_w // 8,
            dtype=torch.int8,
        )
        input4_int = torch.ones(
            2,
            self.inchannels[3],
            self.input_h // 16,
            self.input_w // 16,
            dtype=torch.int8,
        )
        input5_int = torch.ones(
            2,
            self.inchannels[4],
            self.input_h // 32,
            self.input_w // 32,
            dtype=torch.int8,
        )

        self.inputs = [input1, input2, input3, input4, input5]
        self.quantized_inputs = [
            input1_int,
            input2_int,
            input3_int,
            input4_int,
            input5_int,
        ]

        self.model = StereoNetHeadPlus(
            maxdisp=maxdisp,
            bn_kwargs=bn_kwargs,
            refine_levels=3,
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs[0], torch.Tensor)
        assert outputs[0].ndim == 4
        assert outputs[0].shape[2] == self.h_tmp
        assert outputs[0].shape[3] == self.w_tmp

        assert isinstance(outputs[1], torch.Tensor)
        assert outputs[1].ndim == 4
        assert outputs[1].shape[2] == self.h_tmp
        assert outputs[1].shape[3] == self.w_tmp

        assert isinstance(outputs[2], torch.Tensor)
        assert outputs[2].ndim == 4
        assert outputs[2].shape[2] == self.input_h
        assert outputs[2].shape[3] == self.input_w

    def test_fuse_model(self):
        pass

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs[0], torch.Tensor)
        assert qat_outputs[0].ndim == 4
        assert qat_outputs[0].shape[2] == self.h_tmp
        assert qat_outputs[0].shape[3] == self.w_tmp

        assert isinstance(qat_outputs[1], torch.Tensor)
        assert qat_outputs[1].ndim == 4
        assert qat_outputs[1].shape[2] == self.h_tmp
        assert qat_outputs[1].shape[3] == self.w_tmp

        assert isinstance(qat_outputs[2], torch.Tensor)
        assert qat_outputs[2].ndim == 4
        assert qat_outputs[2].shape[2] == self.input_h
        assert qat_outputs[2].shape[3] == self.input_w

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs[0], torch.Tensor)
        assert quantized_outputs[0].ndim == 4
        assert quantized_outputs[0].shape[2] == self.h_tmp
        assert quantized_outputs[0].shape[3] == self.w_tmp

        assert isinstance(quantized_outputs[1], torch.Tensor)
        assert quantized_outputs[1].ndim == 4
        assert quantized_outputs[1].shape[2] == self.h_tmp
        assert quantized_outputs[1].shape[3] == self.w_tmp

        assert isinstance(quantized_outputs[2], torch.Tensor)
        assert quantized_outputs[2].ndim == 4
        assert quantized_outputs[2].shape[2] == self.input_h
        assert quantized_outputs[2].shape[3] == self.input_w


if __name__ == "__main__":
    pytest.main(["-s", __file__])
