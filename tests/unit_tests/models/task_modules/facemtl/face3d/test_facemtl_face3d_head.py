import torch

from hat.models.task_modules.facemtl.face3d.facemtl_face3d_head import (
    FaceMtlFace3dHead,
)
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate


class TestFace3dHead(HeadTemplate):
    def setup(self):
        super(TestFace3dHead, self).setup()
        self.inputs = torch.rand(1, 128, 5, 5)
        self.quantized_inputs = torch.ones(1, 128, 5, 5, dtype=torch.int8)
        self.model = FaceMtlFace3dHead(128, 128)
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, tuple)
        assert len(outputs) == 7

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert isinstance(qat_outputs, tuple)
        assert len(qat_outputs) == 7

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert isinstance(quantized_outputs, tuple)
        assert len(quantized_outputs) == 7
