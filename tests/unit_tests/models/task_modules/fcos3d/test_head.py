import torch

from hat.models.task_modules.fcos3d import FCOS3DHead
from tests.unit_tests.models.task_modules.head_template import (
    HeadTemplate,
    qtensor_test,
)


class TestFCOS3DHead(HeadTemplate):
    def setup(self):
        super(TestFCOS3DHead, self).setup()
        P3 = torch.rand(2, 8, 64, 64)
        P4 = torch.rand(2, 8, 32, 32)
        P5 = torch.rand(2, 8, 16, 16)
        P6 = torch.rand(2, 8, 8, 8)
        P7 = torch.rand(2, 8, 4, 4)
        self.inputs = [P3, P4, P5, P6, P7]
        P3 = torch.ones(2, 8, 64, 64, dtype=torch.int8)
        P4 = torch.ones(2, 8, 32, 32, dtype=torch.int8)
        P5 = torch.ones(2, 8, 16, 16, dtype=torch.int8)
        P6 = torch.ones(2, 8, 8, 8, dtype=torch.int8)
        P7 = torch.ones(2, 8, 4, 4, dtype=torch.int8)
        self.quantized_inputs = [P3, P4, P5, P6, P7]
        self.model = FCOS3DHead(
            num_classes=10,
            in_channels=8,
            feat_channels=8,
            stacked_convs=2,
            strides=[8, 16, 32, 64, 128],
            group_reg_dims=(2, 1, 3, 1, 2),
            use_direction_classifier=True,
            pred_attrs=True,
            num_attrs=9,
            cls_branch=(256,),
            reg_branch=((8,), (8,), (8,), (8,), ()),  # velo
            dir_branch=(8,),
            attr_branch=(8,),
            centerness_branch=(8,),
            centerness_on_reg=True,
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert len(outputs) == 5
        for i in range(len(outputs)):
            assert len(outputs[i]) == len(self.inputs)

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert len(qat_outputs) == 5
        for i in range(len(qat_outputs)):
            assert len(qat_outputs[i]) == len(self.inputs)

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert len(quantized_outputs) == 5
        for i in range(len(quantized_outputs)):
            assert len(quantized_outputs[i]) == len(self.inputs)

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for i in range(len(outputs)):
            assert all(
                torch.allclose(output, fuse_output, atol=1e-5)
                for output, fuse_output in zip(outputs[i], fuse_outputs[i])
            )
