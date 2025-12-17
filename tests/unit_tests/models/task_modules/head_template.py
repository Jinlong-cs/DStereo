from abc import abstractmethod

import torch

from tests.unit_tests.models.base import ModelTemplate, qtensor_test


class HeadTemplate(ModelTemplate):
    @abstractmethod
    def setup(self):
        x1 = torch.rand(1, 16, 64, 64)
        x2 = torch.rand(1, 16, 32, 32)
        x3 = torch.rand(1, 16, 16, 16)
        self.inputs = [x1, x2, x3]
        x1 = torch.ones(1, 16, 64, 64, dtype=torch.int8)
        x2 = torch.ones(1, 16, 32, 32, dtype=torch.int8)
        x3 = torch.ones(1, 16, 16, 16, dtype=torch.int8)
        self.quantized_inputs = [x1, x2, x3]

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert len(outputs[0]) == len(self.inputs)
        assert len(outputs[1]) == len(self.inputs)
        for cls_score in outputs[0]:
            assert isinstance(cls_score, torch.Tensor)
            assert len(cls_score.shape) == 4
            assert cls_score.shape[1] == self.num_anchors * self.num_classes
        for bbox_pred in outputs[1]:
            assert isinstance(bbox_pred, torch.Tensor)
            assert len(bbox_pred.shape) == 4
            assert bbox_pred.shape[1] == self.num_anchors * 4

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert len(qat_outputs[0]) == len(self.inputs)
        assert len(qat_outputs[1]) == len(self.inputs)
        for cls_score in qat_outputs[0]:
            assert isinstance(cls_score, torch.Tensor)
            assert len(cls_score.shape) == 4
            assert cls_score.shape[1] == self.num_anchors * self.num_classes
        for bbox_pred in qat_outputs[1]:
            assert isinstance(bbox_pred, torch.Tensor)
            assert len(bbox_pred.shape) == 4
            assert bbox_pred.shape[1] == self.num_anchors * 4

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert len(quantized_outputs[0]) == len(self.inputs)
        assert len(quantized_outputs[1]) == len(self.inputs)
        for cls_score in quantized_outputs[0]:
            assert isinstance(cls_score, torch.Tensor)
            assert len(cls_score.shape) == 4
            assert cls_score.shape[1] == self.num_anchors * self.num_classes
        for bbox_pred in quantized_outputs[1]:
            assert isinstance(bbox_pred, torch.Tensor)
            assert len(bbox_pred.shape) == 4
            assert bbox_pred.shape[1] == self.num_anchors * 4

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        for i in range(len(outputs)):
            assert all(
                torch.allclose(output, fuse_output, atol=1e-5)
                for output, fuse_output in zip(outputs[i], fuse_outputs[i])
            )
