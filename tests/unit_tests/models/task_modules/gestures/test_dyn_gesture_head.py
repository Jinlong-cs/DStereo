import numpy as np
import torch

from hat.models.task_modules.gesture.dyn_gest_head import DynGestureHead
from tests.unit_tests.models.base import qtensor_test
from tests.unit_tests.models.task_modules.head_template import HeadTemplate

BATCHSIZE = 2
NUM_CLASS = 59


def setup_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TestDynGestureHead(HeadTemplate):
    def setup(self):
        setup_seed()
        self.inputs = torch.rand(BATCHSIZE, 128, 1, 1)
        self.quantized_inputs = torch.ones(
            BATCHSIZE, 128, 1, 1, dtype=torch.int8
        )
        self.model = DynGestureHead(num_classes=NUM_CLASS)
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert outputs.shape == (BATCHSIZE, NUM_CLASS)

    def test_qat_model(self):
        qat_inputs = qtensor_test(self.inputs)
        qat_outputs = self.qat_model(qat_inputs)
        assert qat_outputs.shape == (BATCHSIZE, NUM_CLASS)

    def test_quantize_model(self):
        quantized_inputs = qtensor_test(self.quantized_inputs)
        quantized_outputs = self.quantized_model(quantized_inputs)
        assert quantized_outputs.shape == (BATCHSIZE, NUM_CLASS)

    def test_fuse_model(self):
        outputs = self.float_model(self.inputs)
        fuse_outputs = self.fuse_model(self.inputs)
        assert torch.allclose(outputs, fuse_outputs, atol=1e-5)
