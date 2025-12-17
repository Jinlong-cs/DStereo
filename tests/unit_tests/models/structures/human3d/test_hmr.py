import torch

from hat.models.backbones.resnet import ResNet50
from hat.models.structures.human3d import HMR
from hat.models.task_modules.human3d.head import HMRHead
from tests.unit_tests.models.base import ModelTemplate


class TestHMR(ModelTemplate):
    def setup(self):
        super(TestHMR, self).setup()
        self.inputs = torch.rand((1, 3, 224, 224))
        self.quantized_inputs = torch.ones(1, 3, 224, 224, dtype=torch.int8)
        self.model = HMR(
            backbone=ResNet50(1000, {}, include_top=False),
            head=HMRHead(
                smpl_mean_params="./tmp_orig_data/human3d/spin_params/data/smpl_mean_params.npz"  # noqa
            ),
        )
        self.build_model()

    def test_float_model(self):
        outputs = self.float_model(self.inputs)
        assert isinstance(outputs, tuple)
        assert len(outputs) == 3

    def test_qat_model(self):
        qat_outputs = self.qat_model(self.inputs)
        assert isinstance(qat_outputs, tuple)
        assert len(qat_outputs) == 3

    def test_quantize_model(self):
        quantized_outputs = self.quantized_model(self.inputs)
        assert isinstance(quantized_outputs, tuple)
        assert len(quantized_outputs) == 3
