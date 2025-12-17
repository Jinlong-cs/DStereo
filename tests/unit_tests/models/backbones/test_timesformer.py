import pytest
import torch

from hat.models.backbones.timesformer import TimeSformer
from tests.unit_tests.models.backbones.backbone_template import (
    ClassifierTemplate,
)


@pytest.mark.serial_task
class TestTimeSformer(ClassifierTemplate):
    def setup(self):
        super(TestTimeSformer, self).setup()
        self.model = TimeSformer(
            img_size=self.input_size,
            patch_size=16,
            num_classes=self.num_classes,
            num_frames=2,
            test_mode=False,
        )
        self.inputs = torch.randn(1, 3, 2, self.input_size, self.input_size)
        self.build_model()
