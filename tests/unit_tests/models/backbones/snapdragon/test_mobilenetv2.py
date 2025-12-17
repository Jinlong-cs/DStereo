from hat.models.backbones.snapdragon import SNDRMobileNetV2
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
)


class TestSNDRMobileNetV2(BackboneTemplate):
    def setup(self):
        super(TestSNDRMobileNetV2, self).setup()
        self.in_strides = [2, 4, 8, 16, 32]
        self.model = SNDRMobileNetV2(
            num_classes=self.num_classes,
            bn_kwargs={},
            in_chls=[[32], [32, 32], [32, 32], [32, 64], [128, 128]],
            out_chls=[[32], [32, 32], [32, 32], [64, 128], [128, 128]],
            expand_ratio=2,
            alpha=1.0,
            bias=True,
            include_top=False,
        )
        self.build_model()
