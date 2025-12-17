from hat.models.backbones.snapdragon import SNDRMobileNeXt
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
)


class TestSNDRMobileNeXt(BackboneTemplate):
    def test_fuse_model(self):
        pass

    def setup(self):
        super(TestSNDRMobileNeXt, self).setup()
        self.in_strides = [2, 4, 8, 16, 32]
        self.model = SNDRMobileNeXt(
            num_classes=self.num_classes,
            bn_kwargs={},
            in_chls=[
                [32],
                [16, 24],
                [24, 32, 32],
                [32] + [64] * 4 + [96] * 2,
                [96] + [160] * 3,
            ],
            out_chls=[
                [16],
                [24, 24],
                [32, 32, 32],
                [64] * 4 + [96] * 3,
                [160] * 3 + [320],
            ],
            expand_ratio=2,
            alpha=1.0,
            bias=True,
            include_top=False,
        )
        self.build_model()
