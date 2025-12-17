from hat.models.backbones.snapdragon import SNDREfficientnet
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
)


class TestSNDREfficientnet(BackboneTemplate):
    def test_fuse_model(self):
        pass

    def setup(self):
        super(TestSNDREfficientnet, self).setup()
        self.in_strides = [2, 4, 8, 16, 32]
        self.model = SNDREfficientnet(
            num_classes=self.num_classes,
            coefficient_params=(1.0, 1.0, 128, 0.2),
            model_type="lite",
            activation="relu",
            use_se_block=False,
            include_top=False,
        )
        self.build_model()
