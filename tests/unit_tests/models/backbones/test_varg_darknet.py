from hat.models.backbones.vargdarknet import VarGDarkNet53
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
    ClassifierTemplate,
)


class TestVarGDarkNet53(ClassifierTemplate):
    def setup(self):
        super(TestVarGDarkNet53, self).setup()
        self.model = VarGDarkNet53(
            max_channels=1024,
            bn_kwargs={},
            num_classes=self.num_classes,
            include_top=True,
            flat_output=True,
        )
        self.build_model()


class TestVarGDarkNet53Backbone(BackboneTemplate):
    def setup(self):
        super(TestVarGDarkNet53Backbone, self).setup()
        self.in_strides = [1, 2, 4, 8, 16, 32]
        self.model = VarGDarkNet53(
            max_channels=1024,
            bn_kwargs={},
            num_classes=self.num_classes,
            include_top=False,
        )
        self.build_model()
