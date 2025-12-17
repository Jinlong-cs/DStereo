from hat.models.backbones.resnet import ResNet18, ResNet50, ResNet50V2
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
    ClassifierTemplate,
)


class TestResNet18(ClassifierTemplate):
    def setup(self):
        super(TestResNet18, self).setup()
        self.model = ResNet18(
            num_classes=self.num_classes,
            bn_kwargs={},
            bias=True,
            include_top=True,
            flat_output=True,
        )
        self.build_model()


class TestResNet18Backbone(BackboneTemplate):
    def setup(self):
        super(TestResNet18Backbone, self).setup()
        self.in_strides = [4, 4, 8, 16, 32]
        self.model = ResNet18(
            num_classes=self.num_classes,
            bn_kwargs={},
            bias=True,
            include_top=False,
        )
        self.build_model()


class TestResNet50(ClassifierTemplate):
    def setup(self):
        super(TestResNet50, self).setup()
        self.model = ResNet50(
            num_classes=self.num_classes,
            bn_kwargs={},
            bias=True,
            include_top=True,
            flat_output=True,
        )
        self.build_model()


class TestResNet50Backbone(BackboneTemplate):
    def setup(self):
        super(TestResNet50Backbone, self).setup()
        self.in_strides = [4, 4, 8, 16, 32]
        self.model = ResNet50(
            num_classes=self.num_classes,
            bn_kwargs={},
            bias=True,
            include_top=False,
        )
        self.build_model()


class TestResNet50V2(ClassifierTemplate):
    def setup(self):
        super(TestResNet50V2, self).setup()
        self.model = ResNet50V2(
            num_classes=self.num_classes,
            group_base=8,
            bn_kwargs={},
            bias=True,
            extend_features=False,
            include_top=True,
            flat_output=True,
        )
        self.build_model()


class TestResNet50V2Backbone(BackboneTemplate):
    def setup(self):
        super(TestResNet50V2Backbone, self).setup()
        self.in_strides = [2, 4, 8, 16, 32]
        self.model = ResNet50V2(
            num_classes=self.num_classes,
            group_base=8,
            bn_kwargs={},
            bias=True,
            extend_features=False,
            include_top=False,
        )
        self.build_model()
