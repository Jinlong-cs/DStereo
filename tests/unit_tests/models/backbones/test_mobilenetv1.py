from hat.models.backbones.mobilenetv1 import MobileNetV1
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
    ClassifierTemplate,
)


class TestMobileNetV1Alpha05(ClassifierTemplate):
    def setup(self):
        super(TestMobileNetV1Alpha05, self).setup()
        self.model = MobileNetV1(
            num_classes=self.num_classes,
            bn_kwargs={},
            alpha=0.5,
            bias=True,
            dw_with_relu=True,
            include_top=True,
            flat_output=True,
        )
        self.build_model()


class TestMobileNetV1Alpha10(ClassifierTemplate):
    def setup(self):
        super(TestMobileNetV1Alpha10, self).setup()
        self.model = MobileNetV1(
            num_classes=self.num_classes,
            bn_kwargs={},
            alpha=1.0,
            bias=True,
            dw_with_relu=True,
            include_top=True,
            flat_output=True,
        )
        self.build_model()


class TestMobileNetV1BackboneAlpha05(BackboneTemplate):
    def setup(self):
        super(TestMobileNetV1BackboneAlpha05, self).setup()
        self.in_strides = [2, 4, 8, 16, 32]
        self.model = MobileNetV1(
            num_classes=self.num_classes,
            bn_kwargs={},
            alpha=0.5,
            bias=True,
            dw_with_relu=True,
            include_top=False,
        )
        self.build_model()


class TestMobileNetV1BackboneAlpha10(BackboneTemplate):
    def setup(self):
        super(TestMobileNetV1BackboneAlpha10, self).setup()
        self.in_strides = [2, 4, 8, 16, 32]
        self.model = MobileNetV1(
            num_classes=self.num_classes,
            bn_kwargs={},
            alpha=1.0,
            bias=True,
            dw_with_relu=True,
            include_top=False,
        )
        self.build_model()
