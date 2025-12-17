import torch.nn as nn

from hat.models.backbones.mobilenetv2 import MobileNetV2
from hat.models.base_modules.conv_module import ConvModule2d
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
    ClassifierTemplate,
)


class TestMobileNetV2DepthWiseAsPool(ClassifierTemplate):
    def setup(self):
        super(TestMobileNetV2DepthWiseAsPool, self).setup()
        self.model = MobileNetV2(
            num_classes=self.num_classes,
            bn_kwargs={},
            bias=True,
            include_top=True,
            flat_output=True,
            use_dw_as_avgpool=True,
        )
        self.build_model()
        assert isinstance(self.model.output[1], ConvModule2d)


class TestMobileNetV2Alpha05(ClassifierTemplate):
    def setup(self):
        super(TestMobileNetV2Alpha05, self).setup()
        self.model = MobileNetV2(
            num_classes=self.num_classes,
            bn_kwargs={},
            alpha=0.5,
            bias=True,
            include_top=True,
            flat_output=True,
        )
        self.build_model()
        assert isinstance(self.model.output[1], nn.AvgPool2d)


class TestMobileNetV2Alpha10(ClassifierTemplate):
    def setup(self):
        super(TestMobileNetV2Alpha10, self).setup()
        self.model = MobileNetV2(
            num_classes=self.num_classes,
            bn_kwargs={},
            alpha=1.0,
            bias=True,
            include_top=True,
            flat_output=True,
        )
        self.build_model()
        assert isinstance(self.model.output[1], nn.AvgPool2d)


class TestMobileNetV2BackboneAlpha05(BackboneTemplate):
    def setup(self):
        super(TestMobileNetV2BackboneAlpha05, self).setup()
        self.in_strides = [2, 4, 8, 16, 32]
        self.model = MobileNetV2(
            num_classes=self.num_classes,
            bn_kwargs={},
            alpha=0.5,
            bias=True,
            include_top=False,
        )
        self.build_model()


class TestMobileNetV2BackboneAlpha10(BackboneTemplate):
    def setup(self):
        super(TestMobileNetV2BackboneAlpha10, self).setup()
        self.in_strides = [2, 4, 8, 16, 32]
        self.model = MobileNetV2(
            num_classes=self.num_classes,
            bn_kwargs={},
            alpha=1.0,
            bias=True,
            include_top=False,
        )
        self.build_model()
