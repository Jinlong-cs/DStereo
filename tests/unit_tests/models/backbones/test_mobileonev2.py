import torch

from hat.models.backbones.mobileonev2 import MobileOneV2
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
    ClassifierTemplate,
)


class TestPreMobieleOneV2(ClassifierTemplate):
    def setup(self):
        super(TestPreMobieleOneV2, self).setup()
        self.model = MobileOneV2(
            model_type="s1",
            num_classes=self.num_classes,
            deploy=False,
            include_top=True,
            flat_output=True,
        )
        self.build_model(use_fx=True)

    def test_deploy_model(self):
        res1 = self.float_model(self.inputs)
        self.float_model.switch_to_deploy()
        res2 = self.float_model(self.inputs)
        diff = res1 - res2
        distance = torch.norm(diff)
        assert distance < 1e-5

    def test_fuse_model(self):
        pass


class TestMobieleOneV2Backbone(BackboneTemplate):
    def setup(self):
        super(TestMobieleOneV2Backbone, self).setup()
        self.in_strides = [2, 4, 8, 16, 32]
        self.model = MobileOneV2(
            model_type="s1",
            num_classes=self.num_classes,
            deploy=False,
            include_top=False,
        )
        self.build_model(use_fx=True)

    def test_fuse_model(self):
        pass


class TestPostMobieleOneV2(ClassifierTemplate):
    def setup(self):
        super(TestPostMobieleOneV2, self).setup()
        self.model = MobileOneV2(
            model_type="s1",
            num_classes=self.num_classes,
            deploy=True,
            include_top=True,
            flat_output=True,
        )
        self.build_model(use_fx=True)

    def test_fuse_model(self):
        pass


class TestPostMobieleOneV2Backbone(BackboneTemplate):
    def setup(self):
        super(TestPostMobieleOneV2Backbone, self).setup()
        self.in_strides = [2, 4, 8, 16, 32]
        self.model = MobileOneV2(
            model_type="s1",
            num_classes=self.num_classes,
            deploy=True,
            include_top=False,
        )
        self.build_model(use_fx=True)

    def test_fuse_model(self):
        pass
