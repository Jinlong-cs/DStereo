import torch

from hat.models.backbones.repvgg import RepVGG
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
    ClassifierTemplate,
)


class TestPreRepVGG(ClassifierTemplate):
    def setup(self):
        super(TestPreRepVGG, self).setup()
        self.model = RepVGG(
            model_type="A0",
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
        assert distance < 1e-6, f"deploy mode diff {distance}"

    def test_fuse_model(self):
        pass


class TestPreRepVGGBackbone(BackboneTemplate):
    def setup(self):
        super(TestPreRepVGGBackbone, self).setup()
        self.in_strides = [4, 8, 16, 32]
        self.model = RepVGG(
            model_type="A0",
            num_classes=self.num_classes,
            deploy=False,
            include_top=False,
        )
        self.build_model(use_fx=True)

    def test_fuse_model(self):
        pass


class TestPostRepVGG(ClassifierTemplate):
    def setup(self):
        super(TestPostRepVGG, self).setup()
        self.model = RepVGG(
            model_type="A0",
            num_classes=self.num_classes,
            deploy=True,
            include_top=True,
            flat_output=True,
        )
        self.build_model(use_fx=True)

    def test_fuse_model(self):
        pass


class TestPostRepVGGBackbone(BackboneTemplate):
    def setup(self):
        super(TestPostRepVGGBackbone, self).setup()
        self.in_strides = [4, 8, 16, 32]
        self.model = RepVGG(
            model_type="A0",
            num_classes=self.num_classes,
            deploy=True,
            include_top=False,
        )
        self.build_model(use_fx=True)

    def test_fuse_model(self):
        pass
