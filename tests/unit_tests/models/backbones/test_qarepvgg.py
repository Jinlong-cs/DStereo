import torch

from hat.models.backbones.qa_repvgg import QARepVGG
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
    ClassifierTemplate,
)


class TestPreQARepVGG(ClassifierTemplate):
    def setup(self):
        super(TestPreQARepVGG, self).setup()
        self.model = QARepVGG(
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
        assert distance < 1e-5

    def test_fuse_model(self):
        pass


class TestPreQARepVGGBackbone(BackboneTemplate):
    def setup(self):
        super(TestPreQARepVGGBackbone, self).setup()
        self.in_strides = [4, 8, 16, 32]
        self.model = QARepVGG(
            model_type="A0",
            num_classes=self.num_classes,
            deploy=False,
            include_top=False,
        )
        self.build_model(use_fx=True)

    def test_fuse_model(self):
        pass


class TestPostQARepVGG(ClassifierTemplate):
    def setup(self):
        super(TestPostQARepVGG, self).setup()
        self.model = QARepVGG(
            model_type="A0",
            num_classes=self.num_classes,
            deploy=True,
            include_top=True,
            flat_output=True,
        )
        self.build_model(use_fx=True)

    def test_fuse_model(self):
        pass


class TestPostQARepVGGBackbone(BackboneTemplate):
    def setup(self):
        super(TestPostQARepVGGBackbone, self).setup()
        self.in_strides = [4, 8, 16, 32]
        self.model = QARepVGG(
            model_type="A0",
            num_classes=self.num_classes,
            deploy=True,
            include_top=False,
        )
        self.build_model(use_fx=True)

    def test_fuse_model(self):
        pass
