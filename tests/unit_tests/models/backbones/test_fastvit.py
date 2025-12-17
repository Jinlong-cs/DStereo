import torch
import torch.nn as nn

from hat.models.backbones.fastvit import FastViT, RepMixerBlock
from tests.unit_tests.models.backbones.backbone_template import (
    ClassifierTemplate,
)


def test_repMixer():
    fake_data = torch.rand([8, 16, 56, 56])
    bb_module = RepMixerBlock(
        16,
        kernel_size=3,
        mlp_ratio=4,
        act_layer=nn.GELU,
        dropout=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        deploy=False,
    )
    bb_module.eval()
    res1 = bb_module(fake_data)
    bb_module.switch_to_deploy()
    res2 = bb_module(fake_data)
    diff = res1 - res2
    distance = torch.norm(diff)
    assert distance < 1e-4, f"deploy mode diff {distance}"


class TestPreFastViT(ClassifierTemplate):
    def setup(self):
        super(TestPreFastViT, self).setup()
        self.model = FastViT(
            model_type="t8",
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


class TestPreFastViTqa(ClassifierTemplate):
    def setup(self):
        super(TestPreFastViTqa, self).setup()
        self.model = FastViT(
            model_type="t8",
            num_classes=self.num_classes,
            post_bn=True,
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
