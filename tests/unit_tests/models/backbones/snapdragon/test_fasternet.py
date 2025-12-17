from hat.models.backbones.snapdragon import SNDRFasterNet
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
)


class TestSNDRFasterNet(BackboneTemplate):
    def test_fuse_model(self):
        pass

    def setup(self):
        super(TestSNDRFasterNet, self).setup()
        self.in_strides = [4, 8, 16, 32]
        self.model = SNDRFasterNet(
            in_chans=3,
            mlp_ratio=2,
            embed_dim=32,
            depths=[1, 2, 8, 2],
            patch_size=4,
            patch_stride=4,
            patch_size2=2,
            patch_stride2=2,
            drop_path_rate=0.0,
            norm_layer="BN",
            act_layer="RELU",
            n_div=[1, 2, 4, 4],
        )
        self.build_model()
