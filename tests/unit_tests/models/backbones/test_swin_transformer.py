from hat.models.backbones.swin_transformer import SwinTransformer
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
    ClassifierTemplate,
)


class TestSwinTransformer(ClassifierTemplate):
    def setup(self):
        super(TestSwinTransformer, self).setup()
        self.model = SwinTransformer(
            depth_list=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            num_classes=self.num_classes,
            embedding_dims=96,
            drop_path_ratio=0.2,
            include_top=True,
        )
        self.build_model()


class TestSwinTransformerBackbone(BackboneTemplate):
    def setup(self):
        super(TestSwinTransformerBackbone, self).setup()
        self.in_strides = [4, 8, 16, 32]
        self.model = SwinTransformer(
            depth_list=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            num_classes=self.num_classes,
            embedding_dims=96,
            drop_path_ratio=0.2,
            include_top=False,
        )
        self.build_model()
