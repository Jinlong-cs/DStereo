from hat.models.backbones.horizon_swin_transformer import (
    HorizonSwinTransformer,
)
from tests.unit_tests.models.backbones.backbone_template import (
    BackboneTemplate,
    ClassifierTemplate,
)


class TestHorizonSwinTransformer(ClassifierTemplate):
    def setup(self):
        super(TestHorizonSwinTransformer, self).setup()
        self.model = HorizonSwinTransformer(
            depth_list=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            num_classes=self.num_classes,
            embedding_dims=96,
            drop_path_ratio=0.2,
            include_top=True,
        )
        self.build_model()


class TestHorizonSwinTransformerBackbone(BackboneTemplate):
    def setup(self):
        super(TestHorizonSwinTransformerBackbone, self).setup()
        self.in_strides = [4, 8, 16, 32]
        self.model = HorizonSwinTransformer(
            depth_list=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            num_classes=self.num_classes,
            embedding_dims=96,
            drop_path_ratio=0.2,
            include_top=False,
        )
        self.build_model()
