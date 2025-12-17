from hat.models.backbones.vision_transformer import VisionTransformer
from tests.unit_tests.models.backbones.backbone_template import (
    ClassifierTemplate,
)


class TestVisionTransformer(ClassifierTemplate):
    def setup(self):
        super(TestVisionTransformer, self).setup()
        self.model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=96,
            depth=6,
            drop_path_rate=0.1,
            include_top=True,
            set_int16_qconfig=False,
            num_classes=self.num_classes,
        )
        self.build_model(use_fx=True)
