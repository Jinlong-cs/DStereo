from hat.models.task_modules.retinanet.head import RetinaNetHead
from tests.unit_tests.models.task_modules.head_template import HeadTemplate


class TestHeadInt8(HeadTemplate):
    def setup(self):
        super(TestHeadInt8, self).setup()
        self.num_classes = 10
        self.num_anchors = 4
        self.model = RetinaNetHead(
            num_classes=self.num_classes,
            num_anchors=self.num_anchors,
            in_channels=16,
            feat_channels=16,
            stacked_convs=4,
            int16_output=True,
        )
        self.build_model()


class TestHeadInt32(HeadTemplate):
    def setup(self):
        super(TestHeadInt32, self).setup()
        self.num_classes = 10
        self.num_anchors = 4
        self.model = RetinaNetHead(
            num_classes=self.num_classes,
            num_anchors=self.num_anchors,
            in_channels=16,
            feat_channels=16,
            stacked_convs=4,
            int16_output=False,
        )
        self.build_model()
