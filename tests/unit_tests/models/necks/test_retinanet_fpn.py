from hat.models.necks.retinanet_fpn import RetinaNetFPN
from tests.unit_tests.models.necks.neck_template import NeckTemplate


class TestRetinaNetFPN(NeckTemplate):
    def setup(self):
        super(TestRetinaNetFPN, self).setup()
        self.model = RetinaNetFPN(
            in_strides=self.in_strides,
            in_channels=self.in_channels,
            out_strides=self.out_strides,
            out_channels=self.out_channels,
            fix_out_channel=64,
        )
        self.build_model()
        self.out_strides += [64, 128]
