import pytest

from hat.models.necks.fpn import FPN
from tests.unit_tests.models.necks.neck_template import NeckTemplate


def test_fpn_error():
    # Test len(in_strides) == len(in_channels)
    model = None
    with pytest.raises(AssertionError):
        model = FPN(
            in_strides=[8, 16, 32],
            in_channels=[128, 256],
            out_strides=[8, 16, 32],
            out_channels=[64, 128, 256],
        )

    # Test in_strides must be in [2, 4, 8, 16, 32, 64, 128, 256]
    with pytest.raises(AssertionError):
        model = FPN(
            in_strides=[64, 128, 256, 512],
            in_channels=[256, 256, 256, 256],
            out_strides=[64, 128, 256],
            out_channels=[256, 256, 256],
        )

    # Test in_strides must be continuous and in ascending order
    with pytest.raises(AssertionError):
        model = FPN(
            in_strides=[8, 16, 32, 128],
            in_channels=[64, 128, 256, 256],
            out_strides=[32, 64, 128],
            out_channels=[256, 256, 256],
        )

    # Test len(out_strides) == len(out_channels)
    with pytest.raises(AssertionError):
        model = FPN(
            in_strides=[8, 16, 32],
            in_channels=[64, 128, 256],
            out_strides=[8, 16, 32],
            out_channels=[128, 256],
        )

    # Test out_strides must be continuous and in ascending order
    with pytest.raises(AssertionError):
        model = FPN(
            in_strides=[8, 16, 32],
            in_channels=[64, 128, 256],
            out_strides=[8, 16, 32, 128],
            out_channels=[64, 128, 256, 256],
        )

    # Test all stride of output stride must be in input stride
    with pytest.raises(AssertionError):
        model = FPN(
            in_strides=[8, 16, 32],
            in_channels=[64, 128, 256],
            out_strides=[8, 16, 32, 64],
            out_channels=[64, 128, 256, 256],
        )
    assert model is None


class TestFPNNoneNone(NeckTemplate):
    def setup(self):
        super(TestFPNNoneNone, self).setup()
        self.model = FPN(
            in_strides=self.in_strides,
            in_channels=self.in_channels,
            out_strides=self.out_strides,
            out_channels=self.out_channels,
            fix_out_channel=None,
            bn_kwargs=None,
        )
        self.build_model()

    def test_float_model(self):
        self.float_model._init_weights()
        super(TestFPNNoneNone, self).test_float_model()


class TestFPN64None(NeckTemplate):
    def setup(self):
        super(TestFPN64None, self).setup()
        self.model = FPN(
            in_strides=self.in_strides,
            in_channels=self.in_channels,
            out_strides=self.out_strides,
            out_channels=self.out_channels,
            fix_out_channel=64,
            bn_kwargs=None,
        )
        self.build_model()

    def test_float_model(self):
        self.float_model._init_weights()
        super(TestFPN64None, self).test_float_model()


class TestFPN64BN(NeckTemplate):
    def setup(self):
        super(TestFPN64BN, self).setup()
        self.model = FPN(
            in_strides=self.in_strides,
            in_channels=self.in_channels,
            out_strides=self.out_strides,
            out_channels=self.out_channels,
            fix_out_channel=64,
            bn_kwargs={},
        )
        self.build_model()

    def test_float_model(self):
        self.float_model._init_weights()
        super(TestFPN64BN, self).test_float_model()
