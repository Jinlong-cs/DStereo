from copy import deepcopy

import pytest
from horizon_plugin_pytorch.quantization import prepare_qat

from hat.models.backbones.patcher import ResizePatcher, ZeroPad2DPatcher
from hat.models.backbones.vargnetv2_2631 import VargNetV2Stage2631
from hat.models.base_modules.extend_container import ExtSequential
from hat.utils import qconfig_manager


@pytest.mark.parametrize(
    [
        "input_padding",
    ],
    [
        pytest.param([16, 16, 0, 0]),
        pytest.param([0, 0, 0, 0]),
    ],
)
def test_zeropad2dpatcher(input_padding):

    backbone = VargNetV2Stage2631(
        num_classes=1000,
        multiplier=0.5,
        group_base=4,
        last_channels=1024,
        stages=(1, 2, 3, 4, 5),
        include_top=False,
        extend_features=True,
        bn_kwargs={},
    )

    backbone.fuse_model()
    backbone.qconfig = qconfig_manager.get_default_qat_qconfig()
    backbone.set_qconfig()
    backbone = prepare_qat(backbone, inplace=True)

    backbone_state_dict = deepcopy(backbone.state_dict())

    backbone_with_pad = ZeroPad2DPatcher(
        backbone=VargNetV2Stage2631(
            num_classes=1000,
            multiplier=0.5,
            group_base=4,
            last_channels=1024,
            stages=(1, 2, 3, 4, 5),
            include_top=False,
            extend_features=True,
            bn_kwargs={},
        ),
        input_padding=input_padding,
    )
    backbone_with_pad.fuse_model()
    backbone_with_pad.qconfig = qconfig_manager.get_default_qat_qconfig()
    backbone_with_pad.set_qconfig()
    backbone_with_pad = prepare_qat(backbone_with_pad, inplace=True)

    # check if padded
    if not all([p == 0 for p in input_padding]):
        assert isinstance(backbone_with_pad.quant, ExtSequential)

    # check state_dict compatibility
    backbone_with_pad.load_state_dict(backbone_state_dict, strict=True)
    backbone_with_pad.load_state_dict(
        backbone_with_pad.state_dict(), strict=True
    )


@pytest.mark.parametrize(
    [
        "input_scale",
    ],
    [
        pytest.param((2, 2)),
        pytest.param((1, 1)),
    ],
)
def test_resizepatcher(input_scale):

    backbone = VargNetV2Stage2631(
        num_classes=1000,
        multiplier=0.5,
        group_base=4,
        last_channels=1024,
        stages=(1, 2, 3, 4, 5),
        include_top=False,
        extend_features=True,
        bn_kwargs={},
    )

    backbone.fuse_model()
    backbone.qconfig = qconfig_manager.get_default_qat_qconfig()
    backbone.set_qconfig()
    backbone = prepare_qat(backbone, inplace=True)

    backbone_state_dict = deepcopy(backbone.state_dict())

    backbone_with_resize = ResizePatcher(
        backbone=VargNetV2Stage2631(
            num_classes=1000,
            multiplier=0.5,
            group_base=4,
            last_channels=1024,
            stages=(1, 2, 3, 4, 5),
            include_top=False,
            extend_features=True,
            bn_kwargs={},
        ),
        input_scale=input_scale,
    )
    backbone_with_resize.fuse_model()
    backbone_with_resize.qconfig = qconfig_manager.get_default_qat_qconfig()
    backbone_with_resize.set_qconfig()
    backbone_with_resize = prepare_qat(backbone_with_resize, inplace=True)

    # check if padded
    if not all([p == 1 for p in input_scale]):
        assert isinstance(backbone_with_resize.quant, ExtSequential)

    # check state_dict compatibility
    backbone_with_resize.load_state_dict(backbone_state_dict, strict=True)
    backbone_with_resize.load_state_dict(
        backbone_with_resize.state_dict(), strict=True
    )
