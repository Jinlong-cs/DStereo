from collections import OrderedDict

import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test, qtensor_test


def get_head_config_and_x():
    feat_channels = 64
    real3d_num_classes = 3
    real3d_bias = True
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    use_varg = False
    real3d_stack = 1
    feature_channels = {
        4: feat_channels,
        8: feat_channels,
        16: feat_channels,
        32: feat_channels,
        64: 128,
    }
    cfg = dict(
        type="Real3DHead",
        in_strides=[4, 8, 16, 32, 64],
        out_strides=[4],
        in_channels=feat_channels,
        feature_channels=feature_channels,
        head_channels=OrderedDict(
            hm=real3d_num_classes, dep=1, rot=2, dim=3, loc_offset=2, wh=2
        ),
        use_bias=real3d_bias,
        bn_kwargs=bn_kwargs,
        dw_with_relu=True,
        pw_with_relu=False,
        factor=2,
        group_base=8,
        sep_conv=True,
        use_varg=use_varg,
        stack=real3d_stack,
    )
    x = dict(
        feats=[
            torch.randn(1, feat_channels, 128, 128),
            torch.randn(1, feat_channels, 64, 64),
            torch.randn(1, feat_channels, 32, 32),
            torch.randn(1, feat_channels, 16, 16),
            torch.randn(1, feat_channels, 8, 8),
        ]
    )
    return cfg, x


def test_real3d_head():
    head_cfg, x = get_head_config_and_x()
    head = build_from_registry(head_cfg)
    y = head(x)
    assert isinstance(y, dict)
    assert y["hm"].shape == (1, 3, 128, 128)
    assert y["dep"].shape == (1, 1, 128, 128)
    assert y["rot"].shape == (1, 2, 128, 128)
    assert y["dim"].shape == (1, 3, 128, 128)
    assert y["loc_offset"].shape == (1, 2, 128, 128)
    assert y["wh"].shape == (1, 2, 128, 128)


def test_real3d_head_interpolate():
    head_cfg, x = get_head_config_and_x()
    head_cfg["interpolate_kwargs"] = {
        "scale_factor": 0.5,
        "mode": "bilinear",
        "recompute_scale_factor": True,
    }
    head = build_from_registry(head_cfg)
    y = head(x)
    assert isinstance(y, dict)
    assert y["hm"].shape == (1, 3, 64, 64)
    assert y["dep"].shape == (1, 1, 64, 64)
    assert y["rot"].shape == (1, 2, 64, 64)
    assert y["dim"].shape == (1, 3, 64, 64)
    assert y["loc_offset"].shape == (1, 2, 64, 64)
    assert y["wh"].shape == (1, 2, 64, 64)


def test_real3d_target():
    # TODO(runzhou.ge): add test case #
    pass


def test_real3d_loss():
    # TODO(runzhou.ge): add test case #
    pass


def test_real3d_postprocess():
    # TODO(runzhou.ge): add test case #
    pass


def test_real3d_head_in_train():
    feat_channels = 64
    head_cfg, x = get_head_config_and_x()
    head_cfg["in_channels"] = feat_channels

    cfg = dict(
        type="OutputModule",
        head=head_cfg,
        # TODO(min.du): add loss #
        loss=None,
        prefix="real3d",
    )
    output_module = build_from_registry(cfg)
    y = output_module(x)  # noqa: F841

    # test qat
    x["feats"] = [qtensor_test(feat) for feat in x["feats"]]
    qat_test(output_module, x, False)
    # TODO(min.du): add assert #


def test_real3d_head_in_infer():
    head_cfg, x = get_head_config_and_x()

    cfg = dict(
        type="OutputModule",
        head=head_cfg,
        # TODO(min.du, 0.5): add postprocess #
        postprocess=None,
        prefix="real3d",
    )
    output_module = build_from_registry(cfg)
    y = output_module(x)  # noqa: F841

    # TODO(min.du): add assert #
