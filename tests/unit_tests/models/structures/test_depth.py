import copy

import numpy as np
import torch

from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels
from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


def test_depth_model():
    bn_kwargs = dict(eps=1e-5, momentum=0.1)
    alpha = 0.25
    head_factor = 2
    bifpn_in_strides = [2, 4, 8, 16, 32]
    bifpn_out_strides = [2, 4, 8, 16, 32]
    bifpn_start_level = 0
    bifpn_end_level = -1
    neck_feat_channels = 32
    head_out_strides = [1, 4, 8]
    start_level = 0
    end_level = 3
    pred_out_channel = 2
    head_in_channels = [neck_feat_channels] * len(bifpn_out_strides)
    head_out_channels = [neck_feat_channels] * (end_level - start_level + 1)
    head_out_names = {"depth": "pred_depth"}
    head_parser_strides = {head_out_names["depth"]: head_out_strides}
    losses = [
        dict(
            type="DepthConfidenceCombinedLoss",
            depth_loss=dict(
                type="DepthL1Loss",
                max_gt_depth=150.0,
                loss_weight=1.0,
                loss_name="depth_l1_loss",
                use_weight_map=True,
            ),
            conf_loss=dict(
                type="ConfL1Loss",
                max_gt_depth=150.0,
                loss_weight=1.0,
                loss_name="conf_l1_loss",
                use_weight_map=False,
            ),
            depth_loss_name="depth_loss_s1",
            conf_loss_name="conf_loss_s1",
            loss_weight=[20.0, 2.0],
        )
    ] * len(head_out_strides)
    losses_weight = [1.0] * len(head_out_strides)
    model = dict(
        type="DepthModel",
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            bn_kwargs=bn_kwargs,
            alpha=alpha,
            group_base=8,
            factor=2,
            bias=True,
            include_top=False,
            head_factor=head_factor,
        ),
        neck=dict(
            type="BiFPN",
            fpn_name="bifpn_sum",
            in_strides=bifpn_in_strides,
            out_strides=bifpn_out_strides,
            stride2channels=get_vargnetv2_stride2channels(alpha),
            out_channels=neck_feat_channels,
            stack=3,
            start_level=bifpn_start_level,
            end_level=bifpn_end_level,
            num_outs=5,
        ),
        head=dict(
            type="DepthUnetHead",
            in_strides=bifpn_out_strides,
            out_strides=head_out_strides,
            in_channels=head_in_channels,
            out_channels=head_out_channels,  # need to be improved
            pred_out_channel=pred_out_channel,
            stacked_convs=6,
            start_level=start_level,
            end_level=end_level,
            bn_kwargs=bn_kwargs,
            group_base=8,
            conv_method="conv2d",
            use_auxi_loss=True,
            with_refine=False,
            last_with_relu=FloatingPointError,
            dequant_output=True,
            int8_output=True,  # True
            head_out_names=head_out_names,
        ),
        head_parser=dict(
            type="DepthHeadParserWithScale",
            out_strides=head_parser_strides,
        ),
        target_generator=dict(
            type="DepthMultiTargets",
            downsample_scale=1,
            depth_type="Cylindrical",
            low_depth=0.0,
            high_depth=150.0,
            gt_scale=1.0,
            label_name="gt_depth",
            pred_names=list(head_out_names.values()),
            target_modules=dict(
                type="DepthValueTarget",
                label_name="gt_depth",
                with_smooth=False,
                with_virtual_normal=False,
            ),
            valid_hfov=np.deg2rad(70),
        ),
        decoder=None,
        losses=dict(
            type="SegLoss",
            loss=[
                dict(
                    type="MixSegLossMultipreds",
                    losses=losses,
                    losses_weight=losses_weight,
                ),
            ],
        ),
        head_out_name=head_out_names["depth"],
    )

    segmentor = build_from_registry(model)
    width = 352
    height = 288
    x = {
        "img": torch.rand(1, 3, height, width),
        "gt_depth": torch.randint(
            1,
            2 ** 16,
            (
                1,
                height,
                width,
            ),
        )
        / 256.0,
        "virtual_cam_params": torch.from_numpy(
            np.array(
                [[110, 0, 176], [0, 110, 144], [0, 0, 1]], dtype=np.float32
            ).reshape(1, 3, 3)
        ),
    }
    x_qat = copy.deepcopy(x)
    y = segmentor(x)

    assert y["depth_preds"].shape == (
        1,
        pred_out_channel,
        height // head_out_strides[0],
        width // head_out_strides[0],
    ), y["depth_preds"].shape

    qat_test(segmentor, x_qat, with_quantized=False)
