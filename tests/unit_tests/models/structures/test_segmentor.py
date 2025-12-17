import numpy as np
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


def test_segmentor():
    alpha = 0.25
    input_size = 256
    num_classes = 10
    train_scales = (4, 8, 16, 32, 64)

    config = dict(
        type="Segmentor",
        backbone=dict(
            type="MobileNetV1",
            num_classes=-1,
            bn_kwargs={},
            alpha=alpha,
            dw_with_relu=True,
            include_top=False,
            flat_output=False,
        ),
        neck=dict(
            type="DwUnet",
            base_channels=int(32 * alpha),
            use_deconv=False,
            dw_with_act=True,
            output_scales=train_scales,
        ),
        head=dict(
            type="SegHead",
            num_classes=num_classes,
            in_strides=train_scales,
            out_strides=train_scales,
            stride2channels={
                stride: int(stride * 32 * alpha) for stride in train_scales
            },
            feat_channels=tuple(np.array(train_scales) * int(32 * alpha)),
            stacked_convs=0,
            argmax_output=False,
            dequant_output=True,
            int8_output=True,
            upscale=False,
        ),
        losses=dict(
            type="SoftmaxFocalLoss",
            loss_name="Focal",
            num_classes=num_classes,
            reduction="mean",
            weight=tuple(np.array((256, 128, 64, 32, 16)) / 19),
        ),
    )

    segmentor = build_from_registry(config)

    x = {
        "img": torch.rand(1, 3, input_size, input_size),
        "gt_seg": [
            torch.rand(
                1,
                num_classes,
                input_size // 2 ** (i + 2),
                input_size // 2 ** (i + 2),
            ).round()
            for i in range(5)
        ],
    }

    y = segmentor(x)

    assert len(y[0]) == 5
    for i in range(5):
        assert y[0][i].shape == (
            1,
            10,
            input_size // 2 ** (i + 2),
            input_size // 2 ** (i + 2),
        )

    qat_test(segmentor, x, with_quantized=False)


def test_segmentor_v2():
    alpha = 0.25
    input_size = 256
    num_classes = 10
    train_scales = (4, 8, 16, 32, 64)

    config = dict(
        type="SegmentorV2",
        backbone=dict(
            type="MobileNetV1",
            num_classes=-1,
            bn_kwargs={},
            alpha=alpha,
            dw_with_relu=True,
            include_top=False,
            flat_output=False,
        ),
        neck=dict(
            type="DwUnet",
            base_channels=int(32 * alpha),
            use_deconv=False,
            dw_with_act=True,
            output_scales=train_scales,
        ),
        head=dict(
            type="SegHead",
            num_classes=num_classes,
            in_strides=train_scales,
            out_strides=train_scales,
            stride2channels={
                stride: int(stride * 32 * alpha) for stride in train_scales
            },
            feat_channels=tuple(np.array(train_scales) * int(32 * alpha)),
            stacked_convs=0,
            argmax_output=False,
            dequant_output=True,
            int8_output=True,
            upscale=False,
        ),
        loss=dict(
            type="SoftmaxFocalLoss",
            loss_name="Focal",
            num_classes=num_classes,
            reduction="mean",
            weight=tuple(np.array((256, 128, 64, 32, 16)) / 19),
        ),
    )

    segmentor = build_from_registry(config)

    x = {
        "img": torch.rand(1, 3, input_size, input_size),
        "labels": [
            torch.rand(
                1,
                num_classes,
                input_size // 2 ** (i + 2),
                input_size // 2 ** (i + 2),
            ).round()
            for i in range(5)
        ],
    }

    y = segmentor(x)

    assert len(y["pred"]) == 5
    for i in range(5):
        assert y["pred"][i].shape == (
            1,
            10,
            input_size // 2 ** (i + 2),
            input_size // 2 ** (i + 2),
        )

    qat_test(segmentor, x, with_quantized=False)


def test_bmsegmentor():
    alpha = 0.25
    input_size = 256
    num_classes = 10
    train_scales = (4, 8, 16, 32, 64)

    config = dict(
        type="BMSegmentor",
        backbone=dict(
            type="MobileNetV1",
            num_classes=-1,
            bn_kwargs={},
            alpha=alpha,
            dw_with_relu=True,
            include_top=False,
            flat_output=False,
        ),
        neck=dict(
            type="DwUnet",
            base_channels=int(32 * alpha),
            use_deconv=False,
            dw_with_act=True,
            output_scales=train_scales,
        ),
        head=dict(
            type="SegHead",
            num_classes=num_classes,
            in_strides=train_scales,
            out_strides=train_scales,
            stride2channels={
                stride: int(stride * 32 * alpha) for stride in train_scales
            },
            feat_channels=tuple(np.array(train_scales) * int(32 * alpha)),
            stacked_convs=0,
            argmax_output=False,
            dequant_output=True,
            int8_output=True,
            upscale=False,
        ),
    )

    segmentor = build_from_registry(config)

    x = {
        "img": torch.rand(1, 3, input_size, input_size),
        "labels": [
            torch.rand(
                1,
                num_classes,
                input_size // 2 ** (i + 2),
                input_size // 2 ** (i + 2),
            ).round()
            for i in range(5)
        ],
    }

    y = segmentor(x)

    assert len(y["pred"]) == 5
    for i in range(5):
        assert y["pred"][i].shape == (
            1,
            10,
            input_size // 2 ** (i + 2),
            input_size // 2 ** (i + 2),
        )

    qat_test(segmentor, x, with_quantized=False)

    config.update(
        dict(
            loss=dict(
                type="SoftmaxFocalLoss",
                loss_name="Focal",
                num_classes=num_classes,
                reduction="mean",
                weight=tuple(np.array((256, 128, 64, 32, 16)) / 19),
            )
        )
    )

    segmentor = build_from_registry(config)
    y = segmentor(x)

    assert "Focal" in y, y
