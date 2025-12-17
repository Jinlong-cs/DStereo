import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


def test_encoderdecoder():
    input_size = 512
    bn_kwargs = {}

    config = dict(
        type="EncoderDecoder",
        backbone=dict(
            type="efficientnet",
            bn_kwargs=bn_kwargs,
            model_type="b0",
            num_classes=1000,
            include_top=False,
            activation="relu",
            use_se_block=False,
        ),
        decode_head=dict(
            type="Deeplabv3plusHead",
            in_channels=320,
            feat_channels=128,
            num_classes=19,
            c1_index=2,
            dilations=[1, 2, 4, 4],
            num_repeats=[1, 1, 1, 2],
            c1_in_channels=40,
            bn_kwargs=bn_kwargs,
            argmax_output=False,
            dequant_output=True,
            int8_output=False,
            dropout_ratio=0.1,
            upsample_decode_scale=4,
            upsample_output_scale=None,
        ),
        target=dict(
            type="FCNTarget",
            num_classes=19,
        ),
        loss=dict(
            type="CrossEntropyLoss",
            loss_name="decode",
            reduction="mean",
            ignore_index=-1,
            loss_weight=1.0,
        ),
        auxiliary_heads=[
            dict(
                head=dict(
                    type="FCNHead",
                    input_index=3,
                    in_channels=112,
                    feat_channels=128,
                    num_classes=19,
                    dropout_ratio=0.1,
                    num_convs=1,
                    bn_kwargs=bn_kwargs,
                ),
                target=dict(
                    type="FCNTarget",
                    num_classes=19,
                ),
                loss=dict(
                    type="CrossEntropyLoss",
                    loss_name="aux1",
                    ignore_index=-1,
                    reduction="mean",
                    loss_weight=0.4,
                ),
            ),
            dict(
                head=dict(
                    type="FCNHead",
                    input_index=2,
                    in_channels=40,
                    feat_channels=128,
                    num_classes=19,
                    dropout_ratio=0.1,
                    num_convs=1,
                    bn_kwargs=bn_kwargs,
                ),
                target=dict(
                    type="FCNTarget",
                    num_classes=19,
                ),
                loss=dict(
                    type="CrossEntropyLoss",
                    loss_name="aux2",
                    ignore_index=-1,
                    reduction="mean",
                    loss_weight=0.4,
                ),
            ),
        ],
    )

    module = build_from_registry(config)

    x = {
        "img": torch.rand(2, 3, input_size, input_size),
        "gt_seg": torch.rand(
            2,
            input_size,
            input_size,
        ),
    }

    y = module(x)
    y["decode"].backward()
    assert len(y) == 3
    qat_test(module, x, with_quantized=False)
