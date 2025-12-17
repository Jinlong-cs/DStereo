import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


def test_lipmovemodel():
    input_size = 96
    batchsize = 8
    seqlen = 80
    bn_kwargs = {}

    config = dict(
        type="LipmoveModel",
        vfea_extractor=dict(
            type="CocktailVargNetV2",
            bn_kwargs=bn_kwargs,
            flat_output=False,
            top_layer=dict(
                type="CocktailTopLayer",
                num_channels=256,
            ),
        ),
        lipmove_net=dict(
            type="GeneralTemporal",
            bn_kwargs=bn_kwargs,
            num_channels=512,
            num_classes=2,
            lookahead_conv_op=dict(
                type="AdaptiveLookAheadConv",
                bn_kwargs=bn_kwargs,
                num_channels=512,
                use_stride_2_conv=True,
            ),
        ),
        loss=dict(
            type="CrossEntropyLoss",
            loss_name="decode",
            reduction="mean",
            ignore_index=-1,
            loss_weight=1.0,
        ),
        num_cached_frames=15,
        ftr_in_framewise_operation="fdd",
        replicate_type_in_tile_and_pad_net="zero",
        no_cut_seqlen_in_tile_and_padding=False,
    )
    module = build_from_registry(config)
    imgs = torch.rand(batchsize, seqlen, 3, input_size, input_size)
    x = {
        "images": imgs,
        "label": torch.rand(batchsize, seqlen),
    }
    y = module(x)
    print("test_loss:{}".format(y))
    qat_test(module, x, with_quantized=False)
