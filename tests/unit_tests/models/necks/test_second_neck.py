import torch

from hat.models.necks.second_neck import SECONDNeck


def test_sequential_bottleneck():
    input_data = torch.randn(1, 2, 32, 32)

    module = SECONDNeck(
        in_feature_channel=2,
        down_layer_nums=[1, 2, 3],
        down_layer_strides=[2, 2, 2],
        down_layer_channels=[2, 4, 8],
        up_layer_strides=[1, 2, 4],
        up_layer_channels=[2, 2, 2],
        bn_kwargs=None,
        use_relu6=True,
        quantize=True,
    )
    output = module(input_data)

    assert output.shape == (1, 6, 16, 16)
