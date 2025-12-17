import horizon_plugin_pytorch as horizon
import pytest
import torch

from hat.models.necks.sequential_bottleneck import SequentialBottleNeck


@pytest.mark.parametrize(
    [
        "use_scnet",
        "use_res2net",
        "use_repvgg",
        "use_plugin_quan",
        "use_tconv",
        "use_secnet",
    ],
    [
        pytest.param(False, False, False, False, False, False),  # default
        pytest.param(True, False, False, False, False, False),  # scnet
        pytest.param(False, True, False, False, False, False),  # res2net
        pytest.param(False, False, True, False, False, False),  # repvgg
        pytest.param(False, False, False, True, False, False),  # plugin
        pytest.param(False, False, False, False, True, False),  # tconv
        pytest.param(False, False, False, False, True, True),  # secnet, tconv
    ],
)
def test_sequential_bottleneck(
    use_scnet: bool,
    use_res2net: bool,
    use_repvgg: bool,
    use_plugin_quan: bool,
    use_tconv: bool,
    use_secnet: bool,
):
    # Ensure that under each param, model can build and can run.
    # That should be enough.
    input_data = torch.randn(1, 64, 128, 128)

    layer_nums = [1, 3, 5, 5]
    ds_layer_strides = [2, 2, 2, 2]
    ds_num_filters = [48, 48, 96, 192]
    us_layer_strides = [1, 2, 4]
    us_num_filters = [96, 96, 96]
    num_input_features = 64
    bn_kwargs = dict(eps=1e-3, momentum=0.01)
    quantize = True

    module = SequentialBottleNeck(
        layer_nums=layer_nums,
        ds_layer_strides=ds_layer_strides,
        ds_num_filters=ds_num_filters,
        us_layer_strides=us_layer_strides,
        us_num_filters=us_num_filters,
        num_input_features=num_input_features,
        bn_kwargs=bn_kwargs,
        quantize=quantize,
        ds_maxpool=False,
        ds_avgpool=True,
        use_scnet=use_scnet,
        use_res2net=use_res2net,
        use_repvgg=use_repvgg,
        use_secnet=use_secnet,
        use_tconv=use_tconv,
    )
    if use_plugin_quan:
        module.fuse_model()
        module.set_qconfig()
        horizon.quantization.prepare_qat(module, inplace=True)

    output, stride_features = module(input_data)

    output_channle = ds_num_filters[-1] + ds_num_filters[-2]
    assert output.shape == (1, output_channle, 32, 32)
    assert len(stride_features) == 4
