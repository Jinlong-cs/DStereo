import horizon_plugin_pytorch as horizon
import pytest
import torch

from hat.models.task_modules.lidar import AfdetHead

# from tests.unit_tests.models.base import qat_test, qtensor_test


@pytest.mark.parametrize("use_plugin_quan", [False, True])
def test_AfdetHead(use_plugin_quan: bool):

    in_channels = sum([96, 96, 96])
    common_heads = {
        "reg": (2, 2),
        "height": (1, 2),
        "dim": (3, 2),
        "rot": (2, 2),
    }

    model = AfdetHead(
        num_class=3,
        in_channels=in_channels,
        common_heads=common_heads,
        share_conv_channel=48,
        quantize=True,
    )

    if use_plugin_quan:
        model.fuse_model()
        model.set_qconfig()
        horizon.quantization.prepare_qat(model, inplace=True)

    input = torch.randn((1, in_channels, 256, 256))
    stride_feats = [
        torch.randn((1, 16, 256, 256)),
        torch.randn((1, 16, 128, 128)),
        torch.randn((1, 32, 64, 64)),
        torch.randn((1, 64, 32, 32)),
    ]
    ret_dicts = model(input, stride_feats)

    assert "reg" in ret_dicts
    assert "height" in ret_dicts
    assert "dim" in ret_dicts

    # TODO by shijie.sun
    # make issues:
    # 1.how to pass qat_test.
    # 2.how to rewrite forward.
    # input = qtensor_test(input)
    # stride_feats =qtensor_test(stride_feats)
    # input_tuple = [input,stride_feats]
    # input_dict =dict(input_tensor=input,
    #     stride_feats=stride_feats)

    # qat_test(model, input_dict, with_quantized=False)
