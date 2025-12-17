import torch

from hat.models.necks.yolov3_group import YoloGroupNeck
from tests.unit_tests.models.base import qat_test, qtensor_test


def test_yolo_group_neck():
    neck = YoloGroupNeck(
        head_group=True,
        backbone_idx=[-1, -2, -3],
        in_channels_list=[128, 64, 32],
        out_channels_list=[512, 256, 128],
        bn_kwargs={},
    )
    input_size = 416
    x = [
        torch.randn((1, 16, input_size // 2, input_size // 2)),
        torch.randn((1, 16, input_size // 4, input_size // 4)),
        torch.randn((1, 32, input_size // 8, input_size // 8)),
        torch.randn((1, 64, input_size // 16, input_size // 16)),
        torch.randn((1, 128, input_size // 32, input_size // 32)),
    ]
    y = neck(x)
    assert len(y) == 3
    assert y[0].shape[1] == 512 * 2
    assert y[1].shape[1] == 256 * 2
    assert y[2].shape[1] == 128 * 2

    x = qtensor_test(x)
    qat_test(neck, x, with_quantized=False)
