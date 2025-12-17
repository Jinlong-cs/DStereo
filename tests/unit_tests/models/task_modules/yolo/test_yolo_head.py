import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test, qtensor_test


def test_yolo_head():
    anchors = [
        [(116, 90), (156, 198), (373, 326)],
        [(30, 61), (62, 45), (59, 119)],
        [(10, 13), (16, 30), (33, 23)],
    ]
    num_classes = 20
    head = dict(
        type="YOLOV3Head",
        bn_kwargs={},
        feature_idx=[-3, -2, -1],
        in_channels_list=[32, 64, 128],
        num_classes=num_classes,
        anchors=anchors,
        reverse_feature=False,
        int16_output=False,
    )

    input_size = 416
    yolo_head = build_from_registry(head)
    x = [
        torch.randn((1, 32, input_size // 8, input_size // 8)),
        torch.randn((1, 64, input_size // 16, input_size // 16)),
        torch.randn((1, 128, input_size // 32, input_size // 32)),
    ]
    yolo_head.eval()
    y = yolo_head(x)
    assert len(y) == 3
    assert y[0].shape[-1] == 52
    assert y[1].shape[-1] == 26
    assert y[2].shape[-1] == 13

    yolo_head.train()
    y = yolo_head(x)
    assert isinstance(y, list)
    assert isinstance(y[0], torch.Tensor)
    assert y[0].size(0) == 1

    x = qtensor_test(x)
    qat_test(yolo_head, x, with_quantized=False)
