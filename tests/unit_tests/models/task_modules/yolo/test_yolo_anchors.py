# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.task_modules.yolo import YOLOV3AnchorGenerator


def test_yolo_anchors():
    anchors = [
        [(116, 90), (156, 198), (373, 326)],
        [(30, 61), (62, 45), (59, 119)],
        [(10, 13), (16, 30), (33, 23)],
    ]
    bs = 3
    input_size = 416
    yolo_anchor = YOLOV3AnchorGenerator(
        anchors=anchors,
        strides=(32, 16, 8),
        image_size=(input_size, input_size),
    )

    yolo_anchor.train()
    x = torch.randn((bs, 75, input_size // 8, input_size // 8))
    train_out = yolo_anchor(x)
    assert isinstance(train_out, torch.Tensor)

    x = [
        torch.randn((bs, 75, input_size // 8, input_size // 8)),
        torch.randn((bs, 75, input_size // 16, input_size // 16)),
        torch.randn((bs, 75, input_size // 32, input_size // 32)),
    ]
    yolo_anchor.eval()
    eval_out = yolo_anchor(x)
    assert isinstance(eval_out, list)
    assert len(eval_out) == 3
    assert len(eval_out[0]) == 4
