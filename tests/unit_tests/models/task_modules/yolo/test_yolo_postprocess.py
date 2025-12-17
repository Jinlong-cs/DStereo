# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.task_modules.yolo import YOLOV3PostProcess


def test_yolo_postprocess():
    anchors = [
        [(116, 90), (156, 198), (373, 326)],
        [(30, 61), (62, 45), (59, 119)],
        [(10, 13), (16, 30), (33, 23)],
    ]

    bs = 3
    input_size = 416
    num_classes = 20
    yolo_pp = YOLOV3PostProcess(
        anchors=anchors,
        strides=[32, 16, 8],
        num_classes=num_classes,
        score_thresh=0.01,
        nms_thresh=0.45,
        topK=20,
    )
    x = [
        torch.randn((bs, 75, input_size // 32, input_size // 32)),
        torch.randn((bs, 75, input_size // 16, input_size // 16)),
        torch.randn((bs, 75, input_size // 8, input_size // 8)),
    ]
    y = yolo_pp(x)

    assert len(y) == bs
    assert y[0].shape[1] <= 20  # max TopK
