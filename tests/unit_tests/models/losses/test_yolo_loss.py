# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.losses.yolo_losses import YOLOV3Loss


def test_yolo_loss():
    num_classes = 20
    anchors = [
        [(10, 13), (16, 30), (33, 23)],
        [(30, 61), (62, 45), (59, 119)],
        [(116, 90), (156, 198), (373, 326)],
    ]
    yolo_loss = YOLOV3Loss(
        num_classes=num_classes,
        anchors=anchors,
        strides=[8, 16, 32],
        ignore_thresh=0.5,
        loss_xy=torch.nn.BCELoss(reduce=False),
        loss_wh=torch.nn.L1Loss(reduce=False),
        loss_conf=torch.nn.BCELoss(reduction="sum"),
        loss_cls=torch.nn.BCELoss(reduction="sum"),
        lambda_loss=[2.0, 2.0, 1.0, 1.0],
    )

    input = [
        torch.rand(2, 75, 52, 52).cuda(),
        torch.rand(2, 75, 26, 26).cuda(),
        torch.rand(2, 75, 13, 13).cuda(),
    ]

    data = dict(
        gt_bboxes=[
            torch.tensor(
                [
                    [196.4924, 286.8647, 235.0429, 348.6652],
                    [261.2227, 299.4708, 290.8548, 349.5876],
                ]
            ),
            torch.tensor([[71.0866, 0.0000, 257.3105, 297.1428]]),
        ],
        gt_classes=[
            torch.tensor([3.0, 3.0]),
            torch.tensor([14.0]),
        ],
    )
    y = yolo_loss(input, (data["gt_bboxes"], data["gt_classes"]))
    assert isinstance(y, torch.Tensor)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
