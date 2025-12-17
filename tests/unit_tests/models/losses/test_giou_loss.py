# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.losses.giou_loss import GIoULoss


def test_giou_loss():
    pred = torch.Tensor(
        [
            [6, 10, 12, 15],
            [16, 7, 8, 11],
            [11, 18, 3, 8],
            [14, 1, 10, 3],
            [19, 12, 13, 19],
        ]
    )
    target = torch.Tensor(
        [
            [10, 8, 11, 8],
            [7, 8, 4, 0],
            [19, 2, 15, 1],
            [10, 4, 16, 13],
            [18, 19, 3, 19],
        ]
    )
    avg_factor = 20000
    target_loss = torch.FloatTensor([187.7400]).squeeze()
    weight = torch.Tensor([0.6424, 0.1823, 0.6099, 0.1039, 0.0894])

    giou_loss = GIoULoss("loss_bbox")
    loss = giou_loss(pred, target, weight, avg_factor)
    assert torch.abs(loss["loss_bbox"] - target_loss) < 1e-4


def test_3d_giou_loss():
    pred = torch.Tensor(
        [
            [
                [6, 10, 12, 15],
                [16, 7, 8, 11],
                [11, 18, 3, 8],
                [14, 1, 10, 3],
                [19, 12, 13, 19],
            ],
            [
                [3, 7, 11, 13],
                [1, 6, 8, 16],
                [24, 11, 23, 18],
                [14, 1, 10, 3],
                [15, 10, 3, 9],
            ],
        ]
    )
    target = torch.Tensor(
        [
            [
                [10, 8, 11, 8],
                [7, 8, 4, 0],
                [19, 2, 15, 1],
                [10, 4, 16, 13],
                [18, 19, 3, 19],
            ],
            [
                [12, 9, 13, 11],
                [4, 6, 5, 8],
                [20, 21, 18, 1],
                [11, 5, 14, 10],
                [19, 20, 5, 11],
            ],
        ]
    )
    target_loss = torch.Tensor(
        [
            [1.2857, 4.0000, -1.5000, 1.3611, 4.2000e07],
            [1.1667, 0.97143, 0.42857, 1.7407, -1.3800e08],
        ]
    )

    giou_loss = GIoULoss("loss_bbox", reduction="none")
    loss = giou_loss(pred, target)

    assert torch.all(torch.abs(loss["loss_bbox"] - target_loss) < 1e-4)
