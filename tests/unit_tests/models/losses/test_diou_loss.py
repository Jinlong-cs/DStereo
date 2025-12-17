# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.losses.diou_loss import DIoULoss


def test_diou_loss():
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
    target_loss = torch.FloatTensor([1.0691]).squeeze()
    weight = torch.Tensor([0.6424, 0.1823, 0.6099, 0.1039, 0.0894])

    diou_loss = DIoULoss()
    loss = diou_loss(pred, target, weight)

    assert torch.abs(loss - target_loss) < 1e-4


def test_3d_diou_loss():
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
            [1.2647, 4.9559, 5.4663, 1.2403, 1.8673],
            [1.2224, 1.0788, 1.5603, 1.3389, 46.0000],
        ]
    )

    diou_loss = DIoULoss(reduction="none")
    loss = diou_loss(pred, target)

    assert torch.all(torch.abs(loss - target_loss) < 1e-4)
