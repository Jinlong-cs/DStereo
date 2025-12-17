# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.losses.ciou_loss import CIoULoss


def test_ciou_loss():
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
    target_loss = torch.FloatTensor([8.1395e-05]).squeeze()
    weight = torch.Tensor([0.6424, 0.1823, 0.6099, 0.1039, 0.0894])
    # 8.1395e-05
    ciou_loss = CIoULoss("loss_bbox")
    loss = ciou_loss(pred, target, weight, avg_factor)
    assert torch.abs(loss["loss_bbox"] - target_loss) < 1e-4
