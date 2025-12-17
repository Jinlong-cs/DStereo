# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.losses.softmax_ce_loss import SoftmaxCELoss

pred = torch.Tensor(
    [
        [
            [0.9140, 0.3939, 0.8436, 0.8709, 0.3387],
            [0.7222, 0.1991, 0.7227, 0.0187, 0.8135],
            [0.5644, 0.7513, 0.0774, 0.3974, 0.2366],
        ],
        [
            [0.8459, 0.2419, 0.1675, 0.3813, 0.7049],
            [0.2227, 0.8864, 0.8276, 0.9101, 0.4410],
            [0.6678, 0.2443, 0.5320, 0.6253, 0.0654],
        ],
    ]
)
target = torch.Tensor(
    [
        [
            [0.4135, 0.6855, 0.8427, 0.3257, 0.9094],
            [0.1176, 0.4234, 0.7032, 0.2256, 0.4893],
            [0.4502, 0.8952, 0.6433, 0.0331, 0.3308],
        ],
        [
            [0.9196, 0.7138, 0.6535, 0.2618, 0.0073],
            [0.3535, 0.3894, 0.0118, 0.0079, 0.0777],
            [0.0233, 0.7960, 0.5167, 0.4476, 0.1777],
        ],
    ]
)
weight = torch.Tensor([0.6424, 0, 0.6099, 0.1039, 0.0894])

target_loss = torch.FloatTensor([0.1456]).squeeze()


def test_softmax_ce_loss():
    softmax_ce_loss = SoftmaxCELoss(dim=1, loss_weight=1.0, reduction="mean")
    loss = softmax_ce_loss(pred, target, weight=weight)

    assert torch.abs(loss - target_loss) < 1e-4
