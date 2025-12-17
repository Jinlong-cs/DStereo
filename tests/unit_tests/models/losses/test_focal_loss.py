# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.losses.focal_loss import FocalLoss, FocalLossV2


def test_focal_loss():
    alpha_list = [0.25, 0.5]
    gamma_list = [2.0, 3.0]
    pred = torch.Tensor(
        [
            [-1.2061],
            [0.0617],
            [1.1632],
            [-1.5008],
            [-1.5944],
            [-0.0187],
            [-2.1325],
            [-0.5270],
            [-0.1021],
            [0.0099],
        ]
    )
    target = torch.Tensor([1, 1, 1, 1, 0, 0, 0, 1, 1, 0]).type(torch.int64)
    target_loss = [
        torch.FloatTensor([0.4461]).squeeze(),
        torch.FloatTensor([0.4540]).squeeze(),
    ]
    for ii, (alpha, gamma) in enumerate(zip(alpha_list, gamma_list)):
        focal_loss = FocalLoss("loss_cls", 2, alpha, gamma)
        loss = focal_loss(pred, target, avg_factor=4)
        assert torch.abs(target_loss[ii] - loss["loss_cls"]) < 1e-4


@pytest.mark.parametrize(
    ["from_logits"],
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
def test_focal_loss_v2(from_logits):
    alpha_list = [0.25, 0.5]
    gamma_list = [2.0, 3.0]
    loss_weight_list = [1.0, 2.0]
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
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [-1, -1, -1, -1, -1],
            ],
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )
    weight = torch.Tensor(
        [
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [False, False, False, False, False],
            ],
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ],
        ]
    )
    if from_logits:
        target_loss = [
            torch.FloatTensor([2.5831]).squeeze(),
            torch.FloatTensor([3.4000]).squeeze(),
        ]
    else:
        target_loss = [
            torch.FloatTensor([1.5012]).squeeze(),
            torch.FloatTensor([1.3462]).squeeze(),
        ]

    for ii, (alpha, gamma, loss_weight) in enumerate(
        zip(alpha_list, gamma_list, loss_weight_list)
    ):
        focal_loss_v2 = FocalLossV2(
            alpha, gamma, loss_weight, from_logits=from_logits
        )
        loss = focal_loss_v2(pred, target, weight, 5)
        assert torch.abs(target_loss[ii] - loss) < 1e-4
