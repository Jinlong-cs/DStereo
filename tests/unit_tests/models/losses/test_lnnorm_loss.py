# Copyright (c) Horizon Robotics. All rights reserved.
import torch

from hat.models.losses.lnnorm_loss import LnNormLoss


def test_lnnorm_loss():
    output = torch.Tensor(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ],
                [
                    [1, 1],
                    [2, 2],
                ],
            ]
        ]
    )

    label = torch.Tensor(
        [
            [
                [
                    [-1, 1],
                    [2, 3],
                ],
                [
                    [0, 1],
                    [2, 1],
                ],
            ]
        ]
    )

    loss = LnNormLoss()
    computed_result = loss(output, label)
    expected_result = torch.FloatTensor([5.650281539])

    assert torch.allclose(expected_result, computed_result, rtol=0.001)
