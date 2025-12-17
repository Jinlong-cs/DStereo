# Copyright (c) Horizon Robotics. All rights reserved.
import torch

from hat.models.losses.lidar_losses import (
    AfdetFocalLoss,
    BoundaryLoss,
    LidarFastFocalLoss,
    LidarRegLoss,
    LidarSmoothL1RegLoss,
    WeightedSmoothL1Loss,
)


def test_lidar_reg_loss():
    output = torch.Tensor(
        [
            [
                [
                    [0.8369, 0.8963, 0.2457],
                    [0.3743, 0.4708, 0.4358],
                    [0.4356, 0.2927, 0.6098],
                ],
                [
                    [0.8484, 0.5972, 0.4551],
                    [0.2372, 0.8343, 0.7602],
                    [0.0240, 0.7468, 0.8073],
                ],
            ]
        ]
    )

    mask = torch.ones(1, 5)
    ind = torch.zeros(1, 5, dtype=torch.int64)
    target = torch.ones(1, 5, 2)

    loss = LidarRegLoss()
    computed_result = loss(output, mask, ind, target)

    expected_result = torch.FloatTensor([0.1631, 0.1516])

    assert torch.allclose(expected_result, computed_result, rtol=0.0001)


def test_lidar_fast_focal_loss():
    output = torch.Tensor(
        [
            [
                [
                    [0.6521, 0.3426, 0.6092],
                    [0.1880, 0.5562, 0.6506],
                    [0.2064, 0.1235, 0.3754],
                ]
            ]
        ]
    )
    target = torch.zeros(1, 1, 3, 3)
    target[0, 0, 1, 1] = 1.0
    ind = torch.zeros(1, 3).type(torch.int64)
    mask = torch.ones(1, 3)
    cat = torch.zeros(1, 3).type(torch.int64)

    loss = LidarFastFocalLoss()
    computed_result = loss(output, target, ind, mask, cat)

    expected_result = torch.FloatTensor([0.5109]).squeeze()

    assert torch.allclose(expected_result, computed_result, rtol=0.0001)


def test_lidar_smooth_l1_reg_loss():
    output = torch.Tensor(
        [
            [
                [
                    [0.8809, 0.0015, 0.7214],
                    [0.0344, 0.3668, 0.6909],
                    [0.2287, 0.3963, 0.6871],
                ]
            ]
        ]
    )
    target = torch.Tensor([[[0.0107], [0.3450], [0.6716]]])
    ind = torch.zeros(1, 3).type(torch.int64)
    mask = torch.ones(1, 3)

    loss = LidarSmoothL1RegLoss()
    computed_result = loss(output, mask, ind, target)

    expected_result = torch.FloatTensor([0.1814])

    assert torch.allclose(expected_result, computed_result, rtol=0.001)


def test_AfdetFocalLoss():
    output = torch.Tensor(
        [
            [
                [
                    [0.8809, 0.0015, 0.7214],
                    [0.0344, 0.3668, 0.6909],
                    [0.2287, 0.3963, 0.6871],
                ]
            ]
        ]
    )
    target = torch.Tensor(
        [
            [
                [
                    [0.8809, 0.0015, 0.7214],
                    [0.0344, 0.3668, 0.6909],
                    [0.2287, 0.3963, 0.6871],
                ]
            ]
        ]
    )

    loss = AfdetFocalLoss()
    computed_result = loss(output, target)

    expected_result = torch.FloatTensor([0.04])

    assert torch.allclose(expected_result, computed_result, rtol=0.001)


def test_BoundaryLoss():
    output = torch.Tensor(
        [
            [
                [
                    [0.8809, 0.0015, 0.7214],
                    [0.0344, 0.3668, 0.6909],
                    [0.2287, 0.3963, 0.6871],
                ]
            ]
        ]
    )
    target = torch.Tensor(
        [
            [
                [
                    [0.8809, 0.0015, 0.7214],
                    [0.0344, 0.3668, 0.6909],
                    [0.2287, 0.3963, 0.6871],
                ]
            ]
        ]
    )

    loss = BoundaryLoss()
    computed_result = loss(output, target)

    expected_result = torch.FloatTensor([1.0])

    assert torch.allclose(expected_result, computed_result, rtol=0.001)


def test_lidar_weighted_smoothL1_loss():

    hm_pred_stu = torch.Tensor(
        [
            [0.8809, 0.0015, 0.7214],
            [0.0344, 0.3668, 0.6909],
            [0.2287, 0.3963, 0.6871],
        ]
    )

    hm_pred_tea = torch.Tensor(
        [
            [0.8809, 0.0015, 0.7214],
            [0.0344, 0.3668, 0.6909],
            [0.2287, 0.3963, 0.6871],
        ]
    )

    loss = WeightedSmoothL1Loss()
    computed_result = loss(hm_pred_stu, hm_pred_tea)
    expected_result = torch.zeros(3, 3).type(torch.float32)

    assert torch.allclose(expected_result, computed_result, rtol=0.0001)
