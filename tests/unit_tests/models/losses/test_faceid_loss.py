# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.losses.faceid import ArcFace, CosFace


def test_arcface():
    pred_list = [
        torch.FloatTensor(
            [
                [0.4714, -1.1909, 1.4327, -0.3126, -0.7205],
                [0.8871, 0.8595, -0.6365, 0.0156, -2.2426],
            ]
        ),
    ]
    target_list = [
        torch.LongTensor([-1, 3]),
    ]
    target_loss = [
        torch.FloatTensor(
            [
                [30.1696, -64.0000, 64.0000, -20.0064, -46.1120],
                [56.7744, 55.0080, -40.7360, -29.8033, -64.0000],
            ]
        ).squeeze(),
    ]

    loss = ArcFace()
    for ii, (pred, target) in enumerate(zip(pred_list, target_list)):
        loss = loss(pred, target)
        assert torch.max(torch.abs((loss - target_loss[ii]))) < 1e-4


def test_cosface():
    pred_list = [
        torch.FloatTensor(
            [
                [0.4714, -1.1909, 1.4327, -0.3126, -0.7205],
                [0.8871, 0.8595, -0.6365, 0.0156, -2.2426],
            ]
        ),
    ]
    target_list = [
        torch.LongTensor([-1, 4]),
    ]
    target_loss = [
        torch.FloatTensor(
            [
                [30.1696, -64.0000, 64.0000, -20.0064, -46.1120],
                [56.7744, 55.0080, -40.7360, 0.9984, -89.6000],
            ]
        ).squeeze(),
    ]

    loss = CosFace()
    for ii, (pred, target) in enumerate(zip(pred_list, target_list)):
        loss = loss(pred, target)
        assert torch.max(torch.abs((loss - target_loss[ii]))) < 1e-4
