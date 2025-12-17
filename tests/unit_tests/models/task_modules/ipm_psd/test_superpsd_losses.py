# Copyright (c) Horizon Robotics. All rights reserved.

import torch

from hat.models.losses.focal_loss import GaussianFocalLoss


def test_bce_loss():

    guassian_focal_loss = GaussianFocalLoss()
    input = torch.randn(1, 4, 100, 100, requires_grad=True)
    input = torch.sigmoid(input)
    target = torch.randint(2, (1, 4, 100, 100))
    loss = guassian_focal_loss(input, target).sum()
    loss.backward()
