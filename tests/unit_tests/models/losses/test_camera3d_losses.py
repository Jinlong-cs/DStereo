# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.losses.camera3d_losses import HMFocalLoss, HML1Loss


@pytest.mark.parametrize(
    ["pred", "gt", "ignore_mask", "target_loss"],
    [
        pytest.param(
            0.98 * torch.ones(1, 1, 128, 240),
            0.98 * torch.ones(1, 1, 128, 240),
            torch.ones(1, 1, 128, 240),
            0.0,
        ),
        pytest.param(
            0.98 * torch.ones(1, 1, 128, 240),
            0.98 * torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            0.0185,
        ),
        pytest.param(
            0.5 * torch.ones(1, 1, 128, 240),
            0.5 * torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            332.7107,
        ),
        pytest.param(
            0.02 * torch.ones(1, 1, 128, 240),
            0.02 * torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            0.2290,
        ),
    ],
)
def test_HMFocalLoss(pred, gt, ignore_mask, target_loss):
    hmfocalloss = HMFocalLoss()
    loss = hmfocalloss(pred, gt, ignore_mask)
    assert torch.abs(loss - target_loss) < 1e-4


@pytest.mark.parametrize(
    [
        "pred",
        "target",
        "weight_mask",
        "ignore_mask",
        "heatmap_type",
        "target_loss",
    ],
    [
        pytest.param(
            torch.zeros(1, 2, 128, 240),
            0.5 * torch.ones(1, 2, 128, 240),
            torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            "dense",
            0.5,
        ),
        pytest.param(
            torch.ones(1, 2, 128, 240),
            torch.ones(1, 2, 128, 240),
            torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            "point",
            0.0,
        ),
        pytest.param(
            torch.zeros(1, 2, 128, 240),
            torch.ones(1, 2, 128, 240),
            torch.ones(1, 1, 128, 240),
            torch.zeros(1, 1, 128, 240),
            "dense",
            1.0,
        ),
    ],
)
def test_HML1Loss(
    pred, target, weight_mask, ignore_mask, heatmap_type, target_loss
):
    hml1loss = HML1Loss(heatmap_type)
    loss = hml1loss(pred, target, weight_mask, ignore_mask)
    assert torch.abs(loss - target_loss) < 1e-4
