# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.losses.mse_loss import MSELoss

pred = torch.rand(3, 5, 20, 20) * 5
label = torch.rand(3, 5, 20, 20) * 5

loss_weight = torch.rand(3, 5, 20, 20)
valid_mask = torch.ones(3, 5, 20, 20)


@pytest.mark.parametrize(
    ["clip_val", "weight"],
    [
        pytest.param(None, None),
        pytest.param(1.5, loss_weight),
        pytest.param(2, None),
        pytest.param(None, loss_weight),
    ],
)
def test_mse_loss(clip_val, weight):
    mse_loss = MSELoss(clip_val=clip_val)

    _pred = torch.tensor(pred.clone(), requires_grad=True)
    _label = label.clone()

    loss = mse_loss(_pred, _label, valid_mask=valid_mask, weight=weight).sum()
    loss.backward()

    # check gradient
    grad = _pred.grad

    diff = pred - label

    # handle clip value
    if clip_val:
        diff[diff > clip_val] = clip_val
        diff[diff < -clip_val] = -clip_val

    # handle weight
    if weight is not None:
        diff *= weight

    assert torch.all(torch.abs(grad - diff) < 1e-5)
