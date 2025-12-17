# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.losses.hinge_loss import (
    ElementwiseL1HingeLoss,
    ElementwiseL2HingeLoss,
    WeightedSquaredHingeLoss,
)

pred = torch.rand(3, 5, 20, 20)
label = torch.randint(2, (3, 5, 20, 20)).float()
weight = torch.randint(2, (3, 5, 20, 20)).float()


@pytest.mark.parametrize(
    ["loss_bound"],
    [
        pytest.param(0.7),
        pytest.param(0),
    ],
)
def test_elt_l1_hingeloss(loss_bound):
    elt_loss = ElementwiseL1HingeLoss(
        loss_bound_l1=loss_bound,
        reduction="none",
    )

    _pred = pred.clone()
    _label = label.clone()
    _weight = weight.clone()

    loss = elt_loss(_pred, _label, weight=weight)

    # effect of weight
    assert loss[_weight == 0].sum() == 0

    # hinge loss
    assert (loss[(_pred >= _label) * (_label >= 1)]).sum() == 0
    assert (loss[(_pred <= _label) * (_label <= 0)]).sum() == 0

    # loss_bound
    if loss_bound > 0:
        assert torch.all(torch.abs(loss) <= loss_bound)
    else:
        assert torch.all(torch.abs(loss) <= 1)


@pytest.mark.parametrize(
    ["loss_bound"],
    [
        pytest.param(0.7),
        pytest.param(0),
    ],
)
def test_elt_l2_hingeloss(loss_bound):
    elt_loss = ElementwiseL2HingeLoss(
        loss_bound_l1=loss_bound,
        reduction="none",
    )

    _pred = pred.clone()
    _label = label.clone()
    _weight = weight.clone()

    loss = elt_loss(_pred, _label, weight=weight)

    # effect of weight
    assert loss[_weight == 0].sum() == 0

    # hinge loss
    assert (loss[(_pred >= _label) * (_label >= 1)]).sum() == 0
    assert (loss[(_pred <= _label) * (_label <= 0)]).sum() == 0

    # loss_bound
    if loss_bound > 0:
        assert torch.all(torch.abs(loss) <= loss_bound ** 2)
    else:
        assert torch.all(torch.abs(loss) <= 1)


def test_weighted_squared_hingeloss():
    # no test case yet
    _ = WeightedSquaredHingeLoss(reduction="mean")
