# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import torch

from hat.models.losses.lovasz_loss import LovaszSoftmaxLoss
from hat.utils.seed import seed_everything


def test_lovasz_softmax_loss():
    seed_everything(17)
    pred = torch.Tensor(np.random.rand(48)).view((2, 2, 3, 4))

    target = torch.Tensor([0])
    target_loss = 0.6386

    loss_lovasz = LovaszSoftmaxLoss(
        classes="present",
        per_image=False,
        loss_weight=1.0,
        ignore_index=255,
        loss_name="loss_lovasz",
    )

    loss = loss_lovasz(pred, target)
    assert torch.abs(loss["loss_lovasz"] - target_loss) < 1e-4
