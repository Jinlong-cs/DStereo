# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Optional

import torch

from hat.models.losses.utils import weight_reduce_loss
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "MDPValueLoss",
]


@OBJECT_REGISTRY.register
class MDPValueLoss(torch.nn.Module):
    """Calculate the loss of markov process outputs.

    The loss is supervised from experts demonstrations,
    including binary cross entropy and l1 loss.

    Args:
        pi_weight:
        svf_weight:

    Returns:
        losses dict with 'pi_loss' and 'svf_loss'.

    """

    def __init__(
        self,
        pi_weight: float = 1.0,
        svf_weight: float = 1.0,
        reduction: str = "mean",
        avg_factor: Optional[float] = None,
    ):
        super(MDPValueLoss, self).__init__()
        self.pi_weight = pi_weight
        self.svf_weight = svf_weight
        self.reduction = reduction
        self.avg_fac = avg_factor

    def forward(self, pred, target=None):
        """Forward."""
        pi_pred = pred["reward_head_pi"]
        pi_target = target["plan_bc_targets"]
        pi_pred = pi_pred.reshape(pi_target.shape)

        pi_loss = -pi_pred[pi_target].log()
        pi_loss = weight_reduce_loss(
            pi_loss, self.pi_weight, self.reduction, self.avg_fac
        )

        # svf path L1 loss for auxiliary
        svf_pred = pred["reward_head_svf"]
        svf_target = target["plan_svf_path"]

        svf_loss = torch.abs(svf_pred - svf_target)
        svf_loss = weight_reduce_loss(
            svf_loss, self.svf_weight, self.reduction
        )

        results = {"pi_loss": pi_loss, "svf_loss": svf_loss}

        return results
