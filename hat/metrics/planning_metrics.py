# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Union

import torch
from scipy.stats import wasserstein_distance as w_dist

from hat.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = ["PlanningRewardMetric"]


@OBJECT_REGISTRY.register
class PlanningRewardMetric(EvalMetric):
    """Calculate validation metrics for imitation learning. Only heads \
    in SUPPORTED_HEADS is allowed.

    Args:
        head_names: the dict of used head names.
        name: name of this metric instance for display. It should be
            consistent with the key's prefix of prediction outputs.
    """

    def __init__(
        self,
        name: Union[List[str], str],
    ):

        self.name = name
        super(PlanningRewardMetric, self).__init__(self.name)

    def _init_states(self):
        """Initialize state variables."""
        self.add_state(
            "svf_diff",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state(
            "cdf_dist",
            default=[],
            dist_reduce_fx="cat",
        )

    def update(self, output):
        """Update internal buffer with the latest prediction results."""
        pred_svf = output["reward_head_svf"]
        target_svf = output["plan_svf_path"]

        batch_svf_diff = torch.abs(pred_svf - target_svf).mean()

        batch_cdf_dist = torch.tensor(
            w_dist(pred_svf.reshape(-1).cpu(), target_svf.reshape(-1).cpu())
        )
        batch_cdf_dist = batch_cdf_dist.float().to(batch_svf_diff.device)

        self.svf_diff.append(batch_svf_diff)
        self.cdf_dist.append(batch_cdf_dist)

    def reset(self):
        """Clear and rest states after each iteration of validation."""
        self.svf_diff = []
        self.was_disp = []

    def compute(self):
        """Override this method to compute final results from metric states.

        All states variables registered with `self.add_state` are synchronized
        across devices before the execution of this method.
        """
        values = [self.svf_diff.mean().item(), self.cdf_dist.mean().item()]

        return values
