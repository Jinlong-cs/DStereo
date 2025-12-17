# Copyright (c) Horizon Robotics. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class FocalCTCLoss(nn.Module):
    r"""Focal Connectionist Temporal Classification Loss.

    Args:
        alpha: The hyperparameter for focal loss.
        gamma: The hyperparameter for focal loss.
        blank: The blank label index.
        reduction: The reduction method for loss.
        zero_infinity: Whether to zero infinite losses.

    References
    ----------
        `Connectionist Temporal Classification: Labelling Unsegmented
        Sequence Data with Recurrent Neural Networks
        <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_
        <https://www.hindawi.com/journals/complexity/2019/9345861/>
    """

    def __init__(
        self,
        alpha: int = 0.25,
        gamma: int = 0.5,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """Forward function, calculate the loss.

        Args:
            log_probs: Tensor of size (seq_len, batch, num_classes).
            targets: Tensor of size (batch, max_target_len).
            input_lengths: Tensor of size (batch,).
            target_lengths: Tensor of size (batch,).
        """
        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            self.blank,
            "none",
            self.zero_infinity,
        )
        prob = torch.exp(-loss.detach())
        loss = self.alpha * ((1 - prob) ** self.gamma) * loss

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


class GHMCTCLoss(nn.Module):
    r"""GHM Connectionist Temporal Classification Loss.

    Args:
        bins: The number of bins for gradient length.
        momentum: The momentum for moving average.
        blank: The blank label index.
        reduction: The reduction method for loss.
        zero_infinity: Whether to zero infinite losses.

    References
    ----------
        `Connectionist Temporal Classification: Labelling Unsegmented
        Sequence Data with Recurrent Neural Networks
        <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_
        `Gradient Harmonized Single-stage Detector
        <https://arxiv.org/abs/1811.05181>`
    """

    def __init__(
        self,
        bins: int = 10,
        momentum: int = 0.9,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = False,
    ):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        assert reduction in ["mean", "sum", "none"]
        self.zero_infinity = zero_infinity
        self.bins = bins

        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer("edges", edges)
        self.edges[-1] += 1e-6

        self.momentum = momentum
        if momentum > 0:
            acc_sum = torch.ones(bins) * 32
            self.register_buffer("acc_sum", acc_sum)
            self.fist_step = True

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """Forward function, calculate the loss.

        Args:
            log_probs: Tensor of size (seq_len, batch, num_classes).
            targets: Tensor of size (batch, max_target_len).
            input_lengths: Tensor of size (batch,).
            target_lengths: Tensor of size (batch,).
        """
        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            self.blank,
            "none",
            self.zero_infinity,
        )

        # gradient length
        prob = torch.exp(-loss.detach())
        grad = torch.abs(prob - 1.0)
        valid = loss != 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        weights = torch.zeros_like(loss)
        for i in range(self.bins):
            inds = (grad >= self.edges[i]) & (grad < self.edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = (
                        self.momentum * self.acc_sum[i]
                        + (1 - self.momentum) * num_in_bin
                    )
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin

                n += 1
        if n > 0:
            weights = weights / n

        loss = loss * weights

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss
