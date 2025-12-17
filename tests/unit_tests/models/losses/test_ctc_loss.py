# Copyright (c) Horizon Robotics. All rights reserved.
import torch

from hat.models.losses.ctc_loss import FocalCTCLoss, GHMCTCLoss


def test_ghm_ctc_loss():
    """Tests :py:class:`hat.models.losses.ctc_loss.GHMCTCLoss` class."""

    # Test initialization
    ghm_ctc_loss = GHMCTCLoss(
        bins=10,
        momentum=0.9,
        reduction="sum",
        zero_infinity=True,
    )
    probs = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.5, 0.5, 0.0, 0.0]],
            [[0.5, 0.5, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.5, 0.0, 0.5]],
            [[0.0, 0.0, 0.0, 0.5, 0.5]],
            [[0.0, 0.0, 0.5, 0.0, 0.5]],
        ]
    )
    log_probs = torch.log(probs)
    target = torch.tensor([[0, 1, 2, 3, 4]])
    input_lengths = torch.tensor([6])
    target_lengths = torch.tensor([5])

    # Test forward
    loss = ghm_ctc_loss(
        log_probs,
        target,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
    )
    assert loss - 0.0959 < 1e-4


def test_focal_ctc_loss():
    """Tests :py:class:`hat.models.losses.ctc_loss.FocalCTCLoss` class."""

    # Test initialization
    focal_ctc_loss = FocalCTCLoss(
        alpha=0.25,
        gamma=0.5,
        reduction="sum",
        zero_infinity=True,
    )
    probs = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.5, 0.5, 0.0, 0.0]],
            [[0.5, 0.5, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.5, 0.0, 0.5]],
            [[0.0, 0.0, 0.0, 0.5, 0.5]],
            [[0.0, 0.0, 0.5, 0.0, 0.5]],
        ]
    )
    log_probs = torch.log(probs)
    target = torch.tensor([[0, 1, 2, 3, 4]])
    input_lengths = torch.tensor([6])
    target_lengths = torch.tensor([5])

    # Test forward
    loss = focal_ctc_loss(
        log_probs,
        target,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
    )
    assert loss - 0.6711 < 1e-4
