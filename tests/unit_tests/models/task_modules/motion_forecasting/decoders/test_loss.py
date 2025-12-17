import torch

from hat.models.task_modules.motion_forecasting.decoders.densetnt.loss import (
    DensetntLoss,
)


def test_densetnt_loss():
    loss = DensetntLoss()

    goals_target = {
        "goals_preds": torch.rand(1, 1, 1, 2048),
        "goals_labels": torch.ones(1).long(),
    }

    traj_target = {
        "traj_preds": torch.rand(1, 1, 30, 2),
        "traj_labels": torch.rand(1, 1, 30, 2),
    }

    losses = loss(
        goals_target,
        traj_target,
    )
    assert "goals_loss" in losses
    assert "traj_loss" in losses
