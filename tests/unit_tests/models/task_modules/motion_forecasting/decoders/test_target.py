import torch

from hat.models.task_modules.motion_forecasting.decoders.densetnt.target import (  # noqa
    DensetntTarget,
)


def test_densetnt_target():
    target = DensetntTarget()
    goals_preds = torch.randn(1, 1, 1, 2048)
    traj_preds = torch.randn(1, 1, 30, 2)
    data = {
        "goals_2d_labels": torch.randn(1, 2048),
        "goals_2d_mask": torch.randn(1, 2048),
        "traj_labels": torch.rand(1, 30, 2),
    }

    target(goals_preds, traj_preds, data)
