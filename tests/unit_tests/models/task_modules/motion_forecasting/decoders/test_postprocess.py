import torch

from hat.models.task_modules.motion_forecasting.decoders.densetnt.post_process import (  # noqa
    DensetntPostprocess,
)


def test_densetnt_postprocess():
    post_process = DensetntPostprocess(
        threshold=2.0, pred_steps=30, mode_num=6
    )
    goals_scores = torch.randn(1, 1, 1, 150)
    traj_preds = torch.randn(1, 60, 1, 150)
    pred_goals = torch.randn(1, 2, 1, 150)

    post_process(goals_scores, traj_preds, pred_goals, {})
