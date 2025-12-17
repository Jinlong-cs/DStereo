import torch

from hat.models.losses.planning_loss import MDPValueLoss


def test_mdp_loss():

    bs = 8
    horizon = 50
    action_len = 9
    grid_dim = 65

    pi = (
        torch.randn(bs, horizon, action_len, grid_dim, grid_dim)
        .float()
        .sigmoid()
    )
    pi_target = torch.randn(
        bs, horizon, action_len, grid_dim, grid_dim
    ).sigmoid()
    pi_target = pi_target > 0.9
    svf = torch.randn(bs, 1, grid_dim, grid_dim).float()
    svf_target = torch.randn(bs, 1, grid_dim, grid_dim).float()

    pred = {
        "reward_head_pi": pi,
        "reward_head_svf": svf,
    }

    targets = {"plan_bc_targets": pi_target, "plan_svf_path": svf_target}

    mdp_loss = MDPValueLoss(
        pi_weight=1.0,
        svf_weight=1.0,
        reduction="mean",
        avg_factor=bs,
    )

    output = mdp_loss(pred, targets)

    assert "pi_loss" in output
    assert "svf_loss" in output
