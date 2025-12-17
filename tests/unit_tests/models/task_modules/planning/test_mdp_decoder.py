import torch

from hat.models.task_modules.planning.mdp_decoder import MDPValueDecoder


def test_mdp_decoder():

    bs = 8
    grid_dim = 65
    action_len = 9
    horizon = 50

    model = MDPValueDecoder(
        action_len=9,
        mdp_horizon=50,
        initial_state=[45, 32],
        grid_dim=[65, 65],
    )
    reward = torch.randn(bs, 1, grid_dim, grid_dim)
    goal = torch.randn(bs, 1, grid_dim, grid_dim)

    data = {
        "reward": reward,
        "plan_svf_goal": goal,
    }

    output = model(data)

    assert "pi" in output
    assert "svf" in output
    assert output["pi"].shape == (bs, action_len * horizon, grid_dim, grid_dim)
    assert output["svf"].shape == (bs, 1, grid_dim, grid_dim)
