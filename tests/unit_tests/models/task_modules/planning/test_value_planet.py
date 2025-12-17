from collections import OrderedDict

import torch

from hat.models.task_modules.planning.mdp_decoder import MDPValueDecoder
from hat.models.task_modules.planning.reward_encoder import (
    PatialResNet,
    RewardModel,
)
from hat.models.task_modules.planning.value_planet import ValuePlanNet


def test_value_planet():

    bs = 4
    grid_dim = 65
    action_len = 9
    horizon = 50
    img_h, img_w = 512, 512

    model = ValuePlanNet(
        backbone=PatialResNet(
            input_channels=5, bn_kwargs=dict(eps=1e-5, momentum=0.1)
        ),
        neck=RewardModel(
            in_feat_size=256,
            scene_feat_size=128,
            agg_size=[32, 32],
            grid_cell_size=[2, 2],
        ),
        heads=OrderedDict(
            reward_head=MDPValueDecoder(
                action_len=action_len,
                mdp_horizon=horizon,
                initial_state=[45, 32],
                grid_dim=[grid_dim, grid_dim],
                value=-100,
            ),
        ),
        road_map_colored=False,
    )

    road_map = torch.randn(bs, 1, img_h, img_w)
    obs_map = torch.randn(bs, 4, img_h, img_w)
    goal = torch.randn(bs, 1, grid_dim, grid_dim)

    data = {
        "road_map": road_map,
        "rendered_obs": obs_map,
        "plan_svf_goal": goal,
    }

    output = model(data)

    assert "feats" in output
    assert "reward" in output
    assert "reward_head_pi" in output
    assert "reward_head_svf" in output
    assert output["feats"].shape == (bs, 128, grid_dim, grid_dim)
    assert output["reward"].shape == (bs, 1, grid_dim, grid_dim)
    assert output["reward_head_pi"].shape == (
        bs,
        action_len * horizon,
        grid_dim,
        grid_dim,
    )
    assert output["reward_head_svf"].shape == (bs, 1, grid_dim, grid_dim)
