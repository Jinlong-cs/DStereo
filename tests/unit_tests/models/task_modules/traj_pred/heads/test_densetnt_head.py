import pytest
import torch

from hat.models.task_modules.traj_pred.heads.densetnt_head import DenseTNTHead


@pytest.mark.parametrize(
    [
        "k_values",
        "in_channels",
        "hidden_size",
        "traj_feat_hidden_size",
        "road_feat_hidden_size",
        "traj_feat_num",
        "road_feat_num",
        "num_goals",
        "num_dyn_obs",
        "num_road_ele",
        "num_fut_frame",
        "goal_coords_scale",
        "data",
    ],
    [
        pytest.param(
            [1],
            128,
            128,
            128,
            128,
            32,
            128,
            2048,
            32,
            128,
            12,
            0.04,
            {
                "graph_feat": torch.randn((32, 128, 1, 1)),
                "traj_feat": torch.randn((32, 128, 1, 32)),
                "road_feat": torch.randn((32, 128, 1, 128)),
                "goal_coords": torch.randn((32, 2, 1, 2048)),
                "end_points": torch.randn((32, 2, 1, 1)),
                "epoch_id": 0,
            },
        ),
    ],
)
def test_densetnt_head(
    k_values,
    in_channels,
    hidden_size,
    traj_feat_hidden_size,
    road_feat_hidden_size,
    traj_feat_num,
    road_feat_num,
    num_goals,
    num_dyn_obs,
    num_road_ele,
    num_fut_frame,
    goal_coords_scale,
    data,
):
    model = DenseTNTHead(
        k_values=k_values,
        in_channels=in_channels,
        hidden_size=hidden_size,
        traj_feat_hidden_size=traj_feat_hidden_size,
        road_feat_hidden_size=road_feat_hidden_size,
        traj_feat_num=traj_feat_num,
        road_feat_num=road_feat_num,
        num_goals=num_goals,
        num_dyn_obs=num_dyn_obs,
        num_road_ele=num_road_ele,
        num_fut_frame=num_fut_frame,
        goal_coords_scale=goal_coords_scale,
        use_lane_scoring=False,
    )
    predictions = model(
        # graph_feats=graph_feats,
        # traj_feats=traj_feats,
        # road_feats=road_feats,
        data=data,
        # is_int_infer_model=True,
    )

    predict_goal_scores = predictions["predict_goal_scores"]
    predict_trajs = predictions["predict_trajs"]

    assert (
        predict_goal_scores is None
        or predict_goal_scores.shape[3] == num_goals
    )
    assert predict_trajs is None or predict_trajs.shape[2] == num_fut_frame
