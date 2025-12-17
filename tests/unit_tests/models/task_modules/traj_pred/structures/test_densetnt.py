import torch

from hat.models.task_modules.traj_pred.backbones.vectornet_backbone import (
    VectorNetBackbone,
)
from hat.models.task_modules.traj_pred.heads.densetnt_head import DenseTNTHead
from hat.models.task_modules.traj_pred.structures.DenseTNT import DenseTNT
from hat.utils.qconfig_manager import QconfigMode, set_qconfig_mode


def gen_example_densetnt_model():

    test_input = {
        "nearest_goal_idxs": torch.randn((32, 1, 1, 1)),
        "nearest_road_idxs": torch.randn((32, 1, 1, 1)),
        "future_trajectories": torch.randn((32, 1, 12, 2)),
        "goal_coords": torch.randn((32, 2, 1, 2048)),
        "valid_masks": torch.randn((32, 12)),
        "struct_road_feats": torch.randn([32, 7, 128, 9]),
        "struct_traj_feats": torch.randn([32, 6, 32, 3]),
        "struct_road_masks": torch.randn([32, 128]),
        "struct_traj_masks": torch.randn([32, 32]),
        "end_points": torch.randn([32, 2, 1, 1]),
        "ctx_trajectories": None,
        "vis_lanes": None,
        "file_names": None,
        "attention_mask": torch.ones([32, 1, 128 + 32, 128 + 32]).float(),
    }

    encoder = VectorNetBackbone(
        road_feat_channels=7,
        road_polyline_len=9,
        road_polyline_num=128,
        traj_feat_channels=6,
        traj_polyline_len=3,
        traj_polyline_num=32,
        num_hidden_units=128,
        num_attn_hidden_units=128,
        num_sub_graph_layers=3,
        num_attention_heads=4,
        use_out_fc=True,
        fc_out_channels=128,
        output_all_feats=True,
    )

    decoder = {
        "heads": DenseTNTHead(
            k_values=[1],
            in_channels=128,
            hidden_size=128,
            traj_feat_hidden_size=128,
            road_feat_hidden_size=128,
            traj_feat_num=32,
            road_feat_num=128,
            num_goals=2048,
            num_dyn_obs=32,
            num_road_ele=128,
            num_fut_frame=30,
            goal_coords_scale=0.04,
            use_lane_scoring=False,
        )
    }

    net = DenseTNT(
        backbone=encoder,
        heads=decoder,
        is_int_infer_model=True,
    )
    return net, test_input


def test_densetnt_structure():

    k_values = [1]
    num_goals = 2048
    set_qconfig_mode(QconfigMode.QAT)
    model, test_input = gen_example_densetnt_model()
    model.fuse_model()
    model.set_qconfig()
    output = model(test_input)

    assert hasattr(output, "predict_trajs")
    assert hasattr(output, "real_scores")

    predict_trajs = output[0]
    predict_goal_scores = output[1]

    assert (
        predict_goal_scores is None
        or predict_goal_scores.shape[3] == num_goals
    )
    assert predict_trajs is None or predict_trajs.shape[1] == max(k_values)
