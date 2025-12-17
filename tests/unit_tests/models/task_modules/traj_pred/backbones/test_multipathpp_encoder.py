import torch

from hat.models.task_modules.traj_pred.backbones.multipathpp_encoder import (
    MTPPlusEncoder,
)


def test_mtpplusencoder():
    """Test Multipath++ encoder.

    The setting is the same as during the 2022 Waymo challenge.
    """
    model = MTPPlusEncoder(
        use_LSTM=False,
        target_pos_vel_split=True,
        target_agent_feat_size=4,
        target_agent_emb_size=16,
        target_agent_mcg_layers=5,
        target_agent_mcg_hidden_size=256,
        position_time_size=3,
        nbr_feat_size=4,
        nbr_emb_size=16,
        nbr_agent_mcg_layers=5,
        nbr_agent_mcg_hidden_size=1024,
        nbr_enc_size=32,
        custom_lane_encode=True,
        node_feat_size=12,
        node_emb_size=12,
        node_enc_size=16,
        lane_only=True,
        max_node_count=None,
        road_agent_mcg_layers=5,
        road_agent_mcg_hidden_size=2048,
    )

    B = 1
    MN = 100
    MP = 20
    MV = 10
    MPED = 5
    t_h = 11
    target_agent_feat_size = 4
    node_feat_size = 12
    nbr_feat_size = 4

    input_dict = {
        "target_agent_representation": torch.randn(
            B, t_h, target_agent_feat_size
        ),
        "map_representation": {
            "lane_node_feats": torch.randn(B, MN, MP, node_feat_size),
            "lane_node_masks": torch.zeros(B, MN, MP, node_feat_size),
        },
        "surrounding_agent_representation": {
            "vehicles": torch.randn(B, MV, t_h, nbr_feat_size),
            "vehicle_masks": torch.zeros(B, MV, t_h, nbr_feat_size),
            "pedestrians": torch.randn(B, MPED, t_h, nbr_feat_size),
            "pedestrian_masks": torch.zeros(B, MPED, t_h, nbr_feat_size),
            "cyclists": torch.randn(B, MPED, t_h, nbr_feat_size),
            "cyclist_masks": torch.zeros(B, MPED, t_h, nbr_feat_size),
        },
        "agent_node_masks": {
            "vehicles": torch.zeros(B, MN, MV),
            "pedestrians": torch.zeros(B, MN, MPED),
            "cyclists": torch.zeros(B, MN, MPED),
        },
    }

    output = model(input_dict)
    assert output.shape[-1] == 256 + 16 + 1024 + 2048
