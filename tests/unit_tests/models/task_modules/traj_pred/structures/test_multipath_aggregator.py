import torch

from hat.models.task_modules.traj_pred.structures.multipathpp_aggregator import (  # noqa: E501
    MTPPlusAggregator,
    MultiHeadMTPPlusAggregator,
)


def test_MTPPlusAggregator():
    model = MTPPlusAggregator(
        target_agent_mcg_hidden_size=256,
        target_agent_emb_size=16,
        nbr_agent_mcg_hidden_size=1024,
        road_agent_mcg_hidden_size=2048,
        num_anchor=6,
        anchor_emb_size=256,
        pred_mcg_layers=5,
        pred_mcg_hidden_size=4096,
    )

    test_input_context = torch.randn(1, 256 + 16 + 1024 + 2048)

    output = model(test_input_context)
    assert output.shape == (1, 6, 4096)


def test_Multihead_aggregator():
    model = MultiHeadMTPPlusAggregator(
        num_header=2,
        target_agent_mcg_hidden_size=256,
        target_agent_emb_size=16,
        nbr_agent_mcg_hidden_size=1024,
        road_agent_mcg_hidden_size=2048,
        num_anchor=6,
        anchor_emb_size=256,
        pred_mcg_layers=5,
        pred_mcg_hidden_size=4096,
    )

    test_input_context = torch.randn(1, 256 + 16 + 1024 + 2048)

    output = model(test_input_context)
    assert len(output) == 2 and all([i.shape == (1, 6, 4096) for i in output])
