import pytest
import torch

from hat.models.task_modules.traj_pred.heads.http_heads import (
    HeatmapHead,
    TrajRegHead,
)


@pytest.mark.parametrize(
    [
        "output_channel",
        "conv_channels",
        "paddings",
        "fc_channels",
        "pooling_size",
        "agent_feature",
        "endpoint_features",
        "state_feature",
    ],
    [
        pytest.param(
            30,
            [64, 64, 64, 32, 32],
            [0, 0, 0, 0, 0],
            [32, 32, 16],
            -1,
            torch.randn((10, 128, 11, 11)),
            [torch.randn((10, 32, 1, 1)) for _ in range(5)],
            torch.randn((10, 16, 1, 1)),
        ),
        pytest.param(
            20,
            [64, 32, 32],
            [1, 0, 0],
            [32, 32, 16],
            7,
            torch.randn((10, 128, 11, 11)),
            [torch.randn((10, 32, 1, 1)) for _ in range(1)],
            torch.randn((10, 16, 1, 1)),
        ),
    ],
)
def test_traj_reg_head(
    output_channel,
    conv_channels,
    paddings,
    fc_channels,
    pooling_size,
    agent_feature,
    endpoint_features,
    state_feature,
):
    model = TrajRegHead(
        bn_kwargs={"eps": 2e-5, "momentum": 0.1},
        input_channel=agent_feature.shape[1],
        output_channel=output_channel,
        endpoint_feature_channel=endpoint_features[0].shape[1],
        state_feature_channel=state_feature.shape[1],
        conv_channels=conv_channels,
        paddings=paddings,
        fc_channels=fc_channels,
        pooling_size=pooling_size,
        confidence=True,
    )
    trajectory, confidence = model(
        agent_feature, endpoint_features, state_feature
    )
    assert trajectory.shape[0] == sum([x.shape[0] for x in endpoint_features])
    assert trajectory.shape[1] == output_channel
    assert trajectory.shape[2] == 1
    assert trajectory.shape[3] == 1
    assert confidence.shape[0] == trajectory.shape[0]


@pytest.mark.parametrize(
    ["channels", "agent_feature"],
    [pytest.param([16, 8, 8], torch.randn([10, 32, 11, 11]))],
)
def test_heatmap_head(channels, agent_feature):
    model = HeatmapHead(
        bn_kwargs={"eps": 2e-5, "momentum": 0.1},
        input_channel=agent_feature.shape[1],
        channels=channels,
        with_offset=True,
    )
    heatmap, offset = model(agent_feature)
    assert heatmap.shape[0] == agent_feature.shape[0]
    assert heatmap.shape[1] == 1
    assert heatmap.shape[2] == agent_feature.shape[2] * 2 ** len(channels)
    assert heatmap.shape[3] == agent_feature.shape[3] * 2 ** len(channels)
    assert offset.shape[0] == heatmap.shape[0]
    assert offset.shape[1] == 2
    assert offset.shape[2] == heatmap.shape[2]
    assert offset.shape[3] == heatmap.shape[3]
