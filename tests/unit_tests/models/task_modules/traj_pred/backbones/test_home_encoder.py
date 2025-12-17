import pytest
import torch

from hat.models.task_modules.traj_pred.backbones.home_encoder import (
    HomeEncoder,
)
from hat.utils.package_helper import check_packages_available


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.skip(
    reason="""Module requires online download which is not possible in CI/CD"""
    """environment. Will change in the future."""
)
def test_home_encoder():
    """Test HOME encoder module."""
    input_channels = 50
    agent_feat_size = 3
    timesteps = 10
    num_agents = 30

    model = HomeEncoder(
        context_encoding_size=16,
        backbone="resnet50",
        input_channels=input_channels,
        agent_feat_size=agent_feat_size,
        agent_emb_size=64,
        agent_enc_size=128,
        social_encoding_size=128,
    )

    test_input = {
        "surrounding_dynamics": torch.randn(
            1, num_agents, timesteps, agent_feat_size
        ),
        "ego_dynamics": torch.randn(1, 1, timesteps, agent_feat_size),
        "raster": torch.randn(1, input_channels, 320, 320),
    }

    output = model(test_input)
    assert output["social_encoding"].shape == (1, 128, 16, 16)
    assert output["context_encoding"].shape == (1, 512, 16, 16)
