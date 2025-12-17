# Copyright (c) Horizon Robotics. All rights reserved.
import torch

from hat.models.task_modules.traj_pred.heads.home_decoder import (
    HomeConcat,
    HomeDecoder,
)


def test_homeconcat():
    """Test HomeConcat module."""
    model = HomeConcat(in_channels=640, out_channels=32)
    test_input = torch.randn(1, 640, 16, 16)
    test_output = model(test_input)
    assert test_output["heatmap"].shape == (1, 32, 256, 256)


def test_home_decoder():
    """Test HomeDecoder module."""
    model = HomeDecoder(in_channels=512, heatmap_channels=1)
    test_input = torch.randn(1, 512, 16, 16)
    test_output = model(test_input)
    assert test_output["heatmap"].shape == (1, 1, 256, 256)
