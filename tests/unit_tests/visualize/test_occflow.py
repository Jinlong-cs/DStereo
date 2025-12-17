import torch

from hat.data.datasets.occflow_dataset import AgentGrids
from hat.visualize.occflow import flow_rgb_image, occupancy_rgb_image


def test_occupancy_rgb_image():
    agent_grids = AgentGrids()
    road_graph = torch.zeros((1, 256, 256, 1), dtype=torch.float32)
    result_img = occupancy_rgb_image(agent_grids, road_graph)
    assert result_img.shape == (1, 256, 256, 3)


def test_flow_rgb_image():
    flow = torch.zeros((1, 256, 256, 2), dtype=torch.float32)
    road_graph = torch.zeros((1, 256, 256, 1), dtype=torch.float32)
    agent_trails = torch.zeros((1, 256, 256, 1), dtype=torch.float32)
    result_img = flow_rgb_image(flow, road_graph, agent_trails)
    assert result_img.shape == (1, 256, 256, 3)
