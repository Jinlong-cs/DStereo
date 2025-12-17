import pytest
import torch

from hat.core.nus_box3d_utils import (
    adjust_coords,
    bbox3d_nus_transform,
    bbox_bev2ego,
    bbox_ego2bev,
    bbox_ego2img,
    bbox_to_corner,
    get_min_max_coords,
    inverse_bbox3d_nus_transform,
    inverse_sigmoid,
)

try:
    import nuscenes
except ImportError:
    nuscenes = None

bev_size = (51.2, 51.2, 0.8)
grid_size = (128, 128)


def test_inverse_sigmoid():
    x = torch.randn(1, 64, 128, 128)
    inverse_sigmoid(x)


def test_bbox3d_nus_transform():
    bboxes = torch.randn(10, 10)
    bbox3d_nus_transform(bboxes)


def test_get_min_max_coords():
    get_min_max_coords(bev_size)


def test_inverse_bbox3d_nus_transform():
    bboxes = torch.randn(10, 10)
    inverse_bbox3d_nus_transform(bboxes)


def test_adjust_coords():
    coords = torch.randn(1, 128, 128, 2)
    adjust_coords(coords, grid_size)


@pytest.mark.skipif(nuscenes is None, reason="need nuscenes")
def test_bbox_to_corner():
    bboxes = torch.randn(10, 10)
    bbox_to_corner(bboxes, score_thresh=0.0)


def test_bbox_bev2ego():
    bboxes = torch.randn(10, 10)
    bbox_bev2ego(bboxes, bev_size)


def test_bbox_ego2bev():
    bboxes = torch.randn(10, 10)
    bbox_ego2bev(bboxes, bev_size)


@pytest.mark.skipif(nuscenes is None, reason="need nuscenes")
def test_bbox_ego2img():
    bboxes = torch.randn(10, 10)
    ego2img = torch.randn(4, 4).numpy()
    bbox_ego2img(bboxes, ego2img, (512, 960), 0.0)
