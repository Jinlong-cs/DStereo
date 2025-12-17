import numpy as np
import pytest
import torch

from hat.data.collates.nusc_collates import (
    collate_nuscenes,
    collate_nuscenes_sequence,
)


def test_collate_nuscenes():
    img = torch.from_numpy(
        np.random.randint(0, 255, (900, 1600), dtype=np.uint8)
    )
    imgs = [img] * 6

    ego2img = torch.from_numpy(np.random.randn(4, 4))
    ego2imgs = [ego2img] * 6

    cam2ego = torch.from_numpy(np.random.randn(4, 4))
    cam2egos = [cam2ego] * 6

    camera_intrinsic = torch.from_numpy(np.random.randn(4, 4))
    camera_intrinsics = [camera_intrinsic] * 6

    data = {
        "img": imgs,
        "ego2img": ego2imgs,
        "cam2ego": cam2egos,
        "camera_intrinsic": camera_intrinsics,
        "img_name": "file",
        "cam_name": ["CAM0", "CAM1", "CAM2", "CAM3", "CAM4", "CAM5"],
        "bev_bboxes": [np.random.randn(10), np.random.randn(10)],
        "bev_cat_ids": [1, 2],
        "bev_bboxes_labels": [np.random.randn(10), np.random.randn(10)],
        "layout": ["hwc", "hwc"],
        "color_space": ["rgb", "rgb"],
        "sample_token": "000001",
        "ego2global": np.random.randn(4, 4),
        "ego_bboxes_labels": [np.random.randn(10), np.random.randn(10)],
    }
    batch = [data] * 2
    batch = collate_nuscenes(batch)

    assert "img" in batch
    assert "ego2img" in batch
    assert "cam2ego" in batch
    assert "camera_intrinsic" in batch
    assert "cam_name" in batch
    assert "bev_bboxes" in batch
    assert "bev_cat_ids" in batch
    assert "bev_bboxes_labels" in batch
    assert "layout" in batch
    assert "color_space" in batch
    assert "sample_token" in batch
    assert "ego2global" in batch
    assert "ego_bboxes_labels" in batch


def test_collate_nuscenes_sequence():
    img = torch.from_numpy(
        np.random.randint(0, 255, (900, 1600), dtype=np.uint8)
    )
    imgs = [img] * 6
    ego2img = torch.from_numpy(np.random.randn(4, 4))
    ego2imgs = [ego2img] * 6

    cam2ego = torch.from_numpy(np.random.randn(4, 4))
    cam2egos = [cam2ego] * 6

    camera_intrinsic = torch.from_numpy(np.random.randn(4, 4))
    camera_intrinsics = [camera_intrinsic] * 6

    data = {
        "img": imgs,
        "ego2img": ego2imgs,
        "cam2ego": cam2egos,
        "camera_intrinsic": camera_intrinsics,
        "img_name": "file",
        "cam_name": ["CAM0", "CAM1", "CAM2", "CAM3", "CAM4", "CAM5"],
        "bev_bboxes": [np.random.randn(10), np.random.randn(10)],
        "bev_cat_ids": [1, 2],
        "bev_bboxes_labels": [np.random.randn(10), np.random.randn(10)],
        "layout": ["hwc", "hwc"],
        "color_space": ["rgb", "rgb"],
        "sample_token": "000001",
        "ego2global": [np.random.randn(4, 4)],
        "ego_bboxes_labels": [np.random.randn(10), np.random.randn(10)],
    }
    batch = [[data] * 3] * 2
    batch = collate_nuscenes_sequence(batch)

    assert "img" in batch
    assert "ego2img" in batch
    assert "cam2ego" in batch
    assert "camera_intrinsic" in batch
    assert "cam_name" in batch
    assert "bev_bboxes" in batch
    assert "bev_cat_ids" in batch
    assert "bev_bboxes_labels" in batch
    assert "layout" in batch
    assert "color_space" in batch
    assert "sample_token" in batch
    assert "ego2global" in batch
    assert "ego_bboxes_labels" in batch


if __name__ == "__main__":
    pytest.main(["-s", __file__])
