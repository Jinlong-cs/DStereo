import pytest
import torch

from hat.models.task_modules.pointpillars import (
    BatchVoxelization,
    PointPillarsPreProcess,
)


def test_batch_voxelization():
    max_voxels_num = 10
    max_points_in_voxel = 5
    preprocess = BatchVoxelization(
        pc_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=[0.16, 0.16, 4.0],
        max_voxels_num=max_voxels_num,
        max_points_in_voxel=max_points_in_voxel,
    )

    points_list = [
        torch.randn(100, 4),
        torch.randn(100, 4),
    ]
    feature, coords, num_points_per_voxel = preprocess(points_list, True)

    assert feature.shape == (2 * 10, 5, 4)
    assert coords.shape == (2 * 10, 4)
    assert num_points_per_voxel.shape == (2 * 10,)


@pytest.mark.skip(reason="interface will be update, skip this temporarily.")
def test_preprocess():
    max_voxels_num = 1000
    max_points_in_voxel = 5
    preprocess = PointPillarsPreProcess(
        pc_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=[0.16, 0.16, 4.0],
        max_voxels_num=max_voxels_num,
        max_points_in_voxel=max_points_in_voxel,
    )

    points_list = [
        torch.randn(1000, 4),
        torch.randn(1000, 4),
    ]

    feature, coords, _ = preprocess(points_list, True)
    assert feature.shape == (
        1,
        4,
        len(points_list) * max_voxels_num,
        max_points_in_voxel,
    )
    assert coords.shape == (len(points_list) * max_voxels_num, 4)
