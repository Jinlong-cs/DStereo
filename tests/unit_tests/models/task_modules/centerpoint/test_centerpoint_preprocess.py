import pytest
import torch

from hat.models.task_modules.centerpoint.pre_process import (
    CenterPointPreProcess,
)

# Voxelization cfg
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
max_num_points = 20
max_voxels = 40000


@pytest.mark.parametrize(
    ["is_deploy"],
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
def test_centerpoint_preprocess(is_deploy):
    preprocess = CenterPointPreProcess(
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels_num=max_voxels,
        max_points_in_voxel=max_num_points,
        norm_range=[-51.2, -51.2, -5.0, 0.0, 51.2, 51.2, 3.0, 255.0],
        norm_dims=[0, 1, 2, 3],
    )
    points_lst = [torch.rand(300000, 5)]

    features, coors = preprocess(points_lst, is_deploy)
    if is_deploy is True:
        assert features.shape == (1, 5, 20, 40000)
        assert coors.shape == (40000, 4)
    else:
        assert features is not None
        assert coors.shape[-1] == 4
