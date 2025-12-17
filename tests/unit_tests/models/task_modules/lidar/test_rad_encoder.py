import torch

from hat.models.task_modules.lidar import RadGridMaker, RadScatter


def test_RadScatter():

    intensity_range = [-1, 1]
    density_range = [-1, 1]
    scatter_module = RadScatter(
        intensity_range=intensity_range,
        density_range=density_range,
    )
    voxel_features = torch.randn(20000, 500, 64)
    coords = torch.randn(20000, 4)
    coords[:, 0] = 0
    num_voxels = torch.randn(20000)

    voxel_features_pillars = torch.randn(20000, 500, 64)
    coords_pillars = torch.randn(20000, 4)
    coords_pillars[:, 0] = 0
    num_voxels_pillars = torch.randn(20000)

    batch_size = 1
    input_shape = [512, 512, 64]

    output = scatter_module(
        batch_size,
        input_shape,
        voxel_features,
        coords,
        num_voxels,
        voxel_features_pillars,
        coords_pillars,
        num_voxels_pillars,
    )

    assert output.size() == (1, 65, 512, 512)


def test_RadGridMaker():

    gird_module = RadGridMaker(down_scale=2)
    rad_features = torch.randn(1, 64, 512, 512)

    output = gird_module(rad_features, 40, -2)

    assert output.size() == (1, 1, 256, 256)
