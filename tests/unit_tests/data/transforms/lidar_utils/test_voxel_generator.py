import numpy as np

from hat.data.transforms.lidar_utils.voxel_generator import VoxelGenerator


def test_VoxelGenerator():

    pc_range = [-19.2, -72, -4, 96, 72, 2.3]
    voxel_size = [0.15, 0.15, 0.1]
    max_points_in_voxel = 30
    max_voxel_num = 50000
    voxel_generator = VoxelGenerator(
        voxel_size, pc_range, max_points_in_voxel, max_voxel_num
    )

    points_x = np.random.uniform(-19.2, 96, (1000, 1))
    points_y = np.random.uniform(-72, 72, (1000, 1))
    points_z = np.random.uniform(-4, 2.3, (1000, 1))
    intensity = np.random.uniform(0, 1, (1000, 1))
    points = np.concatenate([points_x, points_y, points_z, intensity], axis=1)

    voxels, coordinates, num_points = voxel_generator.generate(points)
    assert len(voxels.shape) == 3
    assert len(coordinates.shape) == 2
    assert len(num_points.shape) == 1
    assert voxels.shape[0] == coordinates.shape[0] == num_points.shape[0]
