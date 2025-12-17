import numpy as np

from hat.data.transforms.lidar import (
    ParsePointCloud,
    Point2VCS,
    Voxelization,
    _dict_select,
)
from hat.data.transforms.lidar_utils.preprocess import (
    filter_gt_box_outside_range,
)


def test_filter_gt_box_outside_range():
    """Test filter_gt_box_outside_range functin."""
    box = np.array(
        [
            [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0],
            [10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    limit_range = np.array([-5, -5, 5, 5], dtype=np.float32)
    filtered = filter_gt_box_outside_range(box, limit_range)
    assert np.all(filtered == np.array([True, False]))


def test_dict_select():

    gt_dicts = {
        "gt_boxes": np.zeros((10, 7)),
        "gt_names": np.array(range(10)),
    }
    select = np.array([2, 4, 7, 9])
    _dict_select(gt_dicts, select)

    assert len(gt_dicts["gt_names"]) == 4
    assert gt_dicts["gt_boxes"].shape[0] == 4


def test_voxelization():
    pc_range = [-50, -50, -5, 50, 50, 3]
    voxel_size = [0.1, 0.1, 0.1]
    nframe = 2
    voxel_generator = Voxelization(
        range=pc_range,
        voxel_size=voxel_size,
        max_points_in_voxel=5,
        max_voxel_num=500000,
        voxel_key="voxel",
        nframe=nframe,
    )

    fake_input = dict(
        point_clouds=[np.random.rand(100, 4) for _ in range(nframe)],
    )

    fake_output = voxel_generator(fake_input)

    assert "voxel_data" in fake_output


def test_point_cloud():
    dtype = np.float32
    data_dim = 4
    fake_raw_data = np.random.rand(100, data_dim).astype(dtype).tobytes()
    data = dict(
        point_clouds=[fake_raw_data],
        meta_info=dict(
            T_lidar2vcs=np.eye(4),
        ),
    )

    point_cloud_parser = ParsePointCloud(
        dtype=dtype,
        load_dim=data_dim,
        keep_dim=4,
    )
    point2vcs = Point2VCS(
        shuffle_points=True,
    )

    data = point_cloud_parser(data)
    data = point2vcs(data)
    assert "point_clouds" in data
    assert data["point_clouds"][0].shape == (100, 4)
