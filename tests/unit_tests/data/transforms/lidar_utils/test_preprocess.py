import numpy as np

from hat.data.transforms.lidar_utils.preprocess import (
    BatchSampler,
    DBFilterByDifficulty,
    DBFilterByMinNumPoint,
    filter_gt_box_outside_range,
    global_rotation,
    global_scaling_v2,
    global_translate_,
)


def test_BatchSampler():

    cls = "Car"
    value = range(10)
    db_batchSampler = BatchSampler(cls, value)
    sample_list = db_batchSampler.sample(2)

    assert len(sample_list) == 2


def test_DBFilterByDifficulty():

    filter_by_difficulty = [-1]
    db_infos = {
        "Car": [
            {"difficulty": -1},
        ],
        "Cyclist": [
            {"difficulty": 1},
        ],
        "Pedestrian": [
            {"difficulty": 1},
        ],
    }
    db_filter = DBFilterByDifficulty(filter_by_difficulty)
    sample_list = db_filter(db_infos)

    assert len(sample_list["Car"]) == 0
    assert len(sample_list["Cyclist"]) == 1
    assert len(sample_list["Pedestrian"]) == 1


def test_DBFilterByMinNumPoint():

    filter_by_min_num_points = {
        "Car": 3,
        "Cyclist": 3,
        "Pedestrian": 3,
    }
    db_infos = {
        "Car": [
            {"num_points_in_gt": 2},
        ],
        "Cyclist": [
            {"num_points_in_gt": 4},
        ],
        "Pedestrian": [
            {"num_points_in_gt": 5},
        ],
    }
    db_filter = DBFilterByMinNumPoint(filter_by_min_num_points)
    sample_list = db_filter(db_infos)

    assert len(sample_list["Car"]) == 0
    assert len(sample_list["Cyclist"]) == 1
    assert len(sample_list["Pedestrian"]) == 1


def test_global_scaling_v2():
    """Test global_scaling_v2 function."""

    box = np.array(
        [[0.5, 0.5, 0.5, 1.0, 1.0, 1.0, np.pi / 2]], dtype=np.float32
    )
    points = np.array(
        [[0.5, 0.5, 0.5, 1.0], [0.8, 0.3, 0.4, 0.6]], dtype=np.float32
    )
    scale_box, scale_points = global_scaling_v2(
        box, points, min_scale=1.0, max_scale=1.0
    )
    assert np.all(box == scale_box)
    assert np.all(points == scale_points)


def test_global_translate_():
    """Test global_translate_ function."""
    box = np.array(
        [[0.5, 0.5, 0.5, 1.0, 1.0, 1.0, np.pi / 2]], dtype=np.float32
    )
    points = np.array(
        [[0.5, 0.5, 0.5, 1.0], [0.8, 0.3, 0.4, 0.6]], dtype=np.float32
    )
    std = 0.0
    trans_box, trans_points = global_translate_(box, points, std)
    assert np.all(box == trans_box)
    assert np.all(points == trans_points)


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


def test_global_rotation():
    """Test global_rotation function."""
    # The function to be tested is probablistic, so it's hard to get a
    # deterministic test result.
    rot_angle = 0.0
    box = np.array(
        [[0.5, 0.5, 0.5, 1.0, 1.0, 1.0, np.pi / 2]], dtype=np.float32
    )
    points = np.array(
        [[0.5, 0.5, 0.5, 1.0], [0.8, 0.3, 0.4, 0.6]], dtype=np.float32
    )
    rot_box, rot_points = global_rotation(box, points, rot_angle)
    assert np.all(box == rot_box)
    assert np.all(points == rot_points)
