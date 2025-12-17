import os

import numpy as np
import pytest

from hat.data.transforms.lidar_utils.sample_ops import DataBaseSampler
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_DataBaseSampler():

    db_database = "dbinfos_train_2022-06-21.pkl"
    data_root = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/lidar/Datasets_m1"
    )

    db_info_path = data_root + "/{}".format(db_database)
    enable = True
    sample_groups_num = [3000, 5000, 5000, 5000, 5000, 5000, 5000]
    sample_groups = [
        dict(Car=sample_groups_num[0]),
        dict(Cyclist=sample_groups_num[1]),
        dict(Tricycle=sample_groups_num[2]),
        dict(Pedestrian=sample_groups_num[3]),
        dict(Construction=sample_groups_num[4]),
        dict(Truck=sample_groups_num[5]),
        dict(Bus=sample_groups_num[6]),
    ]
    db_prep_steps = [
        dict(
            type="DBFilterByMinNumPoint",
            filter_by_min_num_points=dict(
                Car=3,
                Cyclist=3,
                Tricycle=3,
                Pedestrian=3,
                Construction=3,
                Truck=3,
                Bus=3,
            ),
        ),
        dict(
            type="DBFilterByDifficulty",
            filter_by_difficulty=[-1],
        ),
    ]
    global_random_rotation_range_per_object = [0, 0]

    Data_sampler = DataBaseSampler(
        enable,
        db_info_path,
        sample_groups,
        db_prep_steps,
        global_random_rotation_range_per_object,
    )
    root_path = data_root
    gt_boxes = np.array(
        [[0.5, 0.5, 0.5, 1.0, 1.0, 1.0, np.pi / 2]], dtype=np.float32
    )
    gt_names = ["Car"]
    num_point_features = 4

    sampled_dict = Data_sampler.sample_all(
        root_path, gt_boxes, gt_names, num_point_features
    )

    assert "gt_boxes" in sampled_dict.keys()
    assert "difficulty" in sampled_dict.keys()
    assert "points" in sampled_dict.keys()
    assert "gt_masks" in sampled_dict.keys()
    assert "gt_names" in sampled_dict.keys()
