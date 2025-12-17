import copy

import numpy as np
import pytest
import torch

from hat.registry import build_from_registry

# A sample dataset that contains 5 frames of lidar point clouds.
waymo_dataset_config = dict(
    type="WaymoDataset",
    info_path="tmp_data/waymo/infos_train_02sweeps_unittest.pkl",
    root_path="tmp_data/waymo/",
    transforms=None,
    class_names=["VEHICLE", "PEDESTRIAN", "CYCLIST"],
    test_mode=False,
    nsweeps=2,
    load_interval=1,
)


def test_waymo_dataset():
    """Test waymo dataset loading. (Transforms test performed separately)"""
    config = copy.deepcopy(waymo_dataset_config)
    dataset = build_from_registry(config)

    assert len(dataset) == 5

    for i in range(5):
        sample = dataset[i]
        assert "lidar" in sample
        assert "combined" in sample["lidar"]
        assert "annotations" in sample["lidar"]
        assert "metadata" in sample


def test_waymo_evaluation():
    """Test evluation function."""
    waymo_dataset_exist = False
    # Determine if waymo_open_dataset is installed. Also gets around pep8
    # imported module not used check.
    try:
        import waymo_open_dataset

        waymo_dataset_exist = waymo_open_dataset.__file__ is not None
    except ImportError:
        pass
    # create fake data. For each token the following are forged:
    # 1) 20 boxes, ranged in (-75.0, 75.0)m, with random location/size/heading;
    # 2) scores, sampled from uniform distribution;
    # 3) labels, randomly chosen from [0, 1, 2].
    config = copy.deepcopy(waymo_dataset_config)
    dataset = build_from_registry(config)
    fake_dets = dict()
    for i in range(5):
        sample = dataset[i]
        token = sample["metadata"]["token"]
        box3d_lidar = torch.rand(20, 7, dtype=torch.float32) * 150.0 - 75.0
        scores = torch.rand(20, dtype=torch.float32)
        labels = torch.tensor(
            np.random.choice((0, 1, 2), size=20, replace=True)
        )
        detections = {
            "box3d_lidar": box3d_lidar,
            "scores": scores,
            "label_preds": labels,
        }
        fake_dets[token] = detections
    if waymo_dataset_exist:
        # Just make sure the evaluation function runs without any problem.
        dataset.evaluation(fake_dets, output_dir="", write_file=False)
    else:
        with pytest.raises(ModuleNotFoundError):
            dataset.evaluation(fake_dets, output_dir="", write_file=False)
