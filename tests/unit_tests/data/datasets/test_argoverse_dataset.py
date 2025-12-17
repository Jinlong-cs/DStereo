import pytest

from hat.data.datasets.argoverse_dataset import Argoverse1Dataset


def test_argoverse():
    dataset = Argoverse1Dataset(
        data_path="./tmp_data/argoverse-1/val_lmdb",
        map_path="./tmp_data/argoverse-1/map_files",
        pred_step=20,
        max_distance=50.0,
        max_lane_num=64,
        max_traj_num=32,
        max_goals_num=2048,
    )
    state = dataset.__getstate__()
    dataset.__setstate__(state)

    for ind, data in enumerate(dataset):
        assert data["traj_feat"].shape == (9, 19, 32)
        assert data["traj_labels"].shape == (30, 2)
        assert data["lane_feat"].shape == (11, 9, 64)
        assert data["goals_2d"].shape == (2, 1, 2048)
        assert data["goals_2d_mask"].shape == (1, 1, 2048)
        assert data["instance_mask"].shape == (1, 1, 96)
        if ind > 10:
            break


if __name__ == "__main__":
    pytest.main(["-s", __file__])
