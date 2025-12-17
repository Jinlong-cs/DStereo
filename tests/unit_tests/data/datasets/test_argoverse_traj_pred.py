import os
import pickle

import numpy as np
import pytest

try:
    from argoverse.map_representation.map_api import ArgoverseMap
except ModuleNotFoundError:
    ArgoverseMap = None

from hat.data.datasets.argoverse_traj_pred import (
    ArgoverseDataset,
    ArgoverseDatasetFromLMDB,
    ArgoversePacker,
    ArgoverseTdtDataset,
)
from tests import MATRIX_BUCKET_EXISTS, MATRIX_BUCKET_PATH


@pytest.mark.skipif(ArgoverseMap is None, reason="requiring MATRIX bucket")
def test_argoverse_dataset():
    dataset = ArgoverseDataset(
        data_path=f"{MATRIX_BUCKET_PATH}/users/xuewu.lin/argodataset",
        mode="val",
        dataset_size=-1,
        meters_map=100 * 2,
        resolution=0.2,
        only_agent=True,
    )
    dataset_lmdb = ArgoverseDatasetFromLMDB(
        data_path=f"{MATRIX_BUCKET_PATH}/users/xuewu.lin/argodataset",
        mode="val_pack_withtime",
        dataset_size=-1,
        meters_map=100 * 2,
        resolution=0.2,
        only_agent=True,
    )
    dataset_lmdb = pickle.loads(pickle.dumps(dataset_lmdb))

    assert len(dataset) == 39472
    assert len(dataset_lmdb) == 39472

    for i in range(10):
        sample = dataset[i]
        agent_num = len(sample["history"])
        for key in [
            "history_mask",
            "future",
            "future_mask",
            "category",
            "start_point",
            "history_state",
            "valid_flag",
        ]:
            assert len(sample[key]) == agent_num

        assert sample["raster_map"].shape[0] == int(round(100 * 2 / 0.2))
        assert sample["raster_map"].shape[1] == int(round(100 * 2 / 0.2))
        assert sample["raster_map"].shape[2] == 3
        assert sample["history"].shape[1] == 20
        assert sample["future"].shape[1] == 30
        assert sample["history_mask"].shape[1] == 20
        assert sample["future_mask"].shape[1] == 30
        assert sample["valid_flag"].sum() == 1

        sample_lmdb = dataset_lmdb[i]
        sample.pop("category")
        for key, value in sample.items():
            assert np.all(
                sample_lmdb[key].astype("float32") - value.astype("float32")
                < 1e-3
            )


@pytest.mark.skipif(not MATRIX_BUCKET_EXISTS, reason="requiring MATRIX bucket")
def test_argoverse_packer(tmpdir, target_mode="val"):
    packer = ArgoversePacker(
        src_data_path=f"{MATRIX_BUCKET_PATH}/users/xuewu.lin/argodataset",
        mode=target_mode,
        target_data_path=os.path.join(tmpdir, target_mode),
        num_workers=2,
        dataset_size=100,
    )
    packer()
    assert os.path.exists(os.path.join(tmpdir, target_mode, "data.mdb"))
    assert os.path.exists(os.path.join(tmpdir, target_mode, "file_list.txt"))


@pytest.mark.skipif(ArgoverseMap is None, reason="requiring MATRIX bucket")
def test_argoverse_tdt_dataset():
    MATRIX_BUCKET_PATH = "/horizon-bucket/matrix"
    dataset = ArgoverseTdtDataset(
        data_path=f"{MATRIX_BUCKET_PATH}/users/xuewu.lin/argodataset",
        mode="val",
        center="AV",
        meters_map=100 * 2,
        resolution=0.2,
        map_mode="vectorized",
        for_viz=True,
    )
    assert len(dataset) == 39472
    sample = dataset[0]
    assert "vector_map_feats" in sample
    assert "feat_cols" in sample
    assert "adj_mat" in sample
    assert "seq_df" in sample
    assert "seq_index" in sample
