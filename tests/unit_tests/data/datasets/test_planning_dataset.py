# Copyright (c) Horizon Robotics. All rights reserved.
# This file is the unit tests of trajectory prediction datasets.

import os
import pickle

import numpy as np
import pandas as pd
import pytest

from hat.data.datasets.dataset_wrappers import ConcatDataset
from hat.data.datasets.planning_dataset import (
    P3CMultiAgentDataset,
    P3CPickledTdtDataset,
)
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


def generate_dummy_dataset(csv_path):
    csv_file = pd.read_csv(csv_path)
    indices_cols = [
        "map_id",
        "date",
        "data_num",
        "data_version",
        "center_car_id",
    ]

    indices = []
    for ind in csv_file[indices_cols].values.tolist():
        if ind not in indices:
            indices.append(ind)

    dataset = P3CMultiAgentDataset(
        df_path=csv_path,
        traj_group_indices=indices,
        seq_length=16,
        seq_period=1,
        sample_step=1,
        context_frames=4,
        veh_type_id=1,
        transforms=None,
        allow_incomplete_traj=False,
        ego_as_obs=True,
        have_index_col=False,
        load_df_func=None,
        path_prefix="3079" + "_",
    )

    return dataset


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_p3c_agent_dataset():
    csv_path = os.path.join(
        HAT_BUCKET_PATH,
        "users/bikun.wang/unit_test/concat_train.csv",
    )

    dataset = generate_dummy_dataset(csv_path)

    # Test function __len__
    length_dataset = len(dataset)
    assert length_dataset == 80

    # Test function __getitem__
    # -- find required keys.
    sample = dataset[0]
    assert "concat_dataset_index" in sample
    assert "dataset_index" in sample
    assert "seq_index" in sample
    assert "seq_df" in sample
    assert "last_context_frame_id" in sample
    # -- check the trajectory group token.
    df = sample["seq_df"]
    flags = (
        (df["map_id"] == dataset.df_indices[0][0].traj_group_index.map_id)
        & (
            df["date"].astype("str")
            == dataset.df_indices[0][0].traj_group_index.date
        )
        & (
            df["data_num"]
            == dataset.df_indices[0][0].traj_group_index.data_num
        )
        & (
            df["data_version"]
            == dataset.df_indices[0][0].traj_group_index.data_version
        )
        & (
            df["center_car_id"]
            == dataset.df_indices[0][0].traj_group_index.center_car_id
        )
    )
    assert np.all(flags)
    # -- check the ego vehicle.
    ego_track_id = -P3CMultiAgentDataset.ANSWER
    vals = df.values
    cols = df.columns
    track_id_col = cols.get_loc("track_id")
    track_ids = np.unique(vals[:, track_id_col])
    assert ego_track_id in track_ids

    # concat datasets
    concat_dataset = ConcatDataset(datasets=[dataset])

    # Check concat.
    assert len(concat_dataset.datasets) == len([dataset])
    dataset_size = 0
    for i in concat_dataset.datasets:
        assert i.concat_dataset_idx == dataset_size
        dataset_size += len(i)
    assert len(concat_dataset) == dataset_size

    # Test function __getitem__
    # -- find required keys.
    sample = concat_dataset[0]
    assert "concat_dataset_index" in sample
    assert "dataset_index" in sample
    assert "seq_index" in sample
    assert "seq_df" in sample
    assert "last_context_frame_id" in sample

    pkl_save_path = os.path.join(
        HAT_BUCKET_PATH,
        "users/bikun.wang/unit_test/concat_train.pkl",
    )
    with open(pkl_save_path, "wb") as f:
        pickle.dump(concat_dataset, f, 1)

    dataset_pkl = P3CPickledTdtDataset(
        pkl_path=pkl_save_path,
        seq_length=16,
        seq_period=1,
        sample_step=1,
        context_frames=4,
        transforms=None,
    )

    # Test function __len__
    length_dataset = len(dataset_pkl)
    assert length_dataset == 80

    # Test function __getitem__
    # -- find required keys.
    sample = dataset_pkl[0]
    assert "concat_dataset_index" in sample
    assert "dataset_index" in sample
    assert "seq_index" in sample
    assert "seq_df" in sample
    assert "last_context_frame_id" in sample
    assert "dataset_prefix" in sample
    assert "date_token" in sample
    assert "traj_navi" in sample
    assert "date_token" in sample
