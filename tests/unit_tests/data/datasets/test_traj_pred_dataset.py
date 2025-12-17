# Copyright (c) Horizon Robotics. All rights reserved.
# This file is the unit tests of trajectory prediction datasets.

import os
import sys
from distutils.version import LooseVersion
from glob import glob

import numpy as np
import pandas
import pytest
import yaml

from hat.core.traj_pred_typing import TrajGroupIndex
from projects.prediction.configs.processed_dataset import HAT_UNITTEST_PREFIX
from tests import SD_AlGORITHM_BUCKET_EXISTS, SD_AlGORITHM_BUCKET_PATH

from hat.data.datasets.traj_pred_dataset import (  # isort: skip
    BaseTrajDataset,
    ConcatTdtDataset,
    PickledTdtDataset,
    TrajPredLMDBDateset,
    TrajPredFPVPedDataset,
    TrajPredJaadDataset,
    TrajPredBehavDataset,
    PickledTdtDatasetV2,
)


pandas_version = pandas.__version__
PANDAS_VERSION_MATCH = LooseVersion(pandas_version) == LooseVersion("1.1.0")


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_base_traj_dataset():
    tdt_dir = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "datasets/auto_urban_8days/hdfs_save/csv_trans/concat_tdt/",  # noqa: E501
    )
    tdt_yaml_dir = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "datasets/auto_urban_8days/hdfs_save/csv_trans/concat_tdt/tdt_meta.yaml",  # noqa: E501
    )

    # Extract the meta info from the yaml file.
    with open(tdt_yaml_dir, "r") as f:  # noqa: E501
        tdt_yaml = yaml.load(f.read(), Loader=yaml.FullLoader)
    train_csv_info = tdt_yaml["autourban"]["train_files"]
    traj_group_info = list(train_csv_info)[0]
    csv_file = "train_train_part_000.csv"
    traj_group_info = TrajGroupIndex(*traj_group_info)
    tdt_file_dir = os.path.join(tdt_dir, csv_file)

    # Build a dataset instance.
    ego_veh_type_id = 8
    dataset = BaseTrajDataset(
        df_path=tdt_file_dir,
        traj_group_indices=[traj_group_info],
        seq_length=16,
        seq_period=1,
        sample_step=1,
        context_frames=4,
        veh_type_id=ego_veh_type_id,
        transforms=None,
        allow_incomplete_traj=True,
        ego_as_obs=True,
    )

    # Test function __len__
    length_dataset = len(dataset)
    assert length_dataset == 598

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
        (df["map_id"] == traj_group_info.map_id)
        & (df["date"].astype("str") == traj_group_info.date)
        & (df["data_num"] == traj_group_info.data_num)
        & (df["data_version"] == traj_group_info.data_version)
        & (df["center_car_id"] == traj_group_info.center_car_id)
    )
    assert np.all(flags)
    # -- check the ego vehicle.
    ego_track_id = -BaseTrajDataset.ANSWER
    vals = df.values
    cols = df.columns
    track_id_col = cols.get_loc("track_id")
    class_id_col = cols.get_loc("classification")
    track_ids = np.unique(vals[:, track_id_col])
    assert ego_track_id in track_ids
    ego_veh_rows = vals[:, track_id_col] == ego_track_id
    ego_veh_class = vals[ego_veh_rows, class_id_col]
    assert np.all(ego_veh_class == ego_veh_type_id)


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_concat_dataset():
    tdt_dir = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "datasets/auto_urban_8days/hdfs_save/csv_trans/concat_tdt/",  # noqa: E501
    )
    tdt_yaml_dir = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "datasets/auto_urban_8days/hdfs_save/csv_trans/concat_tdt/tdt_meta.yaml",  # noqa: E501
    )

    # Extract the meta info from the json file.
    with open(tdt_yaml_dir, "r") as f:  # noqa: E501
        tdt_yaml = yaml.load(f.read(), Loader=yaml.FullLoader)

    train_csv_info = tdt_yaml["autourban"]["validation_files"]
    traj_group_indices = [TrajGroupIndex(*i) for i in list(train_csv_info)]
    all_val_files = os.path.join(tdt_dir, "val_*.csv")
    all_val_files = sorted(glob(all_val_files))

    files = []
    for file, traj_index in zip(all_val_files, traj_group_indices):
        file_name = file.split("/")[-1]
        files.append({"df": file_name, "traj_indices": [traj_index]})

    # Build a dataset instance.
    ego_veh_type_id = 8
    dataset = ConcatTdtDataset(
        prefix=tdt_dir,
        files=files,
        dataset_name="base",
        seq_length=16,
        seq_period=1,
        sample_step=1,
        context_frames=4,
        veh_type_id=ego_veh_type_id,
        transforms=None,
        ego_as_obs=True,
    )

    # Check concat.
    assert len(dataset.datasets) == len(files)
    dataset_size = 0
    for i in dataset.datasets:
        assert i.concat_dataset_idx == dataset_size
        dataset_size += len(i)
    assert len(dataset) == dataset_size

    # Test function __getitem__
    # -- find required keys.
    sample = dataset[0]
    assert "concat_dataset_index" in sample
    assert "dataset_index" in sample
    assert "seq_index" in sample
    assert "seq_df" in sample
    assert "last_context_frame_id" in sample


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS or "3.8" in sys.version,
    reason="requiring SD_Algorithm bucket",
)
def test_auto_multiagent_navi_dataset():
    pickle_dir = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        f"{HAT_UNITTEST_PREFIX}/datasets/navi_dataset",
        "test_lane_info_hat_pickle_IC_pandar128float_map3.5_vis_loc_1.0.2_part1_val.pkl",  # noqa: E501
    )
    dataset_pkl = PickledTdtDataset(
        pkl_path=pickle_dir,
        seq_length=16,
        seq_period=1,
        sample_step=1,
        context_frames=4,
        transforms=None,
    )
    for ds in dataset_pkl.loaded_ds.datasets:
        ds.navi_save_mode = "local"
        ds.use_struct_road_info = False
    sample = dataset_pkl[0]
    assert "navi_info" in sample
    navi_info = sample["navi_info"]
    for _, v in navi_info.items():
        assert "succ" in v
        assert "is_logical" in v
        assert "bounding_box" in v
        assert "drive_line" in v


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS,
    reason="requiring SD_Algorithm bucket",
)
@pytest.mark.skipif(not PANDAS_VERSION_MATCH, reason="requiring pandas==1.1.0")
def test_PickledTdtDatasetV2():
    pickle_dir = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        f"{HAT_UNITTEST_PREFIX}/datasets/PickledTdtDatasetV2",
        "val_583.pkl",  # noqa: E501
    )

    dataset_pkl = PickledTdtDatasetV2(
        pkl_path=pickle_dir,
        transforms=None,
    )
    for ds in dataset_pkl.loaded_ds.datasets:
        ds.navi_save_mode = "local"
        ds.use_struct_road_info = False
    sample = dataset_pkl[0]
    assert "navi_info" in sample
    navi_info = sample["navi_info"]
    for _, v in navi_info.items():
        assert "succ" in v
        assert "is_logical" in v
        assert "bounding_box" in v
        assert "drive_line" in v


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_traj_pred_lmdb_dataset():
    root = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        f"{HAT_UNITTEST_PREFIX}/datasets/test/dataset_unit_test",
    )

    data_key_name = "data_list.txt"
    bev_lmdb_name = "bev_lmdb"
    label_lmdb_name = "label_lmdb"

    assert os.path.exists(os.path.join(root, data_key_name))
    assert os.path.exists(os.path.join(root, bev_lmdb_name))
    assert os.path.exists(os.path.join(root, label_lmdb_name))

    dataset = TrajPredLMDBDateset(
        root, data_key_name, bev_lmdb_name, label_lmdb_name, transforms=None
    )

    assert len(dataset) > 0

    for data in dataset:
        assert "road_map" in data
        assert "seq_df" in data
        assert "seq_index" in data
        assert "last_context_frame_id" in data
        assert "lcf_timestamp" in data
        assert "dataset_index" in data


@pytest.mark.skipif(not SD_AlGORITHM_BUCKET_EXISTS, reason="require SD bucket")
def test_traj_pred_sgnet():
    FV_Data_Paths = [
        os.path.join(
            SD_AlGORITHM_BUCKET_PATH,
            "11_perception_prediction/02_user/",
            "zhanbo01.li/train_data/HAT_Unit_Test/dataset",
        )
    ] * 2
    dim = 4
    enc_steps = 3
    dec_steps = 8
    latent_dim = 32
    k_value = 1

    dataset1 = TrajPredFPVPedDataset(
        dataset_root_paths=FV_Data_Paths,
        regen_data_cache=False,
        feature_type=["bbox"],
        obs_type={"ped": 2},
        enable_relative_tar_coord=True,
        enable_cvae=True,
        stage="test",
        bbox_type=None,
        normalize_type="zero-one",
    )

    dataset2 = TrajPredFPVPedDataset(
        dataset_root_paths=FV_Data_Paths[:1],
        regen_data_cache=False,
        feature_type=["bbox"],
        obs_type={"ped": 2},
        enable_relative_tar_coord=True,
        enable_cvae=True,
        stage="test",
        bbox_type=None,
        normalize_type="zero-one",
    )

    data = dataset1[0]
    assert "input_traj" in data
    assert "raw_input_traj" in data
    assert "rand_cvae_seed" in data
    assert "target_traj" in data
    assert "raw_target_traj" in data
    assert "position" in data
    assert "height" in data
    assert "vcs_vel" in data
    assert "global_vel" in data
    assert "stamp" in data
    assert "date_token" in data
    assert "id" in data
    assert "type" in data
    assert "stage" in data
    assert "enable_relative" in data

    input_traj = data["input_traj"]
    target_traj = data["target_traj"]
    rand_cvae_seed = data["rand_cvae_seed"]
    position = data["position"]
    height = data["height"]
    vcs_vel = data["vcs_vel"]
    global_vel = data["global_vel"]

    assert input_traj.shape == (dim, 1, enc_steps)
    assert target_traj.shape == (dim, 1, dec_steps)
    assert rand_cvae_seed.shape == (latent_dim, 1, k_value)
    assert position.shape == (enc_steps, 2)
    assert height.shape == (enc_steps,)
    assert vcs_vel.shape == (enc_steps, 2)
    assert global_vel.shape == (enc_steps, 2)

    assert dataset2.__len__() == dataset1.__len__() / 2


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS, reason="requiring SD_Algorithm bucket"
)
def test_traj_pred_sgnet_jaad():
    data_root = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        "09_perception_post_processing_dynamic/02_user/xiaoming.zhang/JJAD",
    )
    enc_steps = 10
    dec_steps = 5
    FPS = 5
    dim = 4
    latent_dim = 32
    k_value = 1

    train_dataset = TrajPredJaadDataset(
        data_root, enc_steps, dec_steps, split="train", FPS=FPS
    )
    val_dataset = TrajPredJaadDataset(
        data_root, enc_steps, dec_steps, split="val", FPS=FPS
    )
    test_dataset = TrajPredJaadDataset(
        data_root, enc_steps, dec_steps, split="test", FPS=FPS
    )

    train_data = train_dataset[0]
    assert "input_traj" in train_data
    assert "raw_input_traj" in train_data
    assert "rand_cvae_seed" in train_data
    assert "target_traj" in train_data
    assert "raw_target_traj" in train_data
    assert "stamp" in train_data
    assert "date_token" in train_data
    assert "id" in train_data
    assert "type" in train_data
    assert "stage" in train_data
    assert "enable_relative" in train_data

    val_data = val_dataset[0]
    assert "input_traj" in val_data
    assert "raw_input_traj" in val_data
    assert "rand_cvae_seed" in val_data
    assert "target_traj" in val_data
    assert "raw_target_traj" in val_data
    assert "stamp" in val_data
    assert "date_token" in val_data
    assert "id" in val_data
    assert "type" in val_data
    assert "stage" in val_data
    assert "enable_relative" in val_data

    test_data = test_dataset[0]
    assert "input_traj" in test_data
    assert "raw_input_traj" in test_data
    assert "rand_cvae_seed" in test_data
    assert "target_traj" in test_data
    assert "raw_target_traj" in test_data
    assert "stamp" in test_data
    assert "date_token" in test_data
    assert "id" in test_data
    assert "type" in test_data
    assert "stage" in test_data
    assert "enable_relative" in test_data

    input_traj = train_data["input_traj"]
    target_traj = train_data["target_traj"]
    rand_cvae_seed = train_data["rand_cvae_seed"]

    assert input_traj.shape == (dim, 1, enc_steps)
    assert target_traj.shape == (dim, 1, dec_steps)
    assert rand_cvae_seed.shape == (latent_dim, 1, k_value)

    enc_steps = 5
    dec_steps = 10
    train_dataset = TrajPredJaadDataset(
        data_root, enc_steps, dec_steps, split="train", FPS=FPS
    )
    train_data = train_dataset[0]
    input_traj = train_data["input_traj"]
    target_traj = train_data["target_traj"]
    rand_cvae_seed = train_data["rand_cvae_seed"]

    assert input_traj.shape == (dim, 1, enc_steps)
    assert target_traj.shape == (dim, 1, dec_steps)
    assert rand_cvae_seed.shape == (latent_dim, 1, k_value)


@pytest.mark.skipif(
    not SD_AlGORITHM_BUCKET_EXISTS or "3.8" in sys.version,
    reason="requiring SD_Algorithm bucket",
)
def test_traj_pred_behav_dataset():
    dataset_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        "SD_Algorithm/11_perception_prediction/03_pack_package/zepei.sun/sd_dataset/behav_full_dataset/behav_full_train_1042555samples.pkl",  # noqa: E501
    )
    behav_dataset = TrajPredBehavDataset(
        dataset_pkl=dataset_path,
        road_scale=[1 for _ in range(12)],
        traj_scale=[1 for _ in range(9)],
        stage="train",
        down_ratio=1,
    )
    sample = behav_dataset[0]
    assert "struct_road_feats" in sample
    assert "struct_traj_feats" in sample
    assert "struct_road_masks" in sample
    assert "struct_traj_masks" in sample
    assert "behav_state_vectors" in sample
