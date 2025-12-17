# Copyright (c) Horizon Robotics. All rights reserved.

import os
import random

import numpy as np
import pytest
from torch.utils.data import DataLoader

from hat.data.collates.traj_pred_collates import (
    CatStackCollator,
    collate_SGNet,
    collate_vectornet,
)
from hat.data.datasets.traj_pred_dataset import TrajPredFPVPedDataset
from hat.utils.package_helper import check_packages_available
from projects.prediction.configs.processed_dataset import (
    DENSETNT_UNITTEST_PREFIX,
    HAT_UNITTEST_PREFIX,
)
from tests import SD_AlGORITHM_BUCKET_EXISTS, SD_AlGORITHM_BUCKET_PATH
from tests.unit_tests.core.test_traj_pred_utils import (  # noqa: E501
    gen_autourban_example_data_loader,
    gen_data_loader_v2,
    gen_densetnt_transforms,
    gen_example_transforms,
)

pd_available = check_packages_available("pandas==1.1.0", raise_exception=False)


@pytest.mark.skipif(True, reason="requiring New Dataset")
def test_multipath_collate():
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
    img_dir = os.path.join(SD_AlGORITHM_BUCKET_PATH, "mengyuan")

    train_dataloader = gen_autourban_example_data_loader(
        tdt_dir, tdt_yaml_dir, img_dir, enable_high_freq=True
    )

    aa = iter(train_dataloader)
    sample = next(aa)

    required_key = [
        "dataset_index",
        "seq_center",
        "valid_track_ids",
        "valid_img_coords",
        "valid_masks",
        "valid_class",
        "state_vectors",
        "resize_ratio",
        "road_map",
        "rendered_obs",
        "future_trajectories",
    ]
    optional_key = ["img_homographys", "affine_transforms"]
    for key in required_key + optional_key:
        assert key in sample
    num_obj, traj_len, _ = sample["future_trajectories"].shape
    batch_size = 16
    assert len(sample["dataset_index"]) == batch_size
    assert len(sample["seq_center"]) == batch_size
    assert list(sample["road_map"].shape) == [batch_size, 1, 512, 512]
    assert list(sample["rendered_obs"].shape) == [batch_size, 4, 512, 512]
    assert list(sample["valid_masks"].shape) == [num_obj, traj_len]
    assert list(sample["state_vectors"].shape) == [num_obj, 3, 1, 1]
    assert list(sample["resize_ratio"].shape) == [num_obj, 2]
    assert list(sample["img_homographys"].shape) == [num_obj, 8, 8, 2]
    num_valid_track_ids = 0
    for tracks, coords in zip(
        sample["valid_track_ids"], sample["valid_img_coords"]
    ):
        assert len(tracks) == len(coords)
        num_valid_track_ids += len(tracks)
    assert num_valid_track_ids == num_obj


def test_cat_stack_collator():

    collator = CatStackCollator(
        keys_to_stack=["img"], keys_to_concat=["bbox"], max_agent_num=5
    )

    batch_size = 6
    for _test_index in range(100):
        batch = []
        for _ in range(batch_size):
            img = np.random.uniform(size=[224, 224, 3])
            agent_num = int(random.random() * 10) + 1
            bbox = np.random.uniform(size=[agent_num, 4])
            batch.append({"img": img, "bbox": bbox})

        batch_output = collator(batch)
        assert batch_output["img"].shape == (batch_size, 224, 224, 3)
        assert batch_output["bbox"].shape[0] <= 5 * batch_size
        assert batch_output["bbox"].shape[1] == 4
        assert (
            batch_output["bbox"].shape[0]
            == batch_output["batch_index"].shape[0]
        )
        assert batch_output["agent_num"].shape[0] == batch_size
        assert batch_output["agent_num"].sum() == batch_output["bbox"].shape[0]
        assert batch_output["batch_index"].max() < batch_size


@pytest.mark.skipif(not SD_AlGORITHM_BUCKET_EXISTS, reason="require SD bucket")
def test_sgnet_collate():
    FV_Data_Path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        "11_perception_prediction/02_user/",
        "zhanbo01.li/train_data/HAT_Unit_Test/dataset",
    )
    batch_size = 4
    dim = 4
    enc_steps = 3
    dec_steps = 8
    latent_dim = 32
    k_value = 1

    dataset = TrajPredFPVPedDataset(
        dataset_root_paths=[FV_Data_Path],
        regen_data_cache=False,
        feature_type=["bbox"],
        obs_type={"ped": 2, "cyclist": 18},
        enable_relative_tar_coord=True,
        enable_cvae=True,
        stage="test",
        bbox_type="cxcywh",
        normalize_type=None,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        collate_fn=collate_SGNet,
    )

    for _, data in enumerate(dataloader):
        assert "input_traj" in data
        assert "raw_input_traj" in data
        assert "rand_cvae_seed" in data
        assert "target_traj" in data
        assert "raw_target_traj" in data
        assert "stamp" in data
        assert "date_token" in data
        assert "id" in data
        assert "type" in data
        assert "stage" in data
        assert "enable_relative" in data
        assert "position" in data
        assert "height" in data
        assert "vcs_vel" in data
        assert "global_vel" in data

        input_traj = data["input_traj"]
        target_traj = data["target_traj"]
        stamp = data["stamp"]
        date_token = data["date_token"]
        rand_cvae_seed = data["rand_cvae_seed"]
        enable_releative = data["enable_relative"]
        position = data["position"]
        height = data["height"]
        vcs_vel = data["vcs_vel"]
        global_vel = data["global_vel"]

        assert input_traj.shape == (batch_size, dim, 1, enc_steps)
        assert target_traj.shape == (batch_size, dim, 1, dec_steps)
        assert rand_cvae_seed.shape == (batch_size, latent_dim, 1, k_value)
        assert len(stamp) == batch_size
        assert len(date_token) == batch_size
        assert len(enable_releative) == batch_size
        assert position.shape == (batch_size, enc_steps, 2)
        assert height.shape == (batch_size, enc_steps)
        assert vcs_vel.shape == (batch_size, enc_steps, 2)
        assert global_vel.shape == (batch_size, enc_steps, 2)
        break


@pytest.mark.skipif(
    not (SD_AlGORITHM_BUCKET_EXISTS and pd_available),
    reason="requiring SD_Algorithm bucket",
)
def test_densetnt_collate():
    basic_trans, obstacle_trans, _, _ = gen_example_transforms()
    vectornet_trans = gen_densetnt_transforms()
    all_transforms = basic_trans + obstacle_trans + vectornet_trans

    root_pkl_path = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        DENSETNT_UNITTEST_PREFIX,
        "densetnt/val_his6_tiny.pkl",
    )
    dataset = gen_data_loader_v2(root_pkl_path)
    sample = dataset[0]

    batch = [sample]
    transformed_batch = []

    for data in batch:
        for trans in all_transforms:
            data = trans(data)
        assert "dataset_index" in data
        assert "seq_center" in data
        assert "valid_track_ids" in data
        assert "valid_img_coords" in data
        assert "valid_masks" in data
        assert "valid_class" in data
        assert "future_trajectories" in data
        assert "struct_road_feats" in data
        assert "struct_road_masks" in data
        assert "struct_traj_feats" in data
        assert "struct_traj_masks" in data
        assert "itp_fut_obs_trajs" in data
        assert "itp_fut_obs_masks" in data
        assert "road_feat_scale" in data
        assert "traj_feat_scale" in data

        assert "goal_coords" in data
        assert "nearest_goal_idxs" in data
        assert "nearest_road_idxs" in data
        assert "end_points" in data
        transformed_batch.append(data)

    transformed_batch = collate_vectornet(transformed_batch)
    goal_coords = transformed_batch["goal_coords"]
    nearest_goal_idxs = transformed_batch["nearest_goal_idxs"]
    nearest_road_idxs = transformed_batch["nearest_road_idxs"]
    end_points = transformed_batch["end_points"]

    assert goal_coords.shape[1:] == (2, 1, all_transforms[-1].num_goals)
    assert nearest_goal_idxs.shape[1:] == (1, 1, 1)
    assert nearest_road_idxs.shape[1:] == (1, 1, 1)
    assert end_points.shape[1] == 2
