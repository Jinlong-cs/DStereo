# Copyright (c) Horizon Robotics. All rights reserved.

import os
import pickle

import pytest

from hat.data.datasets.psd_dataset import PSDSlotDataset, PSDTestSlotDataset

train_lmdb_path = (
    "/horizon-bucket/SuperParking/song.ding/2021-12-28_lmdb/train_lmdb"
)
val_lmdb_path = (
    "/horizon-bucket/SuperParking/song.ding/2021-12-28_lmdb/val_lmdb"
)


@pytest.mark.skipif(
    not os.path.exists(train_lmdb_path), reason="lmdb_path doesn't exists."
)
def test_slot_lmdb():
    input_size = (896, 896)
    dataset = PSDSlotDataset(train_lmdb_path, input_size=input_size)
    dataset = pickle.dumps(dataset)
    dataset = pickle.loads(dataset)
    assert isinstance(dataset[0], dict)
    need_meta_keys = ["img", "ori_img", "label", "img_name"]
    print(dataset[0]["label"])
    for key in need_meta_keys:
        assert key in dataset[0].keys()


@pytest.mark.skipif(
    not os.path.exists(val_lmdb_path), reason="lmdb_path doesn't exists."
)
def test_slot_test_lmdb():
    input_size = (896, 896)
    dataset = PSDTestSlotDataset(val_lmdb_path, input_size)
    dataset = pickle.dumps(dataset)
    dataset = pickle.loads(dataset)
    assert isinstance(dataset[0], dict)
    need_meta_keys = ["img", "ori_img", "img_name"]
    for key in need_meta_keys:
        assert key in dataset[0].keys()
