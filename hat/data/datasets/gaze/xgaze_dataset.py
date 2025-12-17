import json
import os
import random
from typing import List, Optional

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset

from hat.registry import OBJECT_REGISTRY

__all__ = ["XGazeDataset"]


@OBJECT_REGISTRY.register
class XGazeDataset(Dataset):
    """Dataset for public gaze dataset ETH-XGaze.

    This dataset was realeased on ECCV2020. Please refer to more details to
    the original paper `ETH-XGaze: A Large Scale Dataset for Gaze Estimation
    under Extreme Head Pose and Gaze Variation`

    The code is based on official dataset code, and improved to be adaptive
    for HAT training pipeline. The original source code can be found at:
    https://github.com/xucong-zhang/ETH-XGaze/blob/master/data_loader.py

    Args:
        dataset_path: data path
        sub_folder: train or test folder. Defaults to "train".
        transform: data transforms. Defaults to None.
        is_shuffle: shuffle the index or not. Defaults to True.
        index_file: index file. Defaults to None.
    """

    def __init__(
        self,
        dataset_path: str,
        sub_folder: str = "train",
        transform: Optional[List] = None,
        is_shuffle: bool = True,
        index_file: Optional[str] = None,
    ):
        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder

        refer_list_file = os.path.join(dataset_path, "train_test_split.json")
        print("load the train file list from: ", refer_list_file)

        with open(refer_list_file, "r") as f:
            datastore = json.load(f)

        keys_to_use = datastore[sub_folder]
        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        self.selected_keys = keys_to_use

        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(
                self.path, self.sub_folder, self.selected_keys[num_i]
            )
            self.hdfs[num_i] = h5py.File(file_path, "r", swmr=True)
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and
        # person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                n = self.hdfs[num_i]["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)]
        else:
            print("load the file: ", index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(
                self.idx_to_kv
            )  # random the order to stable the training

        self.hdf = None
        self.transform = transform

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self, idx):
        data = {}
        key, index = self.idx_to_kv[idx]

        self.hdf = h5py.File(
            os.path.join(self.path, self.sub_folder, self.selected_keys[key]),
            "r",
            swmr=True,
        )
        assert self.hdf.swmr_mode

        # Get face image
        image = self.hdf["face_patch"][index, :]
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        data["img"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gaze_label = self.hdf["face_gaze"][index, :]
        data["gaze_label"] = gaze_label.astype("float")

        if self.transform is not None:
            data = self.transform(data)

        return data
