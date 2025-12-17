# Copyright (c) Horizon Robotics. All rights reserved.
import collections
import os
import random
from typing import List, Optional

import cv2
import msgpack
import numpy as np
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.lmdb import Lmdb

__all__ = ["EVEDataset"]


@OBJECT_REGISTRY.register
class EVEDataset(data.Dataset):
    """Dataset for public gaze dataset EVE.

    More details can be found in the documentation of
    <https://e2qq6pi6j9.feishu.cn/wiki/FCEiwllfQidogOkgvbsc9qVVnlJ>

    Args:
        lmdb_path: The input needs to include three parts of lmdb,
            lmdb_path/anno/xxx: contains meta information and label.
            lmdb_path/data/xxx: contains image.
            lmdb_path/idx/xxx: contains fundamental data including scene_token,
            timestamp and index, mainly for achieving sorting and splitting
            of samples.
        num_history_frames: Number of history frames to load in a sample.
        num_future_frames: Number of future frames to load in a sample.
        max_interval: Max interval for load historical and future frames.
        max_interval: Min interval for load historical and future frames.
        fix_intra_interval: Whether to use a fix interval in a single sample.
        num_seq_split: Number of segments to divide a scene data.
        transforms: Transforms.
    """

    def __init__(
        self,
        lmdb_path: str,
        num_history_frames: int = 0,
        num_future_frames: int = 0,
        max_interval: int = 1,
        min_interval: int = 1,
        fix_intra_interval: bool = True,
        max_seq_length: Optional[int] = None,
        transforms: Optional[List] = None,
    ):
        self.lmdb_path = lmdb_path
        self.idx_path = os.path.join(lmdb_path, "idx_lmdb")
        self.anno_path = os.path.join(lmdb_path, "anno_lmdb")
        self.data_path = os.path.join(lmdb_path, "data_lmdb")
        lmdb_config = {
            "writable": False,
            "map_size": 10485760,
            "meminit": True,
            "map_async": False,
            "sync": True,
        }
        self._idx_lmdb = Lmdb(uri=self.idx_path, **lmdb_config)
        self._anno_lmdb = Lmdb(uri=self.anno_path, **lmdb_config)
        self._data_lmdb = Lmdb(uri=self.data_path, **lmdb_config)
        self.indices = []
        for key, value in self._idx_lmdb.txn.cursor():
            if key == b"__len__":
                continue
            value = msgpack.unpackb(value, raw=False)
            self.indices.append(value)
        self.indices.sort(
            key=lambda x: (x["scene_token"], float(x["timestamp"]))
        )

        self.history_frames = num_history_frames
        self.future_frames = num_future_frames
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.fix_intra_interval = fix_intra_interval
        self.transforms = transforms
        self.max_seq_length = max_seq_length
        self.num_samples = len(self.indices)

        self.set_flag()

    def set_flag(self):
        """Set scene flag for class 'DistributedGroupInBatchSampler'."""
        self.flag = []
        scenes = [x["scene_token"] for x in self.indices]
        counter = collections.Counter(scenes)

        current_flag = 0
        for _, count in counter.items():
            if self.max_seq_length is None:
                self.flag.extend([current_flag] * count)
                current_flag += 1
            else:
                quotient, remainder = divmod(count, self.max_seq_length)
                groups = []
                if quotient > 0:
                    groups.extend([self.max_seq_length] * quotient)
                if remainder > 0:
                    groups.extend([remainder])
                for group in groups:
                    self.flag.extend([current_flag] * group)
                    current_flag += 1
        self.flag = np.array(self.flag, dtype=np.int64)
        assert len(self.flag) == self.num_samples

    def get_seq_index(self, index):
        scene_token = self.indices[index]["scene_token"]
        interval = random.choice(
            range(self.min_interval, self.max_interval + 1)
        )

        indices = [[index], [index]]
        for i, frames in enumerate([self.history_frames, self.future_frames]):
            for _ in range(frames):
                if i == 0:
                    next_index = indices[i][-1] - interval
                    next_index = max(next_index, 0)
                else:
                    next_index = indices[i][-1] + interval
                    next_index = min(next_index, self.num_samples - 1)
                if scene_token != self.indices[next_index]["scene_token"]:
                    break
                indices[i].append(next_index)
                if not self.fix_intra_interval:
                    interval = random.choice(
                        range(self.min_interval, self.max_interval + 1)
                    )
        indice_history, indice_future = indices[0][1:], indices[1][1:]
        if len(indice_history) != self.history_frames:
            indice_history += [indices[0][-1]] * (
                self.history_frames - len(indice_history)
            )
        indice_history.reverse()
        if len(indice_future) != self.future_frames:
            indice_future += [indices[1][-1]] * (
                self.future_frames - len(indice_future)
            )
        return indice_history, indice_future

    def __len__(self):
        return self.num_samples

    def get_frame_data(self, index):
        key = str(self.indices[index]["key"]).encode("ascii")
        frame_data = msgpack.unpackb(self._anno_lmdb.get(key), raw=False)
        img_bytes = msgpack.unpackb(self._data_lmdb.get(key), raw=False)
        img = cv2.imdecode(
            np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        split_w = int(img.shape[1] / 2)
        eye_img = {"left": img[:, :split_w, :], "right": img[:, :split_w, :]}
        data = []
        for side in ["left", "right"]:
            tmp = {}
            tmp["img"] = eye_img[side].astype(np.float32)
            tmp["flag"] = index
            tmp["scene"] = self.flag[index]
            tmp["layout"] = "hwc"
            tmp["inv_cam_trans"] = np.array(
                frame_data["inv_camera_transformation"], np.float32
            )
            tmp["pixels_per_millimeter"] = np.array(
                frame_data["pixels_per_millimeter"], np.float32
            )
            tmp["gt_gaze"] = np.array(frame_data[f"{side}_g_tobii"])
            tmp["gt_pog"] = np.array(frame_data[f"{side}_PoG_tobii"])
            tmp["validity"] = np.array(
                frame_data[f"{side}_PoG_tobii_validity"]
            )
            tmp["R"] = np.array(frame_data[f"{side}_R"], np.float32)
            tmp["orign"] = np.array(frame_data[f"{side}_o"], np.float32)
            tmp["side"] = side
            data.append(tmp)

        return data

    def __getitem__(self, idx):
        indice_history, indice_future = self.get_seq_index(idx)
        video_data = []
        for i in indice_history + [idx] + indice_future:
            video_data.append(self.get_frame_data(i))

        outputs = []
        for i in range(len(video_data)):
            for frame in video_data[i]:
                if self.transforms is not None:
                    frame = self.transforms(frame)
                outputs.append(frame)

        return outputs
