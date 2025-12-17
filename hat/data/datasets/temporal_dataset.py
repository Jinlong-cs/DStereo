import collections
import copy
import json
import logging
import os
import pickle
import random
from abc import abstractmethod
from typing import Callable, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.deprecate import deprecated_warning
from hat.utils.pack_type.lmdb import Lmdb

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class TemporalLMDBDataset(Dataset):
    """Temporal dataset with lmdb format annotations.

    This dataset only load annotations from lmdb,  and the loading of the raw
    data (images and lidar points) depends on the custom transform.

    Args:
        lmdb_path: The input needs to include two parts of lmdb,
            lmdb_path/anno/xxx: contains meta information and label.
            lmdb_path/idx/xxx: contains fundamental data including scene_token,
            timestamp and index, mainly for achieving sorting and splitting
            of samples.
        dataroot: Path to raw data, such as images and lidar points.
        history_frames: Number of history frames to load in a sample.
        future_frames: Number of future frames to load in a sample.
        max_interval: Max interval for load historical and future frames.
        max_interval: Min interval for load historical and future frames.
        fix_interval: Whether to use a fix interval in a single sample.
        num_seq_split: Number of segments to divide a scene data.
        classes: Names of the category to be loaded.
        transforms: Transforms.
        test_mode: Whether in test mode.
        data_aug_configs: The configs for data augmentation including resize,
            crop, rotate, and rotate3d.
        mode: The mode of the data to be loaded is required if it is not None.
            It should be included as 'mode' in 'idx'.
            Common modes can be 'train', 'test', or 'val'.
        instance_anno_keys: Used to filter instance labels by class names.
    """

    def __init__(
        self,
        lmdb_path,
        dataroot=None,
        history_frames=0,
        future_frames=0,
        max_interval=1,
        min_interval=1,
        fix_interval=True,
        num_seq_split=1,
        classes=None,
        transforms=None,
        test_mode=True,
        data_aug_configs=None,
        mode=None,
        instance_anno_keys=None,
    ):
        deprecated_warning(
            author="zixiang.pei",
            old_name="TemporalLMDBDataset",
            deprecation_version="v1.4.1",
            removal_version="v1.5.0",
        )
        self.lmdb_path = lmdb_path
        self.idx_path = os.path.join(lmdb_path, "idx")
        self.anno_path = os.path.join(lmdb_path, "anno")
        lmdb_config = {
            "writable": False,
            "map_size": 10485760,
            "meminit": True,
            "map_async": False,
            "sync": True,
        }
        self._idx_lmdb = Lmdb(uri=self.idx_path, **lmdb_config)
        self._anno_lmdb = Lmdb(uri=self.anno_path, **lmdb_config)
        self.indices = []
        for key, value in self._idx_lmdb.txn.cursor():
            if key == b"__len__":
                continue
            value = pickle.loads(value)
            if mode is not None and value.get("mode") not in mode:
                continue
            self.indices.append(value)
        self.indices.sort(
            key=lambda x: (x["scene_token"], float(x["timestamp"]))
        )

        self.dataroot = dataroot
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.fix_interval = fix_interval
        self.num_seq_split = num_seq_split
        self.classes = classes
        self.transforms = transforms
        self.test_mode = test_mode
        self.data_aug_configs = {
            "resize_range": (1.0, 1.0),
            "rotation_range": (0.0, 0.0),
            "rand_flip": False,
            "crop_vertical_range": (0.0, 0.0),
            "rotation_3d_range": (0.0, 0.0),
        }
        if data_aug_configs is not None:
            self.data_aug_configs.update(data_aug_configs)
        self.instance_anno_keys = instance_anno_keys
        if self.instance_anno_keys is None:
            self.instance_anno_keys = [
                "gt_bboxes_3d",
                "instance_inds",
            ]
        self.set_flag()

    def set_flag(self):
        self.flag = []
        scenes = [x["scene_token"] for x in self.indices]
        counter = collections.Counter(scenes)

        current_flag = 0
        counted = []
        for scene in scenes:
            if scene in counted:
                continue
            counted.append(scene)
            count = counter[scene]
            clips = [count // self.num_seq_split] * self.num_seq_split
            remain = count - sum(clips)
            for i in range(remain):
                clips[i] += 1
            for len_clip in clips:
                self.flag.extend([current_flag] * len_clip)
                current_flag += 1
        self.flag = np.array(self.flag, dtype=np.int64)
        assert len(self.flag) == self.__len__()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, dict):
            aug_configs = index["aug_configs"]
            index = index["index"]
        elif self.transforms is not None:
            aug_configs = self.get_aug_configs()
        indice_history, indice_future = self.get_seq_index(index)
        video_data = []
        for i in [index] + indice_history + indice_future:
            video_data.append(self.get_frame_data(i))
        if self.transforms is not None:
            for i in range(len(video_data)):
                video_data[i]["aug_configs"] = copy.deepcopy(aug_configs)
                video_data[i] = self.transforms(video_data[i])
        output = video_data[0]
        output["data_queue"] = video_data[1 : 1 + len(indice_history)]
        output["future_data_queue"] = video_data[1 + len(indice_history) :]
        return output

    def get_seq_index(self, index):
        scene_token = self.indices[index]["scene_token"]
        interval = int(random.random() * self.max_interval) + self.min_interval

        indices = [[index], [index]]
        for i, frames in enumerate([self.history_frames, self.future_frames]):
            for _ in range(frames):
                if indices[i][-1] == (0 if i == 0 else self.__len__() - 1):
                    break
                if i == 0:
                    next_index = indices[i][-1] - interval
                    next_index = max(next_index, 0)
                else:
                    next_index = indices[i][-1] + interval
                    next_index = min(next_index, self.__len__() - 1)

                if scene_token != self.indices[next_index]["scene_token"]:
                    break
                indices[i].append(next_index)
                if not self.fix_interval:
                    interval = (
                        int(random.random() * self.max_interval)
                        + self.min_interval
                    )
        indice_history, indice_future = indices[0][1:], indices[1][1:]
        return indice_history, indice_future

    def get_frame_data(self, index):
        key = str(self.indices[index]["index"]).encode("ascii")
        frame_data = pickle.loads(self._anno_lmdb.get(key))
        frame_data["dataroot"] = copy.deepcopy(self.dataroot)
        if self.classes is not None and "names" in frame_data:
            frame_data["gt_labels_3d"] = np.array(
                [
                    self.classes.index(name) if name in self.classes else -1
                    for name in frame_data["names"]
                ]
            )
            mask = frame_data["gt_labels_3d"] != -1
            for key in ["gt_labels_3d", "names"] + self.instance_anno_keys:
                if key not in frame_data:
                    continue
                frame_data[key] = frame_data[key][mask]
        return frame_data

    def get_aug_configs(self):
        if "origin_img_hw" not in self.data_aug_configs:
            return None
        H, W = self.data_aug_configs["origin_img_hw"]
        if "output_img_hw" in self.data_aug_configs:
            fH, fW = self.data_aug_configs["output_img_hw"]
        else:
            fH, fW = self.data_aug_configs["origin_img_hw"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_configs["resize_range"])
            resize_dims = (int(W * resize), int(H * resize))
            crop_top_proportion = np.random.uniform(
                *self.data_aug_configs["crop_vertical_range"]
            )
            crop_w = int(np.random.uniform(0, max(0, resize_dims[0] - fW)))
            flip = self.data_aug_configs["rand_flip"]
            flip = flip and np.random.choice([True, False])
            rotate = np.random.uniform(
                *self.data_aug_configs["rotation_range"]
            )
            rotate_3d = np.random.uniform(
                *self.data_aug_configs["rotation_3d_range"]
            )
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            crop_top_proportion = np.mean(
                self.data_aug_configs["crop_vertical_range"]
            )
            crop_h = int((1 - crop_top_proportion) * resize_dims[1]) - fH
            crop_w = int(max(0, resize_dims[0] - fW) / 2)
            flip = False
            rotate = 0
            rotate_3d = 0
        crop_h = int((1 - crop_top_proportion) * resize_dims[1]) - fH
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        aug_configs = {
            "resize": resize,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_configs


class TemporalLabelDataset(Dataset):
    """Temporal dataset with all kinds of annotations..

    Args:
        sorted_ts_list: List of timestamp item. Each item include a dict
            of timestamp and event_id.
        transforms: Config dict of transformations.
        max_interval: max_interval between 2 clip. if abs(cur_ts - pre_ts) >
            max_interval, cur_ts will be regarded as another clip.
        max_len_in_clip: max length of one clip. if length of a clip >
            max_len_in_clip, it will be splited to different clips.
            if max_len_in_clip <= 0, length of a clip can be any
            positive integer.
        event_key: name of clip_id, it can be `event_id`,
            `scene_token` or None.
    """

    def __init__(
        self,
        sorted_ts_list: List[Dict[str, str]],
        transforms: Optional[List[Callable]] = None,
        max_interval: int = 600,
        max_len_in_clip: int = -1,
        event_key: Optional[str] = None,  # "event_id", "scene_token"
    ):
        if transforms is not None:
            self.transforms = _as_list(transforms)
        else:
            self.transforms = transforms

        self.event_key = event_key

        self.len = len(sorted_ts_list)

        if max_interval > 0:
            self._set_sequence_group_flag(
                sorted_ts_list, max_interval, max_len_in_clip
            )
        elif max_interval == 0:
            self._set_default_group_flag()
        else:
            self.flag = None

    def _set_default_group_flag(self):
        self.flag = np.arange(self.len, dtype=np.int64)

    def _set_sequence_group_flag(
        self, sorted_ts_list, max_interval, max_len_in_clip
    ):

        res = []

        curr_sequence = -1
        pre_ts = -99999
        clip_len = 0
        pre_event_id = ""

        for ts_item in sorted_ts_list:
            ts = int(ts_item["timestamp"])
            if (
                (abs(ts - pre_ts) > max_interval)
                or (max_len_in_clip > 0 and clip_len >= max_len_in_clip)
                or (
                    self.event_key
                    and (self.event_key in ts_item)
                    and (ts_item[self.event_key] != pre_event_id)
                )
            ):
                curr_sequence += 1
                clip_len = 1
            else:
                clip_len += 1
            res.append(curr_sequence)
            pre_ts = ts
            if self.event_key and (self.event_key in ts_item):
                pre_event_id = ts_item[self.event_key]
            elif self.event_key:
                pre_event_id = ""

        self.flag = np.array(res, dtype=np.int64)

        assert len(self.flag) == self.__len__()

    @abstractmethod
    def parse_label(self, index):
        raise NotImplementedError

    def __getitem__(self, index):

        label = self.parse_label(index)
        imgs_meta = {"dataset_index": index, "meta": label}

        if self.transforms is not None:
            for transform in self.transforms:
                imgs_meta = transform(imgs_meta)
        return imgs_meta

    def __len__(self):
        return self.len


@OBJECT_REGISTRY.register
class TemporalJsonDataset(TemporalLabelDataset):
    """Temporal dataset with Json annotations..

    Args:
        json_file: the path of json anno file.
        transforms: Config dict of transformations.
        max_interval: max_interval between 2 clip. if abs(cur_ts - pre_ts) >
            max_interval, cur_ts will be regarded as another clip.
        max_len_in_clip: max length of one clip. if length of a clip >
            max_len_in_clip, it will be splited to different clips.
            if max_len_in_clip <= 0, length of a clip can be any
            positive integer.
        event_key: name of clip_id, it can be `event_id`,
            `scene_token` or None.
    """

    def __init__(
        self,
        json_file: str,
        transforms: Optional[List[Callable]] = None,
        max_interval: int = 600,
        max_len_in_clip: int = -1,
        event_key: Optional[str] = None,  # "event_id", "scene_token"
    ):
        ts_list = []
        with open(json_file, "r") as f:
            self.json_lines = list(f.readlines())
            for line in self.json_lines:
                label = json.loads(line)
                ts_item = {
                    "timestamp": label["timestamp"],
                }
                if event_key and event_key in label:
                    ts_item[ts_item] = label[event_key]
                ts_list.append(ts_item)

        super(TemporalJsonDataset, self).__init__(
            sorted_ts_list=ts_list,
            transforms=transforms,
            max_interval=max_interval,
            max_len_in_clip=max_len_in_clip,
            event_key=event_key,
        )

        logger.info(f"Loading {json_file}.")

    def parse_label(self, index):
        label = json.loads(self.json_lines[index])
        return label


@OBJECT_REGISTRY.register
class TemporalLmdbDataset(TemporalLabelDataset):
    """Temporal dataset with Lmdb annotations..

    Args:
        idx_path: the path of lmdb idx dir.
        anno_path: the path of lmdb idx dir.
        transforms: Config dict of transformations.
        max_interval: max_interval between 2 clip. if abs(cur_ts - pre_ts) >
            max_interval, cur_ts will be regarded as another clip.
        max_len_in_clip: max length of one clip. if length of a clip >
            max_len_in_clip, it will be splited to different clips.
            if max_len_in_clip <= 0, length of a clip can be any
            positive integer.
        event_key: name of clip_id, it can be `event_id`,
            `scene_token` or None.
    """

    def __init__(
        self,
        idx_path: str,
        anno_path: str,
        transforms: Optional[List[Callable]] = None,
        max_interval: int = 600,
        max_len_in_clip: int = 20,
        event_key: Optional[str] = None,  # "event_id", "scene_token"
    ):
        lmdb_common_config = {
            "writable": False,
            "map_size": 10485760,
            "meminit": True,
            "map_async": False,
            "sync": True,
        }
        self._idx_lmdb = Lmdb(
            uri=idx_path,
            **lmdb_common_config,
        )
        self._anno_lmdb = Lmdb(
            uri=anno_path,
            **lmdb_common_config,
        )
        assert len(self._anno_lmdb) == len(self._idx_lmdb)
        self.len = len(self._idx_lmdb)
        ts_list = []
        for key, value in self._idx_lmdb.txn.cursor():
            if key == b"__len__":
                continue
            value = pickle.loads(value)
            ts_list.append(value)

        assert len(ts_list) > 0
        if event_key and event_key in ts_list[0]:
            ts_list.sort(
                key=lambda x: (str(x[event_key]), int(x["timestamp"]))
            )
        else:
            ts_list.sort(key=lambda x: int(x["timestamp"]))

        super(TemporalLmdbDataset, self).__init__(
            sorted_ts_list=ts_list,
            transforms=transforms,
            max_interval=max_interval,
            max_len_in_clip=max_len_in_clip,
            event_key=event_key,
        )
        self.sorted_ts_list = ts_list

        logger.info(f"Loading {idx_path}.")

    def parse_label(self, index):
        key = self.sorted_ts_list[index]["idx"]
        raw_anno = self._anno_lmdb.get(key)
        label = pickle.loads(raw_anno)
        label["lmdb_key"] = key
        return label
