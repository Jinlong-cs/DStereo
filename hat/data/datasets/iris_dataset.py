# Copyright (c) Horizon Robotics. All rights reserved.

import collections
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.mxrecord import unpack, unpack_img

__all__ = ["IrisDataset"]


class RecDataset(Dataset):
    def __init__(
        self,
        rec_path: str,
        idx_path: str,
        transforms: Optional[List] = None,
        visible_index: Optional[Tuple] = (0, 2),
        invisible_index: Optional[Tuple] = (1, 3, 4),
        shuffle: Optional[bool] = False,
    ):
        super(RecDataset, self).__init__()
        self.path_imgrec = rec_path
        self.path_imgidx = idx_path
        self.transforms = transforms
        self.visible_index = visible_index
        self.invisible_index = invisible_index
        self.pack_type = PackTypeMapper["mxrecord"]

        self._init_pack()
        # balance category
        self._balance_category()
        if shuffle:
            np.random.shuffle(self._indices)

    def _init_pack(self):
        self.pack_file = self.pack_type(
            uri=self.path_imgrec, idx_path=self.path_imgidx, writable=False
        )
        self.pack_file.open()

        self._size = len(self.pack_file.record.keys)
        self._indices = list(np.arange(self._size))

        self.cnt_category = collections.defaultdict(int)
        self.targets_label = []
        for index in range(len(self.pack_file.record.keys)):
            s = self.pack_file.read(self.pack_file.record.keys[index])
            header0, _ = unpack(s)
            label = header0.label
            self.cnt_category[label[0]] += 1
            self.targets_label.append(label[0])

    def _balance_category(self):
        tmp_indices = []
        cur_index = 0
        max_cnt = max(self.cnt_category.values())
        for _, cnt in self.cnt_category.items():
            tmp_cnt = max_cnt
            while tmp_cnt > cnt:
                tmp_indices += self._indices[cur_index : cur_index + cnt]
                tmp_cnt -= cnt
            tmp_indices += self._indices[cur_index : cur_index + tmp_cnt]
            cur_index += cnt
        self._indices = list(np.array(tmp_indices))
        self._size = len(self._indices)

    def _parse_label(self, label, visible_index, invisible_index):
        label = torch.cat((label[0], label[0][3:]), dim=0).reshape(1, -1)
        if label[0][3].numpy() in visible_index:
            label[0][3] = 0
        if label[0][3].numpy() in invisible_index:
            label[0][3] = 1
        if label[0][4].numpy() in visible_index:
            label[0][4] = 0
        if label[0][4].numpy() in invisible_index:
            label[0][4] = 1

        return label.type(torch.float32)

    def __getitem__(self, index: int):
        idx = self._indices[index]
        record = self.pack_file.read(self.pack_file.record.keys[idx])
        header, img = unpack_img(record, iscolor=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = torch.tensor(header.label).unsqueeze(dim=0)
        label = self._parse_label(
            label, self.visible_index, self.invisible_index
        ).squeeze()
        label = tuple([_as_list(label[_])[0] for _ in range(len(label))])
        data = {"img": img, "labels": label, "layout": "hwc"}
        if self.transforms:
            data = self.transforms(data)
        return data

    def __len__(self):
        return self._size


@OBJECT_REGISTRY.register
class IrisDataset(ConcatDataset):
    """IrisDataset provides the method of reading mx iris data from rec list.

    Args:
        rec_list : Path to iris rec list.
        transfroms : Transfroms of data before using.
        shuffle: Shuffle data.
        visible_index: Which label index regarded as visible.
        invisible_index: Which label index regarded as invisible.
    """

    def __init__(
        self,
        rec_list: list,
        transforms: Optional[List] = None,
        shuffle: Optional[bool] = False,
        visible_index: Optional[Tuple] = (0, 2),
        invisible_index: Optional[Tuple] = (1, 3, 4),
    ):
        self.transforms = transforms
        self.path_reclist = rec_list
        self.path_idxlist = [rec.replace(".rec", ".idx") for rec in rec_list]
        self.visible_index = visible_index
        self.invisible_index = invisible_index
        self.shuflle = shuffle

        self._init_pack()

        super(IrisDataset, self).__init__(self.dataset_list)

    def _init_pack(self):
        self.dataset_list = []
        for rec, idx in zip(self.path_reclist, self.path_idxlist):
            self.dataset_list.append(
                RecDataset(
                    rec_path=rec,
                    idx_path=idx,
                    transforms=self.transforms,
                    visible_index=self.visible_index,
                    invisible_index=self.invisible_index,
                    shuffle=self.shuflle,
                )
            )
