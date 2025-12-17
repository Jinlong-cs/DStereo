# Copyright (c) Horizon Robotics. All rights reserved.
import os
from typing import List, Optional, Union

import cv2
import msgpack
import numpy as np
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.mxrecord import MXRecord, unpack_img
from hat.utils.pack_type.utils import get_packtype_from_path

__all__ = ["LdmkRecDataset", "LdmkLmdbDataset", "LdmkDataset"]


class LdmkRecDataset(data.Dataset):
    """Landmark dataset which reads image from mxnet rec file.

    This dataset is created based on GluonFace-style rec and label.
    The label info consists landmark(2N or 3N), landmark_attr(N), img_attr(3),
    where N is the number of landmarks.
    """

    def __init__(
        self,
        filename: str,
        num_ldmk: int,
        data_desc: str = "",
        use_3d: bool = False,
        task_type: str = "face",
        transforms: Optional[List] = None,
        ldmk_pairs: Optional[List] = None,
    ):
        self._rec = filename
        self._idx = os.path.splitext(filename)[0] + ".idx"
        self.data_desc = data_desc
        self.num_ldmk = num_ldmk
        self.dim = 3 if use_3d else 2
        self.task_type = task_type.lower()
        self.transforms = transforms
        self.ldmk_pairs = ldmk_pairs
        self.load_data()

    def load_data(self):
        self.pack_file = MXRecord(self._rec, self._idx, writable=False)
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()

    def __getitem__(self, idx):
        item = self.pack_file.read(self.samples[idx])
        header, img = unpack_img(item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color_space = "rgb"
        label = header.label
        gt_ldmk_attr = label[self.dim * self.num_ldmk : -3]
        gt_ldmk_weight = np.ones((self.num_ldmk, self.dim))
        gt_ldmk_weight[gt_ldmk_attr < 0] = 0
        data = {
            "img": img,
            "img_shape": img.shape,
            "gt_ldmk": label[: self.dim * self.num_ldmk].reshape(
                (self.num_ldmk, self.dim)
            ),
            "gt_ldmk_weight": gt_ldmk_weight,
            "gt_ldmk_attr": gt_ldmk_attr,
            "layout": "hwc",
            "ldmk_pairs": self.ldmk_pairs,
            "color_space": color_space,
            "data_desc": self.data_desc,
        }
        if self.task_type == "face":
            data["head_pose"] = label[-3:]
        elif self.task_type == "hand":
            right_hand = label[-1]
            data["right_hand"] = right_hand
            data["right_hand_weight"] = 0 if right_hand < 0 else 1

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.samples)


class LdmkLmdbDataset(data.Dataset):
    """Landmark dataset which reads image from lmdb file.

    This dataset is created based on `projects/halo/cv/tools/landmark`
    """

    def __init__(
        self,
        image_lmdb: str,
        anno_lmdb: str,
        num_ldmk: int,
        task_type: str = "face",
        transforms: Optional[List] = None,
        ldmk_pairs: Optional[List] = None,
        data_desc: str = "",
    ):
        self.pack_type = get_packtype_from_path(image_lmdb)
        self.data_pack = self.pack_type(image_lmdb, writable=False)
        self.data_pack.open()
        self.anno_pack = self.pack_type(anno_lmdb, writable=False)
        self.anno_pack.open()
        self.samples = self.anno_pack.get_keys()

        self.num_ldmk = num_ldmk
        self.task_type = task_type.lower()
        self.transforms = transforms
        self.ldmk_pairs = ldmk_pairs
        self.data_desc = data_desc

    def __getitem__(self, idx):
        key = self.samples[idx]

        img_data = msgpack.unpackb(self.data_pack.read(key), raw=False)
        img = cv2.imdecode(
            np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color_space = "rgb"
        anno_data = msgpack.unpackb(self.anno_pack.read(key), raw=False)
        gt_bboxes = np.array(anno_data["gt_bboxes"])
        gt_ldmk = np.array(anno_data["gt_ldmk"])
        gt_ldmk_attr = np.array(anno_data["ldmk_attr"])
        gt_ldmk_weight = np.ones_like(gt_ldmk)
        gt_ldmk_weight[gt_ldmk_attr < 0] = 0
        data = {
            "img": img,
            "img_shape": img.shape,
            "gt_bboxes": gt_bboxes,
            "gt_ldmk": gt_ldmk,
            "gt_ldmk_weight": gt_ldmk_weight,
            "gt_ldmk_attr": gt_ldmk_attr,
            "layout": "hwc",
            "ldmk_pairs": self.ldmk_pairs,
            "color_space": color_space,
            "data_desc": self.data_desc,
        }

        if self.task_type == "hand":
            data["right_hand"] = anno_data["right_hand"]
            data["right_hand_weight"] = 0 if anno_data["right_hand"] < 0 else 1

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.samples)


@OBJECT_REGISTRY.register
class LdmkDataset(data.ConcatDataset):
    """Landmark dataset for lmdb or rec files.

    Args:
        image_lmdb_list: List of image lmdb. Defaults to None.
        anno_lmdb_list: List of annotation lmdb. Defaults to None.
        rec_list: List of rec file. Defaults to None.
        num_ldmk: number of landmark. Defaults to 68.
        task_type: landmark type, e.g. "face", "hand", "body".
            Defaults to "face".
        data_type: use "rec" or "lmdb" dataset. Defaults to "lmdb".
        ldmk_pairs: left/right pairs of landmark for flip. Defaults to None.
        transforms: List of transforms. Defaults to None.
        use_3d: 3d landmark or 2d. Only available for rec dataset.
            Defaults to False.
        data_desc: dataset description. Defaults to "".
    """

    def __init__(
        self,
        image_lmdb_list: Union[List[str], str] = None,
        anno_lmdb_list: Union[List[str], str] = None,
        rec_list: Optional[Union[List[str], str]] = None,
        num_ldmk: int = 68,
        task_type: str = "face",
        data_type: str = "lmdb",
        ldmk_pairs: Optional[List] = None,
        transforms: Optional[List] = None,
        use_3d: bool = False,
        data_desc: List[str] = None,
    ):
        if data_desc is None:
            if data_type == "lmdb":
                data_desc = [""] * len(image_lmdb_list)
            elif data_type == "rec":
                data_desc = [""] * len(rec_list)
        self.transforms = transforms
        datasets = []

        if data_type.lower() == "lmdb":
            if isinstance(image_lmdb_list, str):
                image_lmdb_list = [image_lmdb_list]
            if isinstance(anno_lmdb_list, str):
                anno_lmdb_list = [anno_lmdb_list]
            assert isinstance(image_lmdb_list, list)
            assert isinstance(anno_lmdb_list, list)
            assert len(image_lmdb_list) == len(anno_lmdb_list)
            for image, anno, desc in zip(
                image_lmdb_list, anno_lmdb_list, data_desc
            ):
                datasets.append(
                    LdmkLmdbDataset(
                        image_lmdb=image,
                        anno_lmdb=anno,
                        num_ldmk=num_ldmk,
                        task_type=task_type,
                        transforms=transforms,
                        ldmk_pairs=ldmk_pairs,
                        data_desc=desc,
                    )
                )
        elif data_type.lower() == "rec":
            if isinstance(rec_list, str):
                rec_list = [rec_list]
            assert isinstance(rec_list, list)
            for rec, desc in zip(rec_list, data_desc):
                datasets.append(
                    LdmkRecDataset(
                        filename=rec,
                        num_ldmk=num_ldmk,
                        task_type=task_type,
                        transforms=transforms,
                        ldmk_pairs=ldmk_pairs,
                        data_desc=desc,
                        use_3d=use_3d,
                    )
                )

        super(LdmkDataset, self).__init__(datasets)
