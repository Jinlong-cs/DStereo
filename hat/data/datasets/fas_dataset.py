# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Union

import cv2
import msgpack
import numpy as np
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.mxrecord import MXRecord, unpack_img
from hat.utils.pack_type.utils import get_packtype_from_path

__all__ = ["FasRecDataset"]


class FasSingleRecDataset(data.Dataset):
    """Anti-spoof dataset read image from rec file.

    Args:
        rec_path: Path to rec file.
        transforms: List of data transform. Default to None.
        car_classes: Car type of imgs in the dataset. Defaults to 0.
    """

    def __init__(
        self,
        rec_path: str,
        transforms: Optional[List] = None,
        car_classes: int = 0,
    ):
        self.rec_path = rec_path
        self.car_classes = car_classes
        self.dataset = None
        self.samples = 0

        self.load_data()

        self.transforms = transforms

    def load_data(self):
        idx_path = self.rec_path[:-4] + ".idx"
        self.dataset = MXRecord(self.rec_path, idx_path, writable=False)
        self.dataset.open()
        self.samples = self.dataset.get_keys()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        attr_s = self.dataset.read(self.samples[index])
        header, img = unpack_img(attr_s, iscolor=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fas_label = header.label
        data = {
            "img": img,
            "img_shape": img.shape,
            "fas_label": int(fas_label),
            "database_labels": self.car_classes,
            "layout": "hwc",
            "color_space": "rgb",
        }
        if self.transforms is not None:
            data = self.transforms(data)
        return data


@OBJECT_REGISTRY.register
class FasRecDataset(data.ConcatDataset):
    """
    FasRecDataset read image from rec file list.

    Args:
        imgrec_path_list: List of path to rec file.
        transforms: List of data transform. Default to None.
        database_labels: Car types of imgs in datasets.
    """

    def __init__(
        self,
        imgrec_path_list: List[str],
        transforms: Optional[List] = None,
        database_labels: Optional[List] = None,
    ):
        self.transforms = transforms
        self.imgrec_path_list = imgrec_path_list
        self.database_labels = database_labels
        if self.database_labels is None:
            self.database_labels = [0] * len(imgrec_path_list)

        self._init_pack()

        super(FasRecDataset, self).__init__(self.dataset_list)

    def _init_pack(self):
        assert len(self.imgrec_path_list) == len(self.database_labels)
        self.dataset_list = []
        for rec_path, car_cls in zip(
            self.imgrec_path_list, self.database_labels
        ):
            self.dataset_list.append(
                FasSingleRecDataset(
                    rec_path=rec_path,
                    transforms=self.transforms,
                    car_classes=int(car_cls),
                )
            )


class FasSingleLmdbDataset(data.Dataset):
    def __init__(
        self,
        image_path: str,
        anno_path: str,
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms
        self.pack_type = get_packtype_from_path(image_path)
        self.data_pack = self.pack_type(image_path, writable=False)
        self.data_pack.open()

        self.anno_pack = self.pack_type(anno_path, writable=False)
        self.anno_pack.open()
        self.samples = self.anno_pack.get_keys()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {}
        key = self.samples[index]
        img_data = msgpack.unpackb(self.data_pack.read(key), raw=False)
        img = cv2.imdecode(
            np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert img.ndim == 3
        data["img"] = img
        data["img_shape"] = img.shape
        data["layout"] = "hwc"
        data["color_space"] = "rgb"

        anno_data = msgpack.unpackb(self.anno_pack.read(key), raw=False)
        data["fas_label"] = anno_data["liveness"]
        data["car_type"] = anno_data["car_type"]

        if self.transforms is not None:
            data = self.transforms(data)

        return data


@OBJECT_REGISTRY.register
class FasLmdbDataset(data.ConcatDataset):
    """Face antispoofing datasets for lmdb file.

    This dataset is designed for face multi-tasks.
    """

    def __init__(
        self,
        image_lmdb_list: Union[List[str], str],
        anno_lmdb_list: Union[List[str], str],
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms
        if isinstance(image_lmdb_list, str):
            image_lmdb_list = [image_lmdb_list]
        if isinstance(anno_lmdb_list, str):
            anno_lmdb_list = [anno_lmdb_list]

        assert isinstance(image_lmdb_list, list)
        assert isinstance(anno_lmdb_list, list)

        assert len(image_lmdb_list) == len(anno_lmdb_list)
        datasets = []
        for image, anno in zip(image_lmdb_list, anno_lmdb_list):
            datasets.append(FasSingleLmdbDataset(image, anno, transforms))
        super(FasLmdbDataset, self).__init__(datasets)
