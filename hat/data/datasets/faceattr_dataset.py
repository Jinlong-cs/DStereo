# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional

import cv2
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.mxrecord import MXRecord, unpack_img

__all__ = ["FaceAttrRecDataset"]


class FaceAttrSingleDataset(data.Dataset):
    """Genger age dataset read image from rec file.

    Args:
        rec_path: Path to rec file.
        transforms: List of data transform. Default to None.
        age_classes: Predict age in [0, age_classes]. Defaults to 85.
    """

    def __init__(
        self,
        rec_path: str,
        transforms: Optional[List] = None,
        age_classes: int = 85,
    ):

        self.rec_path = rec_path
        self.dataset = None
        self.samples = 0
        self.age_classes = age_classes

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
        age, gender = header.label
        if age < 0 or age > self.age_classes:
            age = -1
        if int(gender) not in [0, 1]:
            gender = -1
        data = {
            "img": img,
            "age": int(age),
            "gender": int(gender),
            "layout": "hwc",
            "color_space": "rgb",
            "img_shape": img.shape,
        }
        if self.transforms is not None:
            data = self.transforms(data)
        return data


@OBJECT_REGISTRY.register
class FaceAttrRecDataset(data.ConcatDataset):
    """FaceAttrRecDataset read image from rec file list.

    Args:
        imgrec_path_list: List of path to rec file.
        transforms: List of data transform. Default to None.
        age_classes: Predict age in [0, age_classes]. Defaults to 85.
    """

    def __init__(
        self,
        imgrec_path_list: List[str],
        transforms: Optional[List] = None,
        age_classes: int = 85,
    ):

        self.transforms = transforms
        self.imgrec_path_list = imgrec_path_list
        self.age_classes = age_classes

        self._init_pack()

        super(FaceAttrRecDataset, self).__init__(self.dataset_list)

    def _init_pack(self):
        self.dataset_list = []
        for rec_path in self.imgrec_path_list:
            self.dataset_list.append(
                FaceAttrSingleDataset(
                    rec_path=rec_path,
                    transforms=self.transforms,
                    age_classes=self.age_classes,
                )
            )
