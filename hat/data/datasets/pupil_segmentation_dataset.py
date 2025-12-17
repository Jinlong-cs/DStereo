# Copyright (c) Horizon Robotics. All rights reserved.
import copy
from typing import List, Optional

import cv2
import msgpack
import numpy as np
import torch
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.utils import get_packtype_from_path

__all__ = ["PupilSegDataset"]


@OBJECT_REGISTRY.register
class PupilSegDataset(data.Dataset):
    """Dataset for pupil segmentation.

    Read raw image, bbox, pupil ellipse parameters from single lmdb file.
    Return cropped eye roi. Pupil mask, pupil center, dist map,
    spatial weights, normed pupil ellipse parameters will alse be
    returned when data_type is not `predict`.

    Args:
        image_path: The path of image lmdb file.
        anno_path: The path of anno lmdb file.
        transforms: Transforms of data augmentation.
        data_type: Indicates whether the data is used for training,
            validation or prediction. Only `train`, `val` and `predict` are
            supported. When `predict`, anno_path can be None.
        pack_kwargs: Kwargs for pack type.
    """

    def __init__(
        self,
        image_path: str,
        anno_path: Optional[str] = None,
        data_type: str = "train",
        transforms: Optional[List] = None,
        pack_kwargs: Optional[dict] = None,
    ):
        self.data_type = data_type
        assert self.data_type in [
            "train",
            "val",
            "predict",
        ], "data_type must be `train`, `val` or `predict`!"
        self.transforms = transforms
        self.image_path = image_path
        self.anno_path = anno_path
        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        self.pack_type = get_packtype_from_path(image_path)
        if self.data_type != "predict":
            assert (
                self.anno_path is not None
            ), "When data_type is not `predict`, anno path cannot be None!"
            self.anno_pack = self.pack_type(
                self.anno_path, writable=False, **self.kwargs
            )
            self.anno_pack.open()

        self.data_pack = self.pack_type(
            image_path, writable=False, **self.kwargs
        )
        self.data_pack.open()
        self.samples = self.data_pack.get_keys()
        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    def __repr__(self):
        return "PupilSegDataset"

    def process_predict_data(self, data):
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def process_train_data(self, data):
        if self.transforms is not None:
            data = self.transforms(data)
        data["gt_pupil_ellipse_param"] = torch.from_numpy(
            data["gt_pupil_ellipse_param"]
        ).to(torch.float32)
        data["gt_pupil_center"] = copy.deepcopy(
            data["gt_pupil_ellipse_param"][:2]
        )
        if "gt_pupil_mask" in data.keys():
            data["gt_pupil_mask"] = torch.from_numpy(data["gt_pupil_mask"]).to(
                torch.long
            )
        if "spat_weights" in data.keys():
            data["spat_weights"] = torch.from_numpy(data["spat_weights"]).to(
                torch.float32
            )
        if "dist_map" in data.keys():
            data["dist_map"] = torch.from_numpy(data["dist_map"]).to(
                torch.float32
            )
        if "gt_norm_pupil_ellipse_param" in data.keys():
            data["gt_norm_pupil_ellipse_param"] = torch.from_numpy(
                data["gt_norm_pupil_ellipse_param"]
            ).to(torch.float32)
        return data

    def __getitem__(self, index):
        key = self.samples[index]
        img_data = msgpack.unpackb(self.data_pack.read(key), raw=False)
        img = cv2.imdecode(
            np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = {"img": img, "layout": "hwc"}
        if self.data_type == "predict":
            return self.process_predict_data(data)
        else:
            anno_data = msgpack.unpackb(self.anno_pack.read(key), raw=False)
            if "gt_bboxes" in anno_data.keys():
                gt_bbox = np.array(anno_data["gt_bboxes"]).astype(np.float32)
                data.update({"gt_bboxes": gt_bbox})
            gt_pupil_ellipse_param = np.array(
                anno_data["gt_pupil_ellipse_param"]
            ).astype(np.float32)
            data.update({"gt_pupil_ellipse_param": gt_pupil_ellipse_param})
            return self.process_train_data(data)
