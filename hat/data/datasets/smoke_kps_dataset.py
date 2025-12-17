# Copyright (c) Horizon Robotics. All rights reserved.
"""Dataset for mx-record rec data, used in smoke keypoints."""

try:
    import mxnet as mx
except ImportError:
    mx = None

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.mxrecord import MXRecord
from hat.utils.package_helper import require_packages

__all__ = ["SmokeKpsRecDataset"]
logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SmokeKpsRecDataset(data.Dataset):
    """Smoke Kps dataset which reads image from mxnet rec file.

    This dataset is created based on GluonFace-style rec and label.
    The label info consists landmark(2N),
    where N is the number of landmarks.

    Args:
        filename: .rec file name.
        num_ldmk: number of landmarks.
        num_class: class number of smoke classification.
        task_type: Defaults to "smoke_kps".
        transforms: transorm list. Defaults to None.
    """

    @require_packages("mxnet")
    def __init__(
        self,
        filename: str,
        num_ldmk: int = 4,
        num_class: int = 2,
        roi_scale: float = 1.0,
        task_type: str = "smoke_kps",
        net_input_size: Tuple[int, int] = (128, 128),
        transforms: Optional[List] = None,
        ldmk_pairs: Optional[List] = None,
        img_params: Optional[Dict] = None,
        _ldmk_vector_head_params: Optional[Dict] = None,
    ):

        self._rec = filename
        self._idx = os.path.splitext(filename)[0] + ".idx"

        self.num_ldmk = num_ldmk
        self.num_class = num_class
        self.roi_scale = roi_scale
        self.num_coords = 2
        self.task_type = task_type.lower()
        self.net_input_size = net_input_size
        self.transforms = transforms
        self.ldmk_pairs = ldmk_pairs
        self.img_params = img_params
        self._ldmk_vector_head_params = _ldmk_vector_head_params

        self.pack_file = MXRecord(self._rec, self._idx, writable=False)
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()
        self.ignore_clsid = ["5"]  # "5": cigarette not in mouth

    def __getitem__(self, idx: int):
        """Get img and label info from *.rec file.

        Smoke kps label is array type (ldmk + label_encode),
        Smoke cls label is str type (label_encode).

        The label_encode(e.g. 2000130000) is designed as:
        - Cigaret visibility(0)
        - Smoke class(1-3)
        - Image blur(4)
        - Camera type(5)
        - Car type(6-8)
        - Driver position(9)

        Returns:
            data (dict):
                'img', 'gt_visable', 'gt_classes',
                'gt_ldmk', 'gt_ldmk_attr', 'layout', 'ldmk_paris',
                'gt_vector_x', 'gt_vector_weight_x',
                'gt_vector_y', 'gt_vector_weight_y',
                'gt_heatmap', 'gt_heatmap_weight',
                'img_shape', 'pad_shape'
        """
        item = self.pack_file.read(self.samples[idx])
        header, img = mx.recordio.unpack_img(item, iscolor=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.net_input_size[1], self.net_input_size[0]))
        label = header.label

        # get vis and cls from label encode
        if self.task_type == "smoke_kps":
            if isinstance(header.label, np.ndarray):
                label_encode = str(int(header.label[-1]))
            else:
                label_encode = str(int(header.label))
            vis = int(label_encode[1] not in self.ignore_clsid)
            _cls_label = (int(label_encode[0]) > 1) * vis

        # rebuild the label: [x1,y1,x2,y2,x3,y3,x4,y4,vis,cls]
        label = np.zeros(self.num_coords * self.num_ldmk + 2, dtype="float32")
        label[-2:] = [vis, _cls_label]
        if isinstance(header.label, np.ndarray) and len(header.label) > 1:
            label[:-2] = header.label[: self.num_coords * self.num_ldmk]
        gt_ldmk = label[: self.num_coords * self.num_ldmk].reshape(
            (self.num_ldmk, self.num_coords)
        ) * np.array([128, 128])
        gt_ldmk_attr = np.ones(self.num_ldmk, dtype="float32") * np.any(
            gt_ldmk
        )

        # organize img info in dict
        data = {
            "img": img.astype(np.uint8),
            "gt_visable": vis,
            "gt_classes": _cls_label,
            "gt_ldmk": gt_ldmk,
            "gt_ldmk_attr": gt_ldmk_attr,
            "layout": "hwc",
            "roi_scale": np.float32(self.roi_scale),
            "img_height": self.net_input_size[0],
            "img_width": self.net_input_size[1],
        }

        # data transform
        if self.transforms is not None:
            if isinstance(self.transforms, list):
                for transform in self.transforms:
                    data = transform(data)
            else:
                data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.samples)
