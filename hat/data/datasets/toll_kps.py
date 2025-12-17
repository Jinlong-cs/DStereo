# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import List, Optional

import cv2
import numpy as np

from hat.data.datasets.legacy_densebox import LegacyDenseBoxImageRecordDataset
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "TollgateDataset",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class TollgateDataset(LegacyDenseBoxImageRecordDataset):
    """Tollgate dataset which reads image from rec file.

    Args:
        rec_path: Image record path.
        anno_path: Annotation path.
        rec_idx_file_path: Image record index file path.
            if None use:rec_path + '.idx'
        read_only: Whether output raw content, by default False
        transforms: Transforms of data augmentation.
        to_rgb: Whether output image in rgb color, by default True
        with_img_buf: Whether the raw content with img buf.
        with_seg_label: Whether the raw content with segmentation label.
        seg_label_dtype: The output data type of segmentation label.
    """

    def __init__(
        self,
        rec_path: str,
        anno_path: str,
        rec_idx_file_path: str = None,
        read_only: bool = False,
        transforms: Optional[List] = None,
        with_img_buf: bool = False,
        with_seg_label: bool = False,
        to_rgb: Optional[bool] = False,
        seg_label_dtype: type = np.uint8,
    ):
        super(TollgateDataset, self).__init__(
            rec_path,
            anno_path,
            read_only,
            with_img_buf,
            with_seg_label,
            to_rgb,
            rec_idx_file_path,
            seg_label_dtype,
        )

        self.transforms = transforms
        self.to_rgb = to_rgb

    def __len__(self):
        return len(self._rec_dataset)

    def __getitem__(self, idx):
        image, anno = super().__getitem__(idx)
        color_space = "bgr"
        data = {}
        if self.to_rgb:
            # cv2.cvtColor may be slow.
            # See http://wiki.hobot.cc/pages/viewpage.action?pageId=186775106 for more details.     # noqa
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise NotImplementedError("ERROR: func not implemented!")
            color_space = "rgb"
        labels = np.array(
            [res.points_data + res.attribute for res in anno.instances]
            if (len(anno.instances) > 0)
            else []
        )
        data["img_name"] = anno.img_url.split("/")[-1]
        data["gt_lines"] = (
            np.array([labels[::2], labels[1:][::2]])
            if len(labels) != 0
            else np.array([])
        )
        data["img_height"] = anno.img_h
        data["img_width"] = anno.img_w
        data["img_id"] = np.expand_dims(anno.idx, 0)
        data["img"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        if self.transforms is not None:
            data = self.transforms(data)
        return data
