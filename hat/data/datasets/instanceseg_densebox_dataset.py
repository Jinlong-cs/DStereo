# Copyright (c) Horizon Robotics. All rights reserved.
"""Dataset for densebox mx-record data, used in auto."""
import copy
import os
from typing import Dict, Optional

import cv2
import numpy as np

from hat.data.datasets.densebox_dataset import DenseboxDataset
from hat.registry import OBJECT_REGISTRY

__all__ = ["InstSegDenseboxDataset"]


@OBJECT_REGISTRY.register
class InstSegDenseboxDataset(DenseboxDataset):
    """Densebox record dataset for instance segmentation tasks.

    Args:
        data_path : Path of data relative.
        anno_path : Path of annotation.
        task_type : must be 'instanceseg'.
        ignore_index: Ignore index for ignore regions. Default is 255.
        use_ignore : Whether to use ignore regions in annotation.
            Default is False.
        with_seg_label: Return segmentation ground truth. Default is True.
        with_polygon: Return instance polygons. Default is True.
    """

    def __init__(
        self,
        data_path: str,
        anno_path: str,
        task_type: Optional[str] = "instanceseg",
        ignore_index: int = 255,
        use_ignore: bool = True,
        with_seg_label: bool = True,
        with_polygon: bool = True,
        **kwargs,
    ):
        assert task_type == "instanceseg", task_type
        super().__init__(
            data_path=data_path,
            anno_path=anno_path,
            task_type="segmentation" if with_seg_label else "detection",
            use_ignore=use_ignore,
            **kwargs,
        )
        self.with_seg_label = with_seg_label
        self.with_polygon = with_polygon
        self.ignore_index = ignore_index

    def add_gt_labels(self, data, anno):
        gt_labels = []
        for ins in anno["instances"]:
            gt_attr = ins["attribute"]
            gt_labels.append(np.array(gt_attr, dtype=np.int32))
        data["gt_labels"] = gt_labels
        return data

    def add_gt_polygons(self, data, anno):
        gt_polygons = []
        for ins in anno["instances"]:
            gt_polygon = ins["mask_poly"]
            gt_polygons.append(np.array(gt_polygon))
        data["gt_polygons"] = gt_polygons
        return data

    def add_ignore_regions(self, data, anno):
        for ignore_region in anno["ignore_regions"]:
            ignore_label = copy.deepcopy(data["gt_labels"][0])
            ignore_label[0] = self.ignore_index
            ignore_label[1] = self.ignore_index
            data["gt_labels"].insert(0, ignore_label)
            data["gt_polygons"].insert(0, np.array(ignore_region["contour"]))
        return data

    def __getitem__(self, index: int) -> Dict:
        data = {}
        image, anno = self.dataset[index]
        color_space = "bgr"
        if self.to_rgb:
            # cv2.cvtColor may be slow.
            # See http://wiki.hobot.cc/pages/viewpage.action?pageId=186775106 for more details.     # noqa
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_space = "rgb"

        if self.with_seg_label:
            seg_label = anno[1]
            seg_label = seg_label.astype(np.uint8)
            anno = anno[0]
        else:
            anno = anno
        anno = anno.to_dict()
        data["img_name"] = os.path.basename(anno["img_url"])
        data["img_height"] = anno["img_h"]
        data["img_width"] = anno["img_w"]
        data["img_id"] = np.expand_dims(anno["idx"], 0)
        data["img"] = image
        data["color_space"] = color_space
        data["layout"] = "hwc"
        data["img_shape"] = image.shape
        if self.return_orig_img:
            data["orig_img"] = image.copy()

        if self.with_seg_label:
            data["gt_seg"] = seg_label
            if self.return_orig_gt_seg:
                data["orig_gt_seg"] = seg_label

        # gt attributes of instances
        data = self.add_gt_labels(data, anno)

        # gt mask polygons of instances
        if self.with_polygon:
            data = self.add_gt_polygons(data, anno)
            # polygons of ignore regions
            if self.use_ignore and "ignore_regions" in anno:
                self.add_ignore_regions(data, anno)

            assert len(data["gt_polygons"]) == len(data["gt_labels"]), (
                len(data["gt_polygons"]),
                len(data["gt_labels"]),
                data["img_id"],
            )

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __repr__(self):
        return "InstSegDenseboxDataset"
