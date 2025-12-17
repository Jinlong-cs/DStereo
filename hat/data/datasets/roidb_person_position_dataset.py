# Copyright (c) Horizon Robotics. All rights reserved.
from copy import deepcopy

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY
from .roidb_detection_dataset import RoidbDetectionDataset

__all__ = ["RoidbPersonPositionDataset"]


@OBJECT_REGISTRY.register
class RoidbPersonPositionDataset(RoidbDetectionDataset):
    """A dataset that can read roidb (pickle) and the relation image record \
        (mxnet record) dataset.

    This class is inherits from RoidbDetectionDataset and override __getitem__
    method to organization person position data.
    """

    def __getitem__(self, index):
        data = {}
        img, anno = self.dataset[index]

        color_space = "bgr"
        if self.keep_ori_img:
            data["ori_img"] = deepcopy(img)
        if self.to_rgb:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            color_space = "rgb"

        gt_bboxes = []
        gt_classes = []
        gt_position_dms = []
        gt_position_oms = []
        # if no person position label in data, generate ignore data to cover.
        if "position_oms" not in anno and "position_dms" not in anno:
            gt_position_dms.append(-1)
            gt_bboxes.append([0, 0, 1, 1])
            gt_classes.append(-1)
            anno["camera"] = "DMS"
        else:
            idx = 0
            for bbox, class_id in zip(anno["boxes"], anno["classes"]):
                # positive class id in selected class list.
                if class_id in self.valid_selected_class_ids:
                    gt_bboxes.append(bbox)
                    gt_classes.append(self.class_id_map[class_id])
                    if (
                        "position_dms" in anno
                        and anno["position_dms"].shape[0] > 0
                    ):
                        gt_position_dms.append(anno["position_dms"][idx])
                    if (
                        "position_oms" in anno
                        and anno["position_oms"].shape[0] > 0
                    ):
                        gt_position_oms.append(anno["position_oms"][idx])
                    idx += 1
                # negative class id in selected class list.
                elif -class_id in self.valid_selected_class_ids:
                    gt_bboxes.append(bbox)
                    gt_classes.append(-self.class_id_map[-class_id])
                    if (
                        "position_dms" in anno
                        and anno["position_dms"].shape[0] > 0
                    ):
                        gt_position_dms.append(-1)
                    if (
                        "position_oms" in anno
                        and anno["position_oms"].shape[0] > 0
                    ):
                        gt_position_oms.append(-1)
                    idx += 1

        data = {
            "img": img,
            "img_name": anno["image_name"],
            "img_height": anno["image_height"],
            "img_width": anno["image_width"],
            "img_shape": img.shape,
            "color_space": color_space,
            "layout": "hwc",
            # gt_bboxes contain the coordinates of each box
            "gt_bboxes": np.array(gt_bboxes, dtype=np.float32).reshape(
                (-1, 4)
            ),
            # gt_classes represent class type of each box,
            # 1 represent person box and only use person box in this class.
            "gt_classes": np.array(gt_classes, dtype=np.int64).reshape(
                -1,
            ),
            # Only one attribute has value between gt_position_dms and
            # gt_position_oms, depending on which camera the image was taken.
            # It represent class type of each person box, the class type is:
            # {0: unknown; 1:driver; 2:copilot; 3:passenger}
            "gt_position_dms": np.array(gt_position_dms),
            "gt_position_oms": np.array(gt_position_oms),
        }
        if self.data_desc is not None:
            data["data_desc"] = self.data_desc
        if "camera" in anno:
            data["camera"] = anno["camera"]

        if self.transforms is not None:
            data = self.transforms(data)
        return data
