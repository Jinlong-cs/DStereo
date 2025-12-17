from typing import Dict, List, Tuple, Union

import numpy as np

from hat.registry import OBJECT_REGISTRY
from .real3d import draw_reg_map, fill_mask_by_bbox, get_gaussian2D

__all__ = [
    "CenterNetTargetGenerator",
]


# TODO(xiang.yan): implement target_generator  on gpu to accelerate
@OBJECT_REGISTRY.register
class CenterNetTargetGenerator(object):
    """Generate gound truth labels for Real3dDesensitization.

    Args:
        input_size: The width and heigh of the input images
        head_channels (dict): a dict like {'hm':3,'dep':1,} to config output
            device nums.
        down_stride (int): The downstride between input size to output
            result size

    """

    def __init__(
        self,
        input_size: Union[Tuple, List],
        head_channels: Dict,
        down_stride: int = 4,
    ):
        self.input_size = input_size
        self.head_channels = head_channels
        self.down_stride = down_stride
        width, height = self.input_size
        assert width % self.down_stride == 0 and height % self.down_stride == 0
        self.output_width = int(width // self.down_stride)
        self.output_height = int(height // self.down_stride)

    def __call__(self, data):
        target = {}
        for head_name, channel_num in self.head_channels.items():
            if channel_num > 1 or head_name == "hm":
                target[head_name] = np.zeros(
                    (self.output_height, self.output_width, channel_num),
                    dtype=np.float32,
                )
        ignore_mask = np.zeros(
            (self.output_height, self.output_width), dtype=np.float32
        )

        gt_bboxes = data["gt_bboxes"]
        gt_classes = data["gt_classes"]
        order = np.argsort(gt_classes)
        gt_bboxes = gt_bboxes[order]
        gt_classes = gt_classes[order]
        for gt_class, gt_bbox in zip(gt_classes, gt_bboxes):
            bbox = gt_bbox / self.down_stride
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_width - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_height - 1)
            if gt_class < 0:
                fill_mask_by_bbox(ignore_mask, bbox, value=1.0)
                continue
            fill_mask_by_bbox(ignore_mask, bbox, value=0.0)
            # _wh = bbox[2:] - bbox[:2]
            ct = (bbox[:2] + bbox[2:]) / 2
            _wh = (bbox[2:] - np.rint(ct).astype(np.int32)) * 2
            if (_wh[0] * _wh[1]) < 2:
                fill_mask_by_bbox(ignore_mask, bbox, value=1.0)
                continue
            ct_int = tuple(np.rint(ct).astype(np.int32).tolist())
            insert_hm = get_gaussian2D(_wh, alpha=4.0)
            draw_reg_map(
                target["hm"][:, :, gt_class], insert_hm, ct_int, op="max"
            )
            target["wh"][ct_int[1], ct_int[0], 0] = max(
                bbox[2] - ct_int[0], ct_int[0] - bbox[0]
            )
            target["wh"][ct_int[1], ct_int[0], 1] = max(
                bbox[3] - ct_int[1], ct_int[1] - bbox[1]
            )

        for key, value in target.items():
            target[key] = np.transpose(value, (2, 0, 1))
        target["ignore_mask"] = ignore_mask[None]
        label = {}
        img = data["img"].copy()
        label["img"] = np.transpose(img, (2, 0, 1))
        label["labels"] = target
        return label
