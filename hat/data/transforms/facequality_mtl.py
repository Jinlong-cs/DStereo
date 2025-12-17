# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Any, Dict

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "FaceQualityTransformLabel",
]


@OBJECT_REGISTRY.register
class FaceQualityTransformLabel(object):
    """Transform label to fit multitask framework."""

    def __call__(self, data: Dict[str, Any]):
        gt_task_dict = data.pop("gt_face_quality")
        for task, value in gt_task_dict.items():
            task_gt = {}
            task_gt[f"gt_{task}"] = value
            data[task] = task_gt
        return data
