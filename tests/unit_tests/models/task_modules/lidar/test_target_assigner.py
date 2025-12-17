import numpy as np
import torch

from hat.models.task_modules.lidar import GroundBox3dCoder, LidarTargetAssigner


def test_target_assigner():

    labels = [
        {
            "gt_boxes": torch.randn([3, 7]),
            "gt_classes": torch.ones(3, dtype=torch.int32),
            "gt_names": np.array(["Car", "Car", "Car"]),
        }
    ]

    anchor_list = [torch.randn(2, 2, 2, 7)]
    matched_thresholds = [torch.ones(8)]
    unmatched_thresholds = [torch.zeros(8)]

    target_assigner = LidarTargetAssigner(
        box_coder=GroundBox3dCoder(),
        class_names=["Car"],
        positive_fraction=-1,
        sample_size=2,
    )

    bbox_targets, cls_labels, reg_weights = target_assigner(
        anchor_list, matched_thresholds, unmatched_thresholds, labels
    )

    assert bbox_targets.shape == (1, 8, 7)
    assert cls_labels.shape == (
        1,
        8,
    )
    assert reg_weights.shape == (
        1,
        8,
    )
