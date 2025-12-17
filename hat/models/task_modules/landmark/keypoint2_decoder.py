from typing import Dict, List

import numpy as np
import torch
from torch import nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["KeyPoint2DDecoder"]


class KeyPointSet:
    def __init__(self, keypoints) -> None:
        self.keypoints = keypoints


@OBJECT_REGISTRY.register
class KeyPoint2DDecoder(nn.Module):
    """wheel key points decoder.

    Args:
        num_points: keypoints nums.
        feature_stride: stride of feature map.
        points_type: keypoints names.
        kps_pos_distance_xy: anchor length.
        layout: feature map layout.
    """

    def __init__(
        self,
        num_points: int,
        feature_stride: int,
        points_type: List,
        kps_pos_distance_xy: List,
        layout: str = "NCHW",
    ):
        super().__init__()
        self.num_points = num_points
        self.feature_stride = feature_stride
        self.points_type = points_type
        self.kps_pos_distance_xy = kps_pos_distance_xy
        self.layout = layout

    def forward(self, data, label):
        cls_feature = data["kps_label_pred"]
        reg_feature = data["kps_pos_offset_pred"]

        kps = decode_keypoint2d(
            cls_feature,
            reg_feature,
            num_points=self.num_points,
            feature_stride=self.feature_stride,
            points_type=self.points_type,
            kps_pos_distance_xy=self.kps_pos_distance_xy,
            layout=self.layout,
            label=label,
        )

        kps = KeyPointSet(kps)
        return {"pred_kps": kps}


def decode_keypoint2d(
    cls_feature,
    reg_feature,
    num_points: int,
    feature_stride: int,
    points_type: List,
    label: Dict,
    kps_pos_distance_xy: List,
    layout: str = "NCHW",
):  # noqa
    """wheel keypoints decoder.

    Args:
        cls_feature: feature map of classification.
        reg_feature: feature map of regression.
        num_points: keypoints nums.
        feature_stride: stride of feature map.
        points_type: keypoints names.
        kps_pos_distance_xy: anchor length.
        layout: feature map layout.
    """
    assert len(points_type) == num_points

    assert layout in ["NCHW", "NHWC"]
    if layout == "NHWC":
        cls_feature = torch.permute(cls_feature, dims=(0, 3, 1, 2))
        reg_feature = torch.permute(reg_feature, dims=(0, 3, 1, 2))

    cls_feature = cls_feature.cpu().numpy()
    reg_feature = reg_feature.cpu().numpy()

    batch_size = cls_feature.shape[0]
    feat_width = cls_feature.shape[3]
    assert batch_size == reg_feature.shape[0]
    assert cls_feature.shape[1] == num_points
    assert reg_feature.shape[1] == num_points * 2

    cls_feature = cls_feature.reshape((batch_size * num_points, -1))
    reg_feature = reg_feature.reshape((batch_size * num_points, 2, -1))

    all_inds = np.arange(batch_size * num_points)
    max_inds = cls_feature.argmax(axis=1)  # (batch_size * num_points, )

    max_inds_xy = np.zeros((batch_size * num_points, 2))
    max_inds_xy[:, 0] = max_inds % feat_width
    max_inds_xy[:, 1] = max_inds // feat_width

    max_deltas_xy = reg_feature[all_inds, :, max_inds]

    max_deltas_xy[:, 0] *= kps_pos_distance_xy[0]
    max_deltas_xy[:, 1] *= kps_pos_distance_xy[1]

    pred_kps = (
        feature_stride * max_inds_xy + max_deltas_xy
    )  # (batch_size*num_points, 2)  # noqa
    pred_kps = pred_kps.reshape((batch_size, num_points, 2))
    max_scores = cls_feature[all_inds, max_inds].reshape(
        (batch_size, num_points)
    )  # noqa

    keypoint2ds = []
    for kps_i, score_i in zip(pred_kps, max_scores):
        keypoint_i = dict(  # noqa
            points=[
                dict(  # noqa
                    x=float(kps_i[pid][0]),
                    y=float(kps_i[pid][1]),
                    conf=score_i[pid],
                )
                for pid in range(num_points)
            ],
            points_type=points_type,
            is_valid=True,
        )
        keypoint2ds.append(keypoint_i)

    return keypoint2ds
