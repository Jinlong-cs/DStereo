import numpy as np

from hat.core.box_np_ops import (
    dropout_points_in_gt,
    make_truncate_points,
    points_count_rbbox,
)


def test_points_count_rbbox():
    gt_boxes = np.array(
        [
            [0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.01],
            [0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.01],
        ]
    )
    points = np.array([[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]])
    point_counts = points_count_rbbox(points, gt_boxes)

    assert point_counts.shape[0] == 2
    assert point_counts[0] == 1
    assert point_counts[1] == 0


def test_make_truncate_points():
    gt_boxes = np.array(
        [
            [0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.01],
            [0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.01],
        ]
    )
    points = np.array(
        [
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
        ]
    )
    truncate_angle = (10, 170)
    points_filter = make_truncate_points(points, gt_boxes, truncate_angle)

    assert points_filter.shape[0] == 2
    assert points_filter[0, 1] == -0.5
    assert points_filter[1, 1] == -0.5


def test_dropout_points_in_gt():
    gt_boxes = np.array(
        [
            [0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.0],
            [0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.0],
        ]
    )
    points = np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]])
    points_drop = dropout_points_in_gt(points, gt_boxes, 1.0)

    assert points_drop.shape[0] == 0
