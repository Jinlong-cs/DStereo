import numpy as np
import pytest
import torch

from hat.core.box_utils import (
    bbox_filter_by_hw,
    bbox_overlaps,
    center_to_corner_box2d,
    corners_nd,
    get_bev_bbox,
    is_center_of_bboxes_in_roi,
    minmax_to_corner_2d,
    rotation_2d,
    zoom_boxes,
)

bboxes1 = torch.FloatTensor(
    [
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [32, 32, 38, 42],
    ]
)
bboxes2 = torch.FloatTensor(
    [
        [0, 0, 10, 20],
        [0, 10, 10, 19],
        [10, 10, 20, 20],
    ]
)
bboxes3 = torch.FloatTensor(
    [
        [
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [32, 32, 38, 42],
        ],
        [[0, 0, 10, 20], [0, 10, 10, 20], [0, 10, 20, 20]],
    ]
)


bboxes4 = torch.FloatTensor(
    [
        [
            [0, 0, 10, 20],
            [0, 10, 10, 19],
            [10, 10, 20, 20],
        ],
        [
            [0, 0, 10, 20],
            [0, 2, 10, 10],
            [10, 10, 20, 20],
        ],
    ]
)


empty = torch.empty(0, 4)
nonempty = torch.FloatTensor([[0, 0, 10, 9]])

ans_overlaps1 = torch.tensor(
    [
        [0.5000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000],
    ]
)
ans_overlaps2 = torch.tensor([0.5000, 0.0000, 0.0000])
ans_overlaps3 = torch.empty((0, 1))
ans_overlaps4 = torch.empty((1, 0))
ans_overlaps5 = torch.empty((0, 0))
ans_overlaps6 = torch.tensor(
    [[0.5000, 0.0000, 0.0000], [1.0000, 0.0000, 0.5000]]
)
ans_overlaps7 = torch.tensor(
    [
        [
            [0.5000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000],
            [0.0000, 0.0000, 0.0000],
        ],
        [
            [1.0000, 0.4000, 0.0000],
            [0.5000, 0.0000, 0.0000],
            [1 / 3.0, 0.0000, 0.5000],
        ],
    ]
)


@pytest.mark.parametrize(
    ["bboxes1", "bboxes2", "ans_overlaps", "is_aligned"],
    [
        pytest.param(bboxes1, bboxes2, ans_overlaps1, False),
        pytest.param(bboxes1, bboxes2, ans_overlaps2, True),
        pytest.param(empty, nonempty, ans_overlaps3, False),
        pytest.param(nonempty, empty, ans_overlaps4, False),
        pytest.param(empty, empty, ans_overlaps5, False),
        pytest.param(
            bboxes1.numpy(), bboxes2.numpy(), ans_overlaps1.numpy(), False
        ),
        pytest.param(
            bboxes1.numpy(), bboxes2.numpy(), ans_overlaps2.numpy(), True
        ),
        pytest.param(
            bboxes3.numpy(), bboxes4.numpy(), ans_overlaps6.numpy(), True
        ),
        pytest.param(bboxes3, bboxes4, ans_overlaps6, True),
        pytest.param(bboxes3, bboxes4, ans_overlaps7, False),
    ],
)
def test_bbox_overlaps(bboxes1, bboxes2, ans_overlaps, is_aligned):
    overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=is_aligned)
    assert isinstance(bboxes1, type(overlaps))
    if isinstance(overlaps, np.ndarray):
        overlaps = torch.from_numpy(overlaps)
        ans_overlaps = torch.from_numpy(ans_overlaps)
    assert torch.equal(overlaps, ans_overlaps)


def test_is_center_of_bboxes_in_roi():
    bboxes1 = np.array([[20, 25, 40, 55], [10, 30, 50, 50]])
    bboxes2 = np.array([25, 35, 50, 60])
    mask = is_center_of_bboxes_in_roi(bboxes1, bboxes2)
    assert mask.all()


def test_corners_nd():
    """Test corners_nd function. 3d boxes only."""
    # Prepare 3 3-d boxes. Numbers represent dimensions.
    dimensions = np.array(
        [[1.0, 1.0, 1.0], [4.0, 2.0, 1.0], [0.5, 2.0, 3.0]], dtype=np.float32
    )
    # Origin point. To center the box around (0, 0, 0), set to 0.5.
    origin = 0.5
    corners = corners_nd(dimensions, origin)

    target_corners = np.array(
        [
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
            ],
            [
                [-2.0, -1.0, -0.5],
                [-2.0, -1.0, 0.5],
                [-2.0, 1.0, 0.5],
                [-2.0, 1.0, -0.5],
                [2.0, -1.0, -0.5],
                [2.0, -1.0, 0.5],
                [2.0, 1.0, 0.5],
                [2.0, 1.0, -0.5],
            ],
            [
                [-0.25, -1.0, -1.5],
                [-0.25, -1.0, 1.5],
                [-0.25, 1.0, 1.5],
                [-0.25, 1.0, -1.5],
                [0.25, -1.0, -1.5],
                [0.25, -1.0, 1.5],
                [0.25, 1.0, 1.5],
                [0.25, 1.0, -1.5],
            ],
        ],
        dtype=np.float32,
    )

    assert np.all(corners == target_corners)


def test_rotation_2d():
    """Test 2d point rotation function."""
    # Prepare 1 point set containing 2 points.
    points = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    # Prepare a rotation angle for this point set.
    rot_angle = np.array([np.pi / 4], dtype=np.float32)
    rotated_points = rotation_2d(points, rot_angle)
    target_points = np.array(
        [[[0.707, -0.707], [0.707, 0.707]]], dtype=np.float32
    )
    assert np.allclose(rotated_points, target_points, atol=0.001)


def test_center_to_corner_box2d():
    """Test center_to_corner_2d function."""
    # Prepare a single box located at (1, 0) with size (1, 1), rotated pi/4.
    box_centers = np.array([[1.0, 0.0]], dtype=np.float32)
    box_dimension = np.array([[1.0, 1.0]], dtype=np.float32)
    angle = np.array([np.pi / 4], dtype=np.float32)
    corners_box = center_to_corner_box2d(box_centers, box_dimension, angle)
    target_representation = np.array(
        [[[0.2928, 0.0], [1, 0.707], [1.707, 0.0], [1.0, -0.707]]],
        dtype=np.float32,
    )
    assert np.allclose(corners_box, target_representation, atol=0.001)


def test_minmax_to_corner_2d():
    """Test minmax_to_corner_2d function."""
    box = np.array([[-76.8, -76.8, 76.8, 76.8]], dtype=np.float32)
    corner_box = minmax_to_corner_2d(box)
    target_representation = np.array(
        [[[-76.8, -76.8], [-76.8, 76.8], [76.8, 76.8], [76.8, -76.8]]],
        dtype=np.float32,
    )
    assert np.all(corner_box == target_representation)


@pytest.mark.parametrize(
    ["coordinate", "size", "yaw"],
    [
        pytest.param(
            np.random.uniform(size=[1, 2]),
            np.random.uniform(size=[1, 2]),
            np.array([np.pi]),
        ),
        pytest.param(
            np.random.uniform(size=[10, 2]),
            np.random.uniform(size=[10, 2]),
            np.random.uniform(size=[10], low=-4, high=4),
        ),
    ],
)
def test_get_bev_bbox(coordinate, size, yaw):
    bev_bbox = get_bev_bbox(coordinate, size, yaw)
    assert bev_bbox.shape[-1] == 2
    assert bev_bbox.shape[-2] == 4


def test_zoom_boxes():

    expect_bboxes = torch.FloatTensor(
        [
            [-5, -5, 15, 15],
            [5, 5, 25, 25],
            [29, 27, 41, 47],
        ]
    )
    ret = zoom_boxes(bboxes1, (2, 2))
    assert (expect_bboxes == ret).all()


@pytest.mark.parametrize(
    ["legacy_bbox", "min_filter_hw", "expect_mask"],
    [
        pytest.param(True, (2, 2), torch.BoolTensor([[True], [True], [True]])),
        pytest.param(
            False, (2, 2), torch.BoolTensor([[False], [True], [True]])
        ),
    ],
)
def test_bbox_filter_by_hw(legacy_bbox, min_filter_hw, expect_mask):
    example_bbox = torch.FloatTensor(
        [
            [1, 1, 2, 2],
            [1, 1, 100, 100],
            [-1, -1, 2, 2],
        ]
    )

    example_batch_bbox = example_bbox.unsqueeze(0)
    mask = bbox_filter_by_hw(example_bbox, min_filter_hw, legacy_bbox)
    print(mask)
    print(expect_mask)
    assert (mask == expect_mask).all()
    mask = bbox_filter_by_hw(example_batch_bbox, min_filter_hw, legacy_bbox)
    assert (mask == expect_mask.unsqueeze(0)).all()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
