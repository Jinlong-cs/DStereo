import numpy as np
import pytest

from hat.core.point_geometry import (
    coor_transformation,
    get_rotation_matrix,
    points_count_convex_polygon_3d_jit,
    points_in_convex_polygon_3d_jit,
    points_in_convex_polygon_jit,
    rotation_points_single_angle,
)


@pytest.mark.parametrize(
    ["axis"],
    [pytest.param(1), pytest.param(2), pytest.param(0), pytest.param(999)],
)
def test_rotation_points_single_angle(axis: int):
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    angle = np.pi / 2
    if axis == 999:
        with pytest.raises(ValueError):
            rotated_points = rotation_points_single_angle(points, angle, axis)
    else:
        rotated_points = rotation_points_single_angle(points, angle, axis)
        if axis == 0:
            target = np.array(
                [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
            )
        elif axis == 1:
            target = np.array(
                [[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32
            )
        else:
            target = np.array(
                [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32
            )
        assert np.allclose(rotated_points, target, atol=1e-7)


def test_points_count_convex_polygon_3d_jit():
    """Test points_count_convex_polygon_3d_jit function."""
    # Prepare 2 points and a cuboid. 1 point is inside the cuboid and 1 is out.
    # For each surface of the cuboid, the points are ordered so that the
    # surface normal must face outwards. Use right-hand rule to judge that.
    points = np.array([[0.5, 0.5, 0.5], [2, 2, 2]], dtype=np.float32)
    surfaces = np.array(
        [
            [
                [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # bottom
                [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # left
                [[1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]],  # right
                [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],  # front
                [[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]],  # top
                [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # back
            ]
        ],
        dtype=np.float32,
    )
    points_count = points_count_convex_polygon_3d_jit(points, surfaces)
    assert np.all(points_count == [1])


def test_points_in_convex_polygon_3d_jit():
    """Test points_in_convex_polygon_3d_jit function."""
    points = np.array([[0.5, 0.5, 0.5], [2, 2, 2]], dtype=np.float32)
    surfaces = np.array(
        [
            [
                [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # bottom
                [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # left
                [[1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]],  # right
                [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],  # front
                [[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]],  # top
                [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # back
            ]
        ],
        dtype=np.float32,
    )
    points_in = points_in_convex_polygon_3d_jit(points, surfaces)
    assert np.all(points_in == np.array([[True], [False]]))


def test_points_in_convex_polygon_jit():
    """Test points_in_convex_polygon_jit function."""
    points = np.array([[0.5, 0.5], [2, 2]], dtype=np.float32)
    surfaces = np.array([[[0, 0], [0, 1], [1, 1], [1, 0]]], dtype=np.float32)
    points_in = points_in_convex_polygon_jit(points, surfaces)
    assert np.all(points_in == np.array([[True], [False]]))


@pytest.mark.parametrize(
    ["theta", "translation"],
    [
        pytest.param(0, None),
        pytest.param(np.pi, None),
        pytest.param(np.pi / 2, None),
        pytest.param(np.pi * 2, [1, 1]),
        pytest.param(
            np.random.uniform(size=[100]), np.random.uniform(size=[100, 2])
        ),
    ],
)
def test_get_rotation_matrix(theta, translation):
    ret = get_rotation_matrix(theta, translation)
    if translation is None:
        assert ret.shape[-2] == 2 and ret.shape[-1] == 2
    else:
        assert ret.shape[-2] == 3 and ret.shape[-1] == 3
    assert np.all(ret[..., 0, 0] == ret[..., 1, 1])
    assert np.all(ret[..., 0, 1] == -ret[..., 1, 0])
    assert np.all(
        np.abs(
            ret[..., 0, 0] * ret[..., 1, 1]
            - ret[..., 0, 1] * ret[..., 1, 0]
            - 1
        )
        < 1e-3
    )


@pytest.mark.parametrize(
    ["x", "theta", "translation"],
    [
        pytest.param(
            np.random.uniform(size=[2]),
            np.pi / 2,
            None,
        ),
        pytest.param(
            np.random.uniform(size=[10, 2]),
            np.pi,
            [1, 1],
        ),
        pytest.param(
            np.random.uniform(size=[10, 20, 2]),
            np.pi,
            [20, 20],
        ),
    ],
)
def test_coordinate_transformation(x, theta, translation):
    y = coor_transformation(x, theta, translation)
    assert y.shape == x.shape
    if translation is not None:
        x_2 = coor_transformation(y - translation, -theta)
    else:
        x_2 = coor_transformation(y, -theta)
    assert np.all(np.abs(x - x_2) < 1e-3)
