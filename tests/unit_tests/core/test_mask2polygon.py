import numpy as np
import pytest

from hat.core.mask2polygon import (
    LaneMask2Polygon,
    Mask2Polygon,
    Parsing2Polygon,
)

mask0 = np.zeros([128, 256], dtype=np.uint8)
mask0[50:100, 50:100] = 1

mask1 = np.zeros([128, 256], dtype=np.uint8)
mask1[50:100, 50:100] = 1
mask1[50, 60] = 0
mask1[51, 59:62] = 0
mask1[52, 58:63] = 0
mask1[53, 57:64] = 0

mask2 = np.zeros([128, 256], dtype=np.uint8)
mask2[50, 50:100] = 1


@pytest.mark.parametrize(
    ["method", "mask", "max_size_only"],
    [
        pytest.param("kcos", mask0, True),
        pytest.param("simple", mask0, False),
        pytest.param("simple", mask1, False),
        pytest.param("simple", mask2, False),
    ],
)
def test_mask2polygon(method, mask, max_size_only):
    mask2polygon_fn = Mask2Polygon(method, max_size_only=max_size_only)
    polygon = mask2polygon_fn(mask)
    if max_size_only:
        assert isinstance(polygon, np.ndarray)
        polygon = [polygon]
    assert isinstance(polygon, list)
    assert polygon[0].ndim == 2
    assert polygon[0].shape[1] == 2


@pytest.mark.parametrize(
    ["method", "mask"],
    [
        pytest.param("kcos", mask0),
        pytest.param("simple", mask1),
        pytest.param("simple", mask2),
    ],
)
def test_lanemask2polygon(method, mask):
    mask2polygon_fn = LaneMask2Polygon(method)
    polygon = mask2polygon_fn(mask)
    assert isinstance(polygon, np.ndarray)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2


@pytest.mark.parametrize(
    ["method", "mask", "polygon_num"],
    [
        pytest.param("simple", mask0, 2),
    ],
)
def test_parsing2polygon(method, mask, polygon_num):
    mask2polygon_fn = Parsing2Polygon(method)
    index_polygon_classids = mask2polygon_fn(mask)
    assert len(index_polygon_classids) == polygon_num
    polygon = index_polygon_classids[0][1]
    assert isinstance(polygon, np.ndarray)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2
