import numpy as np

from hat.core.center_utils import (
    draw_umich_gaussian,
    gaussian2D,
    gaussian_radius,
)


def test_gaussian_radius():

    length = 400
    width = 400
    gaussian_overlap = 0.1
    radius = gaussian_radius((length, width), min_overlap=gaussian_overlap)
    assert int(radius) == 172


def test_gaussian2D():

    diameter = 2 * 172 + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    assert gaussian.shape == (diameter, diameter)
    assert gaussian[diameter // 2, diameter // 2] == 1.0


def test_draw_umich_gaussian():

    hm = np.zeros((400, 400))
    center = np.array([200, 200])
    radius = 172
    draw_umich_gaussian(hm, center, radius)

    assert hm.sum() != 0
    assert np.where(hm == np.max(hm)) == (200, 200)
