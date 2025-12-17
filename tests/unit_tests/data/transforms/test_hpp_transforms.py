import numpy as np
import pytest

from hat.data.transforms.hpp_transforms import (
    ChangeIntensity,
    GaussianNoiseWithChannel,
    HPPPoint2Map,
    RandomHorizontalFilpWithPoints,
    RandomRotateWithPoints,
    Shadow,
)


def get_hpp_fake_data(h, w, c, num_points):
    data = {}
    data["img"] = np.random.randint(0, 255, (h, w, c), dtype=np.uint8)
    data["points"] = np.random.randint(0, 80, (num_points, 2), dtype=np.uint8)
    return data


@pytest.mark.parametrize(
    ["prob", "mean", "sigma"],
    [
        pytest.param(1.0, 0, 0),
        pytest.param(1.0, 20, 60),
        pytest.param(0.0, 0, 20),
        pytest.param(0.5, 20, 20),
    ],
)
def test_gaussian(prob, mean, sigma):
    data = get_hpp_fake_data(256, 512, 3, 50)
    gaussian = GaussianNoiseWithChannel(prob, mean, sigma)
    gaussian_data = gaussian(data).copy()
    assert isinstance(gaussian_data["img"], np.ndarray)


@pytest.mark.parametrize(
    ["prob", "rescale"],
    [
        pytest.param(1.0, 0),
        pytest.param(1.0, 40),
        pytest.param(0.0, 20),
        pytest.param(0.5, 60),
    ],
)
def test_change_intensity(prob, rescale):
    data = get_hpp_fake_data(256, 512, 3, 50)
    change_intensity = ChangeIntensity(prob, rescale)
    change_intensity_data = change_intensity(data).copy()
    assert isinstance(change_intensity_data["img"], np.ndarray)


@pytest.mark.parametrize(
    ["prob", "min_alpha", "max_alpha"],
    [
        pytest.param(1.0, 0.5, 0.75),
        pytest.param(1.0, 0, 0),
        pytest.param(0.0, 0.5, 0.6),
        pytest.param(0.5, 0.2, 0.5),
    ],
)
def test_shadow(prob, min_alpha, max_alpha):
    data = get_hpp_fake_data(256, 512, 3, 50)
    shadow = Shadow(prob, min_alpha, max_alpha)
    shadow_data = shadow(data).copy()
    assert isinstance(shadow_data["img"], np.ndarray)


@pytest.mark.parametrize(
    ["filp_prob"],
    [
        pytest.param(1.0),
        pytest.param(1.0),
        pytest.param(0.0),
        pytest.param(0.2),
        pytest.param(0.5),
    ],
)
def test_random_h_filp_with_points(filp_prob):
    data = get_hpp_fake_data(256, 512, 3, 50)
    h_filp = RandomHorizontalFilpWithPoints(filp_prob)
    h_filp_data = h_filp(data).copy()
    assert isinstance(h_filp_data["img"], np.ndarray)
    assert (h_filp_data["points"] == data["points"]).all()


@pytest.mark.parametrize(
    ["prob", "neg_filter", "angle"],
    [
        pytest.param(1.0, True, 10),
        pytest.param(1.0, True, 90),
        pytest.param(1.0, True, 180),
        pytest.param(0.2, False, 40),
        pytest.param(0, True, 0),
    ],
)
def test_random_rotate_with_points(prob, neg_filter, angle):
    data = get_hpp_fake_data(256, 512, 3, 50)
    rotate = RandomRotateWithPoints(prob, neg_filter, angle)
    rotate_data = rotate(data).copy()
    assert isinstance(rotate_data["img"], np.ndarray)
    assert (
        np.max(rotate_data["img"]) <= 255 and np.min(rotate_data["img"]) >= 0
    )
    assert rotate_data["img"].shape == data["img"].shape
    assert len(rotate_data["points"]) <= len(data["points"])


@pytest.mark.parametrize(
    ["output_stride"],
    [
        pytest.param(8),
        pytest.param(8),
        pytest.param(8),
    ],
)
def test_hpp2points(output_stride):
    data = get_hpp_fake_data(256, 512, 3, 50)
    hpp_point2map = HPPPoint2Map(output_stride)
    hpp_point2map_data = hpp_point2map(data).copy()
    assert "labels" in hpp_point2map_data
    assert len(hpp_point2map_data["labels"]) == 2


if __name__ == "__main__":
    pytest.main(["-s", __file__])
