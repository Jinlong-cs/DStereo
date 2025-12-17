import copy

import numpy as np
import pytest
import torch

from hat.data.transforms.landmark import (
    CropRecROI,
    GaussianNoise,
    GenerateGaussianHeatmap,
    GenerateGaussianVector,
    Lighting,
    RandomNoise,
    SaltPepperNoise,
)

NUM_LDMK = 2
DATA = {
    "img": np.zeros((128, 128, 3)),
    "gt_ldmk": np.random.uniform(0, 128, (NUM_LDMK, 2)),
    "gt_ldmk_attr": np.ones((NUM_LDMK,)),
}
DATA_NONSQUARE = {
    "img": np.zeros((96, 160, 3)),
    "gt_ldmk": np.concatenate(
        [
            np.random.uniform(0, 160, (NUM_LDMK, 1)),
            np.random.uniform(0, 96, (NUM_LDMK, 1)),
        ],
        axis=1,
    ),
    "gt_ldmk_attr": np.ones((NUM_LDMK,)),
}


@pytest.mark.parametrize(["encoding_method"], [["standard"], ["unbiased"]])
def test_heatmap(encoding_method):
    generator = GenerateGaussianHeatmap(
        num_ldmk=NUM_LDMK,
        feat_stride=4,
        heatmap_shape=(32, 32),
        sigma=2,
        encoding_method=encoding_method,
    )
    data = generator(copy.deepcopy(DATA))
    assert "gt_heatmap" in data and "gt_heatmap_weight" in data
    assert data["gt_heatmap"].shape == (NUM_LDMK, 32, 32)
    assert data["gt_heatmap_weight"].shape == (NUM_LDMK, 32, 32)
    max_value = data["gt_heatmap"].max()
    min_value = data["gt_heatmap"].min()
    assert min_value == 0
    if encoding_method == "standard":
        assert max_value == 2
    else:
        assert max_value <= 1


@pytest.mark.parametrize(
    ["encoding_method"],
    [["standard"], ["unbiased"]],
)
def test_vector(encoding_method):
    generator = GenerateGaussianVector(
        num_ldmk=NUM_LDMK,
        feat_stride=4,
        vector_size=(40, 24),
        sigma=2,
        encoding_method=encoding_method,
    )
    data = generator(copy.deepcopy(DATA_NONSQUARE))
    assert "gt_vector_x" in data and "gt_vector_y" in data
    assert "gt_vector_weight_x" in data and "gt_vector_weight_y" in data
    assert data["gt_vector_x"].shape == (NUM_LDMK, 40)
    assert data["gt_vector_y"].shape == (NUM_LDMK, 24)
    assert data["gt_vector_weight_x"].shape == (NUM_LDMK, 40)
    assert data["gt_vector_weight_y"].shape == (NUM_LDMK, 24)
    max_value_x = data["gt_vector_x"].max()
    max_value_y = data["gt_vector_y"].max()
    min_value_x = data["gt_vector_x"].min()
    min_value_y = data["gt_vector_y"].min()
    assert min_value_x == 0
    assert min_value_y == 0
    if encoding_method == "standard":
        assert max_value_x == 2
        assert max_value_y == 2
    else:
        assert max_value_x <= 1
        assert max_value_y <= 1


@pytest.mark.parametrize(["crop_type"], [["random"], ["scale"], ["center"]])
def test_crop_rec_roi(crop_type):
    crop = CropRecROI(
        crop_type=crop_type,
        target_shape=(128, 128, 3),
        base_roi=[48, 48, 208, 208],
        crop_jitter_range=0.25,
        center_shift_range=0.0,
        random_type="gaussian",
    )
    data = {
        "img": np.zeros((256, 256, 3)),
        "gt_ldmk": np.random.uniform(64, 192, (NUM_LDMK, 2)),
        "gt_ldmk_attr": np.ones((NUM_LDMK,)),
        "gt_pupil_ellipse_param": np.array([64.0, 64.0, 10.0, 15.0, 20.0]),
    }
    data = crop(data)
    assert isinstance(data["img"], np.ndarray)
    assert isinstance(data["gt_ldmk"], np.ndarray)
    assert data["img"].shape == (128, 128, 3)
    assert data["gt_ldmk"].shape == (NUM_LDMK, 2)
    if crop_type == "center":
        assert (
            abs(
                data["gt_pupil_ellipse_param"]
                - np.array([12.8, 12.8, 8.0, 12.0, 20.0])
            ).sum()
            < 1e-5
        )


@pytest.mark.parametrize(["alphastd"], [["0.2"], ["0.5"], ["1.0"], ["1.5"]])
def test_lighting(alphastd):
    lighting = Lighting(1.0, float(alphastd))
    data = {"img": torch.zeros((3, 256, 256))}
    data = lighting(data)
    assert isinstance(data["img"], torch.Tensor)


@pytest.mark.parametrize(
    ["mean", "sigma"], [["0", "1"], ["0", "1.2"], ["0.5", "1.5"]]
)
def test_gaussian_noise(mean, sigma):
    gaussian_noise = GaussianNoise(1.0, float(mean), float(sigma))
    data = {"img": np.zeros((256, 256, 3))}
    data = gaussian_noise(data)
    assert isinstance(data["img"], np.ndarray)
    assert data["img"].shape == (256, 256, 3)


@pytest.mark.parametrize(
    ["min", "max"], [["-5", "5"], ["0.5", "2.5"], ["-1.2", "0.5"]]
)
def test_random_noise(min, max):
    random_noise = RandomNoise(1.0, float(min), float(max))
    data = {"img": np.zeros((256, 256, 3))}
    data = random_noise(data)
    assert isinstance(data["img"], np.ndarray)
    assert data["img"].shape == (256, 256, 3)


@pytest.mark.parametrize(
    ["s_ratio", "p_ratio"], [["0.05", "0.05"], ["0.1", "0.2"], ["0", "0"]]
)
def test_saltpepper_noise(s_ratio, p_ratio):
    saltpepper_noise = SaltPepperNoise(1.0, float(s_ratio), float(p_ratio))
    data = {"img": np.zeros((256, 256, 3))}
    data = saltpepper_noise(data)
    assert isinstance(data["img"], np.ndarray)
    assert data["img"].shape == (256, 256, 3)
