import numpy as np
import pytest

from hat.data.transforms.keypoints import (
    AddGaussianNoise,
    GenerateHeatmapTarget,
    RandomPadLdmkData,
)


@pytest.mark.parametrize(
    ["prob", "mean", "sigma"], [[0, 0, 1], [0.5, 0, 1.2], [1, 0.5, 1.5]]
)
def test_add_gaussian_noise(prob, mean, sigma):
    addgaussiannoise = AddGaussianNoise(
        prob=prob,
        mean=mean,
        sigma=sigma,
    )
    data = {"img": np.zeros((256, 256, 3))}
    data = addgaussiannoise(data)
    assert isinstance(data["img"], np.ndarray)
    assert data["img"].shape == (256, 256, 3)


def test_generate_heatmap_target():
    num_ldmk = 2
    generate_heatmap_target = GenerateHeatmapTarget(
        num_ldmk=num_ldmk,
        feat_stride=4,
        heatmap_shape=(32, 32),
        sigma=2,
    )

    data = {
        "img": np.zeros((128, 128, 3)),
        "gt_ldmk": np.random.uniform(0, 128, (num_ldmk, 2)),
        "gt_ldmk_attr": np.ones((num_ldmk,)),
    }

    data = generate_heatmap_target(data)
    assert "gt_heatmap" in data and "gt_heatmap_weight" in data
    assert data["gt_heatmap"].shape == (num_ldmk, 32, 32)
    assert data["gt_heatmap_weight"].shape == (num_ldmk, 32, 32)
    min_value = data["gt_heatmap"].min()
    assert min_value == 0


@pytest.mark.parametrize(["random"], [[True], [False]])
def test_random_pad_ldmk_data(random):
    random_pad_ldmk_data = RandomPadLdmkData(
        size=(256, 256),
        random=random,
    )
    data = {
        "img": np.zeros((128, 128, 3)),
        "gt_ldmk": np.random.uniform(0, 128, (2, 2)),
    }
    data = random_pad_ldmk_data(data)
    assert isinstance(data["img"], np.ndarray)
    assert data["img"].shape == (256, 256, 3)
    assert data["gt_ldmk"].shape == (2, 2)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
