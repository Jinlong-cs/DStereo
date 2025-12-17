import copy

import numpy as np
import pytest

from hat.data.transforms.face3d import RandomRotateCrop
from hat.data.transforms.landmark import GenerateGaussianHeatmap

NUM_LDMK = 4
DATA = {
    "img": np.random.randint(0, 256, [128, 128, 3]).astype(np.uint8),
    "gt_ldmk": np.random.uniform(0, 128, (NUM_LDMK, 2)),
    "gt_ldmk_attr": np.ones((NUM_LDMK,)),
    "roi_scale": 0.625,
}


@pytest.mark.parametrize(["encoding_method"], [["ellipse"]])
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
    assert data["gt_heatmap"].shape == (NUM_LDMK + 1, 32, 32)
    assert data["gt_heatmap_weight"].shape == (NUM_LDMK + 1, 32, 32)
    max_value = data["gt_heatmap"].max()
    min_value = data["gt_heatmap"].min()
    assert min_value == 0
    assert max_value <= 1


@pytest.mark.parametrize(
    ["net_input_size", "net_target_size", "norm_method"],
    [
        pytest.param(
            (128, 128),
            (128, 128),
            "longside_square",
        )
    ],
)
def test_random_rotate_crop(net_input_size, net_target_size, norm_method):
    rotate_crop = RandomRotateCrop(
        net_input_size=net_input_size,
        rot_prob=1.0,
        rot_angle_range=30,
        center_shift_prob=1.0,
        center_shift_range=0.01,
        norm_ratio=1.25,
        norm_method=norm_method,
        norm_jitter_range=0.25,
        keep_ldmk_complt_ratio=1,
        net_target_size=net_target_size,
        base_len=1.0,
    )
    result = rotate_crop(DATA)
    assert result["img"].shape[0] == net_input_size[1]
    assert result["img"].shape[1] == net_input_size[0]
