import numpy as np
import pytest

from hat.registry import build_from_registry

input_size = 128
x = np.random.randint(0, 256, [input_size, input_size, 3]).astype(
    np.uint8
)  # noqa
data = {
    "img": x,
    "labels": 0,
    "layout": "hwc",
    "roi_scale": 0.625,
}


@pytest.mark.parametrize(
    [
        "data",
    ],
    [pytest.param(data)],
)
def test_SmokeClsTransform(data):

    cfg_transforms = [
        dict(
            type="RandomRotateCrop",
            net_input_size=(128, 128),
            rot_prob=1.0,
            rot_angle_range=30,
            center_shift_prob=1.0,
            center_shift_range=0.01,
            norm_ratio=1.25,
            norm_method="longside_square",
            norm_jitter_range=0.25,
            net_target_size=(128, 128),
            base_len=1.0,
        ),
        dict(
            type="OneFromMultiple",
            transforms=[
                dict(
                    type="RandomNoise",
                    prob=1.0,
                    min=-5.0,
                    max=5.0,
                ),
                dict(
                    type="GaussianNoise",
                    prob=1,
                    mean=0,
                    sigma=1,
                ),
                dict(
                    type="SaltPepperNoise",
                    prob=1,
                    s_ratio=0.05,
                    p_ratio=0.05,
                ),
            ],
            probs=[1, 0.0, 0.0],
        ),
        dict(
            type="RandomFlip",
            px=1,
            py=0,
        ),
        dict(
            type="RandomBrightnessContrast",
            brightness_limit=(-0.2, 0.2),
            contrast_limit=(-0.2, 0.2),
            brightness_by_max=True,
            p=0.5,
        ),
        dict(
            type="HueSaturationValue",
            hue_range=(-20, 20),
            sat_range=(-30, 30),
            val_range=(-20, 20),
            p=0.5,
        ),
    ]

    transforms = build_from_registry(cfg_transforms)
    for transform in transforms:
        data = transform(data)
    assert isinstance(data["img"], np.ndarray)
    assert data["img"].shape == (128, 128, 3)
