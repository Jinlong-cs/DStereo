import copy

import numpy as np

from hat.data.transforms.faceattr import RandomOcclusion, TransformAgeLabel

DATA = {
    "img": np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8),
    "age": np.random.uniform(1, 85),
    "layout": "hwc",
}


def test_random_occlusion():
    occlusion = RandomOcclusion(occ_type="forehead", prob=1)

    data = occlusion(copy.deepcopy(DATA))
    assert "img" in data
    assert isinstance(data["img"], np.ndarray)
    assert data["img"].dtype == DATA["img"].dtype
    assert data["img"].shape == DATA["img"].shape


def test_agetransform():
    age_trans = TransformAgeLabel(
        age_classes=85,
    )
    data = age_trans(copy.deepcopy(DATA))
    assert "ord_age" in data
    assert "dist_age" in data
    assert (DATA["img"] == data["img"]).all()
