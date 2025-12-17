import copy

import numpy as np

from hat.data.transforms.face_fas import EraseBackground

DATA = {
    "img": np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8),
    "layout": "hwc",
}


def test_random_occlusion():
    occlusion = EraseBackground(p=0.5)
    data = occlusion(copy.deepcopy(DATA))
    assert "img" in data
    assert isinstance(data["img"], np.ndarray)
    assert data["img"].dtype == DATA["img"].dtype
    assert data["img"].shape == DATA["img"].shape
