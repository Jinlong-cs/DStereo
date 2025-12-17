import copy

import numpy as np

from hat.registry import build_from_registry

DATA = {
    "img": np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8),
    "gt": 1,
    "layout": "hwc",
}


def test_facemtl_transform_label():
    config = dict(
        type="FaceMtlTransformLabel",
        name="facemtl",
    )
    transform = build_from_registry(config)
    data = transform(copy.deepcopy(DATA))
    assert "img" in data
    assert "facemtl" in data
    assert "gt" in data["facemtl"]
    assert "layout" in data["facemtl"]
    assert (DATA["img"] == data["img"]).all()
