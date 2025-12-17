import copy

import numpy as np

from hat.data.transforms.facequality_mtl import FaceQualityTransformLabel

DATA = {
    "img": np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8),
    "gt_face_quality": {"blur": 0, "mouth": 1},
    "layout": "hwc",
}


def test_facequality_trans_label():
    label_trans = FaceQualityTransformLabel()
    data = label_trans(copy.deepcopy(DATA))
    assert "blur" in data
    assert "gt_blur" in data["blur"]
    assert "mouth" in data
    assert "gt_mouth" in data["mouth"]
    assert (DATA["img"] == data["img"]).all()
