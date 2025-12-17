import copy

import numpy as np

from hat.data.transforms.anon import TransformBboxToFcosFormat

DATA = {
    "img": np.random.randint(0, 255, (320, 490, 3), dtype=np.uint8),
    "layout": "hwc",
    "gt_boxes": np.ones((64, 5)),
}


def test_transform_bbox_to_fcosformat():
    transform = TransformBboxToFcosFormat(change_label_bool=True)
    data = transform(copy.deepcopy(DATA))
    assert "gt_classes" in data
    transform = TransformBboxToFcosFormat(change_label_bool=False)
    data = transform(copy.deepcopy(DATA))
    assert "gt_classes" in data
