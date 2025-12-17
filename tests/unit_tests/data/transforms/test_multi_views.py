# Copyright (c) Horizon Robotics. All rights reserved.

from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image

from hat.data.transforms.multi_views import (
    BevFeatureFlip,
    BevFeatureRotate,
    MultiViewsGridMask,
    MultiViewsImgCrop,
    MultiViewsImgFlip,
    MultiViewsImgResize,
    MultiViewsImgRotate,
)


def test_NuscTransfom():
    img = np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
    img = Image.fromarray(img)
    imgs = [img] * 6

    ego2img = np.random.randn(4, 4)
    ego2imgs = [ego2img] * 6

    data = {"img": imgs, "ego2img": ego2imgs}

    # resize
    resizer = MultiViewsImgResize(size=(540, 960))
    resize_data = resizer(deepcopy(data))
    assert resize_data["img"][0].size == (960, 540)

    resizer = MultiViewsImgResize(scales=(0.8, 1.0))
    resize_data = resizer(deepcopy(data))

    # crop
    croper = MultiViewsImgCrop(size=(512, 960))
    croper(deepcopy(data))

    croper = MultiViewsImgCrop(size=(512, 960), random=True)
    croper(deepcopy(data))

    # flip
    fliper = MultiViewsImgFlip(prob=1.0)
    flip_data = fliper(deepcopy(data))
    assert flip_data["img"][0].size == (1600, 900)

    gridmask = MultiViewsGridMask(
        use_h=True,
        use_w=True,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.7,
    )
    mask_data = gridmask(deepcopy(data))
    assert mask_data["img"][0].size == (1600, 900)

    rotater = MultiViewsImgRotate(rot=(-30, 30))
    rotate_data = rotater(deepcopy(data))
    assert rotate_data["img"][0].size == (1600, 900)

    bev_feat = torch.from_numpy(np.random.randn(2, 64, 256, 256)).float()
    bev_seg = torch.from_numpy(np.random.randn(2, 512, 512)).float()
    bev_bboxes = [np.random.randn(10, 10), np.random.randn(11, 10)]

    bev_rotater = BevFeatureRotate(
        bev_size=(512, 512, 0.2), rot=(-0.3925, 0.3925)
    )
    bev_data = {"bev_bboxes_labels": bev_bboxes, "bev_seg_indices": bev_seg}
    new_feat, new_data = bev_rotater(bev_feat, bev_data)
    assert new_feat.shape == (2, 64, 256, 256)
    assert new_data["bev_seg_indices"].shape == (2, 512, 512)

    bev_fliper = BevFeatureFlip(
        bev_size=(512, 512, 0.2), prob_x=1.0, prob_y=1.0
    )
    new_feat, new_data = bev_fliper(bev_feat, bev_data)
    assert new_feat.shape == (2, 64, 256, 256)
    assert new_data["bev_seg_indices"].shape == (2, 512, 512)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
