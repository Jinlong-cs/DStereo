# Copyright (c) Horizon Robotics. All rights reserved.

import random

import numpy as np
import pytest
import torch

from hat.models.task_modules.fcos import FCOSDecoder
from hat.utils.package_helper import check_packages_available
from tests.utils import gen_fake_det_label_pred_data

seed = 20
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type="nms", iou_threshold=0.6),
    max_per_img=100,
)
transforms = [
    dict(type="Resize", img_scale=(640, 640), keep_ratio=True),
    dict(type="FixedCrop", size=(0, 0, 640, 640)),
]
inverse_transform_key = ["scale_factor", "crop_offset", "before_crop_shape"]


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
@pytest.mark.parametrize(
    ["h", "w", "strides", "num_classes"],
    [
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2),
    ],
)
def test_fcos_decoder(h, w, strides, num_classes):
    label, pred = gen_fake_det_label_pred_data(h, w, strides, num_classes)
    fcos_decoder = FCOSDecoder(
        num_classes,
        strides,
        test_cfg=test_cfg,
        transforms=transforms,
        inverse_transform_key=inverse_transform_key,
    )
    results = fcos_decoder(pred, label)
    assert len(results["pred_bboxes"]) == pred[0][0].shape[0]
    assert results["pred_bboxes"][0].shape[-1] == 6
