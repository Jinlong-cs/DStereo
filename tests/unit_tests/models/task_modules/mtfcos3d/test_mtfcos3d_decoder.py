# Copyright (c) Horizon Robotics. All rights reserved.

import random

import numpy as np
import pytest
import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.task_modules.mtfcos3d.utils import (
    gen_fake_mtf_real3d_label_pred_data,
)

seed = 20
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


roi_task_test_post_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.3,
    nms=dict(name="nms", iou_threshold=0.5),
    max_per_img=15,
)

transforms = [
    dict(type="Resize", img_scale=(640, 640), keep_ratio=True),
    dict(type="FixedCrop", size=(0, 0, 640, 640)),
]
inverse_transform_key = ["scale_factor", "crop_offset", "before_crop_shape"]


@pytest.mark.parametrize(
    ["h", "w", "strides", "num_classes", "is_train_3d_branch"],
    [
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2, True),
    ],
)
def test_fcos_decoder(h, w, strides, num_classes, is_train_3d_branch):
    multibin_centers = (0.0, np.pi / 2, np.pi, -np.pi / 2)
    head_channels = dict(
        cls=[num_classes],
        offset_2d_reg=[4],
        offset_3d_group_reg=[2],
        depth_3d_group_reg=[1],
        dim_3d_group_reg=[3],
        dir_reg=[len(multibin_centers)],
        ctrness_2d_reg=[1],
        ctrness_3d_reg=[1],
        rot_3d_group_reg=[1],
    )
    label, pred = gen_fake_mtf_real3d_label_pred_data(
        h, w, strides, head_channels
    )

    fcos_decoder = dict(
        type="MTFCOS3DDecoder",
        num_classes=num_classes,
        head_channels=head_channels,
        strides=strides,
        rescale=True,  # pred rescale to org image
        dir_offset=np.pi / 4,  # same as target cfg
        nms_kwargs=dict(
            nms_pre=200,
            score_thr=0.4,
            nms_thr=0.3,
            max_per_img=100,
            iou_threshold=0.4,
            replace=True,
            nms_sqrt=True,
            use_score2d=True,
        ),
        depth_type="Cartesian",
        is_train_3d_branch=is_train_3d_branch,
        use_multibin=True,
        multibin_centers=multibin_centers,
        multibin_margin=0,
    )
    fcos_decoder = build_from_registry(fcos_decoder)

    results = fcos_decoder(pred, label)
    assert len(results["bbox"]) == pred["cls"][0].shape[0]
