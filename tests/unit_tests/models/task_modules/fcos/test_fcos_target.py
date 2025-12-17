# Copyright (c) Horizon Robotics. All rights reserved.

import pytest

from hat.models.losses.focal_loss import FocalLoss
from hat.models.losses.giou_loss import GIoULoss
from hat.models.task_modules.fcos import DynamicFcosTarget, FCOSTarget
from tests.utils import gen_fake_det_label_pred_data

INF = 1e8
default_regress_ranges = (
    (-1, 64),
    (64, 128),
    (128, 256),
    (256, 512),
    (512, INF),
)


@pytest.mark.parametrize(
    ["h", "w", "strides", "num_classes", "use_iou_replace_ctrness"],
    [
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2, False),
        pytest.param(512, 512, (8, 16, 32, 64), 5, False),
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2, True),
        pytest.param(512, 512, (8, 16, 32, 64), 5, True),
    ],
)
def test_fcos_target(h, w, strides, num_classes, use_iou_replace_ctrness):
    regress_ranges = default_regress_ranges[: len(strides)]
    background_label = num_classes
    fcos_target = FCOSTarget(
        strides,
        regress_ranges,
        num_classes,
        background_label,
        use_iou_replace_ctrness=use_iou_replace_ctrness,
    )
    label, pred = gen_fake_det_label_pred_data(h, w, strides, num_classes)
    cls_target, giou_target, centerness_target = fcos_target(label, pred)
    assert cls_target["pred"].shape[-1] == num_classes
    assert giou_target["pred"].shape[-1] == 4
    assert giou_target["pred"].shape[0] == centerness_target["pred"].shape[0]


@pytest.mark.parametrize(
    [
        "h",
        "w",
        "strides",
        "num_classes",
        "center_sampling",
        "center_sampling_radius",
    ],
    [
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2, False, 2.5),
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2, True, 2.5),
        pytest.param(512, 512, (8, 16, 32, 64), 5, False, 2.5),
        pytest.param(512, 512, (8, 16, 32, 64), 5, True, 2.5),
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2, False, 2.5),
        pytest.param(640, 640, (4, 8, 16, 32, 64), 2, True, 2.5),
        pytest.param(512, 512, (8, 16, 32, 64), 5, False, 2.5),
        pytest.param(512, 512, (8, 16, 32, 64), 5, True, 2.5),
        pytest.param(512, 512, (8, 16, 32, 64), 5, True, 100.0),
        pytest.param(512, 512, (8, 16, 32, 64), 5, True, 0.0),
    ],
)
def test_fcos_dynamic_target(
    h, w, strides, num_classes, center_sampling, center_sampling_radius
):
    background_label = num_classes
    fcos_target = DynamicFcosTarget(
        strides=strides,
        topK=10,
        loss_cls=FocalLoss(
            loss_name="cls",
            num_classes=num_classes + 1,
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            reduction="none",
        ),
        loss_reg=GIoULoss(loss_name="reg", loss_weight=2.0, reduction="none"),
        cls_out_channels=num_classes,
        background_label=background_label,
        center_sampling=center_sampling,
        center_sampling_radius=center_sampling_radius,
    )
    label, pred = gen_fake_det_label_pred_data(h, w, strides, num_classes)
    cls_target, giou_target, centerness_target = fcos_target(label, pred)
    assert cls_target["pred"].shape[-1] == num_classes
    assert giou_target["pred"].shape[-1] == 4
    assert giou_target["pred"].shape[0] == centerness_target["pred"].shape[0]
