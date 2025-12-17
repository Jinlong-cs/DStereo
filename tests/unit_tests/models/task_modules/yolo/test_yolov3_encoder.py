import torch

from hat.models.base_modules.target import BBoxTargetGenerator
from hat.registry import build_from_registry


def test_yolov3_labelencoder():
    matcher_cfg = dict(type="YOLOV3Matcher", ignore_thresh=0.6)

    label_encoder_cfg = dict(
        type="YOLOV3LabelEncoder",
        class_encoder=dict(
            type="OneHotClassEncoder",
            num_classes=2,
        ),
    )

    target_gen = BBoxTargetGenerator(
        build_from_registry(matcher_cfg),
        build_from_registry(label_encoder_cfg),
    )
    assert not target_gen.with_ig_region_matcher

    anchors = (
        torch.tensor([[2, 3, 5, 6], [1, 2, 3, 4]])
        .unsqueeze(0)
        .repeat(2, 1, 1)
        .float()
    )
    target = [
        torch.tensor([[1, 2, 3, 4, 1], [2, 3, 5, 6, 0]]).float(),
        torch.tensor([[1, 2, 3, 4, 0], [2, 3, 5, 6, 0]]).float(),
    ]
    res = target_gen(anchors, target)
    assert isinstance(res[1], dict)
    assert res[1]["mask"].size(0) == 2
    assert res[1]["mask"].size(1) == 2
    assert res[1]["tconf"].size(0) == 2
    assert res[1]["tconf"].size(1) == 2
