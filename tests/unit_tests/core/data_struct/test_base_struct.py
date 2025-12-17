from typing import Dict

import numpy as np
import pytest
import torch

from hat.core.data_struct.base_struct import (
    ClsLabel,
    ClsLabels,
    DetBox2D,
    DetBox3D,
    DetBoxes2D,
    DetBoxes3D,
    Mask,
    Masks,
)

cls_idx = torch.tensor(0)
score = torch.tensor(0.5)
cls_name = "vehicle"


@pytest.mark.parametrize(
    ["cls_idx", "score", "cls_name", "with_score"],
    [
        pytest.param(torch.tensor(0), torch.tensor(0.5), "vehicle", False),
        pytest.param(
            torch.tensor([0, 0, 0]),
            torch.tensor([0.5, 0.6, 0.7]),
            ["vehicle", "vehicle", "vehicle"],
            False,
        ),
    ],
)
def test_ClsLabel(cls_idx, score, cls_name, with_score):
    if isinstance(cls_name, list) and len(cls_name) > 1:
        with pytest.raises(AssertionError):
            cls_label = ClsLabel(
                cls_idx=cls_idx,
                score=score,
                cls_name=cls_name,
            )
    else:
        cls_label = ClsLabel(
            cls_idx=cls_idx,
            score=score,
            cls_name=cls_name,
        )

        aidi_data = cls_label.to_aidi_eval()
        assert isinstance(aidi_data, Dict)
        text = cls_label.to_text(with_score=with_score)
        assert isinstance(text, str)
        if not with_score:
            assert text == cls_label.cls_name
        assert cls_label.device


cls_idxs = torch.tensor([0, 1, 0, 2], dtype=torch.int32)
scores = torch.tensor([0.2, 0.3, 0.4, 0.5])
cls_name_mapping = {0: "vehicle", 1: "rear", 2: "Cyclist"}


def test_ClsLabels():
    cls_labels = ClsLabels(
        cls_idxs=cls_idxs,
        scores=scores,
        cls_name_mapping=cls_name_mapping,
    )

    #  score > 0.3, [0.4, 0.5]
    assert len(cls_labels.with_scores_gt(threshold=0.3)) == 2
    # score >=3, [0.3, 0.4, 0.5]
    assert len(cls_labels.with_scores_ge(threshold=0.3)) == 3
    assert len(cls_labels.with_cls_idxs_in(0)) == 2
    assert len(cls_labels.with_cls_idxs_in(1)) == 1
    assert cls_labels.device


def test_DetBox2D():
    bbox = torch.FloatTensor([10, 10, 20, 20])
    score = torch.tensor(0.5)
    det_box2d = DetBox2D(cls_idx=0, score=score, box=bbox)

    aidi_data = det_box2d.to_aidi_eval()
    assert isinstance(aidi_data, Dict)
    assert aidi_data["bbox"] == bbox.tolist()
    assert aidi_data["bbox_score"] == score

    det_box2d.rescale(0.5)
    assert det_box2d.box.tolist() == [5, 5, 10, 10]
    assert det_box2d.device


def test_DetBoxes2D():
    bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [32, 32, 38, 42],
            [36, 36, 52, 52],
        ]
    )
    DetBoxes2D(
        cls_idxs=cls_idxs,
        scores=scores,
        cls_name_mapping=cls_name_mapping,
        boxes=bboxes,
    )


@pytest.mark.parametrize(
    ["cls_idx", "score", "h", "w", "l", "x", "y", "z", "yaw"],
    [
        pytest.param(
            torch.tensor(0),
            torch.tensor(0.5),
            torch.tensor(0.6),
            torch.tensor(0.8),
            torch.tensor(0.5),
            torch.tensor(0.5),
            torch.tensor(0.6),
            torch.tensor(0.4),
            torch.tensor(np.pi / 2),
        ),
    ],
)
def test_DetBox3D(cls_idx, score, h, w, l, x, y, z, yaw):  # noqa E741

    det_box3d = DetBox3D(
        cls_idx=cls_idx,
        score=score,
        h=h,
        w=w,
        l=l,
        x=x,
        y=y,
        z=z,
        yaw=yaw,
    )

    assert torch.equal(
        det_box3d.location, torch.tensor([x, y, z], device=x.device)
    )
    assert torch.equal(
        det_box3d.dimension, torch.tensor([h, w, l], device=h.device)
    )
    assert det_box3d.device
    assert det_box3d.to_aidi_eval()


@pytest.mark.parametrize(
    ["cls_idxs", "scores", "h", "w", "l", "x", "y", "z", "yaw"],
    [
        pytest.param(
            torch.zeros(4),
            torch.randn([4]),
            torch.randn([4]),
            torch.randn([4]),
            torch.randn([4]),
            torch.randn([4]),
            torch.randn([4]),
            torch.randn([4]),
            torch.randn([4]),
        ),
    ],
)
def test_DetBoxes3D(cls_idxs, scores, h, w, l, x, y, z, yaw):  # noqa E741

    boxes_3d = DetBoxes3D(
        cls_idxs=cls_idxs,
        scores=scores,
        h=h,
        w=w,
        l=l,
        x=x,
        y=y,
        z=z,
        yaw=yaw,
    )

    assert torch.equal(boxes_3d.locations, torch.stack([x, y, z], dim=-1))
    assert torch.equal(boxes_3d.dimensions, torch.stack([h, w, l], dim=-1))
    assert boxes_3d.device
    assert boxes_3d.to_aidi_eval()


def test_Mask():

    data_h, data_w = 32, 32
    scale_h, scale_w = 0.5, 0.5
    mask = torch.zeros([data_h, data_w], dtype=torch.int32)
    mask_class = Mask(mask=mask)
    mask_class.rescale(scale_w=scale_w, scale_h=scale_h)
    assert mask_class.mask.shape == torch.Size(
        [int(data_h * scale_h), int(data_w * scale_w)]
    )

    padding = [3, 4, 5, 6]
    mask_class.inv_pad(*padding)
    expected_shape = (
        int(data_h * scale_h - padding[3] - padding[2]),
        int(data_w * scale_w - padding[1] - padding[0]),
    )
    assert mask_class.mask.shape == torch.Size(expected_shape)
    assert mask_class.device


def test_Masks():
    masks = torch.zeros([3, 32, 32], dtype=torch.int32)
    masks_class = Masks(masks=masks)
    masks_class.rescale(0.5, 0.5)

    assert masks_class.masks.shape == torch.Size([3, 16, 16])
    assert masks_class.device
