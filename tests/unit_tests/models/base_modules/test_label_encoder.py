import pytest
import torch

from hat.models.base_modules.label_encoder import (
    ClassWiseTrackIdEncoder,
    MatchLabelTrackEncoder,
    PersonPositionLabelFromMatch,
)


def test_person_position_encoder():
    label_encoder = PersonPositionLabelFromMatch([2.0, 2.0], [2.0, 4.0])
    boxes = torch.tensor([[[0, 0, 100, 100]]], dtype=torch.float32)
    gt_boxes = torch.tensor(
        [[[0, 0, 100, 100, 1, 0, 1]]],
        dtype=torch.float32,
    )
    match_pos_flag = torch.randint(1, 2, (1, 1))
    match_gt_id = torch.randint(0, 1, (1, 1))
    label = label_encoder(boxes, gt_boxes, match_pos_flag, match_gt_id)

    assert "dms_cls_label" in label
    assert "dms_cls_label_weight" in label
    assert "oms_cls_label" in label
    assert "oms_cls_label_weight" in label
    assert label["dms_cls_label"].size() == boxes.size()[:2]
    assert label["dms_cls_label_weight"].size() == boxes.size()[:2]
    assert label["oms_cls_label"].size() == boxes.size()[:2]
    assert label["oms_cls_label_weight"].size() == boxes.size()[:2]


def test_match_label_track_encoder():
    track_label_encoder = ClassWiseTrackIdEncoder(
        num_classes=2,
        exclude_background=True,
    )
    match_label_track_encoder = MatchLabelTrackEncoder(
        track_use_pos_only=True,
        track_on_hard=False,
        track_label_encoder=track_label_encoder,
    )
    boxes = torch.tensor([[[0, 0, 100, 100]]], dtype=torch.float32)
    gt_boxes = torch.tensor(
        [[[0, 0, 100, 100, 1, 1]]],
        dtype=torch.float32,
    )
    match_pos_flag = torch.ones((1, 1))
    match_gt_id = torch.randint(0, 1, (1, 1))
    ig_flag = torch.zeros((1, 1))
    label = match_label_track_encoder(
        boxes, gt_boxes, match_pos_flag, match_gt_id, ig_flag
    )
    assert "track_id" in label
    # batch
    assert label["track_id"].shape[0] == 1
    # num_rois
    assert label["track_id"].shape[1] == 1
    # num_fg_class
    assert label["track_id"].shape[2] == 1
    # check track_id
    assert label["track_id"][0, 0, 0].item() == 1

    ig_flag = torch.ones((1, 1))
    label = match_label_track_encoder(
        boxes, gt_boxes, match_pos_flag, match_gt_id, ig_flag
    )
    # check track_id
    assert label["track_id"][0, 0, 0].item() == -1


if __name__ == "__main__":
    pytest.main(["-s", __file__])
