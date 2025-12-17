# TODO(tian.li): test cases for matchers. Need to involve some real data
import pytest
import torch

from hat.core.box_utils import bbox_overlaps
from hat.models.base_modules.matcher import MaxIoUMatcher


def test_iou_match():
    max_iou_matcher = MaxIoUMatcher(
        pos_iou=0.7,
        neg_iou=0.3,
        allow_low_quality_match=True,
        low_quality_match_iou=0.0,
        legacy_bbox=True,
        overlap_type="iou",
    )

    anchor = torch.tensor(
        [
            [0.0, 0, 100, 100],
            [100.0, 100, 180, 180],
            [200.0, 200, 230, 230],
            [200.0, 200, 240, 240],
            [200.0, 200, 500, 500],
            [500.0, 500, 600, 600],
        ]
    ).reshape(1, -1, 4)

    gt = torch.tensor(
        [
            [0.0, 0.0, 10, 10, 1],
            [100.0, 100, 175, 175, 1],
            [200.0, 200, 250, 250, 1],
        ]
    ).reshape(1, -1, 5)

    gt_bbox_num = torch.tensor([gt.shape[1]], dtype=torch.float)

    ious = bbox_overlaps(anchor[0, ...], gt[0, :, 0:4], mode="iou")
    flag, matched_gt_id = max_iou_matcher(anchor, gt, gt_bbox_num)

    # check low quality match result
    gt_max_iou_anchor_indexs = torch.argmax(ious, dim=0)
    assert torch.all(flag[0][gt_max_iou_anchor_indexs] == 1)

    # check pos match result
    anchor_max_ious = torch.max(ious, dim=1)[0]
    pos_indx = anchor_max_ious > 0.7
    assert torch.all(flag[0][pos_indx] == 1)

    # check neg match result (remove low quality gt assign from bg iou match)
    neg_indx = set(
        torch.nonzero(anchor_max_ious < 0.3, as_tuple=False)
        .reshape(-1)
        .cpu()
        .numpy()
    ) - set(gt_max_iou_anchor_indexs.cpu().numpy())
    assert torch.all(flag[0][torch.tensor(list(neg_indx))] == 0)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
