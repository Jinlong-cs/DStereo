import pytest
import torch

from hat.models.task_modules.person_position.person_position_gt import (
    GetPersonPositionGT,
)


@pytest.mark.parametrize(
    ["roi_nums", "batch_size"],
    [
        pytest.param(8, 2),
        pytest.param(16, 4),
        pytest.param(32, 8),
        pytest.param(64, 16),
    ],
)
def test_person_position_gt(roi_nums, batch_size):
    gt_boxes = torch.randn((batch_size, 100, 6))
    gt_boxes_num = torch.ones((batch_size))
    head = GetPersonPositionGT(roi_nums)
    output = head(gt_boxes=gt_boxes, gt_boxes_num=gt_boxes_num)

    assert "gt_pred" in output
    assert output["gt_pred"].size() == (batch_size, roi_nums, 6)
