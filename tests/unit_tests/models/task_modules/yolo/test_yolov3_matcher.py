import torch

from hat.models.task_modules.yolo import YOLOV3Matcher


def test_yolov3_matcher():
    gt_x1y1 = torch.randn(3, 4, 2)
    gt_x2y2 = gt_x1y1 + 1.0
    gt_box = torch.cat([gt_x1y1, gt_x2y2], -1)

    box_x1y1 = torch.randn(3, 10, 2)
    box_x2y2 = box_x1y1 + 2.0
    box = torch.cat([box_x1y1, box_x2y2], -1)

    matcher = YOLOV3Matcher(0.4)

    flag, gt_id = matcher(box, gt_box, torch.tensor([4, 4, 4]))

    assert flag.shape == box.shape[:2]
    assert gt_id.shape == gt_box.shape[:2]
