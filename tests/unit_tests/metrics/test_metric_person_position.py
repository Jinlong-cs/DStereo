import torch

from hat.metrics.metric_person_position import PersonPostionMetric


def test_person_position_metric():
    metric = PersonPostionMetric(0.5, 4)
    outs = {
        "pred_position": torch.zeros((1, 1)),
        "pred_boxes": torch.randn((1, 1, 4)),
    }
    batch = {
        "image_name": ["test_img.jpg"],
        "gt_position": torch.zeros((1, 1)),
        "gt_boxes": torch.randn((1, 1, 4)),
    }
    metric.update(batch, outs)
    metric.get()
