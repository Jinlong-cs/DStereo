import numpy as np
import pytest
import torch

from hat.metrics.recall_precision import RecallPrecision
from tests.unit_tests.metrics.testers import MetricTester


@pytest.mark.parametrize("class_names", [None, ["class_test"]])
def test_recall_precision(class_names):
    metric = RecallPrecision(
        score_thresh=0.5, num_classes=1, class_names=class_names
    )
    metric.reset()

    batch = 1
    num_bbox = 2
    targets = dict()
    targets["gt_bboxes"] = torch.ones((batch, num_bbox, 4))
    for i in range(num_bbox):
        width, height = 100, 100
        targets["gt_bboxes"][:, i, 0] = i * width
        targets["gt_bboxes"][:, i, 1] = i * height
        targets["gt_bboxes"][:, i, 2] = targets["gt_bboxes"][:, i, 0] + width
        targets["gt_bboxes"][:, i, 3] = targets["gt_bboxes"][:, i, 1] + height
    targets["gt_classes"] = torch.zeros((batch, num_bbox))
    targets["gt_difficult"] = torch.zeros((batch, num_bbox))
    outputs = torch.zeros((batch, num_bbox, 6))
    outputs[:, :, 0:4] = targets["gt_bboxes"]
    # set pred score to 1.0
    outputs[:, :, 5] = 1
    targets["pred_bboxes"] = outputs
    metric.update(targets)
    result = metric.get()
    # check recall
    assert np.allclose(result[1][0], 1.0)
    # check precision
    assert np.allclose(result[1][0], 1.0)

    # change half pred bbox to no matched gt bbox
    metric.reset()
    outputs[:, num_bbox // 2 :, 0:4] = torch.tensor([1000, 1000, 2000, 2000])
    targets["pred_bboxes"] = outputs
    metric.update(targets)
    result = metric.get()
    # check recall
    assert np.allclose(result[1][0], 0.5)
    # check precision
    assert np.allclose(result[1][1], 0.5)

    # set half pred bbox score to 0.0
    metric.reset()
    outputs[:, num_bbox // 2 :, 5] = 0.0
    targets["pred_bboxes"] = outputs
    metric.update(targets)
    result = metric.get()
    # check recall
    assert np.allclose(result[1][0], 0.5)
    # check precision
    assert np.allclose(result[1][1], 1.0)

    # set low score bbox relation gt to difficult
    metric.reset()
    targets["gt_difficult"][:, num_bbox // 2 :] = 1
    metric.update(targets)
    result = metric.get()
    # check recall
    assert np.allclose(result[1][0], 1.0)
    # check precision
    assert np.allclose(result[1][1], 1.0)


class TestDistRecallPrecision(MetricTester):
    def setup_class(self):
        super(TestDistRecallPrecision, self).setup_class(self)
        num_classes = 5
        batch = 10
        num_bbox_per_batch = 10
        metric1 = RecallPrecision(score_thresh=0.5, num_classes=num_classes)
        metric2 = RecallPrecision(score_thresh=0.5, num_classes=num_classes)

        targets = dict()
        targets["gt_bboxes"] = torch.ones((batch, num_bbox_per_batch, 4))
        for i in range(num_bbox_per_batch):
            width, height = 100, 100
            targets["gt_bboxes"][:, i, 0] = i * width
            targets["gt_bboxes"][:, i, 1] = i * height
            targets["gt_bboxes"][:, i, 2] = (
                targets["gt_bboxes"][:, i, 0] + width
            )
            targets["gt_bboxes"][:, i, 3] = (
                targets["gt_bboxes"][:, i, 1] + height
            )
        targets["gt_classes"] = torch.zeros((batch, num_bbox_per_batch))
        for i in range(num_classes):
            targets["gt_classes"][:, i::num_classes] = i
        targets["gt_difficult"] = torch.zeros((batch, num_bbox_per_batch))

        preds = []
        for i in range(10):
            great_preds = torch.cat(
                [
                    targets["gt_bboxes"][i],
                    targets["gt_classes"][i].unsqueeze(-1),
                    torch.ones((10, 1)),
                ],
                dim=-1,
            )
            preds.append(great_preds)

        targets["pred_bboxes"] = preds
        metric1.update(targets)
        _, values = metric1.get()
        self.args = (preds, targets, values, metric2)

    def func_test(self, rank, preds, targets, values, metric):
        torch.cuda.set_device(rank)
        metric = metric.cuda()
        rank_pred = preds[rank * 5 : rank * 5 + 5]
        rank_pred = [pred.cuda() for pred in rank_pred]
        rank_target = dict()
        for k, v in targets.items():
            if not k == "pred_bboxes":
                rank_target[k] = v[rank * 5 : rank * 5 + 5].cuda()
        rank_target["pred_bboxes"] = rank_pred
        metric.update(rank_target)

        _, m_values = metric.get()
        assert m_values == values


if __name__ == "__main__":
    pytest.main(["-s", __file__])
