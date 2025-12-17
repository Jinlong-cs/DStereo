import pytest
import torch

from hat.metrics.recall import Recall
from tests.unit_tests.metrics.testers import MetricTester

sigmoid_data = {
    "pred": torch.Tensor([10, 10, 10, 10, 10, -10, -10, 10]),
    "label": torch.Tensor([-1, 0.5, -1, 0.5, 1, 1, 0, 0]),
}
softmax_data = {
    "pred": torch.Tensor(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 4, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 4, 3],
            [1, 2, 4, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 4, 2, 3],
        ]
    ),
    "label": torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
}


@pytest.mark.parametrize(
    "task_name, expect_acc",
    [("glass", [0.5, 0.5]), ("brightness", [0, 0.25, 0.5, 0.75])],
)
def test_recall_metric(task_name, expect_acc):
    if task_name == "glass":
        label = sigmoid_data["label"]
        pred = sigmoid_data["pred"]
        # ignore the data labeled -1 or 0.5
        pred = pred[label != -1]
        label = label[label != -1]
        pred = pred[label != 0.5]
        label = label[label != 0.5]
        pred = (pred > 0).int()
        cls_num = 2
    else:
        label = softmax_data["label"]
        pred = softmax_data["pred"]
        cls_num = 4

    acc_metric = Recall(axis=1, cls_num=cls_num, task_name=task_name)
    acc_metric.update(label, pred)
    name, val = acc_metric.get()
    assert name == [f"{task_name}_recall_{i}" for i in range(cls_num)]
    assert val == expect_acc


# ddp recall metric
class TestDistRecallMetric(MetricTester):
    def setup_class(self):
        super(TestDistRecallMetric, self).setup_class(self)
        metric1 = Recall(axis=1, cls_num=4, task_name="brightness")
        metric2 = Recall(axis=1, cls_num=4, task_name="brightness")
        metric1.update(softmax_data["label"], softmax_data["pred"])
        _, values = metric1.get()
        self.args = (values, metric2)

    def func_test(self, rank, values, metric):
        torch.cuda.set_device(rank)
        metric = metric.cuda()
        rank_label = softmax_data["label"][rank * 8 : rank * 8 + 8].cuda()
        rank_pred = softmax_data["pred"][rank * 8 : rank * 8 + 8].cuda()
        metric.update(rank_label, rank_pred)
        _, m_values = metric.get()
        assert m_values == values
