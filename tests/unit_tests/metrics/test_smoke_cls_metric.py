import pytest
import torch

from hat.metrics.smoke_cls_metric import SmokeClsPrecision, SmokeClsRecall

torch.manual_seed(0)
batch_size = 128
num_classes = 3
target = torch.randint(0, num_classes, [batch_size, 1])
labels = torch.zeros(batch_size, num_classes)
labels = labels.scatter_(1, target, 1)
preds = torch.randn([batch_size, num_classes])


@pytest.mark.parametrize(["labels", "preds"], [pytest.param(labels, preds)])
def test_SmokeClsRecall(labels, preds):

    smoke_recall_metric = SmokeClsRecall(cls_id=1, thresh=0.8)
    smoke_recall_metric.update(labels, preds)
    num_inst = smoke_recall_metric.num_inst
    sum_metric = smoke_recall_metric.sum_metric
    assert num_inst == 51
    assert sum_metric == 6


@pytest.mark.parametrize(["labels", "preds"], [pytest.param(labels, preds)])
def test_SmokeClsPrecision(labels, preds):

    smoke_precision_metric = SmokeClsPrecision(thresh=0.8)
    smoke_precision_metric.update(labels, preds)
    num_inst = smoke_precision_metric.num_inst
    sum_metric = smoke_precision_metric.sum_metric
    assert num_inst == 40
    assert sum_metric == 4
