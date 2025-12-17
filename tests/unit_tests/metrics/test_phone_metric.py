import pytest
import torch

from hat.metrics.phone_metric import PrecisionThresh, RecallThresh

torch.manual_seed(0)
batch_size = 128
num_classes = 3
labels = torch.randint(0, num_classes, [batch_size, 1])
preds = torch.randn([batch_size, num_classes, 1, 1])


@pytest.mark.parametrize(["labels", "preds"], [pytest.param(labels, preds)])
def test_RecallThresh(labels, preds):

    phone_recall_metric = RecallThresh(thresh=0.8)
    phone_recall_metric.update(labels, preds)
    num_inst = phone_recall_metric.num_inst
    sum_metric = phone_recall_metric.sum_metric
    assert num_inst == 51
    assert sum_metric == 6


@pytest.mark.parametrize(["labels", "preds"], [pytest.param(labels, preds)])
def test_PrecisionThresh(labels, preds):

    phone_precision_metric = PrecisionThresh(thresh=0.8)
    phone_precision_metric.update(labels, preds)
    num_inst = phone_precision_metric.num_inst
    sum_metric = phone_precision_metric.sum_metric
    assert num_inst == 40
    assert sum_metric == 4
