import pytest
import torch

from hat.metrics.eye_status_metrics import (
    BinEyeStatusMetrics,
    EyeStatusMetrics,
    MixEyeStatusMetrics,
)
from hat.metrics.metric import EvalMetric
from tests.unit_tests.metrics.testers import MetricTester

preds1 = torch.tensor(
    [
        [0.8655, 0.6915, 0.4705, 0.7215, 0.1122],
        [0.9003, 0.2780, 0.8987, 0.8659, 0.5704],
        [0.3755, 0.6302, 0.9484, 0.7850, 0.2123],
        [0.0737, 0.6213, 0.2202, 0.0146, 0.9572],
        [0.3753, 0.5160, 0.0784, 0.9733, 0.8263],
    ]
)

preds1_bin = torch.tensor(
    [[-0.2649], [-2.6998], [0.5813], [-2.3554], [1.8791]]
)
labels = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.03, 0.0, 0.97, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)


class TestEyeStatusMetrics(MetricTester):
    def setup_class(self):
        super(TestEyeStatusMetrics, self).setup_class(self)
        self.metric1 = EyeStatusMetrics(
            num_classes=5,
            name="eye-status-1",
        )
        self.metric2 = EyeStatusMetrics(
            num_classes=5,
            name="eye-status-2",
        )
        self.labels = labels
        self.preds = preds1

        self.metric1.update(self.labels, self.preds)
        self.name1, self.result1 = self.metric1.get()
        self.args = (2, self.preds, self.labels, self.metric2, self.result1)

    def func_test(
        self,
        rank: int,
        worldsize: int,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metric: EvalMetric,
        ref_result: float,
    ):
        """Utility function doing the actual comparison between
        ddp metric and reference metric value.

        Args:
            rank: rank of current process.
            worldsize: number of processes.
            preds: torch tensor with predictions.
            labels: torch tensor with targets.
            metric: ddp metric class that should be tested.
            ref_result: value that is used for comparison.
        """
        torch.cuda.set_device(rank)
        assert preds.shape[0] == labels.shape[0]
        num_batches = preds.shape[0]
        metric = metric.cuda(rank)
        preds = preds.cuda(rank)
        labels = labels.cuda(rank)

        for i in range(rank, num_batches, worldsize):
            metric.update(labels[i], preds[i])

        name, result = self.check_states_values(metric)


class TestBinEyeStatusMetrics(MetricTester):
    def setup_class(self):
        super(TestBinEyeStatusMetrics, self).setup_class(self)
        self.metric1 = BinEyeStatusMetrics(
            bin_inds=[0, 3],
            sigmoid_thresh=0.7,
            name="bin-eye-status-1",
        )
        self.metric2 = BinEyeStatusMetrics(
            bin_inds=[0, 3],
            sigmoid_thresh=0.7,
            name="bin-eye-status-2",
        )
        self.labels = labels
        self.preds = preds1_bin

        self.metric1.update(self.labels, self.preds)
        self.name1, self.result1 = self.metric1.get()
        self.args = (2, self.preds, self.labels, self.metric2, self.result1)

    def func_test(
        self,
        rank: int,
        worldsize: int,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metric: EvalMetric,
        ref_result: float,
    ):
        """Utility function doing the actual comparison between
        ddp metric and reference metric value.

        Args:
            rank: rank of current process.
            worldsize: number of processes.
            preds: torch tensor with predictions.
            labels: torch tensor with targets.
            metric: ddp metric class that should be tested.
            ref_result: value that is used for comparison.
        """
        torch.cuda.set_device(rank)
        assert preds.shape[0] == labels.shape[0]
        num_batches = preds.shape[0]
        metric = metric.cuda(rank)
        preds = preds.cuda(rank)
        labels = labels.cuda(rank)

        for i in range(rank, num_batches, worldsize):
            metric.update(labels[i], preds[i])

        name, result = self.check_states_values(metric)


class TestMixEyeStatusMetrics(MetricTester):
    def setup_class(self):
        super(TestMixEyeStatusMetrics, self).setup_class(self)
        self.metric1 = MixEyeStatusMetrics(
            num_classes=5,
            name="mix-eye-status-1",
        )
        self.metric2 = MixEyeStatusMetrics(
            num_classes=5,
            name="mix-eye-status-2",
        )
        self.labels = labels
        self.preds = preds1
        self.preds_bin = preds1_bin

        self.metric1.update(self.labels, self.preds, self.preds_bin)
        self.name1, self.result1 = self.metric1.get()
        self.args = (
            2,
            self.preds,
            self.preds_bin,
            self.labels,
            self.metric2,
            self.result1,
        )

    def func_test(
        self,
        rank: int,
        worldsize: int,
        preds: torch.Tensor,
        preds_bin: torch.Tensor,
        labels: torch.Tensor,
        metric: EvalMetric,
        ref_result: float,
    ):
        """Utility function doing the actual comparison between
        ddp metric and reference metric value.

        Args:
            rank: rank of current process.
            worldsize: number of processes.
            preds: torch tensor with predictions.
            preds_bin: torch tensor with binary predictions.
            labels: torch tensor with targets.
            metric: ddp metric class that should be tested.
            ref_result: value that is used for comparison.
        """
        torch.cuda.set_device(rank)
        assert preds.shape[0] == labels.shape[0]
        assert preds_bin.shape[0] == labels.shape[0]
        num_batches = preds.shape[0]
        metric = metric.cuda(rank)
        preds = preds.cuda(rank)
        preds_bin = preds_bin.cuda(rank)
        labels = labels.cuda(rank)

        for i in range(rank, num_batches, worldsize):
            metric.update(labels[i], preds[i], preds_bin[i])

        name, result = self.check_states_values(metric)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
