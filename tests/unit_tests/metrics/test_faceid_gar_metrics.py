import pytest
import torch

from hat.metrics.faceid_gar_metric import FaceIDGARMetrics
from hat.metrics.metric import EvalMetric
from tests.unit_tests.metrics.testers import MetricTester

preds1 = torch.tensor(
    [
        [0.8655, 0.6915, 0.4705, 0.7215, 0.1122],
        [0.9003, 0.2780, 0.8987, 0.8659, 0.5704],
        [0.3755, 0.6302, 0.9484, 0.7850, 0.2123],
        [0.0737, 0.6213, 0.2202, 0.0146, 0.9572],
        [0.3753, 0.5160, 0.0784, 0.9733, 0.8263],
        [0.8655, 0.6915, 0.4705, 0.7215, 0.1122],
        [0.9003, 0.2780, 0.8987, 0.8659, 0.5704],
        [0.3755, 0.6302, 0.9484, 0.7850, 0.2123],
        [0.0737, 0.6213, 0.2202, 0.0146, 0.9572],
        [0.3753, 0.5160, 0.0784, 0.9733, 0.8263],
    ],
)

labels = torch.tensor(
    [
        0,
        1,
        2,
        0,
        1,
        0,
        1,
        2,
        0,
        1,
    ]
)


class TestFaceidGARMetrics(MetricTester):
    def setup_class(self):
        super(TestFaceidGARMetrics, self).setup_class(self)
        save_dir = "./tmp"

        self.labels = labels
        self.preds = preds1

        self.metric1 = FaceIDGARMetrics(
            total_num_list=[5],
            save_dir=save_dir,
            name_list=["valid1"],
        )
        self.metric2 = FaceIDGARMetrics(
            total_num_list=[5],
            save_dir=save_dir,
            name_list=["valid2"],
        )

        self.metric1.update(self.labels, self.preds)
        self.name1, self.result1 = self.metric1.get()

        self.args = (2, self.preds, self.labels, self.metric2)

    def func_test(
        self,
        rank: int,
        worldsize: int,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metric: EvalMetric,
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
        if metric is None:
            return
        torch.cuda.set_device(rank)
        assert preds.shape[0] == labels.shape[0]
        num_batches = preds.shape[0]
        metric = metric.cuda(rank)
        preds = preds.cuda(rank)
        labels = labels.cuda(rank)

        feat_splits = torch.split(preds, 1)
        label_splits = torch.split(labels, 1)

        for i in range(rank, num_batches, worldsize):
            metric.update(label_splits[i], feat_splits[i])

        names, vals = metric.get()
        assert vals[0] == "./tmp/valid2-res.txt"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
