from typing import Dict

import pytest
import torch

from hat.metrics.acc import (
    Accuracy,
    AccuracyAttrMultiLabel,
    AccuracySeg,
    TopKAccuracy,
)
from hat.metrics.metric import EvalMetric
from tests.unit_tests.metrics.testers import MetricTester

labels = torch.tensor([1, 0, 5, 9, 3, 1, 4, 9, 1, 4])
preds1 = torch.tensor(
    [
        [
            0.8655,
            0.6915,
            0.4705,
            0.7215,
            0.1122,
            0.7973,
            0.5034,
            0.0751,
            0.3912,
            0.1354,
        ],
        [
            0.9003,
            0.2780,
            0.8987,
            0.8659,
            0.5704,
            0.5836,
            0.5290,
            0.6487,
            0.8558,
            0.5800,
        ],
        [
            0.3755,
            0.6302,
            0.9484,
            0.7850,
            0.9928,
            0.0835,
            0.8184,
            0.4992,
            0.0247,
            0.5489,
        ],
        [
            0.0737,
            0.6213,
            0.2202,
            0.0146,
            0.9572,
            0.3956,
            0.8198,
            0.9099,
            0.8418,
            0.9808,
        ],
        [
            0.3753,
            0.5160,
            0.0784,
            0.9733,
            0.8263,
            0.6443,
            0.6565,
            0.2593,
            0.9523,
            0.5371,
        ],
        [
            0.0383,
            0.9859,
            0.3392,
            0.3061,
            0.0329,
            0.7138,
            0.0017,
            0.5426,
            0.4997,
            0.5700,
        ],
        [
            0.5915,
            0.0923,
            0.7026,
            0.7100,
            0.8277,
            0.0974,
            0.6773,
            0.2553,
            0.1046,
            0.3375,
        ],
        [
            0.4916,
            0.2696,
            0.5526,
            0.2494,
            0.2944,
            0.8649,
            0.3116,
            0.9943,
            0.4896,
            0.4823,
        ],
        [
            0.6753,
            0.8476,
            0.0500,
            0.7402,
            0.0839,
            0.7886,
            0.6231,
            0.7379,
            0.2946,
            0.2442,
        ],
        [
            0.7367,
            0.3269,
            0.7218,
            0.8573,
            0.1016,
            0.4089,
            0.0781,
            0.0671,
            0.1142,
            0.7306,
        ],
    ]
)
preds2 = torch.tensor([0, 0, 4, 9, 3, 1, 4, 7, 1, 3])

gt_seg = torch.tensor(
    [
        [0, 3, 1, 3],
        [1, 2, 1, 4],
        [1, 3, 2, 3],
        [1, 0, 2, 2],
    ]
)
pred_seg1 = torch.tensor(
    [
        [
            [7.8079e-01, 6.5239e-01, 3.9478e-03, 1.2957e-01],
            [3.2944e-01, 1.5677e-01, 2.9564e-01, 7.6431e-01],
            [2.0847e-02, 3.1063e-01, 2.1820e-01, 1.4351e-01],
            [4.0191e-04, 5.5312e-01, 4.1960e-01, 9.0267e-01],
        ],
        [
            [7.4966e-01, 9.3016e-01, 3.5233e-01, 5.9173e-01],
            [5.4737e-01, 8.5359e-01, 1.4353e-01, 6.5262e-01],
            [5.2334e-01, 8.1792e-01, 6.9801e-01, 2.6988e-01],
            [1.2185e-01, 7.2639e-01, 7.1257e-01, 1.6151e-01],
        ],
        [
            [2.5570e-01, 3.5982e-01, 2.9959e-01, 4.3506e-01],
            [2.3949e-01, 2.6966e-01, 3.9645e-01, 8.6738e-01],
            [4.7211e-02, 5.7131e-01, 4.0772e-01, 4.8490e-01],
            [4.3155e-01, 5.2268e-01, 2.4329e-01, 9.0456e-01],
        ],
        [
            [4.6491e-01, 8.5505e-01, 6.2105e-01, 3.7887e-02],
            [2.6620e-01, 2.6384e-01, 1.1146e-01, 2.0618e-01],
            [8.8737e-01, 2.9978e-01, 9.9043e-01, 7.9432e-01],
            [8.9752e-01, 6.1232e-01, 9.5320e-01, 4.2486e-01],
        ],
    ]
)
pred_seg2 = torch.tensor(
    [
        [0, 3, 1, 3],
        [1, 1, 1, 1],
        [3, 3, 1, 3],
        [1, 0, 2, 2],
    ]
)
outputs1 = {
    "gt_seg": gt_seg,
    "pred_seg": pred_seg1,
}
outputs2 = {
    "gt_seg": gt_seg,
    "pred_seg": pred_seg2,
}


class TestAccuracy(MetricTester):
    def setup_class(self):
        super(TestAccuracy, self).setup_class(self)
        self.metric1 = Accuracy(name="accuracy1")
        self.metric2 = Accuracy(name="accuracy2")
        self.preds = torch.randn(100, 10)
        self.labels = torch.randint(high=10, size=(100, 1))
        self.metric1.update(self.labels, self.preds)
        self.name1, self.result1 = self.metric1.get()
        self.args = (2, self.preds, self.labels, self.metric2, self.result1)

    @pytest.mark.parametrize(
        "labels, preds",
        [
            (labels, preds1),
            (labels, preds2),
            ([labels, labels], [preds1, preds2]),
        ],
    )
    def test_acc(self, labels, preds):
        acc = Accuracy(name="accuracy")
        acc.update(labels, preds)
        name, result = acc.get()
        assert name == "accuracy"
        assert abs(result - 0.60) < 1e-6

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
            metric.update(labels[i], preds[i].unsqueeze(0))

        name, result = self.check_states_values(metric)
        assert abs(result - ref_result) < 1e-6


class TestTop5Accuracy(TestAccuracy):
    def setup_class(self):
        super(TestTop5Accuracy, self).setup_class(self)
        self.metric1 = TopKAccuracy(top_k=5, name="accuracy1")
        self.metric2 = TopKAccuracy(top_k=5, name="accuracy2")
        self.preds = torch.randn(100, 10)
        self.labels = torch.randint(high=10, size=(100, 1))
        self.metric1.update(self.labels, self.preds)
        self.name1, self.result1 = self.metric1.get()
        self.args = (2, self.preds, self.labels, self.metric2, self.result1)

    @pytest.mark.parametrize(
        "labels, preds",
        [(labels, preds1), ([labels, labels], [preds1, preds1])],
    )
    def test_acc(self, labels, preds):
        top5acc = TopKAccuracy(top_k=5, name="top_5_accuracy")
        top5acc.update(labels, preds)
        name, result = top5acc.get()
        assert name == "top_5_accuracy_5"
        assert abs(result - 0.70) < 1e-6


class TestAccuracySeg(MetricTester):
    def setup_class(self):
        super(TestAccuracySeg, self).setup_class(self)
        self.metric1 = AccuracySeg(axis=-1, name="accuracy1")
        self.metric2 = AccuracySeg(axis=-1, name="accuracy2")
        pred_seg = torch.randn(100, 10, 10)
        gt_seg = torch.randint(high=10, size=(100, 10))
        self.outputs = {
            "gt_seg": gt_seg,
            "pred_seg": pred_seg,
        }
        self.metric1.update(self.outputs)
        self.name1, self.result1 = self.metric1.get()
        self.args = (2, self.outputs, self.metric2, self.result1)

    @pytest.mark.parametrize(
        "output",
        [
            (outputs1),
            (outputs2),
        ],
    )
    def test_acc_seg(self, output):
        accseg = AccuracySeg(axis=-1, name="accuracy")
        accseg.update(output)
        name, result = accseg.get()
        assert name == "accuracy"
        assert abs(result - 0.75) < 1e-6

    def func_test(
        self,
        rank: int,
        worldsize: int,
        output: Dict[str, torch.Tensor],
        metric: EvalMetric,
        ref_result: float,
    ):
        """Utility function doing the actual comparison between
        ddp metric and reference metric value.

        Args:
            rank: rank of current process.
            worldsize: number of processes.
            output: dict including gt_seg and pred_seg.
            metric: ddp metric class that should be tested.
            ref_result: value that is used for comparison.
        """
        torch.cuda.set_device(rank)
        gt_seg = output["gt_seg"]
        pred_seg = output["pred_seg"]
        assert gt_seg.shape[0] == pred_seg.shape[0]
        num_batches = gt_seg.shape[0]
        metric = metric.cuda(rank)
        gt_seg = gt_seg.cuda(rank)
        pred_seg = pred_seg.cuda(rank)
        for i in range(rank, num_batches, worldsize):
            output_i = {
                "gt_seg": gt_seg[i],
                "pred_seg": pred_seg[i],
            }
            metric.update(output_i)
        name, result = self.check_states_values(metric)
        assert abs(result - ref_result) < 1e-6


class TestAccuracyAttrMultiLabel(MetricTester):
    def setup_class(self):
        super(TestAccuracyAttrMultiLabel, self).setup_class(self)
        self.metric1 = AccuracyAttrMultiLabel(
            name="accuracy1",
            attr_type_name="type",
            attr_type_list=["type"],
            attr_type_numcls=[10],
        )
        self.metric2 = AccuracyAttrMultiLabel(
            name="accuracy2",
            attr_type_name="type",
            attr_type_list=["type"],
            attr_type_numcls=[10],
        )
        self.preds = torch.randn(100, 10)
        self.labels = torch.randint(high=10, size=(100, 1))
        self.labels = torch.zeros(100, 10).scatter_(1, self.labels, 1)
        self.metric1.update(self.labels, self.preds)
        self.name1, self.result1 = self.metric1.get()
        self.args = (2, self.preds, self.labels, self.metric2, self.result1)

    @pytest.mark.parametrize(
        "labels, preds, attr_type_name",
        [
            (labels, preds1, "type"),
            (labels, preds1, "all_attribute"),
        ],
    )
    def test_acc(self, labels, preds, attr_type_name):
        labels = torch.zeros(10, 10).scatter_(1, labels.unsqueeze(1), 1)
        acc = AccuracyAttrMultiLabel(
            name="accuracy",
            attr_type_name=attr_type_name,
            attr_type_list=["type"],
            attr_type_numcls=[10],
        )
        acc.update(labels, preds)
        name, result = acc.get()
        assert name == "accuracy" + "_" + attr_type_name
        assert abs(result - 0.60) < 1e-6

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
            metric.update(labels[i].unsqueeze(0), preds[i].unsqueeze(0))

        name, result = self.check_states_values(metric)
        assert abs(result - ref_result) < 1e-6
