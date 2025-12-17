import torch

from hat.metrics.voc_detection import VOC07MApMetric, VOCMApMetric
from hat.registry import build_from_registry
from tests.unit_tests.metrics.testers import MetricTester


def test_voc_map():
    cfg = VOCMApMetric(num_classes=1)
    metric = build_from_registry(cfg)
    metric.reset()

    targets = dict()
    targets["gt_bboxes"] = torch.rand((5, 10, 4))
    targets["gt_classes"] = torch.rand((5, 10))
    targets["gt_difficult"] = torch.rand((5, 10))
    outputs = [torch.rand((200, 6)) for i in range(5)]
    targets["pred_bboxes"] = outputs
    metric.update(targets)
    result = metric.get()
    assert result is not None


def test_voc07_map():
    cfg = VOC07MApMetric(num_classes=1)
    metric = build_from_registry(cfg)
    metric.reset()

    targets = dict()
    targets["gt_bboxes"] = torch.rand((5, 10, 4))
    targets["gt_classes"] = torch.rand((5, 10))
    targets["gt_difficult"] = torch.rand((5, 10))
    outputs = [torch.rand((200, 6)) for i in range(5)]
    targets["pred_bboxes"] = outputs
    metric.update(targets)
    result = metric.get()
    assert result is not None


class TestDistVocMap(MetricTester):
    def setup_class(self):
        super(TestDistVocMap, self).setup_class(self)
        metric1 = VOCMApMetric(num_classes=5)
        metric2 = VOCMApMetric(num_classes=5)

        targets = dict()
        targets["gt_bboxes"] = torch.rand((10, 10, 4))
        targets["gt_classes"] = torch.randint(0, 5, (10, 10))
        targets["gt_difficult"] = torch.zeros((10, 10))

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


class TestDistVocMapWithRandParams(TestDistVocMap):
    def setup_class(self):
        super(TestDistVocMap, self).setup_class(self)
        metric1 = VOCMApMetric(num_classes=5)
        metric2 = VOCMApMetric(num_classes=5)

        targets = dict()
        targets["gt_bboxes"] = torch.rand((10, 10, 4))
        targets["gt_classes"] = torch.randint(0, 5, (10, 10))
        targets["gt_difficult"] = torch.zeros((10, 10))

        preds = [torch.rand((200, 6)) for i in range(10)]

        targets["pred_bboxes"] = preds
        metric1.update(targets)
        _, values = metric1.get()
        self.args = (preds, targets, values, metric2)


class TestDistVoc07Map(MetricTester):
    def setup_class(self):
        super(TestDistVoc07Map, self).setup_class(self)
        metric1 = VOC07MApMetric(num_classes=5)
        metric2 = VOC07MApMetric(num_classes=5)

        targets = dict()
        targets["gt_bboxes"] = torch.rand((10, 10, 4))
        targets["gt_classes"] = torch.randint(0, 5, (10, 10))
        targets["gt_difficult"] = torch.zeros((10, 10))

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
