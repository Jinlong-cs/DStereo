import torch

from hat.metrics.roi_cls_prec_rec import ROICLSPrecRec
from hat.metrics.roi_kps import ROIKPSMetric, ROIKPSMetricCoupling
from tests.unit_tests.metrics.testers import MetricTester


class TestDistROICLSPrecRec(MetricTester):
    def setup_class(self):
        super().setup_class(self)
        bs, tk, nc = 10, 10, 5
        metric1 = ROICLSPrecRec(num_classes=nc)
        metric2 = ROICLSPrecRec(num_classes=nc)

        gt_bboxes_xy1 = torch.rand((bs, tk, 2))
        gt_bboxes_xy2 = gt_bboxes_xy1 + 15
        gt_bboxes = torch.cat([gt_bboxes_xy1, gt_bboxes_xy2], dim=-1)
        gt_classes = torch.randint(0, nc, (bs, tk)).unsqueeze(-1)
        pred_scores = torch.zeros((bs, tk, nc))
        pred_scores = pred_scores.scatter(2, gt_classes, 1)

        preds = []
        for i in range(10):
            great_preds = torch.cat(
                [
                    gt_bboxes[i],
                    pred_scores[i],
                ],
                dim=-1,
            )
            preds.append(great_preds)

        targets = torch.cat([gt_bboxes, gt_classes + 1], dim=-1)

        metric1.update(targets, preds)
        _, values = metric1.get()
        self.args = (preds, targets, values, metric2)

    def func_test(self, rank, preds, targets, values, metric):
        torch.cuda.set_device(rank)
        metric = metric.cuda()
        rank_pred = preds[rank * 5 : rank * 5 + 5]
        rank_pred = [pred.cuda() for pred in rank_pred]
        rank_target = targets[rank * 5 : rank * 5 + 5]
        rank_target = [target.cuda() for target in rank_target]
        metric.update(rank_target, rank_pred)

        _, m_values = metric.get()
        assert m_values == values


class TestDistROIKPSMetric(MetricTester):
    def setup_class(self):
        bs, tk, nk = 10, 10, 2
        metric1, metric2 = self.get_metrics(num_kps=nk)

        gt_bboxes_xy1 = torch.rand((bs, tk, 2))
        gt_bboxes_xy2 = gt_bboxes_xy1 + 15
        gt_bboxes = torch.cat([gt_bboxes_xy1, gt_bboxes_xy2], dim=-1)
        gt_kps = gt_bboxes_xy1.unsqueeze(-1) + torch.rand((bs, tk, 2, nk))
        gt_kps = torch.cat(
            [gt_kps, torch.randint(0, 3, (bs, tk, 1, nk))], dim=-2
        )
        gt_kps = gt_kps.reshape((bs, tk, -1))

        preds = []
        for i in range(10):
            great_preds = torch.cat(
                [
                    gt_bboxes[i],
                    gt_kps[i],
                ],
                dim=-1,
            )
            preds.append(great_preds)

        targets = preds

        metric1.update(targets, preds)
        _, values = metric1.get()
        self.args = (preds, targets, values, metric2)

    def func_test(self, rank, preds, targets, values, metric):
        torch.cuda.set_device(rank)
        metric = metric.cuda()
        rank_pred = preds[rank * 5 : rank * 5 + 5]
        rank_pred = [pred.cuda() for pred in rank_pred]
        rank_target = targets[rank * 5 : rank * 5 + 5]
        rank_target = [target.cuda() for target in rank_target]
        metric.update(rank_target, rank_pred)

        _, m_values = metric.get()
        assert m_values == values

    def get_metrics(num_kps):
        metric1 = ROIKPSMetric(kps_num=num_kps)
        metric2 = ROIKPSMetric(kps_num=num_kps)
        return metric1, metric2


class TestDistROIKPSMetricCoupling(TestDistROIKPSMetric):
    def get_metrics(num_kps):
        metric1 = ROIKPSMetricCoupling(kps_num=num_kps)
        metric2 = ROIKPSMetricCoupling(kps_num=num_kps)
        return metric1, metric2


if __name__ == "__main__":
    import pytest

    pytest.main(["-s", __file__])
