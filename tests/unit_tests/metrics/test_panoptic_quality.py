import numpy.testing as npt
import torch

from hat.metrics.panoptic_quality import PanopticQualityWithAttributes


def test_panotic_quality():
    attributes = dict(
        double_line=["no", "yes"],
        color=["white", "yellow"],
    )

    metric1 = PanopticQualityWithAttributes(
        attributes=attributes,
        ignore_index=255,
        void_index=0,
        iou_thr=0.5,
        verbose=False,
    )
    metric2 = PanopticQualityWithAttributes(
        attributes=attributes,
        ignore_index=5,
        void_index=0,
        iou_thr=0.8,
        verbose=True,
    )
    metric3 = PanopticQualityWithAttributes(
        ignore_index=255,
        void_index=0,
        iou_thr=0.5,
        verbose=True,
    )

    gt_attributes = [
        torch.tensor([[1, 1, 0], [2, 1, 1], [3, 0, 0], [5, 0, 1]])
    ]
    gt_indexes = [gt_attributes[0][:, :1]]

    gt_id_maps = [
        torch.tensor(
            [
                [2, 2, 0, 1, 0, 5],
                [2, 2, 0, 1, 3, 5],
                [2, 2, 0, 1, 3, 5],
                [2, 2, 0, 1, 3, 5],
                [2, 2, 0, 1, 3, 5],
                [2, 2, 0, 0, 3, 5],
            ]
        )
    ]

    pred_attributes = [
        torch.tensor([[1, 0, 0], [2, 1, 1], [3, 0, 0], [4, 1, 1]])
    ]
    pred_indexes = [pred_attributes[0][:, :1]]

    pred_id_maps = [
        torch.tensor(
            [
                [2, 0, 0, 0, 3, 0],
                [2, 2, 0, 1, 3, 0],
                [2, 2, 0, 1, 3, 4],
                [2, 2, 0, 1, 3, 4],
                [2, 2, 0, 1, 3, 4],
                [2, 2, 0, 1, 3, 0],
            ]
        )
    ]
    metric1.update(gt_attributes, gt_id_maps, pred_attributes, pred_id_maps)
    metric2.update(gt_attributes, gt_id_maps, pred_attributes, pred_id_maps)
    metric3.update(gt_indexes, gt_id_maps, pred_indexes, pred_id_maps)

    iou1 = 4.0 / 6.0
    iou2 = 11.0 / 12.0
    iou3 = 5.0 / 6.0
    npt.assert_almost_equal(metric1.tp, 3, decimal=2)
    npt.assert_almost_equal(metric1.fp, 1, decimal=2)
    npt.assert_almost_equal(metric1.fn, 1, decimal=2)
    npt.assert_almost_equal(
        metric1.double_line,
        [[1, 0, 1, iou3], [1, 1, 0, iou1 + iou2]],
        decimal=2,
    )
    npt.assert_almost_equal(
        metric1.color, [[2, 0, 0, iou1 + iou3], [1, 0, 0, iou2]], decimal=2
    )

    npt.assert_almost_equal(metric2.tp, 2, decimal=2)
    npt.assert_almost_equal(metric2.fp, 2, decimal=2)
    npt.assert_almost_equal(metric2.fn, 1, decimal=2)
    npt.assert_almost_equal(
        metric2.double_line, [[1, 0, 0, iou3], [1, 0, 0, iou2]], decimal=2
    )
    npt.assert_almost_equal(
        metric2.color, [[1, 0, 0, iou3], [1, 0, 0, iou2]], decimal=2
    )

    _, result1 = metric1.get()
    _, result2 = metric2.get()
    _, result3 = metric3.get()
    npt.assert_almost_equal(result1, 0.6042, decimal=2)
    npt.assert_almost_equal(result2[0], 0.5, decimal=2)
    npt.assert_almost_equal(result2[1], 0.8750, decimal=2)
    npt.assert_almost_equal(result2[2], 0.5714, decimal=2)
    npt.assert_almost_equal(result2[3], 0.5, decimal=2)
    npt.assert_almost_equal(result2[4], 0.6667, decimal=2)
    npt.assert_almost_equal(result3[0], 0.6042, decimal=2)
    npt.assert_almost_equal(result3[1], 0.8056, decimal=2)
    npt.assert_almost_equal(result3[2], 0.7500, decimal=2)
    npt.assert_almost_equal(result3[3], 0.7500, decimal=2)
