import numpy.testing as npt
import torch

from hat.metrics.mean_iou import MeanIOU


def test_mean_iou():
    seg_class1 = ["class1", "class2", "class3"]
    seg_class2 = ["class1", "class2", "class3", "class4"]
    ignore_index = 255
    global_ignore_index = 0
    global_ignore_index_v2 = [0, 1]

    metric1 = MeanIOU(seg_class=seg_class1, ignore_index=ignore_index)
    metric2 = MeanIOU(
        seg_class=seg_class1, global_ignore_index=global_ignore_index
    )
    metric3 = MeanIOU(
        seg_class=seg_class1,
        ignore_index=ignore_index,
        verbose=True,
    )
    metric4 = MeanIOU(
        seg_class=seg_class1, global_ignore_index=global_ignore_index_v2
    )
    metric5 = MeanIOU(seg_class=seg_class2, ignore_index=ignore_index)

    gt_seg = torch.tensor(
        [
            [
                [0, 0, 0, 1, 0, 0],
                [0, 2, 0, 1, 1, 0],
                [2, 2, 1, 1, 1, 0],
                [2, 2, 0, 1, 0, 0],
                [2, 2, 0, 1, 0, 0],
                [2, 2, 0, 1, 0, 0],
            ]
        ]
    )
    pred_seg = torch.tensor(
        [
            [
                [0, 0, 0, 1, 0, 0],
                [0, 1, 0, 1, 1, 0],
                [2, 2, 1, 1, 1, 0],
                [2, 2, 0, 1, 0, 0],
                [2, 2, 0, 1, 0, 0],
                [2, 1, 0, 0, 0, 0],
            ]
        ]
    )
    metric1.update(gt_seg, pred_seg)
    _, result1 = metric1.get()
    metric2.update(gt_seg, pred_seg)
    _, result2 = metric2.get()
    metric3.update(gt_seg, pred_seg)
    _, result3 = metric3.get()
    metric4.update(gt_seg, pred_seg)
    _, result4 = metric4.get()
    metric5.update(gt_seg, pred_seg)
    _, result5 = metric5.get()

    npt.assert_almost_equal(result1, [0.82], decimal=2)
    npt.assert_almost_equal(result2, [0.75], decimal=2)
    npt.assert_almost_equal(result3[0], [0.82], decimal=2)
    npt.assert_almost_equal(result3[1], [0.89], decimal=2)
    npt.assert_almost_equal(result3[2], [0.92], decimal=2)
    npt.assert_almost_equal(result3[3], [0.95, 0.73, 0.78], decimal=2)
    npt.assert_almost_equal(result3[4], [1.00, 0.89, 0.78], decimal=2)
    assert result3[5] == ["class1", "class2", "class3"]
    npt.assert_almost_equal(result4, [0.78], decimal=2)
    npt.assert_almost_equal(result5, [0.82], decimal=2)


test_mean_iou()
