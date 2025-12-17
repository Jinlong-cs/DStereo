import numpy.testing as npt
import torch

from hat.metrics.confusion_matrix import ConfusionMatrix


def test_confusion_matrix():
    seg_class = ["class1", "class2", "class3"]
    ignore_index = 255

    metric = ConfusionMatrix(seg_class=seg_class, ignore_index=ignore_index)
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

    result_matrix = [
        [1.0000, 0.0000, 0.0000],
        [0.1111, 0.8889, 0.0000],
        [0.0000, 0.2222, 0.7778],
    ]
    result_label = [18.0, 9.0, 9.0]
    result_pred = [19.0, 10.0, 7.0]
    metric.update(gt_seg, pred_seg)
    _, result = metric.get()
    confusion_matrix, label, pred_label = result
    npt.assert_almost_equal(confusion_matrix, result_matrix, decimal=2)
    npt.assert_almost_equal(label, result_label, decimal=2)
    npt.assert_almost_equal(pred_label, result_pred, decimal=2)


test_confusion_matrix()
