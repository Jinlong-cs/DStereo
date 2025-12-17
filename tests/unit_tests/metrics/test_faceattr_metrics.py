import numpy.testing as npt
import torch

from hat.metrics.faceattr_metrics import (
    CumulativeAccuracy,
    SigmoidOrdinalAccuracy,
    SigmoidOrdinalMAE,
)

MODEL_OUTS = {
    "age": torch.Tensor([2, 18]),
    "pred_age": torch.Tensor(2, 20).uniform_(0.01, 2),
}


def test_cumulative_acc():
    cum_acc = CumulativeAccuracy(offset=5, classes=20)
    cum_acc.update(MODEL_OUTS)
    _, val = cum_acc.get()
    npt.assert_almost_equal(val, 0.50, decimal=5)


def test_sigmoid_mae():
    mae = SigmoidOrdinalMAE(from_sigmoid=False)
    mae.update(MODEL_OUTS)
    _, val = mae.get()
    npt.assert_almost_equal(val, 10, decimal=5)


def test_seg_acc():
    seg_acc_0 = SigmoidOrdinalAccuracy(offset=0, age_bins=[6, 12, 19])
    seg_acc_0.update(MODEL_OUTS)
    _, val = seg_acc_0.get()
    npt.assert_almost_equal(val, 0, decimal=5)

    seg_acc_1 = SigmoidOrdinalAccuracy(offset=1, age_bins=[6, 12, 19])
    seg_acc_1.update(MODEL_OUTS)
    _, val = seg_acc_1.get()
    npt.assert_almost_equal(val, 0.5, decimal=5)
