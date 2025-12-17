import numpy.testing as npt
import torch

# age_cost_sensitive_loss
from hat.models.losses.age_cost_sensitive_loss import CostSensitiveLoss

DATA = {
    "age": torch.Tensor([1, 5]),
    "age_ord": torch.Tensor(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
    ),
    "pred_age": torch.Tensor(
        [
            [0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0, 0, 0, 0],
            [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0, 0],
        ]
    ),
}


def test_cost_sensitive_loss():
    cost_sensitive = CostSensitiveLoss(
        num_classes=10,
        error_max=3,
        from_sigmoid=True,
    )
    pred = DATA["pred_age"]
    label_age = DATA["age"]
    label_ord = DATA["age_ord"]
    loss = cost_sensitive(pred, label_age, label_ord)
    npt.assert_almost_equal(loss, 1.66, decimal=5)
