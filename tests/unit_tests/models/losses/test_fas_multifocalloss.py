import torch

from hat.registry import build_from_registry


def test_multihead_sigmoid_loss():
    multihead_loss = build_from_registry(dict(type="FasMultiheadFocalLoss"))
    preds = [torch.tensor([-1000, 1000]), torch.tensor([-1000, -1000])]
    fas_label = torch.tensor([1, 0])
    car_cls = torch.tensor([0, 1])
    loss = multihead_loss(preds, fas_label, car_cls)
    assert abs(loss - 27.631) < 1e-4
