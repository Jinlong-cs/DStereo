import torch

from hat.models.losses.centernet_loss import CenterNetLoss


def test_centernet_loss():
    centernet_loss = CenterNetLoss(
        task="face",
        loss_weights={"hm": 1.0, "wh": 1.0},
        heatmap_type={"wh": "point"},
        reg_keys=["wh"],
    )
    assert centernet_loss.task == "face"
    assert centernet_loss.loss_weights["hm"] == 1.0
    assert centernet_loss.loss_weights["wh"] == 1.0
    assert centernet_loss.heatmap_type["wh"] == "point"
    assert centernet_loss.reg_keys == ["wh"]
    pred = {
        "hm": torch.rand(size=(1, 1, 48, 240), dtype=torch.float32),
        "wh": torch.rand(size=(1, 2, 48, 240), dtype=torch.float32),
    }
    target = {
        "hm": torch.rand(size=(1, 1, 48, 240), dtype=torch.float32),
        "wh": torch.rand(size=(1, 2, 48, 240), dtype=torch.float32),
        "ignore_mask": torch.zeros(size=(1, 1, 48, 240), dtype=torch.float32),
    }
    all_losses = centernet_loss(pred, target)
    assert (len(all_losses)) == 2
