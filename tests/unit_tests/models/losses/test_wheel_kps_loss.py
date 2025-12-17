import torch

from hat.models.task_modules.landmark.wheel_kps_loss import Lmks2Loss


def gen_fake_data():
    batch_size = 10
    data = {}
    data["pred"] = {
        "kps_label_pred": torch.randn((batch_size, 2, 8, 8)),
        "kps_pos_offset_pred": torch.randn((batch_size, 4, 8, 8)),
    }
    data["label"] = {
        "kps_cls_label": torch.randn((batch_size, 2, 8, 8)),
        "kps_cls_label_weight": abs(torch.randn((batch_size, 2, 8, 8))),
        "kps_pos_offset": torch.randn((batch_size, 4, 8, 8)),
        "kps_pos_offset_weight": abs(torch.randn((batch_size, 4, 8, 8))),
    }
    return data


def test_face3d_loss():
    data = gen_fake_data()
    cyckps_loss = Lmks2Loss()
    res = cyckps_loss(data["pred"], data["label"])
    assert "wheel_kps_loss" in res.keys()
    assert res["wheel_kps_loss"]["kps_class_loss"] > 0
    assert res["wheel_kps_loss"]["kps_reg_loss"] > 0
