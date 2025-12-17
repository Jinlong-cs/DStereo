import torch

from hat.models.losses.pccr.pccr_losses import PCCRLoss


def gen_fake_data():
    id_batch_size = 2
    imgn_single_id = 4
    pbn = 6
    imgn = id_batch_size * imgn_single_id
    pred = {
        "angle": torch.randn((imgn, 2)),
        "R": torch.randn((imgn, 1)),
        "K": torch.randn((imgn, 1)),
        "alpha": torch.randn((imgn, 1)),
        "beta": torch.randn((imgn, 1)),
        "pitch": torch.randn((imgn, 1)),
        "kq_result": torch.randn((imgn, 2)),
        "yaw": torch.randn((imgn, 1)),
        "pb": torch.randn((imgn, pbn, 3)),
        "p": torch.randn((imgn, 3)),
        "valid_mask": torch.randn((imgn, pbn)) > 0,
        "center_1": torch.randn((imgn, 3)),
        "center_2": torch.randn((imgn, 3)),
        "center": torch.randn((imgn, 3)),
        "screen_coords": torch.randn((imgn, 2)),
    }
    label = {
        "angle": torch.randn((imgn, 2)),
        "eye3d": torch.randn((imgn, 3)),
        "gaze_point": torch.randn((imgn, 2)),
    }
    data = {
        "pred": pred,
        "label": label,
    }
    return data, id_batch_size


def test_pccr_loss():
    data, id_batch_size = gen_fake_data()
    params_range = {
        "R": [3, 20],
        "K": [2, 15],
        "alpha": [-0.174, 0.175],
        "beta": [-0.087, 0.088],
        "pitch": [-20, 20],
        "yaw": [-20, 20],
    }
    eye_params_typical = {
        "R": 7.8,
        "K": 4.75,
        "alpha": 0.088,
        "beta": 0.026,
    }

    loss_weights = {
        "params_consistency": {
            "R": 1,
            "K": 1,
            "alpha": 10,
            "beta": 20,
        },
        "regularization": {
            "R": 1e-3,
            "K": 1e-3,
            "alpha": 1e-2,
            "beta": 1e-2,
        },
        "in_range": {
            "R": 1,
            "K": 1,
            "alpha": 10,
            "beta": 20,
            "pitch": 20,
            "yaw": 20,
            "kq": 20,
        },
        "pupil_dis": 1,
        "angle": 1,
        "center_dis": 0.1,
        "eye3d_center_dis": 1e-3,
        "gaze_point_dis": 1e-3,
    }
    pccr_loss = PCCRLoss(
        params_range,
        eye_params_typical,
        loss_weights,
        id_batch_size,
    )
    output = pccr_loss(data)
    for _, v in output.items():
        assert v >= 0
