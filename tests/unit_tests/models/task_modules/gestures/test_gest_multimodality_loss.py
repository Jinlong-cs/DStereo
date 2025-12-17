import numpy as np
import numpy.testing as npt
import pytest
import torch
from torch import nn

from hat.registry import build_from_registry

BATCHSIZE = 2
NUM_CLASS = 59


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


@pytest.mark.parametrize("fusion_type", ["result_fusion", "feature_fusion"])
def test_gest_multimode_loss(fusion_type):
    setup_seed(2022)
    logit_output = [
        torch.rand(BATCHSIZE, NUM_CLASS),  # logit_kps
        torch.rand(BATCHSIZE, NUM_CLASS),  # logit_frames
        torch.rand(BATCHSIZE, NUM_CLASS),  # logit_kps_frames
    ]

    weight = torch.randint(low=0, high=2, size=(BATCHSIZE,))
    target = torch.randint(low=-1, high=NUM_CLASS, size=(BATCHSIZE,))
    if fusion_type == "result_fusion":
        logit_output[2] = None
        loss_kps = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        loss_frames = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        loss_kps_frames = None
        loss_weight = [1, 1]
    elif fusion_type == "feature_fusion":
        logit_output[0] = None
        logit_output[1] = None
        loss_kps = None
        loss_frames = None
        loss_kps_frames = nn.CrossEntropyLoss(
            ignore_index=-1, reduction="none"
        )
        loss_weight = [1]

    config = dict(
        type="GestMultiModeLoss",
        fusion_type=fusion_type,
        loss_kps=loss_kps,
        loss_frames=loss_frames,
        loss_kps_frames=loss_kps_frames,
        loss_weight=loss_weight,
    )
    gest_multimode_loss = build_from_registry(config)
    loss = gest_multimode_loss(logit_output, target, weight)
    print(fusion_type, loss.detach())
    if fusion_type == "result_fusion":
        npt.assert_almost_equal(loss.detach().numpy(), 8.25573, decimal=5)

    if fusion_type == "feature_fusion":
        npt.assert_almost_equal(loss.detach().numpy(), 4.39511, decimal=5)
