import numpy as np
import numpy.testing as npt
import torch
from torch import nn

from hat.models.losses.utils import weight_reduce_loss
from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test

BATCHSIZE = 2
NUM_CLASS = 59


def setup_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def test_dyn_gesture_classifier():
    setup_seed()
    config = dict(
        type="DynGestureClassifier",
        backbone=dict(
            type="VargNetV2",
            num_classes=1000,
            include_top=False,
            alpha=0.5,
            bn_kwargs={},
        ),
        head=dict(type="DynGestureHead", num_classes=NUM_CLASS),
        loss=nn.CrossEntropyLoss(ignore_index=-1, reduction="none"),
    )
    model = build_from_registry(config)
    data = dict(
        clip_keypoints=torch.rand(BATCHSIZE, 3, 16, 21) * 2 - 1,
        act_label=torch.randint(low=-1, high=NUM_CLASS, size=(BATCHSIZE,)),
        label_weight=torch.randint(low=0, high=2, size=(BATCHSIZE,)),
    )
    model.eval()
    preds, target = model(data)
    assert preds.shape == (BATCHSIZE, NUM_CLASS)
    loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    losses = loss_func(preds, target)
    losses = weight_reduce_loss(
        loss=losses, weight=data["label_weight"], reduction="sum"
    )
    npt.assert_almost_equal(losses.detach().numpy(), 8.0196, decimal=4)

    qat_test(model, data, with_quantized=False)


def test_dyn_mbv2_gesture_classifier():
    setup_seed()
    num_classes = NUM_CLASS
    input_channels = 15
    deploy = False
    in_chls = [
        [32],
        [16, 24],
        [24, 32, 32],
        [32] + [64] * 4 + [96] * 2,
        [96] + [128] * 3,
    ]
    out_chls = [
        [16],
        [24, 24],
        [32, 32, 32],
        [64] * 4 + [96] * 3,
        [128] * 3 + [128],
    ]
    config = dict(
        type="DynGestureClassifier",
        backbone=dict(
            type="SNDRMobileNetV2",
            num_classes=1000,
            input_channels=input_channels,
            include_top=False,
            alpha=1,
            bn_kwargs={},  # default: eps 1e-5, m: 0.1
            in_chls=in_chls,
            out_chls=out_chls,
        ),
        head=dict(
            type="DynGestureHead",
            use_pool=False,
            num_classes=num_classes,
            flat_output=not deploy,
        ),
        loss=nn.CrossEntropyLoss(ignore_index=-1, reduction="none"),
    )
    model = build_from_registry(config)
    data = dict(
        clip_keypoints=torch.rand(BATCHSIZE, input_channels, 32, 21) * 2 - 1,
        act_label=torch.randint(low=0, high=NUM_CLASS, size=(BATCHSIZE,)),
        label_weight=torch.randint(low=0, high=2, size=(BATCHSIZE,)),
    )
    model.eval()
    preds, target = model(data)
    assert preds.shape == (BATCHSIZE, NUM_CLASS)
    loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    losses = loss_func(preds, target)
    losses = weight_reduce_loss(
        loss=losses, weight=data["label_weight"], reduction="sum"
    )
    npt.assert_almost_equal(losses.detach().numpy(), 3.9239, decimal=4)
