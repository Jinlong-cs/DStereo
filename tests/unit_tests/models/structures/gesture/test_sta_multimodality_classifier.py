import copy

import numpy as np
import pytest
import torch

from hat.registry import build_from_registry
from projects.halo.cv.configs.gesture.plugins.get_sta_model import (
    get_sta_network,
)
from tests.unit_tests.models.base import qat_test

BATCHSIZE = 2
NUM_FRAME = 8
NUM_CLASS = 71


bn_kwargs = {
    "eps": 2e-5,
    "momentum": 0.9,
}


def setup_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def test_multimode_classifier():
    setup_seed()

    # data
    data = dict(
        frames=torch.rand(BATCHSIZE, NUM_FRAME, 3, 128, 128) * 2 - 1,
        box_center=torch.rand(BATCHSIZE * NUM_FRAME, 32, 1, 1) * 2 - 1,
        clip_keypoints=torch.rand(BATCHSIZE, 3, NUM_FRAME * 2, 21) * 2 - 1,
        label_weight=torch.randint(low=0, high=2, size=(BATCHSIZE,)),
        act_label=torch.randint(low=-1, high=NUM_CLASS + 1, size=(BATCHSIZE,)),
    )
    data_qat = copy.deepcopy(data)
    # -- build model
    multimodality_classifier_dict = get_sta_network(NUM_CLASS)
    model = build_from_registry(multimodality_classifier_dict)

    logit_output, losses = model(data)

    for preds in logit_output:
        if preds is not None:
            assert preds.shape == (BATCHSIZE, NUM_CLASS)

    qat_test(model, data_qat, with_quantized=False)


def test_multimode_kps_encoder():
    setup_seed()

    # data

    data = torch.rand(BATCHSIZE, 3, NUM_FRAME * 2, 21) * 2 - 1

    kps_backbone = dict(
        type="VargNetV2",
        num_classes=71,
        input_channels=3,
        include_top=False,
        alpha=0.5,
        bn_kwargs=bn_kwargs,
    )

    kps_encoder_dict = dict(
        type="ActKPSEncoder",
        backbone=kps_backbone,
        bn_kwargs=bn_kwargs,
    )

    kps_encoder = build_from_registry(kps_encoder_dict)
    feature_kps = kps_encoder(data)

    assert feature_kps.shape == (BATCHSIZE, 128, 1, 1)


@pytest.mark.parametrize(
    "add_motion_in_rgb_branch, mode",
    [(True, "train"), (False, "train"), (True, "rgb"), (False, "rgb")],
)
def test_multimode_rgb_encoder(add_motion_in_rgb_branch, mode):
    setup_seed()

    # data
    data = (
        torch.rand(BATCHSIZE * NUM_FRAME, 3, 128, 128) * 2 - 1,
        torch.rand(BATCHSIZE * NUM_FRAME, 32, 1, 1) * 2 - 1,
    )

    rgb_backbone = dict(
        type="VargNetV2",
        num_classes=71,
        input_channels=3,
        include_top=False,
        alpha=0.5,
        bn_kwargs=bn_kwargs,
    )

    rgb_encoder_dict = dict(
        type="ActRGBEncoder",
        backbone=rgb_backbone,
        img_seq_len=8,
        bn_kwargs=bn_kwargs,
        add_motion_in_rgb_branch=add_motion_in_rgb_branch,
        disable_quanti_input=True,
        mode=mode,
    )

    # -- build model
    rgb_encoder = build_from_registry(rgb_encoder_dict)

    feature_frames = rgb_encoder(data)
    if mode == "train":
        if add_motion_in_rgb_branch:
            assert feature_frames.shape == (BATCHSIZE, 160, 8, 1)
        else:
            assert feature_frames.shape == (BATCHSIZE, 128, 8, 1)
    elif mode == "rgb":
        if add_motion_in_rgb_branch:
            assert feature_frames.shape == (BATCHSIZE * NUM_FRAME, 160, 1, 1)
        else:
            assert feature_frames.shape == (BATCHSIZE * NUM_FRAME, 128, 1, 1)
    else:
        raise ValueError(mode)


@pytest.mark.parametrize(
    "add_motion_in_rgb_branch, mode",
    [
        (True, "train"),
        (False, "train"),
        (True, "kps"),
        (False, "kps"),
        (True, "head"),
        (False, "head"),
    ],
)
def test_multimode_head(add_motion_in_rgb_branch, mode):
    setup_seed()

    # data
    if mode == "train":
        if add_motion_in_rgb_branch:
            data = (
                torch.rand(BATCHSIZE, 128, 1, 1) * 2 - 1,
                torch.rand(BATCHSIZE, 160, 8, 1) * 2 - 1,
            )
        else:
            data = (
                torch.rand(BATCHSIZE, 128, 1, 1) * 2 - 1,
                torch.rand(BATCHSIZE, 128, 8, 1) * 2 - 1,
            )
    elif mode == "kps":
        data = (torch.rand(BATCHSIZE, 128, 1, 1) * 2 - 1,)
    else:
        if add_motion_in_rgb_branch:
            data = (torch.rand(BATCHSIZE, 160, 8, 1) * 2 - 1,)
        else:
            data = (torch.rand(BATCHSIZE, 128, 8, 1) * 2 - 1,)

    in_channels_dict = {
        "kps": 512,
        "frames": 512,
    }

    head_dict = dict(
        type="StaMultiModalityHead",
        fusion_type="result_fusion",
        use_dropout=False,
        in_channels_dict=in_channels_dict,
        num_classes=NUM_CLASS,
        flat_output=True,
        bn_kwargs=bn_kwargs,
        add_motion_in_rgb_branch=add_motion_in_rgb_branch,
        mode=mode,
    )

    # -- build model
    head = build_from_registry(head_dict)

    logit_output = head(data)
    if mode == "train":
        assert len(logit_output) == 2
        assert logit_output[0].shape == (BATCHSIZE, NUM_CLASS)
        assert logit_output[1].shape == (BATCHSIZE, NUM_CLASS)
    elif mode == "kps" or mode == "head":
        assert len(logit_output) == 1
        assert logit_output[0].shape == (BATCHSIZE, NUM_CLASS)
    else:
        raise ValueError(mode)
