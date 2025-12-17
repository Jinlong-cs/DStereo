import numpy as np
import pytest
import torch
from torch import nn

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test

BATCHSIZE = 2
NUM_FRAME = 8
NUM_CLASS = 59


def setup_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


@pytest.mark.parametrize(
    ["fusion_type", "add_motion_in_rgb_branch", "enable_frames_temporal_gap"],
    [
        ("result_fusion", True, False),
        ("result_fusion", True, True),
        ("result_fusion", False, False),
        ("feature_fusion", True, False),
        ("feature_fusion", True, True),
        ("feature_fusion", False, False),
    ],
)
def test_multimode_classifier(
    fusion_type, add_motion_in_rgb_branch, enable_frames_temporal_gap
):
    setup_seed()

    # data
    data = dict(
        frames=torch.rand(BATCHSIZE * NUM_FRAME, 3, 128, 128) * 2 - 1,
        box_center=torch.rand(BATCHSIZE * NUM_FRAME, 32, 1, 1) * 2 - 1,
        clip_keypoints=torch.rand(BATCHSIZE, 3, NUM_FRAME * 2, 21) * 2 - 1,
        label_weight=torch.randint(low=0, high=2, size=(BATCHSIZE,)),
        act_label=torch.randint(low=-1, high=NUM_CLASS, size=(BATCHSIZE,)),
    )
    # -- build model
    # backbone params
    alpha = 0.5
    channels_frames = 128
    if add_motion_in_rgb_branch:
        channels_frames = 160
    proj_channels_frames = [[channels_frames, 512], [512, 512]]
    if not enable_frames_temporal_gap:
        proj_channels_frames.append([512, 512])
        assert NUM_FRAME == 2 ** len(proj_channels_frames)
    else:
        assert NUM_FRAME == 2 ** (len(proj_channels_frames) + 1)

    proj_channels = [[[128, 512]], proj_channels_frames]

    # head params
    in_channels_dict = {
        "kps": proj_channels[0][-1][-1],
        "frames": proj_channels[1][-1][-1],
    }

    # loss params
    if fusion_type == "result_fusion":
        loss_kps = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        loss_frames = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        loss_kps_frames = None
        loss_weight = [1, 1]
    elif fusion_type == "feature_fusion":
        loss_kps = None
        loss_frames = None
        loss_kps_frames = nn.CrossEntropyLoss(
            ignore_index=-1, reduction="none"
        )
        loss_weight = [1]

    # config
    config = dict(
        type="GestMultiModalityClassifier",
        backbone=dict(
            type="ActMultiModalityEncoder",
            network_kps=dict(
                type="VargNetV2",
                num_classes=1000,
                include_top=False,
                alpha=alpha,
                bn_kwargs={},
            ),
            network_frames=dict(
                type="VargNetV2",
                num_classes=1000,
                include_top=False,
                alpha=alpha,
                bn_kwargs={},
            ),
            img_seq_len=NUM_FRAME,
            proj_channels=proj_channels,
            add_motion_in_rgb_branch=add_motion_in_rgb_branch,
            enable_frames_temporal_gap=enable_frames_temporal_gap,
        ),
        head=dict(
            type="ActMultiModalityHead",
            num_classes=NUM_CLASS,
            in_channels_dict=in_channels_dict,
            fusion_type=fusion_type,
        ),
        loss=dict(
            type="GestMultiModeLoss",
            fusion_type=fusion_type,
            loss_kps=loss_kps,
            loss_frames=loss_frames,
            loss_kps_frames=loss_kps_frames,
            loss_weight=loss_weight,
        ),
    )
    model = build_from_registry(config)

    logit_output, losses = model(data)
    for preds in logit_output:
        if preds is not None:
            assert preds.shape == (BATCHSIZE, NUM_CLASS)
    qat_test(model, data, with_quantized=False)
