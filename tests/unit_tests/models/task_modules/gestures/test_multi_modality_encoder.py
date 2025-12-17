import numpy as np
import pytest
import torch

from hat.registry import build_from_registry

BATCHSIZE = 2
NUM_FRAME = 8


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


@pytest.mark.parametrize(
    "add_motion_in_rgb_branch, enable_frames_temporal_gap",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_multimode_encoder(
    add_motion_in_rgb_branch, enable_frames_temporal_gap
):
    setup_seed(2022)

    frames = torch.rand(BATCHSIZE * NUM_FRAME, 3, 128, 128) * 2 - 1
    motion_input = torch.rand(BATCHSIZE * NUM_FRAME, 32, 1, 1) * 2 - 1
    clip_keypoints = torch.rand(BATCHSIZE, 3, 16, 21) * 2 - 1
    data = dict(
        frames=frames,
        box_center=motion_input,
        clip_keypoints=clip_keypoints,
    )

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
    config = dict(
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
    )

    act_multimode_encoder = build_from_registry(config)
    feature_kps, feature_frames = act_multimode_encoder(data)

    assert feature_kps.shape == (BATCHSIZE, proj_channels[0][-1][-1], 1, 1)
    assert feature_frames.shape == (BATCHSIZE, proj_channels[1][-1][-1], 1, 1)
