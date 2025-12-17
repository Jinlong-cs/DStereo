import torch

from hat.models.task_modules.planning.reward_encoder import (
    PatialResNet,
    RewardModel,
)


def test_patial_resnet():

    bs = 4
    img_h, img_w = 512, 512
    channels = 7
    stride = 4
    data_inputs = torch.randn(bs, channels, img_h, img_w)
    out_channel = 256

    backbone = PatialResNet(
        input_channels=7, bn_kwargs=dict(eps=1e-5, momentum=0.1)
    )

    outputs = backbone(data_inputs)

    assert outputs.shape == (bs, out_channel, img_h // stride, img_w // stride)


def test_reward_model():

    bs = 4
    b_channels = 256
    img_h, img_w = 512, 512
    feat_size = 128
    stride = 4
    grid_dim = 65
    encode_motion = True
    motion_size = 2

    model = RewardModel(
        in_feat_size=b_channels,
        scene_feat_size=feat_size,
        agg_size=[32, 32],
        grid_cell_size=[2, 2],
        encode_motion=encode_motion,
        motion_size=motion_size,
    )

    feats = torch.randn(bs, b_channels, img_h // stride, img_w // stride)
    motions = torch.randn(bs, motion_size, grid_dim, grid_dim)

    data = {"feats": feats, "plan_ego_motion": motions}

    output = model(data)

    assert "feats" in output
    assert "reward" in output
    assert output["feats"].shape == (
        bs,
        feat_size + motion_size,
        grid_dim,
        grid_dim,
    )
    assert output["reward"].shape == (bs, 1, grid_dim, grid_dim)
