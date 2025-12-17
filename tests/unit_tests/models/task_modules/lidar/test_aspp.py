import torch

from hat.models.task_modules.lidar.aspp import ASPPNeck


def test_ASPPNeck():
    lidar_neck = ASPPNeck(in_channels=256, block_nums=3, use_se=True)

    lidar_feats_input = torch.randn([1, 256, 128, 128])
    lidar_feats_output, _ = lidar_neck(lidar_feats_input)

    assert lidar_feats_output.size() == (1, 256, 128, 128)
