import torch

from hat.models.task_modules.lidar import LidarCameraFusionModule


def test_LidarCameraFusionModule():

    lidar_feat_channels = 48
    camera_feat_channels = 48
    out_channel = 128

    model = LidarCameraFusionModule(
        lidar_feat_channels=lidar_feat_channels,
        camera_feat_channels=camera_feat_channels,
        out_channel=out_channel,
    )

    lidar_feats = torch.randn((1, lidar_feat_channels, 128, 128))
    cam_feats = torch.randn((1, camera_feat_channels, 128, 128))
    output = model(cam_feats, lidar_feats)
    assert output.shape == (1, out_channel, 128, 128)
