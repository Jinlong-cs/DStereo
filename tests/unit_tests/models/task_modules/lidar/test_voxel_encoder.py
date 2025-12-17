import torch

from hat.models.task_modules.lidar.voxel_encoder import MeanVFE


def test_MeanVFE():
    num_input_feats = 4
    voxel_encoder = MeanVFE(num_input_features=num_input_feats)

    voxel_features = torch.randn([20000, 15, num_input_feats])
    num_voxels = torch.randint(1, 15, (20000,))

    output = voxel_encoder(voxel_features, num_voxels)

    assert output.size() == (20000, num_input_feats)
