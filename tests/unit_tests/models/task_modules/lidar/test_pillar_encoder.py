import pytest
import torch

from hat.models.task_modules.lidar import PillarFeatureNet, PointPillarScatter


@pytest.mark.parametrize(
    ["use_4dim", "use_conv", "horizon_preprocess", "normalize_xyz"],
    [
        pytest.param(
            True,
            True,
            True,
            False,
        ),
        pytest.param(
            False,
            False,
            False,
            True,
        ),
    ],
)
def test_pillarfeaturenet(
    use_4dim, use_conv, horizon_preprocess, normalize_xyz
):
    pillar_feature_net = PillarFeatureNet(
        num_filters=[2],
        num_input_features=4,
        with_distance=False,
        voxel_size=(0.3, 0.3, 6),
        pc_range=(-76.8, -76.8, -2, 76.8, 76.8, 4),
        bn_kwargs=dict(eps=1e-3, momentum=0.01),
        use_4dim=use_4dim,
        use_conv=use_conv,
        normalize_xyz=normalize_xyz,
    )
    if horizon_preprocess:
        input_features = torch.randn(1, 4, 2, 1)
    else:
        input_features = torch.randn(2, 1, 4)
    input_num_voxels = torch.randn(2)
    input_coors = torch.randn(2, 4)

    output = pillar_feature_net(
        input_features,
        input_num_voxels,
        input_coors,
        horizon_preprocess,
    )
    if use_conv:
        assert len(output.shape) == 4
    else:
        assert len(output.shape) == 2
    assert output.shape[-2] == 2
    assert output.shape[-1] == 2


def test_pointpillarscatter():
    scatter_module = PointPillarScatter(num_input_features=1)

    voxel_features = torch.randn(2, 1)
    coords = torch.randn(2, 4)
    coords[:2, :] = 0
    batch_size = 1
    input_shape = [2, 2, 1]

    output = scatter_module(voxel_features, coords, batch_size, input_shape)

    assert output.size() == (1, 1, 2, 2)
