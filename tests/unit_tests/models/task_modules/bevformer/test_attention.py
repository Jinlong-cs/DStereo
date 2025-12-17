import pytest
import torch

from hat.registry import build_from_registry


def gen_horizontemporalselfattention_data():

    query = torch.randn((2, 2500, 256))
    value = torch.randn((4, 2500, 256))
    query_pos = torch.randn((2, 2500, 256))
    reference_points = torch.randn((4, 2500, 1, 2))
    spatial_shapes = torch.Tensor([[50, 50]]).to(torch.float32)

    return query, value, query_pos, reference_points, spatial_shapes


@pytest.mark.serial_task
def test_horizontemporalselfattention():
    config = dict(
        type="HorizonTemporalSelfAttention",
        embed_dims=256,
        num_levels=1,
        view_gird_in=1,
        view_gird_out=100,
        view_num=8,
        feats_size=[[50, 50]],
    )
    model = build_from_registry(config)
    data = gen_horizontemporalselfattention_data()
    outputs = model(*data)
    assert outputs.shape == (2, 2500, 256)


def gen_horizonspatialcrossattention_data():

    query = torch.randn((2, 2500, 256))
    key = torch.randn((6, 375, 2, 256))
    value = torch.randn((6, 375, 2, 256))
    query_pos = torch.randn((2, 2500, 256))
    reference_points_cam = torch.randn((6, 2, 2500, 4, 2))
    spatial_shapes = torch.Tensor([[25, 15]]).to(torch.float32)

    return query, key, value, reference_points_cam, spatial_shapes, query_pos


@pytest.mark.serial_task
def test_horizonspatialcrossattention():
    config = dict(
        type="HorizonSpatialCrossAttention",
        view_num=8,
        deformable_attention=dict(
            type="HorizonMSDeformableAttention3D",
            embed_dims=256,
            num_points=8,
            num_levels=1,
            view_gird_in=160,
            view_gird_out=100,
            feats_size=[[25, 15]],
        ),
        embed_dims=256,
    )
    model = build_from_registry(config)
    data = gen_horizonspatialcrossattention_data()
    outputs = model(*data)
    assert outputs.shape == (2, 2500, 256)


def gen_horizonmsdeformableattention3d_data():

    query = torch.randn((12, 2500, 256))
    value = torch.randn((12, 375, 256))
    reference_points = torch.randn((12, 2500, 4, 2))
    spatial_shapes = torch.Tensor([[25, 15]]).to(torch.float32)

    return query, value, reference_points, spatial_shapes


@pytest.mark.serial_task
def test_horizonmsdeformableattention3d():
    config = dict(
        type="HorizonMSDeformableAttention3D",
        embed_dims=256,
        num_points=8,
        num_levels=1,
        view_gird_in=160,
        view_gird_out=100,
        feats_size=[[25, 15]],
    )
    model = build_from_registry(config)
    data = gen_horizonmsdeformableattention3d_data()
    outputs = model(*data)
    assert outputs.shape == (12, 256, 2500)


def gen_horizonmsdeformableattention_data():

    query = torch.randn((900, 2, 256))
    query_pos = torch.randn((900, 2, 256))
    value = torch.randn((2500, 2, 256))
    reference_points = torch.randn((2, 900, 1, 2))
    spatial_shapes = torch.Tensor([[50, 50]]).to(torch.float32)

    return query, value, query_pos, reference_points, spatial_shapes


@pytest.mark.serial_task
def test_horizonmsdeformableattention():
    config = dict(
        type="HorizonMSDeformableAttention",
        embed_dims=256,
        num_levels=1,
        view_gird_out=6,
        view_gird_in=4,
        feats_size=[[50, 50]],
    )
    model = build_from_registry(config)
    data = gen_horizonmsdeformableattention_data()
    outputs = model(*data)
    assert outputs.shape == (900, 2, 256)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
