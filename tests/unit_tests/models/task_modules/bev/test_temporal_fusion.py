import pytest
import torch

from hat.models.task_modules.bev.temporal_fusion import (
    ANCBEVFusionRecurrentTemporalModule,
    ANCBEVFusionTemporalModule,
)
from hat.models.task_modules.bev.temporal_utils import (
    ANCSpatialGRU,
    ANCTemporalSimpleFusion,
)


@pytest.mark.parametrize(
    ["temporal_module_type"],
    [
        pytest.param("simple"),
        pytest.param("gru"),
    ],
)
def test_bev_fusion_temporal(temporal_module_type):

    bs = 1
    in_channels = 16
    input_shape = (256, 256)
    temporal_length_each_batch = 3
    frame_num = temporal_length_each_batch

    frames_input = torch.randn(
        bs * temporal_length_each_batch,
        in_channels,
        input_shape[0],
        input_shape[1],
    )
    homography_temporal = torch.randn(
        bs, (temporal_length_each_batch - 1), 3, 3
    )

    fusion_module_dict = {
        "simple": ANCTemporalSimpleFusion(
            in_channels, frame_num, fusion_method="cat"
        ),
        "gru": ANCSpatialGRU(in_channels, in_channels, 3, 1, input_shape),
    }
    fusion_module = fusion_module_dict[temporal_module_type]

    fusion_temporal = ANCBEVFusionTemporalModule(
        input_shape,
        in_channels,
        fusion_module,
        temporal_length_each_batch=temporal_length_each_batch,
    )

    fused_data, dump_data = fusion_temporal(
        frames_input,
        homography_temporal,
    )

    assert (
        fused_data.shape
        == dump_data.shape
        == (bs, in_channels, input_shape[0], input_shape[1])
    )

    # compile_model
    fusion_temporal = ANCBEVFusionTemporalModule(
        input_shape,
        in_channels,
        fusion_module,
        temporal_length_each_batch=temporal_length_each_batch,
        compile_model=True,
    )

    frames_input = [
        torch.randn(bs, in_channels, input_shape[0], input_shape[1])
    ] * frame_num
    homo_offset_temporal = [
        torch.randn(bs, input_shape[0], input_shape[1], 2)
    ] * (frame_num - 1)

    fused_data, dump_data = fusion_temporal(
        frames_input, None, homo_offset_temporal
    )
    assert (
        fused_data.shape
        == dump_data.shape
        == (bs, in_channels, input_shape[0], input_shape[1])
    )


@pytest.mark.parametrize(
    [
        "temporal_module_type",
        "do_cache_feat",
        "init_zero_his_feat",
        "is_relative_homography",
    ],
    [
        pytest.param("simple", False, False, False),
        pytest.param("simple", True, False, False),
        pytest.param("simple", False, True, False),
        pytest.param("simple", False, False, True),
        pytest.param("gru", False, False, False),
        pytest.param("gru", True, False, False),
        pytest.param("gru", False, True, False),
        pytest.param("gru", False, False, True),
    ],
)
def test_bev_recurrent_fusion_temporal(
    temporal_module_type,
    do_cache_feat,
    init_zero_his_feat,
    is_relative_homography,
):
    bs = 1
    in_channels = 16
    input_shape = (256, 256)
    temporal_length_each_batch = 3

    frames_input = torch.randn(
        bs * temporal_length_each_batch,
        in_channels,
        input_shape[0],
        input_shape[1],
    )
    homography_temporal = torch.randn(bs, temporal_length_each_batch, 3, 3)

    fusion_module_dict = {
        "gru": ANCSpatialGRU(in_channels, in_channels, 3, 1, input_shape),
        "simple": ANCTemporalSimpleFusion(in_channels, 2, fusion_method="cat"),
    }
    fusion_module = fusion_module_dict[temporal_module_type]

    fusion_temporal = ANCBEVFusionRecurrentTemporalModule(
        input_shape=input_shape,
        in_channels=in_channels,
        fusion_module=fusion_module,
        temporal_length_each_batch=temporal_length_each_batch,
        do_cache_feat=do_cache_feat,
        init_zero_his_feat=init_zero_his_feat,
        is_relative_homography=is_relative_homography,
    )

    meta = {
        "homography_temporal": homography_temporal,
    }
    fused_data, dump_data = fusion_temporal(
        frames_input,
        meta,
    )

    assert (
        fused_data.shape[1:]
        == dump_data.shape[1:]
        == (in_channels, input_shape[0], input_shape[1])
    )
    assert len(fused_data) == 1

    # compile_model
    fusion_temporal = ANCBEVFusionRecurrentTemporalModule(
        input_shape=input_shape,
        in_channels=in_channels,
        fusion_module=fusion_module,
        temporal_length_each_batch=temporal_length_each_batch,
        do_cache_feat=do_cache_feat,
        init_zero_his_feat=init_zero_his_feat,
        is_relative_homography=is_relative_homography,
        compile_model=True,
    )

    frames_input = torch.randn(
        (
            1,
            in_channels,
            input_shape[0],
            input_shape[1],
        )
    )
    homo_offset_temporal = [
        torch.randn(bs, input_shape[0], input_shape[1], 2)
    ] * (2 - 1)

    pre_fusion_feat = torch.randn(
        (
            1,
            in_channels,
            input_shape[0],
            input_shape[1],
        )
    )
    meta = {
        "homo_offset_temporal": homo_offset_temporal,
        "pre_fusion_feat": pre_fusion_feat,
    }
    fused_data, dump_data = fusion_temporal(frames_input, meta)
    assert (
        fused_data.shape
        == dump_data.shape
        == (bs, in_channels, input_shape[0], input_shape[1])
    )
