import pytest
import torch

from hat.models.task_modules.bev.temporal_utils import (
    ANCSpatialGRU,
    ANCTemporalSimpleFusion,
)


def test_spatial_gru():
    frame_num = 3
    input_dim, hidden_dim = 16, 16
    num_layers = 2
    h, w = 256, 256

    frames_input = [torch.randn(1, input_dim, h, w) for _ in range(frame_num)]
    spatial_gru = ANCSpatialGRU(
        input_dim, hidden_dim, 3, num_layers=num_layers, h_w=(h, w)
    )

    fusion_data, hidden_data = spatial_gru(frames_input)

    assert fusion_data.shape == (1, input_dim, h, w)
    assert len(hidden_data) == frame_num - 1


@pytest.mark.parametrize(
    ["fusion_method"],
    [
        pytest.param("add"),
        pytest.param("cat"),
    ],
)
def test_temporal_simple_fusion(fusion_method):
    frame_num = 3
    input_dim = 16
    h, w = 256, 256
    use_decay = True
    frames_input = [torch.randn(1, input_dim, h, w) for _ in range(frame_num)]

    fusion = ANCTemporalSimpleFusion(
        input_dim, frame_num, fusion_method, use_decay=use_decay
    )

    fused_data, dump_data = fusion(frames_input)

    assert fused_data.shape == dump_data.shape == (1, input_dim, h, w)
