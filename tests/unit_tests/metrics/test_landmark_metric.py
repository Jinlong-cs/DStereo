import pytest
import torch

from hat.metrics.landmark import (
    DecodeHeatmap,
    DecodeVector,
    NormalizedMeanError,
)

NUM_LDMK = 68
DATA = {
    "gt_ldmk": torch.rand((8, NUM_LDMK, 2)),
    "pr_ldmk": torch.rand((8, NUM_LDMK, 2)),
    "pr_heatmap": torch.rand((8, NUM_LDMK, 32, 32)),
    "pr_vector_x": torch.rand((8, NUM_LDMK, 1, 32)),
    "pr_vector_y": torch.rand((8, NUM_LDMK, 32, 1)),
}


@pytest.mark.parametrize(
    ["norm_type", "mode", "decoding_method", "feat_stride"],
    [
        ["ION", "coords", "", ""],
        ["ION", "heatmap", "diff", 4],
        ["ION", "heatmap", "shift", 4],
        ["ION", "heatmap", "taylor", 4],
        ["ION", "vector", "diff", 4],
        ["ION", "vector", "shift", 4],
        ["ION", "vector", "taylor", 4],
        ["SIZE", "coords", "", ""],
        ["DIAG", "coords", "", ""],
    ],
)
def test_nme(norm_type, mode, decoding_method, feat_stride):
    metric = NormalizedMeanError(
        NUM_LDMK, "NME", norm_type, mode, decoding_method, feat_stride
    )
    metric.update(DATA)
    _, value = metric.get()
    assert value > 0


@pytest.mark.parametrize(
    ["decoding_method"], [["diff"], ["shift"], ["taylor"]]
)
def test_decode_heatmap(decoding_method):
    data = DATA["pr_heatmap"][0, 0]
    decode = DecodeHeatmap(decoding_method, 4)
    result = decode(data)
    assert isinstance(result, torch.Tensor)
    assert result.device == data.device


@pytest.mark.parametrize(
    ["decoding_method"], [["diff"], ["shift"], ["taylor"]]
)
def test_decode_vector(decoding_method):
    data = torch.rand((2, 32))
    decode = DecodeVector(decoding_method, 4)
    result = decode(data)
    assert isinstance(result, torch.Tensor)
    assert result.device == data.device
