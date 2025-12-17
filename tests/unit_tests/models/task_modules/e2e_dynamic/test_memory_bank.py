import pytest
import torch

from hat.models.task_modules.e2e_dynamic.memory_bank import MemoryBankModule


@pytest.mark.parametrize(
    [
        "query_pos",
        "output_embedding",
        "mem_bank",
        "mem_padding_mask",
        "odo_input",
        "fps_queue_input",
        "active_mask",
        "out_track_score",
    ],
    [
        pytest.param(
            torch.randn(1, 1, 300, 256),
            torch.randn(1, 1, 300, 256),
            torch.randn(1, 10, 300, 256),
            torch.zeros(1, 10, 300, 1),
            torch.randn(1, 11, 1, 4),
            torch.randn(1, 11, 1, 1),
            torch.ones(1, 1, 300, 1),
            True,
            id="out_track_score is True",
        ),
        pytest.param(
            torch.randn(1, 1, 300, 256),
            torch.randn(1, 1, 300, 256),
            torch.randn(1, 10, 300, 256),
            torch.zeros(1, 10, 300, 1),
            torch.randn(1, 11, 1, 4),
            torch.randn(1, 11, 1, 1),
            torch.ones(1, 1, 300, 1),
            False,
            id="out_track_score is False",
        ),
        pytest.param(
            torch.randn(1, 1, 300, 256),
            torch.randn(1, 1, 300, 256),
            torch.randn(1, 10, 300, 256),
            torch.zeros(1, 10, 300, 1),
            torch.randn(1, 11, 1, 4),
            torch.randn(1, 11, 1, 1),
            None,
            True,
            id="active_mask is None",
        ),
    ],
)
def test_memory_bank(
    query_pos,
    output_embedding,
    mem_bank,
    mem_padding_mask,
    odo_input,
    fps_queue_input,
    active_mask,
    out_track_score,
):
    mb_moduel = MemoryBankModule(
        memory_bank_len=10,
        memory_bank_with_self_attn=True,
        memory_bank_with_temp_attn=True,
        num_heads=8,
        dim_in=256,
        hidden_dim=512,
        dim_out=512,
        out_track_score=out_track_score,
    )
    out = mb_moduel(
        query_pos,
        output_embedding,
        mem_bank,
        mem_padding_mask,
        odo_input,
        fps_queue_input,
        active_mask,
    )
    assert len(out) == 3
    assert out[0].shape == output_embedding.shape
    assert out[0].shape == out[1].shape
    if out_track_score:
        assert isinstance(out[-1], torch.Tensor)
    else:
        assert out[-1] is None
