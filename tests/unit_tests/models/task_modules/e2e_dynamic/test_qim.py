import pytest
import torch

from hat.models.task_modules.e2e_dynamic.qim import (
    QuerySpatialInteractionModule,
)


@pytest.mark.parametrize(
    ["query_embed", "output_embedding", "active_mask"],
    [
        pytest.param(
            torch.randn(1, 1, 60, 512),
            torch.randn(1, 1, 60, 256),
            torch.zeros(1, 1, 60, 1),
            id="active_mask_all_zero",
        ),
        pytest.param(
            torch.randn(1, 1, 60, 512),
            torch.randn(1, 1, 60, 256),
            torch.ones(1, 1, 60, 1),
            id="active_mask_all_one",
        ),
        pytest.param(
            torch.randn(1, 1, 60, 512),
            torch.randn(1, 1, 60, 256),
            None,
            id="active_mask_is_None",
        ),
    ],
)
def test_qim(query_embed, output_embedding, active_mask):
    qim_module = QuerySpatialInteractionModule(
        update_query_pos=True,
        dropout_ratio=0.1,
        dim_in=256,
        hidden_dim=512,
        dim_out=512,
        replace_identity_with_tgt=True,
    )
    out = qim_module(query_embed, output_embedding, active_mask)
    assert out.shape == query_embed.shape
