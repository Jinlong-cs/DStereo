import numpy as np
import pytest
import torch

from hat.models.base_modules.basic_timesformer_module import (
    PatchEmbed,
    SpaceTimeBlock,
)


@pytest.mark.parametrize(
    ["dim", "num_heads"],
    [
        pytest.param(768, 12),
    ],
)
def test_basic_timesformer_block(dim, num_heads):
    data_shape = [2, 14 * 14 * 8 + 1, dim]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    times_module = SpaceTimeBlock(
        dim=dim,
        num_heads=num_heads,
    )
    output = times_module(fake_data, 2, 8, 14)
    assert output is not None


@pytest.mark.parametrize(
    ["img_size", "patch_size", "in_chans", "embed_dim"],
    [
        pytest.param(224, 16, 3, 768),
    ],
)
def test_patchembed(img_size, patch_size, in_chans, embed_dim):
    data_shape = [1, 3, 8, img_size, img_size]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    pe_module = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
    )
    output = pe_module(fake_data)
    assert output is not None
