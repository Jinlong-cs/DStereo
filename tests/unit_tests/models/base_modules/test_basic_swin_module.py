import numpy as np
import pytest
import torch
import torch.nn as nn

from hat.models.base_modules.basic_swin_module import (
    BasicLayer,
    PatchEmbedding,
    PatchMerging,
)


@pytest.mark.parametrize(
    ["dim", "depth", "num_heads", "window_size", "downsample"],
    [
        pytest.param(96, 2, 3, 7, PatchMerging),
        pytest.param(96, 2, 3, 7, None),
    ],
)
def test_basic_swinblock(dim, depth, num_heads, window_size, downsample):
    data_shape = [40, 40 * 50, dim]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    bsw_module = BasicLayer(
        fake_data.shape[-1],
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        downsample=downsample,
    )
    output = bsw_module(fake_data, 40, 50)
    assert output is not None


@pytest.mark.parametrize(
    ["patch_size", "in_channels", "embedding_dims", "norm_layer"],
    [
        pytest.param(4, 3, 96, nn.LayerNorm),
        pytest.param(4, 3, 96, None),
    ],
)
def test_patchembedding(patch_size, in_channels, embedding_dims, norm_layer):
    data_shape = [1, in_channels, 32, 32]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    bpe_module = PatchEmbedding(
        patch_size=patch_size,
        in_channels=in_channels,
        embedding_dims=embedding_dims,
        norm_layer=norm_layer,
    )
    output = bpe_module(fake_data)
    assert output is not None
