import numpy as np
import pytest
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import LayerNorm as LayerNorm2d

from hat.models.base_modules.basic_horizon_swin_module import (
    BasicLayer4d,
    MlpModule2d,
    PatchEmbedding4d,
    PatchMerging4d,
)


@pytest.mark.parametrize(
    ["dim", "depth", "num_heads", "window_size", "downsample"],
    [
        pytest.param(96, 2, 3, 7, PatchMerging4d),
        pytest.param(192, 2, 6, 7, None),
    ],
)
def test_basic_horizon_swinblock(
    dim, depth, num_heads, window_size, downsample
):
    data_shape = [40, dim, 40, 50]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    hsw_module = BasicLayer4d(
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        downsample=downsample,
    )
    output = hsw_module(fake_data)
    assert output is not None


@pytest.mark.parametrize(
    ["patch_size", "in_channels", "embedding_dims", "norm_layer"],
    [
        pytest.param(4, 3, 96, LayerNorm2d),
        pytest.param(4, 3, 196, None),
    ],
)
def test_horizon_patchembedding(
    patch_size, in_channels, embedding_dims, norm_layer
):
    data_shape = [1, in_channels, 32, 32]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    hpe_module = PatchEmbedding4d(
        patch_size=patch_size,
        in_channels=in_channels,
        embedding_dims=embedding_dims,
        norm_layer=norm_layer,
    )
    output = hpe_module(fake_data)
    assert output is not None


@pytest.mark.parametrize(
    ["act_layer", "drop_ratio"],
    [
        pytest.param(nn.GELU(), 0.1),
        pytest.param(None, 0.0),
    ],
)
def test_horizon_mlp_module2d(act_layer, drop_ratio):
    data_shape = np.random.randint(10, 20, size=4)
    fake_data = torch.tensor(
        np.random.random(size=data_shape) * 2 - 1,
        dtype=torch.float,
    )
    mlp_module = MlpModule2d(
        in_channels=fake_data.shape[1],
        act_layer=act_layer,
        drop_ratio=drop_ratio,
    )
    output = mlp_module(fake_data)
    assert output is not None
