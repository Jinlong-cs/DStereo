import horizon_plugin_pytorch as horizon
import numpy as np
import pytest
import torch
import torch.nn as nn

from hat.models.base_modules.mlp_module import FFN, MLP, MlpModule2d


@pytest.mark.parametrize(
    ["in_channels", "act_layer", "drop_ratio"],
    [
        pytest.param(96, nn.GELU(), 0.0),
        pytest.param(96, nn.GELU(), 0.1),
    ],
)
def test_mlp_module2d(in_channels, act_layer, drop_ratio):
    hw = np.random.randint(10, 20, size=2)
    fake_data = torch.tensor(
        np.random.random(size=(4, hw[0] * hw[1], in_channels)),
        dtype=torch.float,
    )
    mlp_module = MlpModule2d(
        in_channels=in_channels,
        act_layer=act_layer,
        drop_ratio=drop_ratio,
    )
    output = mlp_module(fake_data)
    assert output is not None


@pytest.mark.parametrize(
    ["input_dim", "hidden_dim", "output_dim", "num_layers"],
    [
        [8, 16, 16, 3],
        [8, 16, 16, 2],
    ],
)
def test_MLP(input_dim, hidden_dim, output_dim, num_layers):
    fake_data = torch.tensor(
        np.random.random(size=(4, 20, input_dim)),
        dtype=torch.float,
    )

    mlp_module = MLP(input_dim, hidden_dim, output_dim, num_layers)
    mlp_module.eval()
    output = mlp_module(fake_data)
    mlp_module = horizon.quantization.prepare_qat_fx(mlp_module)
    output_fx = mlp_module(fake_data)
    assert output is not None
    assert output_fx is not None
    assert (output == output_fx).all()


@pytest.mark.parametrize(
    ["add_identity"],
    [
        [True],
        [False],
    ],
)
def test_FFN(add_identity):
    embed_dim = 16
    fake_data = torch.tensor(
        np.random.random(size=(4, 20, embed_dim)),
        dtype=torch.float,
    )
    ffn_module = FFN(
        embed_dim=embed_dim,
        feedforward_dim=32,
        output_dim=embed_dim,
        num_fcs=2,
        activation=None,
        ffn_drop=0.1,
        fc_bias=True,
        add_identity=add_identity,
    )
    ffn_module.eval()
    if add_identity:
        output = ffn_module(fake_data, fake_data)
    else:
        output = ffn_module(fake_data)
    assert output is not None
    ffn_module.eval()
    ffn_module_fx = horizon.quantization.prepare_qat_fx(ffn_module)

    if add_identity:
        output_fx = ffn_module_fx(fake_data, fake_data)
    else:
        output_fx = ffn_module_fx(fake_data)
    assert output is not None
    assert output_fx is not None
    assert (output == output_fx).all()
