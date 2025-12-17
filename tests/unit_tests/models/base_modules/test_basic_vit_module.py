import numpy as np
import torch
import torch.nn as nn

from hat.models.base_modules.basic_vit_module import (
    PatchEmbed,
    ResPostBlock,
    ViTBlock,
)


def test_PatchEmbed():
    data_shape = [2, 3, 56, 56]
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    module = PatchEmbed(
        img_size=56,
        patch_size=8,
        in_chans=3,
        embed_dim=12,
    )
    output = module(fake_data)
    assert output.shape == (2, 49, 12)


def test_basic_vitblock():
    data_shape = [2, 50, 32]  # B, N, C
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    module = ViTBlock(
        dim=32,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.1,
        attn_drop=0.1,
        init_values=0.01,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    )
    output = module(fake_data)
    assert output.shape == (2, 50, 32)


def test_ResPostBlock():
    data_shape = [2, 50, 32]  # B, N, C
    fake_data = torch.tensor(
        np.random.random(size=data_shape),
        dtype=torch.float,
    )
    module = ResPostBlock(
        dim=32,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.1,
        attn_drop=0.1,
        init_values=0.01,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    )
    output = module(fake_data)
    assert output.shape == (2, 50, 32)
