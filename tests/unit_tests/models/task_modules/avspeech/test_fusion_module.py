# Copyright (c) Horizon Robotics. All rights reserved.
import torch

from hat.models.task_modules.avspeech.avspeech_fusion import (
    AddFusion,
    CatFusion,
    SimpleMixBlock,
)


def test_add_fusion():
    x1 = torch.rand(3, 16, 112, 112)
    x2 = torch.rand(3, 16, 112, 112)
    fusion = AddFusion(
        bn_kwargs={},
        v_in_channels=16,
        a_in_channels=16,
        a_v_out_channels=16,
        fusion_out_channels=32,
    )
    out = fusion(x1, x2)
    assert out.shape == (3, 32, 112, 112)


def test_cat_fusion():
    x1 = torch.rand(3, 16, 112, 112)
    x2 = torch.rand(3, 16, 112, 112)
    fusion = CatFusion(idim=16, odim=16)
    out = fusion(x1, x2)
    assert out.shape == (3, 16, 112, 112)


def test_simple_mix_block():
    x1 = torch.rand(3, 16, 112, 112)
    x2 = torch.rand(3, 16, 112, 112)
    fusion = SimpleMixBlock(mode="add")
    out = fusion(x1, x2)
    assert out.shape == (3, 16, 112, 112)

    fusion = SimpleMixBlock(mode="cat", dim=1)
    out = fusion(x1, x2)
    assert out.shape == (3, 32, 112, 112)
