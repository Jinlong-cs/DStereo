# Copyright (c) Horizon Robotics. All rights reserved.

import pytest
import torch

from hat.models.embeddings import (
    PositionalEmbedding,
    PositionEmbeddingLearned,
    PositionEmbeddingSine,
    SinePositionalEncoding3D,
)
from tests.unit_tests.models.base import qat_test, qtensor_test


def generate_fake_inputs(input_size, out_stride):
    pe_feat_shape = (
        1,
        3,
        input_size[1] // out_stride,
        input_size[0] // out_stride,
    )
    coordinate_map = torch.randn(pe_feat_shape, dtype=torch.float32)

    return coordinate_map


@pytest.mark.parametrize(
    [
        "pe_channel",
        "is_with_relu",
        "out_stride",
    ],
    [
        pytest.param(3, True, 4),
        pytest.param(3, True, 8),
        pytest.param(3, False, 4),
        pytest.param(3, False, 8),
    ],
)
def test_positionalembedding(
    pe_channel,
    is_with_relu,
    out_stride,
):
    pos_embedding = PositionalEmbedding(
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
        pe_channel=pe_channel,
        is_with_relu=is_with_relu,
    )

    coordinate_map = generate_fake_inputs(
        input_size=(1920 // 2, 1280 // 2),
        out_stride=out_stride,
    )

    output = pos_embedding(coordinate_map)
    assert (
        isinstance(output, torch.Tensor)
        and coordinate_map.shape == output.shape
    )

    coordinate_map = qtensor_test(coordinate_map)
    qat_test(pos_embedding, coordinate_map, with_quantized=True)


def test_position_embedding_sine():
    shape = (1, 10, 25)
    hidden_dim = 64
    mask = torch.zeros(shape, dtype=torch.bool)
    pos_module = PositionEmbeddingSine(hidden_dim // 2)
    pos_embs = pos_module(mask)
    assert pos_embs.shape[1] == hidden_dim
    assert pos_embs.shape[-2:] == mask.shape[-2:]


def test_position_embedding_learned():
    num_pos_feats = 128
    mask = torch.randint(0, 2, size=(2, 8, 8))
    module_learned = PositionEmbeddingLearned(num_pos_feats=num_pos_feats)
    pos_learned = module_learned(mask)

    assert pos_learned.shape[0] == mask.shape[0]
    assert pos_learned.shape[1] == num_pos_feats * 2
    assert pos_learned.shape[2] == mask.shape[1]
    assert pos_learned.shape[3] == mask.shape[2]


def test_position_embedding_sine_3d():
    shape = (1, 6, 10, 25)
    hidden_dim = 32
    mask = torch.zeros(shape, dtype=torch.bool)
    pos_module = SinePositionalEncoding3D(hidden_dim)
    pos_embs = pos_module(mask)
    assert pos_embs.shape[1] == 6
    assert pos_embs.shape[2] == hidden_dim * 3
    assert pos_embs.shape[-2:] == mask.shape[-2:]
