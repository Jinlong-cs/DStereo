import pytest
import torch

from hat.models.base_modules.attention import (
    HorizonMultiheadAttention as MultiheadAttention,
)
from hat.models.base_modules.attention import MultiScaleDeformableAttention4Dim
from tests.unit_tests.models.base import (
    qat_test_with_multi_inputs,
    qtensor_test,
)


def generate_fake_inputs(embed_dim):
    q = torch.randn(2, embed_dim, 10, 10)
    k = torch.randn(2, embed_dim, 20, 20)
    v = torch.randn(2, embed_dim, 20, 20)
    key_padding_mask = torch.randint(
        0, 2, size=(2, 20 * 20), device=k.device
    ).to(torch.bool)
    attn_mask = torch.randint(
        0, 2, size=(10 * 10, 20 * 20), device=k.device
    ).to(torch.bool)

    return q, k, v, key_padding_mask, attn_mask


@pytest.mark.parametrize(
    [
        "embed_dim",
        "num_heads",
        "dropout",
        "bias",
    ],
    [
        pytest.param(16, 8, 0.0, True),
        pytest.param(24, 8, 0.1, False),
    ],
)
def test_multiheadattention(
    embed_dim,
    num_heads,
    dropout,
    bias,
):
    attn_module = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias,
    )

    q, k, v, key_padding_mask, attn_mask = generate_fake_inputs(embed_dim)

    output, _ = attn_module(q, k, v, key_padding_mask, attn_mask)
    assert isinstance(output, torch.Tensor) and q.shape == output.shape

    qat_inputs = qtensor_test((q, k, v))
    qat_test_with_multi_inputs(attn_module, qat_inputs, with_quantized=True)
    qat_inputs += [key_padding_mask]
    qat_test_with_multi_inputs(attn_module, qat_inputs, with_quantized=True)
    qat_inputs += [attn_mask]
    qat_test_with_multi_inputs(attn_module, qat_inputs, with_quantized=True)


def gen_data():

    query = torch.randn(2, 256, 15, 25)
    reference_points = torch.randn(2, 2, 15, 25)
    values = [torch.randn(2, 256, 15, 25)]

    return query, reference_points, values


def test_bevformer():
    attn_module = MultiScaleDeformableAttention4Dim(256, 1, 8, 4)
    data = gen_data()
    outputs = attn_module(*data)
    assert outputs.shape == (2, 256, 15, 25)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
