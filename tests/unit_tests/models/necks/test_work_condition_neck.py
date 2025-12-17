import torch

from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test, qtensor_test


def test_classification_neck():
    neck = dict(
        type="WorkConditionNeck",
        in_channel=128,
        o_channel=256,
        end_feature_idx=5,
        num_units=2,
        group_base=8,
        bn_kwargs={},
    )

    input_size = 512
    classification_neck = build_from_registry(neck)
    x = [
        torch.randn((1, 16, input_size // 2, input_size // 2)),
        torch.randn((1, 16, input_size // 4, input_size // 4)),
        torch.randn((1, 32, input_size // 8, input_size // 8)),
        torch.randn((1, 64, input_size // 16, input_size // 16)),
        torch.randn((1, 128, input_size // 32, input_size // 32)),
    ]
    y = classification_neck(x)
    assert len(y) == 6

    x = qtensor_test(x)
    qat_test(classification_neck, x, with_quantized=False)
