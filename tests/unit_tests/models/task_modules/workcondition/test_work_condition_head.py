import math

import torch

from hat.registry import build_from_registry


def test_workcondition_cls_output():
    num_classes = 6
    input_size = (234, 456)
    neck = dict(
        type="WorkConditionClsHead",
        output_dim=1024,
        bn_kwargs={},
        num_classes=num_classes,
        in_channel=2048,
    )

    workcondition_attn_cls_output = build_from_registry(neck)
    x = [
        torch.randn(
            (
                1,
                2048,
                math.ceil(input_size[0] / 32),
                math.ceil(input_size[1] / 32),
            )
        ),
    ]
    y = workcondition_attn_cls_output(x)
    assert len(y.flatten(1).shape) == 2
    assert y.flatten(1).shape[1] == num_classes
