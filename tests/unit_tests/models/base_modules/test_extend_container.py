# Copyright (c) Horizon Robotics. All rights reserved.

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY, build_from_registry
from tests.data.toy_modules import *  # noqa: F403,F401


@OBJECT_REGISTRY.register
class ToyMultiInputModule(torch.nn.Module):
    def __init__(self):
        super(ToyMultiInputModule, self).__init__()

    def forward(self, *args):
        return args


def test_ext_sequential():
    cfg = dict(
        type="ExtSequential",
        modules=[
            build_from_registry(
                dict(type="ToyBackbone", strides=(1, 2), channels=(3, 8))
            ),
            dict(
                type="ToyHead",
                in_channels=8,
                fc_filter=16,
                num_classes=10,
                with_dequant=True,
            ),
        ],
    )

    model = build_from_registry(cfg)

    pred = model(torch.randn(1, 3, 14, 14))
    assert pred.ndim == 2
    assert pred.shape == (1, 10)
    assert hasattr(model, "fuse_model")
    assert hasattr(model, "set_qconfig")


def test_nest_ext_sequential():
    cfg = dict(
        type="ExtSequential",
        modules=[
            dict(
                type="ExtSequential",
                modules=[
                    build_from_registry(
                        dict(
                            type="ToyBackbone", strides=(1, 2), channels=(3, 8)
                        )
                    )
                ],
            ),
            dict(
                type="ToyHead",
                in_channels=8,
                fc_filter=16,
                num_classes=10,
                with_dequant=True,
            ),
        ],
    )

    model = build_from_registry(cfg)

    pred = model(torch.randn(1, 3, 14, 14))
    assert pred.ndim == 2
    assert pred.shape == (1, 10)


def _check_multi_input_sequential(model: nn.Module, num_inputs: int):
    inputs = [torch.randn(2, 2) for i in range(num_inputs)]
    outs = model(*inputs)
    assert len(inputs) == len(outs)
    for i, o in zip(inputs, outs):
        assert i is o


def test_multi_input_sequential():
    cfg = dict(
        type="MultiInputSequential",
        modules=[
            dict(type="ToyMultiInputModule"),
            build_from_registry(dict(type="ToyMultiInputModule")),
        ],
    )
    model = build_from_registry(cfg)
    assert hasattr(model, "fuse_model")
    assert hasattr(model, "set_qconfig")

    # one input
    _check_multi_input_sequential(model, num_inputs=1)

    # two inputs
    _check_multi_input_sequential(model, num_inputs=2)


def test_nest_multi_input_sequential():
    cfg = dict(
        type="MultiInputSequential",
        modules=[
            dict(
                type="MultiInputSequential",
                modules=[
                    dict(type="ToyMultiInputModule"),
                    build_from_registry(dict(type="ToyMultiInputModule")),
                ],
            ),
            dict(type="ToyMultiInputModule"),
        ],
    )
    model = build_from_registry(cfg)

    # one input
    _check_multi_input_sequential(model, num_inputs=1)

    # two inputs
    _check_multi_input_sequential(model, num_inputs=2)
