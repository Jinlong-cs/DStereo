from typing import Mapping, Sequence

import pytest
import torch

from hat.registry import build_from_registry


@pytest.mark.parametrize(
    [
        "label_dict",
        "new_shape",
    ],
    [
        pytest.param(
            {
                "target": {
                    "annimal": {"dog": torch.randn(1, 2, 540, 960)},
                    "plant": {"tree": torch.randn(1, 2, 540, 960)},
                },
            },
            {"annimal": (2, 1, 540, 960), "plant": (2, 540, 960)},
        ),
        pytest.param(
            {
                "target": [
                    [torch.randn(1, 2, 540, 960)],
                    [torch.randn(1, 2, 540, 960)],
                ]
            },
            [(2, 1, 540, 960), (2, 540, 960)],
        ),
    ],
)
def test_reshape_target(label_dict, new_shape):
    data_name = "target"
    config = dict(type="ReshapeTarget", data_name=data_name, shape=new_shape)
    reshape_target = build_from_registry(config)

    pred_dict = dict()
    label_dict = reshape_target(label_dict, pred_dict)
    if isinstance(label_dict[data_name], Mapping):
        for k, v in label_dict[data_name].items():
            for sub_v in v.values():
                assert sub_v.shape == new_shape[k]
    elif isinstance(label_dict[data_name], Sequence):
        for idx, instance in enumerate(label_dict[data_name]):
            print("instance", instance)
            for sub_ins in instance:
                assert sub_ins.shape == new_shape[idx]
