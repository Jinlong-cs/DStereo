import pytest
import torch

from hat.models.backbones.efficientnet import (
    EfficientNet,
    efficientnet,
    efficientnet_lite,
)
from tests.unit_tests.models.base import qat_test


@pytest.mark.parametrize(
    ["cls", "model_type", "use_se_block", "activation", "num_classes"],
    [  # b0 no swish, no attention, relu6
        # pytest.param("efficientnet", "b0", False, "relu6", 1000),
        # b0 no swish, no attention, relu
        pytest.param(efficientnet, "b0", False, "relu", 2),
        # use swish, use attention
        # pytest.param("efficientnet", "b0", False, "relu6", 1000),
        # b0 no swish, no attention, relu
        pytest.param(efficientnet, "b0", False, "relu", 2),
        # b1 use swish, use attention
        pytest.param(efficientnet, "b1", True, "swish", 2),
        # EfficientNet-Lite ReLU
        pytest.param(efficientnet_lite, "lite0", False, "relu", 2),
        # EfficientNet-Lite ReLU6
        # pytest.param("efficientnet_lite", "lite1", False, "relu6", 1000),
    ],
)
def test_efficientnet(cls, model_type, use_se_block, activation, num_classes):
    model = cls(
        model_type=model_type,
        use_se_block=use_se_block,
        activation=activation,
        num_classes=num_classes,
        bn_kwargs={},
    )
    x = torch.randn((2, 3, 224, 224))
    y = model(x)
    assert isinstance(y, torch.Tensor)

    backbone = cls(
        model_type=model_type,
        use_se_block=use_se_block,
        activation=activation,
        include_top=False,
        num_classes=num_classes,
        bn_kwargs={},
    )
    outputs = backbone(x)
    assert len(outputs) == 5
    assert outputs[0].shape == (2, 16, 112, 112)
    assert outputs[1].shape == (2, 24, 56, 56)
    assert outputs[2].shape == (2, 40, 28, 28)
    assert outputs[3].shape == (2, 112, 14, 14)
    assert outputs[4].shape == (2, 320, 7, 7)

    qat_test(model, x)


@pytest.mark.parametrize(
    ["model_type", "use_se_block", "activation", "num_classes"],
    [  # b0 no swish, no attention, relu6
        # pytest.param("efficientnet", "b0", False, "relu6", 1000),
        # b0 no swish, no attention, relu
        pytest.param("b0", False, "relu", 2),
        # use swish, use attention
        # pytest.param("efficientnet", "b0", False, "relu6", 1000),
        # b0 no swish, no attention, relu
        pytest.param("b0", False, "relu", 2),
        # b1 use swish, use attention
        pytest.param("b1", True, "swish", 2),
        # EfficientNet-Lite ReLU
        pytest.param("lite0", False, "relu", 2),
        # EfficientNet-Lite ReLU6
        # pytest.param("efficientnet_lite", "lite1", False, "relu6", 1000),
    ],
)
def test_EfficientNet(model_type, use_se_block, activation, num_classes):
    _coefficient_params = {
        "lite0": (1.0, 1.0, 224, 0.2),
        "lite1": (1.0, 1.1, 240, 0.2),
        "lite2": (1.1, 1.2, 260, 0.3),
        "lite3": (1.2, 1.4, 280, 0.3),
        "lite4": (1.4, 1.8, 300, 0.3),
        "b0tiny": (0.5, 0.5, 224, 0.2),
        "b0small": (0.5, 1.0, 224, 0.2),
        "b0": (1.0, 1.0, 224, 0.2),
        "b1": (1.0, 1.1, 240, 0.2),
        "b2": (1.1, 1.2, 260, 0.3),
        "b3": (1.2, 1.4, 300, 0.3),
        "b4": (1.4, 1.8, 380, 0.4),
        "b5": (1.6, 2.2, 456, 0.4),
        "b6": (1.8, 2.6, 528, 0.5),
        "b7": (2.0, 3.1, 600, 0.5),
    }
    model = EfficientNet(
        activation=activation,
        bn_kwargs={},
        coefficient_params=_coefficient_params[model_type],
        flat_output=False,
        include_top=True,
        model_type=model_type,
        num_classes=num_classes,
        use_se_block=use_se_block,
    )

    x = torch.randn((2, 3, 224, 224))
    y = model(x)
    assert isinstance(y, torch.Tensor)

    backbone = EfficientNet(
        activation=activation,
        bn_kwargs={},
        coefficient_params=_coefficient_params[model_type],
        flat_output=False,
        include_top=False,
        model_type=model_type,
        num_classes=num_classes,
        use_se_block=use_se_block,
    )
    outputs = backbone(x)
    assert len(outputs) == 5
    assert outputs[0].shape == (2, 16, 112, 112)
    assert outputs[1].shape == (2, 24, 56, 56)
    assert outputs[2].shape == (2, 40, 28, 28)
    assert outputs[3].shape == (2, 112, 14, 14)
    assert outputs[4].shape == (2, 320, 7, 7)

    qat_test(model, x)
