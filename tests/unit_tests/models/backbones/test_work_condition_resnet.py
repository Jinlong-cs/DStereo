import pytest
import torch

from hat.registry import build_from_registry


@pytest.mark.parametrize(
    [
        "layers",
        "output_dim",
        "heads",
        "width",
        "use_attnpool",
    ],
    [
        pytest.param((3, 4, 6, 3), 1024, 32, 64, True),
        pytest.param((3, 4, 6, 3), 1024, 32, 64, False),
    ],
)
def test_modified_resnet(layers, output_dim, heads, width, use_attnpool):
    cfg = dict(
        type="WorkConditionResNet",
        layers=layers,
        output_dim=output_dim,
        heads=heads,
        channel_list=[width // 2, width, width * 2, width * 4, width * 8],
        bn_kwargs={},
        use_attnpool=use_attnpool,
    )
    model = build_from_registry(cfg)
    input_size = 224
    x = torch.randn((1, 3, input_size, input_size))
    y = model(x)

    if not use_attnpool:
        assert isinstance(y, list) and len(y) == 5
    else:
        assert isinstance(y, list) and len(y) == 6
        assert len(y[5].shape) == 2

    assert y[0].shape[-1] == input_size / 2
    assert y[1].shape[-1] == input_size / 4
    assert y[2].shape[-1] == input_size / 8
    assert y[3].shape[-1] == input_size / 16
    assert y[4].shape[-1] == input_size / 32
