import pytest
import torch

from hat.registry import build_from_registry

try:
    import mmcv
except ImportError:
    mmcv = None


@pytest.mark.skipif(not mmcv, reason="require mmcv")
@pytest.mark.parametrize(
    [
        "spec_name",
        "norm_eval",
        "frozen_stages",
        "input_ch",
        "out_features",
        "node_name",
    ],
    [
        pytest.param(
            "V-99-eSE",
            True,
            -1,
            3,
            (
                "stage2",
                "stage3",
                "stage4",
                "stage5",
            ),
            "img_backbone",
        ),
    ],
)
def test_vovnet(
    spec_name,
    norm_eval,
    frozen_stages,
    input_ch,
    out_features,
    node_name,
):
    cfg = dict(
        type="VoVNet",
        spec_name=spec_name,
        norm_eval=norm_eval,
        frozen_stages=frozen_stages,
        input_ch=input_ch,
        out_features=out_features,
        node_name=node_name,
    )
    model = build_from_registry(cfg)
    input_size = 960
    x = torch.randn((1, 3, input_size, input_size))
    y = model(x)
    assert isinstance(y, list) and len(y) == 4
    assert y[0].shape[-1] == input_size // 4
    assert y[1].shape[-1] == input_size // 8
    assert y[2].shape[-1] == input_size // 16
    assert y[3].shape[-1] == input_size // 32


if __name__ == "__main__":
    pytest.main(["-s", __file__])
