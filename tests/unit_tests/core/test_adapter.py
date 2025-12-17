import pytest
import torch

from hat.core.adapter import TorchVisionAdapter

try:
    import torchvision
except ImportError:
    torchvision = None


@pytest.mark.skipif(torchvision is None, reason="need torchvision")
def test_torchvision_adaptor():
    data = dict(img=torch.randn(1, 3, 350, 350))
    trf = TorchVisionAdapter(
        interface=torchvision.transforms.RandomResizedCrop,
        size=224,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
    )

    res = trf(data)
    assert "img" in res
    assert res["img"].shape[2] == 224
    assert res["img"].shape[3] == 224

    trf = TorchVisionAdapter(
        interface="RandomResizedCrop",
        size=224,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
    )

    res = trf(data)
    assert "img" in res
    assert res["img"].shape[2] == 224
    assert res["img"].shape[3] == 224
