import pytest
import torch
from torchvision.transforms import Compose

from hat.data.transforms.detection import Normalize, ToTensor
from hat.data.transforms.gaze import Clip
from hat.registry import build_from_registry

transforms = Compose(
    [
        Clip(),
        ToTensor(),
        Normalize(mean=128.0, std=128.0),
    ]
)


@pytest.mark.parametrize("", [()])
def test_sample_model_dataset():
    config = dict(
        type="SampleModelDataset",
        image_size=(512, 330),
        grid_size=(320, 150),
        num=10,
        transforms=transforms,
    )
    sample_model_dataset = build_from_registry(config)
    item = sample_model_dataset[0]
    assert "img" in item
    assert isinstance(item["img"], torch.Tensor)
    assert "grid" in item
    assert isinstance(item["grid"], torch.Tensor)
    assert "label" in item


if __name__ == "__main__":
    pytest.main(["-s", __file__])
