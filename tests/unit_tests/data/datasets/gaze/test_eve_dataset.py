import pytest
import torch
import torchvision

from hat.data.datasets.gaze.eve_dataset import EVEDataset
from hat.data.transforms.detection import ToTensor

DATA_PATH = "./tmp_orig_data/face/gaze_eve/val"
TRANSFORMS = torchvision.transforms.Compose(
    [
        ToTensor(),
    ]
)


@pytest.mark.parametrize("", [()])
def test_eve_dataset():
    """Test eve dataset."""
    dataset = EVEDataset(
        lmdb_path=DATA_PATH,
        transforms=TRANSFORMS,
    )
    item = dataset[0][0]
    assert isinstance(item, dict)
    assert "img" in item.keys()
    assert isinstance(item["img"], torch.Tensor)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
