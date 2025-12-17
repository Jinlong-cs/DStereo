import os

import pytest
import torchvision

from hat.data.datasets.smoke_kps_dataset import SmokeKpsRecDataset
from hat.data.transforms.detection import ToTensor
from hat.data.transforms.landmark import (
    GenerateGaussianHeatmap,
    GenerateGaussianVector,
)

try:
    import mxnet as mx
except ImportError:
    mx = None

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root_dir = os.path.join(
    bucket_root, "HDLTAlgorithm/data/orig_data/action/smoke_kps"
)
filename = "{}/keypoint_rawdata_0005_210906_train_ir_sorted.rec".format(
    root_dir
)


transforms = torchvision.transforms.Compose(
    [
        GenerateGaussianVector(
            num_ldmk=4,
            feat_stride=4,
            vector_size=(32, 32),
            sigma=2,
            encoding_method="standard",
        ),
        GenerateGaussianHeatmap(
            num_ldmk=4,
            feat_stride=4,
            heatmap_shape=(32, 32),
            sigma=2,
            encoding_method="ellipse",
        ),
        ToTensor(),
    ]
)


@pytest.mark.parametrize(
    ["filename"],
    [
        pytest.param(filename),
    ],
)
@pytest.mark.skipif(
    not mx or not os.path.exists(root_dir), reason="data path doesn't exists."
)
def test_smoke_kps_dataset(filename):
    dataset = SmokeKpsRecDataset(
        filename=filename,
        transforms=transforms,
    )
    item = dataset[0]
    assert "img" in item
    assert "gt_ldmk" in item
