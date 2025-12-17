import torchvision

from hat.data.datasets.human3d_dataset import (
    Human3dDataset,
    Human3dMixedDataset,
)
from hat.data.transforms.detection import ToTensor

DATA_PATH = ["./tmp_orig_data/human3d/test_dataset/image_lmdb"]
ANNO_PATH = ["./tmp_orig_data/human3d/test_dataset/pseudo_anno2_lmdb"]
LDMK_PAIRS = [
    [0, 5],
    [1, 4],
    [2, 3],
    [6, 11],
    [7, 10],
    [8, 9],
    [20, 21],
    [22, 23],
]
SMPL_POSE_PAIRS = [
    [1, 2],
    [4, 5],
    [7, 8],
    [10, 11],
    [13, 14],
    [16, 17],
    [18, 19],
    [20, 21],
    [22, 23],
]
TRANSFORMS = torchvision.transforms.Compose(
    [
        ToTensor(),
    ]
)


def test_humna3d_dataset():
    dataset = Human3dDataset(
        "test_dataset",
        DATA_PATH[0],
        ANNO_PATH[0],
        LDMK_PAIRS,
        SMPL_POSE_PAIRS,
        transforms=TRANSFORMS,
    )
    item = dataset[0]
    assert "img" in item.keys()
    assert "gt_ldmk" in item.keys()


def test_mixhuman3d_dataset():
    dataset = Human3dMixedDataset(
        ["test_dataset"],
        DATA_PATH,
        ANNO_PATH,
        LDMK_PAIRS,
        SMPL_POSE_PAIRS,
        transforms=TRANSFORMS,
    )
    item = dataset[0]
    assert "img" in item.keys()
    assert "gt_ldmk" in item.keys()
