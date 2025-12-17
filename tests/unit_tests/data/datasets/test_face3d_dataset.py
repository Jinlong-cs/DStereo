import numpy as np
import pytest
import torch
import torchvision

from hat.data.datasets.face3d_dataset import Face3dDataset
from hat.data.transforms.detection import ToTensor
from hat.data.transforms.face3d import RandomRotateCrop

INPUT_SIZE = (128, 128)
TARGET_SIZE = (256, 256)
DATA_PATH = ["./tmp_orig_data/face/face3d/hat_test/face3d_pp_data/image_lmdb"]
MASK_PATH = ["./tmp_orig_data/face/face3d/hat_test/face3d_pp_data/mask_lmdb"]
ANNO_PATH = ["./tmp_orig_data/face/face3d/hat_test/face3d_pp_data/anno_lmdb"]
TRANSFORMS = torchvision.transforms.Compose(
    [
        RandomRotateCrop(
            net_input_size=INPUT_SIZE,
            rot_prob=0.0,
            rot_angle_range=0.0,
            center_shift_prob=0.0,
            center_shift_range=0.0,
            norm_ratio=1.2,
            norm_method="longside_square",
            norm_jitter_range=0.0,
            net_target_size=TARGET_SIZE,
            base_len=200,
        ),
        ToTensor(),
    ]
)


@pytest.mark.parametrize(
    ["stage"],
    [pytest.param("pretrain"), pytest.param("finetune")],
)
def test_face3d_dataset(stage):
    """Test face3d dataset using lmdb image, mask and anno."""

    dataset = Face3dDataset(DATA_PATH, MASK_PATH, ANNO_PATH, TRANSFORMS, stage)
    item = dataset[0]
    assert isinstance(item, dict)
    assert "img" in item.keys()
    assert "gt_bboxes" in item.keys()
    assert "gt_ldmk" in item.keys()
    assert isinstance(item["img"], torch.Tensor)
    assert item["img"].shape == (3, INPUT_SIZE[1], INPUT_SIZE[0])
    if stage == "finetune":
        assert "gt_img" in item.keys()
        assert "gt_mask" in item.keys()
        assert isinstance(item["gt_img"], np.ndarray)
        assert isinstance(item["gt_mask"], np.ndarray)
        assert item["gt_img"].shape == (3, TARGET_SIZE[1], TARGET_SIZE[0])
        assert item["gt_mask"].shape == (1, TARGET_SIZE[1], TARGET_SIZE[0])


PP_DATA_PATH = [
    "./tmp_orig_data/face/face3d/hat_test/face3d_pp_data/image_lmdb"
]
PP_MASK_PATH = [
    "./tmp_orig_data/face/face3d/hat_test/face3d_pp_data/mask_lmdb"
]
PP_ANNO_PATH = [
    "./tmp_orig_data/face/face3d/hat_test/face3d_pp_data/anno_lmdb"
]


def test_face3d_pp_dataset():
    """Test face3d dataset for perspective projection."""

    dataset = Face3dDataset(
        PP_DATA_PATH, PP_MASK_PATH, PP_ANNO_PATH, modeltype="pp"
    )
    item = dataset[0]
    assert isinstance(item, dict)
    assert "img" in item.keys()
    assert "intrinsic" in item.keys()
    assert "distortion" in item.keys()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
