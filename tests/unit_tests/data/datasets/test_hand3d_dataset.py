import numpy as np
import pytest
import torch

from hat.data.datasets.hand3d_lmdb_dataset import Hand3dLmdbSingleDataset
from hat.data.transforms.detection import ToTensor
from hat.data.transforms.face3d import (
    SimpleNormGenGridMap,
    SimpleNormPositionEncoding,
)

try:
    import kornia
except ImportError:
    kornia = None


INPUT_IMAGE_SIZE = 128
VIRTUAL_CAMERA_FOCAL = 600
VIRTUAL_IMAGE_HW = [1080, 1920, 3]
VIRTUAL_CROP_SIZE = 480
VIRTUAL_NORM_RATIO = 1.2

virtual_intrinsic = np.diag(
    [VIRTUAL_CAMERA_FOCAL, VIRTUAL_CAMERA_FOCAL, 1]
).astype(np.float32)
virtual_intrinsic[:2, -1] = [
    VIRTUAL_IMAGE_HW[1] // 2,
    VIRTUAL_IMAGE_HW[0] // 2,
]

lmdb_list = [
    [
        "./tmp_orig_data/hand3d/datahub/hat_test/FerihandIndexV1b_training/image_lmdb/",  # noqa
        "./tmp_orig_data/hand3d/datahub/hat_test/FerihandIndexV1b_training/anno_lmdb/",  # noqa
        None,
        None,
        1,
    ],
]


@pytest.mark.skipif(kornia is None, reason="need kornia")
def test_face3d_dataset():
    """Test face3d dataset using lmdb image, mask and anno."""

    dataset = Hand3dLmdbSingleDataset(
        image_path=lmdb_list[0][0],
        anno_path=lmdb_list[0][1],
        mask_path=lmdb_list[0][2],
        depth_path=lmdb_list[0][3],
        set_name="training",
        enable_inshape_uniform=True,
        virtual_img_shape=VIRTUAL_IMAGE_HW,
        transforms=[
            SimpleNormGenGridMap(
                net_input_size=INPUT_IMAGE_SIZE,
                norm_ratio=VIRTUAL_NORM_RATIO,
                expand_crop_hw=VIRTUAL_CROP_SIZE,
                norm_method="longside_square",
                virtual_intrinsic=virtual_intrinsic,
            ),
            SimpleNormPositionEncoding(
                net_input_size=INPUT_IMAGE_SIZE,
            ),
            ToTensor(to_yuv=False),
        ],
    )
    item = dataset[0]
    assert isinstance(item, dict)
    assert "img" in item.keys()
    assert "gt_bboxes" in item.keys()
    assert "gt_ldmk" in item.keys()
    assert "gt_ldmk3d" in item.keys()
    assert "grid_map" in item.keys()
    assert "position_map" in item.keys()
    assert isinstance(item["img"], torch.Tensor)
    assert item["img"].shape == (3, VIRTUAL_CROP_SIZE, VIRTUAL_CROP_SIZE)
    assert item["grid_map"].shape == (2, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    assert item["position_map"].shape == (
        2,
        INPUT_IMAGE_SIZE,
        INPUT_IMAGE_SIZE,
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
