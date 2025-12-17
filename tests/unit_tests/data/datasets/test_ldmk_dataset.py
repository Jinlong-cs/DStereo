import torch
import torchvision

from hat.data.datasets.landmark_dataset import LdmkDataset
from hat.data.transforms.detection import ToTensor
from hat.data.transforms.landmark import CropRecROI


def test_landmark_rec_dataset():
    transforms = torchvision.transforms.Compose(
        [
            CropRecROI(
                crop_type="random",
                target_shape=(128, 128, 3),
                base_roi=[48, 48, 208, 208],
                crop_jitter_range=0.1,
                center_shift_range=0.0,
                random_type="gaussian",
            ),
            ToTensor(),
        ]
    )
    filename = "tmp_orig_data/landmark/hat_test/ldmk_toy_data/face_ldmk.rec"
    dataset = LdmkDataset(
        rec_list=filename,
        data_type="rec",
        num_ldmk=68,
        use_3d=False,
        task_type="face",
        transforms=transforms,
    )
    item = dataset[0]
    assert "img" in item
    assert isinstance(item["img"], torch.Tensor)
    assert "gt_ldmk" in item
    del dataset


def test_landmark_lmdb_dataset():
    prefix = "tmp_orig_data/landmark/hat_test/hand_lmdb/01_data_shujutang_v2"
    dataset = LdmkDataset(
        image_lmdb_list=f"{prefix}/image_lmdb",
        anno_lmdb_list=f"{prefix}/anno_lmdb",
        num_ldmk=21,
        task_type="hand",
        data_type="lmdb",
    )
    item = dataset[0]
    assert "img" in item
    assert "gt_ldmk" in item
    assert "gt_bboxes" in item
    assert "gt_ldmk_weight" in item
