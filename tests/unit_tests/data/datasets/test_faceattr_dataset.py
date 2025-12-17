# import torch
# import torchvision

from hat.data.datasets.faceattr_dataset import FaceAttrRecDataset

FILENAME = ["./tmp_orig_data/face/age_gender/agegender_unitest.rec"]


def test_faceattr_rec_dataset():
    dataset = FaceAttrRecDataset(
        imgrec_path_list=FILENAME,
    )
    item = dataset[0]
    assert "img" in item
    assert "age" in item
    assert "gender" in item
