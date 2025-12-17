import os

import numpy as np
import pytest

from hat.data.datasets.img_dataset import ImgLmdbDataset, NamedIndexDatasetV2
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_named_index_dataset_v2():
    bucket_path = HAT_BUCKET_PATH
    rec_path = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/wenming.meng/NamedIndexDatasetV2/pipeline_test_front_img.rec",  # noqa
    )
    dataset = NamedIndexDatasetV2(
        imgrec_path_list=[rec_path], read_by_name=True
    )

    img = dataset[
        "DG201_20210401_D/20210401-150149_735/camera_front/1617260548614.jpg"
    ]
    assert len(dataset.lst_lmdb) == 1
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_img_lmdb_dataset():
    bucket_path = HAT_BUCKET_PATH
    lmdb_path = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/wenming.meng/ImgLmdbDataset/test_lmdb",  # noqa
    )
    dataset = ImgLmdbDataset(lmdb_path)
    assert len(dataset) == 460

    img = dataset[
        "Site_117_09212_40_14846_2/UT1R3_20220508_132755/camera_rear_right/1651987699900.jpg"  # noqa
    ]
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
