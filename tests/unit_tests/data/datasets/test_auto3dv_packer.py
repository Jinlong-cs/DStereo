import os
import shutil
import warnings

import lmdb
import numpy as np
import pytest
import timeout_decorator

from hat.data.datasets.auto_3dv_packer import (
    FileList2LMDB,
    Json2LMDB,
    RecLst2LMDB,
)
from hat.data.datasets.img_dataset import NamedIndexDatasetV2
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


@timeout_decorator.timeout(60)
def _copy_file(src, dst):
    shutil.copyfile(src, dst)


@pytest.mark.skipif(True, reason="cost time")
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_filelist2lmdb(tmpdir):
    bucket_path = HAT_BUCKET_PATH

    file_name = "train_AutoZGC_v5_bev_3frames.txt"
    file_list_path = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/tools/split_mengyuan/AutoZGC_v5/",
        file_name,
    )
    tmp_path = os.path.join(tmpdir, file_name)
    try:
        _copy_file(file_list_path, tmp_path)
    except timeout_decorator.TimeoutError:
        warnings.warn("Copy from bucket reach timeout.")
        return

    target_data_dir = os.path.join(tmpdir, "lmdb")
    packer = FileList2LMDB(tmp_path, target_data_dir)
    packer()

    # check consistency
    env = lmdb.open(
        target_data_dir,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    txn = env.begin(write=False)
    one_item = txn.get(u"{}".format(0).encode("ascii")).decode()

    with open(tmp_path, "r") as f:
        lines = f.read().splitlines()
    one_item_ori = lines[0]

    assert one_item == one_item_ori


@pytest.mark.skipif(True, reason="cost time")
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_fileDict2lmdb(tmpdir):
    bucket_path = HAT_BUCKET_PATH
    dataset_root = os.path.join(bucket_path, "unit_test_data/J5FSD/AutoZGC_v5")
    target_data_dir = os.path.join(tmpdir, "DG201_bev3d_test_lmdb")

    json_list_path = os.path.join(
        bucket_path,
        "unit_test_data/J5FSD/users/xiangyu.li/AutoZGC_v5/bev3d/pack_list/DG201_test_pack_list.txt",  # noqa
    )
    with open(json_list_path, "r") as f:
        lines = f.read().splitlines()
        json_file_list = [os.path.join(dataset_root, _line) for _line in lines]
    packer = Json2LMDB(json_file_list, target_data_dir)
    packer()


@pytest.mark.skipif(True, reason="cost time")
@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def test_recLst2lmdb(tmpdir):
    bucket_path = HAT_BUCKET_PATH
    lst_path = os.path.join(
        bucket_path, "unit_test_data/J5FSD/users/wenming.meng/recLst2lmdb"
    )  # noqa
    target_data_dir = os.path.join(
        tmpdir, os.path.basename(lst_path).replace(".lst", ".lst.lmdb")
    )

    packer = RecLst2LMDB(lst_path, target_data_dir)
    packer()
