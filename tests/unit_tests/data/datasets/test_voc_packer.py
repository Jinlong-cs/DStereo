import os
import shutil
import tarfile
import warnings

import pytest
from timeout_decorator import timeout

from hat.data.datasets.voc import VOCDetectionPacker
from hat.utils.package_helper import check_packages_available


@timeout(seconds=60)
def copy_from_gpfs(target_dir):
    shutil.copy("tmp_orig_data/voc/VOCtest_06-Nov-2007.tar", target_dir)


@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
def test_voc_packer(tmpdir):
    tmp_dir = os.path.join(tmpdir, "voc")
    os.makedirs(tmp_dir)

    try:
        copy_from_gpfs(tmp_dir)
    except Exception:
        warnings.warn("Time out!! Please check GPFS access.")
        return

    voc_file = os.path.join(tmp_dir, "VOCtest_06-Nov-2007.tar")
    with tarfile.open(voc_file, "r") as tar:
        tar.extractall(path=tmp_dir)

    packer = VOCDetectionPacker(
        tmp_dir, tmp_dir, "test", 1, "lmdb", num_samples=50
    )
    packer()
    assert os.path.exists(os.path.join(tmp_dir, "data.mdb"))
