import os
import shutil
import warnings

import pytest
from timeout_decorator import timeout

from hat.data.datasets.imagenet import ImageNetPacker
from hat.utils.package_helper import check_packages_available


@timeout(seconds=100)
def copy_from_gpfs(target_dir):
    shutil.copytree(
        "tmp_orig_data/imagenet/val",
        target_dir,
    )


@pytest.mark.serial_task
@pytest.mark.skipif(
    not check_packages_available("torchvision", raise_exception=False),
    reason="need torchvision",
)
def test_imagenet_packer(tmpdir):
    tmp_dir = os.path.join(tmpdir, "imagenet")
    try:
        copy_from_gpfs(tmp_dir)
    except Exception:
        warnings.warn("Time out!! Please check GPFS access.")
        return

    packer = ImageNetPacker(tmp_dir, tmp_dir, "val", 1, "lmdb", num_samples=50)
    packer()
    assert os.path.exists(os.path.join(tmp_dir, "data.mdb"))

    packer = ImageNetPacker(
        tmp_dir,
        os.path.join(tmp_dir, "val_mxrecord"),
        "val",
        1,
        "mxrecord",
        num_samples=50,
    )
    packer()
    assert os.path.exists(os.path.join(tmp_dir, "val_mxrecord.rec"))
    idx_path = os.path.join(tmp_dir, "val_mxrecord.idx")
    assert os.path.exists(idx_path)
    assert len(open(idx_path, "r").readlines()) == 50
