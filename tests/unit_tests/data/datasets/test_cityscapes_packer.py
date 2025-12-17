import os
import shutil
import warnings

import pytest
import timeout_decorator

from hat.data.datasets.cityscapes import CityscapesPacker

try:
    import torchvision
except ImportError:
    torchvision = None


@timeout_decorator.timeout(60)
def _copy_cityscapes_data(target_dir):
    shutil.copytree(
        "tmp_orig_data/cityscapes",
        target_dir,
    )


@pytest.mark.serial_task
@pytest.mark.skipif(torchvision is None, reason="need torchvision")
def test_cityscapes_packer(tmpdir):
    tmp_dir = os.path.join(tmpdir, "cityscapes")
    try:
        _copy_cityscapes_data(tmp_dir)
    except timeout_decorator.TimeoutError:
        warnings.warn("Copy from bucket reach timeout.")
        return

    packer = CityscapesPacker(
        tmp_dir, tmp_dir, "val", 1, "lmdb", num_samples=50
    )
    packer()
    assert os.path.exists(os.path.join(tmp_dir, "data.mdb"))
