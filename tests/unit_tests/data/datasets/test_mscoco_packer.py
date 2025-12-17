import os
import shutil
import warnings

import pytest
from timeout_decorator import timeout

from hat.data.datasets.mscoco import CocoDetectionPacker

try:
    import pycocotools
except ImportError:
    pycocotools = None


@timeout(seconds=60)
def copy_from_gpfs(target_dir):
    src = "tmp_orig_data/mscoco"
    shutil.copytree(
        os.path.join(src, "annotations"),
        os.path.join(target_dir, "annotations"),
    )
    shutil.copyfile(
        os.path.join(src, "val2017.zip"),
        os.path.join(target_dir, "val2017.zip"),
    )


@pytest.mark.skipif(pycocotools is None, reason="need pycocotools")
def test_mscoco_packer(tmpdir):
    tmp_dir = os.path.join(tmpdir, "mscoco")
    try:
        copy_from_gpfs(tmp_dir)
    except Exception:
        warnings.warn("Time out!! Please check GPFS access.")
        return

    os.system(
        "unzip {}/val2017.zip -d \
         {} > /dev/null".format(
            tmp_dir, tmp_dir
        )
    )
    packer = CocoDetectionPacker(
        tmp_dir, tmp_dir, "val", 1, "lmdb", num_samples=50
    )
    packer()
    assert os.path.exists(os.path.join(tmp_dir, "data.mdb"))
