import os
import shutil

import pytest

from hat.data.datasets.flyingchairs_dataset import FlyingChairsPacker


def _copy_flyingchairs_data(target_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    dst_zip = os.path.join(target_dir, "FlyingChairs.zip")
    shutil.copy(
        "./tmp_orig_data/FlyingChairs/FlyingChairs.zip",
        dst_zip,
    ),

    shutil.copy(
        "./tmp_orig_data/FlyingChairs/FlyingChairs_train_val.txt",
        os.path.join(target_dir, "FlyingChairs_train_val.txt"),
    ),
    os.system(f"unzip {dst_zip} -d {target_dir}")


@pytest.mark.skip("The testcast unzip orig dataset, very slow.")
def test_flyingchairs_packer(tmpdir):
    tmp_dir = os.path.join(tmpdir, "FlyingChairs")
    _copy_flyingchairs_data(tmp_dir)

    packer = FlyingChairsPacker(
        tmp_dir, tmp_dir, "val", 1, "lmdb", num_samples=50
    )
    packer()
    assert os.path.exists(os.path.join(tmp_dir, "data.mdb"))


if __name__ == "__main__":
    pytest.main(["-s", __file__])
