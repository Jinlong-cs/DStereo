import os
import shutil

import pytest

from hat.data.datasets.nuscenes_dataset import NuscenesPacker


def _copy_nuscenes_data(target_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    dst_tar = os.path.join(target_dir, "v1.0-mini.tar")
    shutil.copy(
        "./tmp_orig_data/nuscenes/v1.0-mini.tar",
        dst_tar,
    )
    shutil.copytree(
        "./tmp_orig_data/nuscenes/maps",
        os.path.join(target_dir, "maps"),
    )
    shutil.copytree(
        "./tmp_orig_data/nuscenes/can_bus",
        os.path.join(target_dir, "can_bus"),
    )
    shutil.copytree(
        "./tmp_orig_data/nuscenes/v1.0-mini",
        os.path.join(target_dir, "v1.0-mini"),
    )

    os.system(f"tar xvf {dst_tar} -C {target_dir}")


@pytest.mark.skip("The testcast unzip orig dataset, very slow.")
@pytest.mark.parametrize(
    "only_lidar",
    [(True), (False)],
)
def test_nuscenes_packer(tmpdir, only_lidar):
    tmpdir = "."
    tmp_dir = os.path.join(tmpdir, "nuscenes")
    _copy_nuscenes_data(tmp_dir)

    packer = NuscenesPacker(
        version="v1.0-mini",
        src_data_dir=tmp_dir,
        target_data_dir=tmp_dir,
        split_name="val",
        pack_type="lmdb",
        num_workers=50,
        only_lidar=only_lidar,
    )
    packer()
    assert os.path.exists(os.path.join(tmp_dir, "data.mdb"))


if __name__ == "__main__":
    pytest.main(["-s", __file__])
