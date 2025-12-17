import os

import pytest

from projects.halo.cv.tools.human3d.human3d_packer import Human3dDatasetPacker


@pytest.mark.parametrize(
    "src_type",
    ["image", "anno"],
)
def test_human3d_packer(src_type, tmpdir):
    tmp_dir = "./tmp_orig_data/human3d/CD569_nonproduct_4ways_trainset_03"
    save_dir = os.path.join(tmpdir, "lmdb")
    pack_type = "lmdb"
    pack_path = os.path.join(
        save_dir,
        "%s_%s" % (src_type, pack_type),
    )

    packer = Human3dDatasetPacker(
        anno_path=os.path.join(tmp_dir, "data_crop.json"),
        src_type="image",
        save_dir=pack_path,
        img_dir=os.path.join(tmp_dir, "data"),
    )
    packer()
    assert os.path.exists(
        os.path.join(tmp_dir, "lmdb", src_type + "_lmdb", "data.mdb")
    )
