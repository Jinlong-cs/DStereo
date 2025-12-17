import os
import shutil
import warnings

from timeout_decorator import timeout

from hat.data.datasets.kitti2d import Kitti2DDetectionPacker


@timeout(seconds=60)
def copy_from_gpfs(dst):
    src = "tmp_orig_data/kitti2d"
    shutil.copy(
        os.path.join(src, "kitti_eval.json"),
        os.path.join(dst, "kitti_eval.json"),
    )
    dst_zip = os.path.join(dst, "eval.zip")
    shutil.copy(
        os.path.join(src, "eval.zip"),
        dst_zip,
    )
    os.system(f"unzip {dst_zip} -d {dst}")


def test_kitti2d_packer(tmpdir):
    tmp_dir = os.path.join(tmpdir, "kitti")
    try:
        copy_from_gpfs(tmp_dir)
    except Exception:
        warnings.warn("Time out!! Please check GPFS access.")
        return

    packer = Kitti2DDetectionPacker(
        os.path.join(tmp_dir, "eval"),
        tmp_dir,
        os.path.join(tmp_dir, "kitti_eval.json"),
        1,
        "lmdb",
        num_samples=50,
    )
    packer()
    assert os.path.exists(os.path.join(tmp_dir, "data.mdb"))
