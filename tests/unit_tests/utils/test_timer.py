import os

import lmdb
import pytest

from hat.utils.timer import BytesTimer
from tests import root


def test_with_rec():
    mx = pytest.importorskip("mxnet")
    timer = BytesTimer(per_iters=0)
    path = os.path.join(root, "tests/data/sample.rec")
    reader = mx.recordio.MXRecordIO(path, "r")
    data = timer.timeit(reader.read)()
    reader.close()
    assert isinstance(data, bytes)


def test_with_lmdb():
    timer = BytesTimer(per_iters=0)
    path = os.path.join(root, "tmp_data/imagenet/train_lmdb")
    env = lmdb.open(
        path, readonly=True, lock=False, readahead=False, meminit=False
    )
    txn = env.begin(write=False)
    data = timer.timeit(txn.get)("0".encode("ascii"))
    env.close()
    assert isinstance(data, bytes)


def test_with_file():
    timer = BytesTimer(per_iters=0)
    path = os.path.join(root, "tests/data/lena.jpg")
    with open(path, "rb") as fb:
        data = timer.timeit(fb.read)()
        assert isinstance(data, bytes)
