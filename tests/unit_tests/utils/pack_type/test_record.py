import os

import msgpack
import msgpack_numpy
import numpy as np
import pytest

from hat.utils.pack_type.mxrecord import MXRecord, MXRecordIO


def test_mxrecordio(tmpdir):
    tmp_file = os.path.join(tmpdir, "./tmp_mxrecordio")
    tmp_file_idx = os.path.join(tmpdir, "./tmp_mxrecordio" + ".idx")
    N = 255

    writer = MXRecordIO(tmp_file, "w")
    for i in range(N):
        writer.write(bytes(str(chr(i)), "utf-8"))
    del writer

    reader = MXRecordIO(tmp_file, "r")
    reader.create_idx_file(os.path.join(tmp_file_idx))
    idx = {}
    keys = []
    with open(tmp_file_idx, "r") as f:
        for line in iter(f.readline, ""):
            line = line.strip().split("\t")
            key = int(line[0])
            idx[key] = int(line[1])
            keys.append(key)
    assert len(keys) == N
    assert len(idx.keys()) == N


def test_mxrecord(tmpdir):
    tmp_file = MXRecord(os.path.join(tmpdir, "./tmp_mxrecord"))

    tmp_file.reset()
    x = np.random.rand(2)
    tmp_file.open()
    tmp_file.write(0, msgpack.packb({"data": x}, default=msgpack_numpy.encode))
    tmp_file.close()

    tmp_file = MXRecord(os.path.join(tmpdir, "./tmp_mxrecord"), writable=False)
    tmp_file.open()
    data = tmp_file.read(0)
    tmp = msgpack.unpackb(data, object_hook=msgpack_numpy.decode)
    assert np.all(tmp["data"] == x)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
