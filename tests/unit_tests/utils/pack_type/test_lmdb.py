import os

import msgpack
import msgpack_numpy
import numpy as np
import pytest

from hat.utils.pack_type import Lmdb, LmdbReadList
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.fixture(
    params=[
        [5, 2, True],
        [5, 2],
        [4, 2],
        [3, 1],
        [5, 2],
        [4, 2],
        [3, 1],
    ]
)
def tmp_params(request, tmpdir):
    num_sample = request.param[0]
    commit_step = request.param[1]
    fixed_read_data = request.param[2] if len(request.param) == 3 else False
    tmp_file = Lmdb(
        os.path.join(tmpdir, "./tmp_lmdb"),
        commit_step=commit_step,
        fixed_read_data=fixed_read_data,
    )
    return tmp_file, num_sample, commit_step, fixed_read_data


def test_lmdb(tmp_params):
    tmp_file, num_sample, commit_step, fixed_read_data = tmp_params
    x = np.random.rand(2)

    tmp_file.open()
    for i in range(num_sample):
        tmp_file.write(
            i, msgpack.packb({"data": x}, default=msgpack_numpy.encode)
        )

    tmp_file.close()

    tmp_file.open()
    data = tmp_file.read(0)
    tmp = msgpack.unpackb(data, object_hook=msgpack_numpy.decode)

    assert len(tmp_file.get_keys()) == num_sample
    assert np.all(tmp["data"] == x)
    if fixed_read_data:
        for i in range(num_sample - 1):
            assert tmp_file.read(i) == tmp_file.read(i + 1)


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
@pytest.mark.parametrize(
    "fixed_read_data",
    [False, True],
)
def test_lmdb_read_from_list(fixed_read_data):
    bucket_path = HAT_BUCKET_PATH
    test_keys = [
        "65U3D_20211115_D/20211115-084331_862/1636937012400",
        "4GD36_20210629_D/20210629-095128_267/1624931489300",
    ]
    lmdb_file = [
        "unit_test_data/J5FSD/users/xiangyu.li/lmdb_test/lmdb_test_4GD36",  # noqa
        "unit_test_data/J5FSD/users/xiangyu.li/lmdb_test/lmdb_test_65U3D",  # noqa
    ]
    lmdb_paths = [os.path.join(bucket_path, path) for path in lmdb_file]
    lmdb_reader = LmdbReadList(
        lmdb_paths,
        readonly=True,
        map_size=1024 ** 2 * 10,
        fixed_read_data=fixed_read_data,
    )
    for key in test_keys:
        assert lmdb_reader.read(key)
    if fixed_read_data:
        assert lmdb_reader.read(test_keys[0]) == lmdb_reader.read(test_keys[1])


if __name__ == "__main__":
    pytest.main(["-s", __file__])
