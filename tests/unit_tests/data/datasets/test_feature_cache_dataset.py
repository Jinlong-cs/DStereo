import os

import msgpack
import msgpack_numpy
import numpy as np

from hat.data.datasets.cache_dataset import CacheDataset
from hat.utils.pack_type.mxrecord import MXRecord


def test_feature_cache_dataset(tmpdir):
    # prepare test rec
    tmp_file_name = os.path.join(tmpdir, "./tmp_mxrecord")
    tmp_file = MXRecord(tmp_file_name)
    tmp_file.reset()
    x = np.random.rand(2)
    tmp_file.open()
    tmp_file.write(0, msgpack.packb({"data": x}, default=msgpack_numpy.encode))
    tmp_file.close()

    dataset = CacheDataset(
        data_path=tmp_file_name,
        input_name="data",
        wrapped_input_name="new_data",
    )
    assert len(dataset) == 1
    sample = dataset[0]
    np.testing.assert_allclose(sample["new_data"]["data"].numpy(), x)
