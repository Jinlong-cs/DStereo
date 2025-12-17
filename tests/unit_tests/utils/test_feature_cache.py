import os

import numpy as np
import pytest
import torch

from hat.utils.cache import Cache, CacheIOType, get_global_cache


def test_cache_memory():
    cache = Cache(method=CacheIOType.Memory)

    cache_data = {"a": 1, "b": 2, "c": [1, 2, 3]}
    cache.write("1", cache_data)
    cache.write("2", cache_data)
    cache.write("3", cache_data)
    assert len(cache) == 3
    assert not cache.is_empty()
    assert "1" in cache
    assert "11" not in cache
    assert cache.read("1") == cache_data
    cache.clear()
    assert cache.is_empty()

    cache_data = {"a": torch.randn((4, 3, 224, 224))}
    cache.write_with_index(cache_data)
    assert len(cache) == 4


def test_cache_rec(tmpdir):
    cache_file = os.path.join(tmpdir, "tmp_cache.rec")

    cache = Cache(
        method=CacheIOType.MXRecord,
        cache_file=cache_file,
        writable=True,
    )
    cache_data = {"a": torch.randn((4, 3, 224, 224))}
    cache.write_with_index(cache_data)
    assert len(cache) == 4
    cache.close()

    cache = Cache(
        method=CacheIOType.MXRecord,
        cache_file=cache_file,
        writable=False,
    )
    assert len(cache) == 4
    np.testing.assert_allclose(
        cache.read(0)["a"], cache_data["a"][0:1].numpy()
    )
    cache.close()


@pytest.mark.skip(reason="Lmdb cache may broken, donot support now.")
def test_cache_lmdb(tmpdir):
    cache_file = os.path.join(tmpdir, "tmp_cache")
    cache = Cache(
        method=CacheIOType.Lmdb, cache_file=cache_file, writable=True
    )
    cache_data = {"a": torch.randn((4, 3, 224, 224))}
    cache.write_with_index(cache_data)
    with pytest.raises(NotImplementedError):
        assert len(cache) == 4
    cache.close()

    cache = Cache(
        method=CacheIOType.Lmdb,
        cache_file=cache_file,
        writable=False,
    )
    np.testing.assert_allclose(
        cache.read(0)["a"].cpu().numpy(), cache_data["a"][0:1].numpy()
    )
    cache.close()


def test_get_global_cache(tmpdir):
    cache_file = os.path.join(tmpdir, "tmp_cache.rec")
    cache = get_global_cache(cache_file, True)
    assert isinstance(cache, Cache)
    cache.close()
    assert os.path.exists(cache_file)
