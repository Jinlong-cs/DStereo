import numpy as np

try:
    import redis
except ImportError:
    redis = None


tendis_kwargs = dict(
    host="aidi-kv-cluster-01.hogpu.cc",
    port=7616,  # 原来是 7617
    db=0,
    socket_connect_timeout=5000,
    password="pB9cA2eN1eD1lJ0",
    socket_keepalive=True,
)


def test_redis_client():
    if redis is None:
        return

    from hat.data.datasets.avspeech.redis_client import (
        TendisClient,
        multimodal_default_rename_handle,
    )

    client = TendisClient(
        rename_handle=multimodal_default_rename_handle,
        kwargs=tendis_kwargs,
    )
    key = "redis_client_unit_test"
    set_data = np.random.randn(4, 3)
    client.set(key, set_data.tobytes())
    get_data = np.frombuffer(client.get(key)).reshape(4, 3)
    client.delete(key)
    assert (set_data == get_data).all
