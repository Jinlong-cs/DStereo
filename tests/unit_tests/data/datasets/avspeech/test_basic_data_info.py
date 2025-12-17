import json
import os

import pytest

from hat.data.datasets.avspeech.basic_data_info import TendisBasicInfoReader
from hat.data.datasets.avspeech.redis_client import (
    TendisClient,
    multimodal_default_rename_handle,
)
from tests import HAT_BUCKET_PATH

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


@pytest.mark.skipif(redis is None, reason="need redis")
def test_basic_data_info():
    json_path = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/avspeech/data/datasets/basic_data_info.json",
    )

    client = TendisClient(
        rename_handle=multimodal_default_rename_handle,
        kwargs=tendis_kwargs,
    )

    key = "basic_data_info_unit_test"

    with open(json_path) as fr:
        value = json.load(fr)

    value = json.dumps(value).encode()
    client.set(f"Key2Label_{key}", value)
    basic_info_reader = TendisBasicInfoReader(tendis_kwargs)
    data_info = basic_info_reader.read_basic_data_info(key)
    client.delete(f"Key2Label_{key}")
    assert data_info is not None
