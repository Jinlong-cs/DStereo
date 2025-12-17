import os

import pytest

try:
    from redis import Redis
except ImportError:
    Redis = None

from hat.data.datasets.avspeech.image import (
    DirectVideoImageReader,
    TendisVideoImageReader,
)
from tests import HAT_BUCKET_PATH

tendis_kwargs = dict(
    host="aidi-kv-cluster-01.hogpu.cc",
    port=7616,  # 原来是 7617
    db=0,
    socket_connect_timeout=5000,
    password="pB9cA2eN1eD1lJ0",
    socket_keepalive=True,
)


@pytest.mark.skipif(Redis is None, reason="need redis")
def test_image():
    mp4_path = os.path.join(
        HAT_BUCKET_PATH,
        "unit_test_data/avspeech/data/datasets/7049569131632266504_1.mp4",
    )

    # 测试DirectVideoImageReader
    image_reader = DirectVideoImageReader(
        fps=25,
        missing_image_complete_mode="zero",
    )
    images = image_reader.read_seg_images(
        mp4_path=mp4_path,
    )
    assert len(images) == 29
    assert images[0].shape == (96, 96, 3)

    # 测试TendisVideoImageReader
    # 上传临时数据到tenis
    key_pref = "multimodal-Key2Img"
    utt = "7049569131632266504_1"
    zfill_num = 6
    client = Redis(**tendis_kwargs)
    batch = {}
    for idx in range(len(images)):
        img_data = images[idx]
        img_key = f"{key_pref}_{utt}_{idx :>0{zfill_num}d}"
        batch[img_key] = img_data.tobytes()

    client.mset(batch)

    image_reader = TendisVideoImageReader(
        fps=25,
        origin_fps=25,
        zfill_num=6,
        missing_image_complete_mode="zero",
        filter_rules=None,
        tendis_kwargs=tendis_kwargs,
    )

    images = image_reader.read_seg_images(
        utt="7049569131632266504_1",
        seg_beg=0.0,
        seg_end=1.13,
    )
    assert len(images) == 29
    assert images[0].shape == (96, 96, 3)

    # 删除临时数据
    keys = batch.keys()
    client.delete(*keys)
