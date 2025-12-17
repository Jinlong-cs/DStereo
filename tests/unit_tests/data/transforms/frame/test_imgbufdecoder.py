import os

import cv2
import numpy as np
import pytest

from hat.data.transforms.frame.reader import ImgBufDecoder
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.mark.parametrize(
    ["to_rgb"],
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS, reason="HAT_BUCKET is required"
)  # noqa
def test_img_buf_decoder(to_rgb):
    img_path = os.path.join(
        HAT_BUCKET_PATH,
        "data/orig_data/voc/VOCdevkit/VOC2007/JPEGImages/000001.jpg",
    )

    decoder = ImgBufDecoder(to_rgb=to_rgb)

    with open(img_path, "rb") as f:
        img_buf = f.read()

    data = dict(img_buf=img_buf)
    data = decoder(data)

    img_h, img_w = data["img_shape"][:2]
    img = cv2.imread(img_path)
    assert (img_h, img_w) == img.shape[:2]
    if to_rgb:
        assert data["color_space"] == "rgb"
        img_rgb = data["img"]
        assert np.abs(img_rgb - img[:, :, ::-1]).sum() == 0
    else:
        assert data["color_space"] == "bgr"
        img_bgr = data["img"]
        assert np.abs(img_bgr - img).sum() == 0
