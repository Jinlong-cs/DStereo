import os

import cv2
import pytest

from hat.data.transforms.frame.reader import YUVTurboJPEGDecoder
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH

try:
    import pyramid_resizer  # noqa: F401

    _PYRAMID_RESIZER = True
except ImportError:
    _PYRAMID_RESIZER = False


@pytest.mark.parametrize(
    ["to_string"],
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
@pytest.mark.skipif(
    not _PYRAMID_RESIZER, reason="pyramid_resizer is required"
)  # noqa
@pytest.mark.skipif(
    not HAT_BUCKET_EXISTS, reason="HAT_BUCKET is required"
)  # noqa
def test_yuv_turbo_jpeg_decoder(tmpdir, to_string):

    img_path = os.path.join(
        HAT_BUCKET_PATH,
        "data/orig_data/voc/VOCdevkit/VOC2007/JPEGImages/000001.jpg",
    )

    # Generate available image can be used in pryamid_resize
    img = cv2.imread(img_path)
    img_w_resize, img_h_resize = 1280, 720
    target_size = (img_w_resize, img_h_resize)
    resized_img = cv2.resize(img, target_size)
    resized_img_path = os.path.join(tmpdir, "resized_img.jpg")
    cv2.imwrite(resized_img_path, resized_img)

    decoder = YUVTurboJPEGDecoder(to_string=to_string)

    with open(resized_img_path, "rb") as f:
        img_buf = f.read()

    data = dict(img_buf=img_buf)
    data = decoder(data)
    if to_string:
        yuvi420_buf = data["img_buf"]
        img_h, img_w = data["img_height"], data["img_width"]
        assert yuvi420_buf
        assert (img_w, img_h) == target_size
    else:
        y_img, u_img, v_img = data
        assert y_img.shape == (img_h_resize, img_w_resize)
        assert (
            u_img.shape
            == v_img.shape
            == (img_h_resize // 2, img_w_resize // 2)
        )
