import os

import cv2
import pytest

from hat.data.datasets.utils import img_to_rgb
from tests import root


def test_img_to_rgb():
    img_path = os.path.join(root, "tests/data/lena.jpg")
    img = cv2.imread(img_path).astype("uint8")
    img_rgb = img_to_rgb(img)
    assert (img_rgb[..., 0] == img[..., 2]).all()
    assert (img_rgb[..., 1] == img[..., 1]).all()
    assert (img_rgb[..., 2] == img[..., 0]).all()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
