import os

import pytest
from PIL import Image

from hat.data.utils import pil_loader
from tests import root


def test_pil_loader():
    img_file = os.path.join(root, "tests/data/lena.jpg")
    img = pil_loader(img_file, size=(100, 100))

    assert isinstance(img, Image.Image)

    with pytest.raises(AssertionError):
        # TODO(wenming.meng, 0.2): fix this bug #
        assert img.size == (100, 100)

    img_file = os.path.join(root, "tests/data/not_exist.jpg")
    with pytest.raises(FileNotFoundError):
        img = pil_loader(img_file, size=(100, 100))

    # TODO(min.du, 0.2): test timeout_decorator.timeout #
