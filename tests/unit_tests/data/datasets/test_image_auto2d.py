# Copyright (c) Horizon Robotics. All rights reserved.

import os

import pytest

from hat.data.datasets.image_auto2d import Auto2dFromImage
from tests import root

data_path = os.path.join(root, "tests/data")
image_types = [".jpeg", ".png", ".jpg"]

PATH_EXIST = False
if os.path.exists(data_path):
    PATH_EXIST = True
    num_img = len(
        [
            file
            for file in os.listdir(data_path)
            if os.path.splitext(file)[1] in image_types
        ]
    )


@pytest.mark.skipif(not PATH_EXIST, reason="path does not exist")
def test_auto2d_from_image():
    dataset = Auto2dFromImage(data_path, image_types=image_types)
    assert len(dataset) == num_img
