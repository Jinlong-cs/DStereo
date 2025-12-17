# Copyright (c) Horizon Robotics. All rights reserved.

import random
from typing import List

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "EraseBackground",
]


@OBJECT_REGISTRY.register
class EraseBackground(object):
    """Black the picture around.

    Args:
            p: Ratio of this operation. Defaults to 0.
            r_left: Randomly sample a ratio from the ratio range and black the
                left area with the ratio. Defaults to [0.07, 0.15].
            r_right: Randomly sample a ratio from the ratio range and black
                the right area with the ratio. Defaults to [0.07, 0.15].
            r_top: Randomly sample a ratio from the ratio range and black the
                top area with the ratio. Defaults to [0.03, 0.07].
            r_bottom: Randomly sample a ratio from the ratio range and black
                the bottom area with the ratio. Defaults to [0.03, 0.07].
    """

    def __init__(
        self,
        p: float = 0,
        r_left: List = (0.07, 0.15),
        r_right: List = (0.07, 0.15),
        r_top: List = (0.03, 0.07),
        r_bottom: List = (0.03, 0.07),
    ):
        assert 0 <= p <= 1
        self.p = p
        self.r_left = list(r_left)
        self.r_right = list(r_right)
        self.r_top = list(r_top)
        self.r_bottom = list(r_bottom)

    def _erase_single_img(self, img, h, w):
        erase_left = int(w * random.uniform(self.r_left[0], self.r_left[1]))
        erase_right = int(w * random.uniform(self.r_right[0], self.r_right[1]))
        erase_top = int(h * random.uniform(self.r_top[0], self.r_top[1]))
        erase_bottom = int(
            h * random.uniform(self.r_bottom[0], self.r_bottom[1])
        )

        img[:, : erase_left + 1] = 0
        img[:, -erase_right:] = 0
        img[: erase_top + 1, :] = 0
        img[-erase_bottom:, :] = 0
        return img

    def __call__(self, data):
        assert isinstance(data["layout"], str)
        assert data["layout"] == "hwc"
        do_erase = random.random() < self.p
        if not do_erase:
            return data

        img = data["img"]
        h, w = img.shape[:2]
        img = self._erase_single_img(img, h, w)
        data["img"] = img
        return data
