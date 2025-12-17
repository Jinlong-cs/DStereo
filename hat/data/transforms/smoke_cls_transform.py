# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import random

import numpy as np

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = [
    "OneFromMultiple",
]


@OBJECT_REGISTRY.register
class OneFromMultiple:
    """Randomly choose a transformer to apply from multiple ones.

    Args:
        transform: list of transform to compose.
        prob: list of probability range in [0, 1].
    """

    def __init__(self, transforms, probs):
        assert isinstance(transforms, list)
        assert isinstance(probs, list)
        self.transforms = transforms
        self.probs = np.array(probs)
        assert self.probs.sum() <= 1
        assert ((self.probs < 0).sum() + (self.probs > 1).sum()) == 0
        self.thresh = np.cumsum(self.probs)

    def __call__(self, data):
        num = random.random()
        for func, thresh in zip(self.transforms, self.thresh):
            if num < thresh:
                data = func(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t, p in zip(self.transforms, self.probs):
            format_string += "\n"
            format_string += "    {0}".format(t)
            format_string += f"   p : {p}"
        format_string += "\n)"
        return format_string
