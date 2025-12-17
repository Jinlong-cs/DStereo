# -*- coding:utf-8 -*-
# Copyright (c) Horizon Robotics, All rights reserved.

import copy
from typing import Any

from torch.utils import data

from hat.registry import OBJECT_REGISTRY

__all__ = ["ToyGenDataset"]


@OBJECT_REGISTRY.register
class ToyGenDataset(data.IterableDataset):
    def __init__(
        self, batch_size, loader_len, example: Any, clone: bool = True
    ):
        self.batch_size = batch_size
        self.loader_len = loader_len
        self.example = example
        self.clone = clone

    def __iter__(self):
        for _ in range(self.loader_len):
            if self.batch_size == 1:
                b = copy.deepcopy(self.example) if self.clone else self.example
            else:
                b = [
                    copy.deepcopy(self.example) if self.clone else self.example
                    for _ in range(self.batch_size)
                ]
            yield b
