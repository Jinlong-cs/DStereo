# Copyright (c) Horizon Robotics. All rights reserved.
import collections
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY

__all__ = ["SampleModelDataset"]


@OBJECT_REGISTRY.register
class SampleModelDataset(data.Dataset):
    """Sample model dataset.

    Args:
        image_size: image input (w, h) for sample_model network
        grid_size: grid input (w, h) for sample_model network
        num: num of data
        transforms: List of transforms
    """

    def __init__(
        self,
        image_size: Tuple,
        grid_size: Tuple,
        num: int,
        transforms: Optional[List] = None,
    ):
        self.image_size = image_size
        self.grid_size = grid_size
        self.num = num
        self.transforms = transforms

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        image = torch.rand((1, self.image_size[1], self.image_size[0])) * 255
        image = torch.tensor(image.clone(), dtype=torch.uint8)
        grid = torch.rand((2, self.grid_size[1], self.grid_size[0])) * 2 - 1

        data = collections.defaultdict(dict)
        data["img"] = image
        data["grid"] = grid
        data["label"] = np.ones((1), dtype=np.float32)
        data["layout"] = "chw"

        # data transoform
        if self.transforms:
            data = self.transforms(data)

        return data
