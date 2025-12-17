# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["PickStrideNeck"]


@OBJECT_REGISTRY.register
class PickStrideNeck(nn.Module):
    """Choose specified stride featuremap.

    Return specified stride feature map from featuremap list from fpn.

    Args:
        in_strides: strides of each input feature map
        out_strides: strides of each output feature map,
            should be a subset of in_strides, and continuous (any
            subsequence of 2, 4, 8, 16, 32, 64 ...). The largest
            stride in in_strides and out_strides should be equal
    """

    def __init__(
        self,
        in_strides: List[int],
        out_strides: List[int],
    ):

        super(PickStrideNeck, self).__init__()
        self._valid_strides = [2, 4, 8, 16, 32, 64, 128, 256]

        # in_strides check
        for s in in_strides:
            assert s in self._valid_strides

        min_idx = self._valid_strides.index(in_strides[0])
        max_idx = self._valid_strides.index(in_strides[-1])

        assert tuple(in_strides) == tuple(
            self._valid_strides[min_idx : max_idx + 1]
        ), "Input strides must be continuous and in ascending order"
        self.in_strides = in_strides

        # out_strides check

        assert len(set(out_strides)) == len(out_strides)

        out_indices = []
        for s in out_strides:
            assert s in in_strides
            out_indices.append(in_strides.index(s))

        assert tuple(out_indices) == tuple(
            sorted(out_indices)
        ), "Out strides must be in ascending order"

        self.out_indices = out_indices

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:

        assert len(features) == len(self.in_strides)

        outputs = []
        for idx in self.out_indices:
            outputs.append(features[idx])

        return outputs

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
