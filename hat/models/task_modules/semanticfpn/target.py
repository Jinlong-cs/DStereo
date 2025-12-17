# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Sequence

import torch

from hat.models.base_modules.resize_parser import resize
from hat.registry import OBJECT_REGISTRY

__all__ = ["BMSegTarget"]


@OBJECT_REGISTRY.register
class BMSegTarget(torch.nn.Module):
    """Generate training targets for Seg task.

    Args:
        ignore_index: Index of ignore class.
            Default is 255.
        label_name: The key corresponding to the gt seg in label.
            Default is gt_seg.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        label_name: str = "gt_seg",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_name = label_name

    def forward(
        self, label: dict, pred: Sequence[torch.Tensor]
    ) -> Sequence[dict]:  # noqa: D205,D400
        """

        Args:
            label: Meta data dict.
            pred: Output corresponding to multiple strides, the
                shape of each element is NHWC, HW is different for each stride.

        Returns:
            Sequence[dict]: Loss inputs list.

        """
        # loss need 3D input
        if label[self.label_name].shape[0] != 1:
            gt_seg = label[self.label_name].squeeze()
        else:
            gt_seg = label[self.label_name].squeeze(-1)
        gt_seg = gt_seg.type(torch.int64)

        loss_inputs = []
        for _ind, out in enumerate(pred):
            loss_input = {}
            assert len(out.shape) == 4, "bilinear need 4D input"
            out = resize(
                input=out,
                size=gt_seg.shape[1:3],
                mode="bilinear",
                align_corners=False,
                warning=False,
            )
            loss_input["pred"] = out
            loss_input["target"] = gt_seg
            avg_factor = (gt_seg != self.ignore_index).sum()
            loss_input["avg_factor"] = avg_factor.clamp(min=1.0)
            loss_inputs.append(loss_input)
        return loss_inputs
