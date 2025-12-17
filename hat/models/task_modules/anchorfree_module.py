# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from .output_module import OutputModule


@OBJECT_REGISTRY.register
class AnchorFreeModule(OutputModule):
    """The container of Anchor-Free detector module.

    This class serves as the container of anchor-free detector,
    which takes feature maps as input and outputs predictions

    All the actaul calculations are implemented in component modules.

    Args:
        head: Anchor-Free head network, transfroms input feature maps
            into predictions (like regression map and classification
            score in FCOS).
        ext_feat: Extra feature module that processes input feature maps
            before head.
        target: Target generator module.
        loss: Loss module, calculates training loss by comparing head
            predictions with training targets.
        postprocess: Postprocess module, applies predictions generated
            by head module to get final prediction.
        desc: Desc module, adds user-defined description to prediction.
        output_head_out: Whether to output raw prediction of head module.
    """

    def __init__(
        self,
        head: nn.Module,
        ext_feat: Optional[nn.Module] = None,
        target: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
        desc: Optional[nn.Module] = None,
        output_head_out: bool = False,
    ):
        super().__init__(
            head,
            loss=loss,
            target=target,
            postprocess=postprocess,
            keep_name=True,
        )

        # submodule
        self.ext_feat = ext_feat
        self.desc = desc

        self._output_head_out = output_head_out

    @property
    def with_ext_feat(self) -> bool:
        return self.ext_feat is not None

    def forward(
        self,
        feat_maps: List[torch.Tensor],
        y: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        # optionally process input features with extra layers
        if self.with_ext_feat:
            feat_maps = self.ext_feat(feat_maps)

        # get rpn feats
        head_out = self.head(feat_maps)

        out_dict = OrderedDict()

        if self._output_head_out:
            out_dict.update(head_out)

        # apply prediction scores to get final predictions
        if self.with_postprocess:
            pred = self.postprocess(head_out)
            if self.desc is not None:
                pred = self.desc(pred)
            out_dict.update(pred)
        # calculate loss between predictions and ground truths
        if self.with_loss:
            assert self.has_target, "target can not be None if cal loss "
            targets = self.target(head_out, y)

            loss = self.loss(head_out, targets)
            out_dict.update(loss)

        return out_dict
