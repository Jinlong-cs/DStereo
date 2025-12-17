# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

import torch.nn as nn

from hat.models.structures.classifier import Classifier
from hat.registry import OBJECT_REGISTRY

__all__ = ["TrafficLightClassifier"]


@OBJECT_REGISTRY.register
class TrafficLightClassifier(Classifier):
    """The basic structure for traffic light classifier.

    Args:
        input_preprocess: Enter the preprocessing operation before entering
            the backbone.
        backbone: Backbone module.
        backbone_extra: Neck module.
        prediction_head: Prediction head using features extracted
            from backbone.
        losses: Losses module.
        multilabel: One head corresponds to the label of multiple attributes.
            For example, in the traffic light time classification,
            the final head output feature have 33 channel,
            [0, 3)represents the countdown digits,
            [3, 13) represents the hundred digits,
            [13, 23) represents the ten digits,
            and [23, 33) represents the single digits.
        mask_in_bpu: In order to solve the problem of adjacent lights,
            whether to do mask operation on the BPU. Please refer to
            https://horizonrobotics.feishu.cn/docx/BFLZdkmbNozzsrxvsMCctLIUnlG
            for more details.
        postprocess: Postprocess op to do in BPU.
        desc: Add a description to the output.
        task_name: Task name for current head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_extra: nn.Module,
        prediction_head: nn.Module,
        losses: nn.Module,
        input_preprocess: nn.Module = None,
        multilabel: Optional[bool] = False,
        mask_in_bpu: Optional[bool] = False,
        postprocess: Optional[nn.Module] = None,
        desc: nn.Module = None,
        task_name: str = None,
    ):
        super(TrafficLightClassifier, self).__init__(backbone, losses)
        self.input_preprocess = input_preprocess
        self.backbone_extra = backbone_extra
        self.prediction_head = prediction_head
        self.multilabel = multilabel
        self.mask_in_bpu = mask_in_bpu
        self.postprocess = postprocess
        self.desc = desc
        self.task_name = task_name

    def forward(self, data):
        x = data["img"]
        if self.mask_in_bpu:
            assert (
                "mask_width" in data.keys()
            ), "when do mask op in bpu, crop_mask must in input"
            height_crop_mask = data["mask_height"]
            width_crop_mask = data["mask_width"]
            assert self.input_preprocess is not None
            x = self.input_preprocess(x, height_crop_mask, width_crop_mask)

        target = data.get("gt_classes", None)
        features = self.backbone(x)
        features = self.backbone_extra(features)
        preds = self.prediction_head(features)

        if self.training and self.desc is not None:
            if self.postprocess is not None:
                preds = self.postprocess(preds)
            preds = self.desc(preds)
        if target is None:
            if "obj_id" in data:
                return {
                    "pred_cls": preds,
                    "obj_id": data["obj_id"],
                    "img_name": data["img_name"],
                }
            return {"pred_cls": preds}

        if not self.training or self.losses is None:
            return {"preds": preds, "target": target}

        if self.multilabel:
            losses = self.losses(preds, data["multi_label"])
            return {
                "preds": preds,
                "nums_digit_loss": losses[0],
                "ones_place_loss": losses[1],
                "tens_place_loss": losses[2],
                "hundreds_place_loss": losses[3],
            }
        else:
            losses = self.losses(
                preds.flatten(1),
                target.view(
                    -1,
                ),
            )
            return {"preds": preds, "classification_loss": losses}
