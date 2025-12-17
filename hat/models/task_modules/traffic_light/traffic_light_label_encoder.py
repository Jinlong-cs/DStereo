# Copyright (c) Horizon Robotics. All rights reserved.
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn

from hat.models.base_modules.label_encoder import MatchLabelSepEncoder
from hat.registry import OBJECT_REGISTRY
from hat.utils.tensor_func import take_row

__all__ = ["TrafficLensAttrlabelEncoder"]


@OBJECT_REGISTRY.register
class TrafficLensAttrlabelEncoder(MatchLabelSepEncoder):
    """Encode gt and matching results to separate bbox and class labels.

    Args:
        bbox_encoder: BBox label encoder
        class_encoder: Class label encoder
        cls_use_pos_only: Whether to use positive labels only during encoding.
        reg_on_hard: Regression on hard label only.
        cls_on_hard: Classification on hard label only.
        attr_encoder_list: Det attr label encoder.
        attr_use_pos_only: Det attr classification on hard label only.
    """

    def __init__(
        self,
        bbox_encoder: Optional[nn.Module] = None,
        class_encoder: Optional[nn.Module] = None,
        cls_use_pos_only: Optional[bool] = False,
        cls_on_hard: Optional[bool] = False,
        reg_on_hard: Optional[bool] = False,
        attr_encoder_list: List = None,
        attr_use_pos_only: bool = False,
    ):
        super(TrafficLensAttrlabelEncoder, self).__init__(
            bbox_encoder,
            class_encoder,
            cls_use_pos_only,
            cls_on_hard,
            reg_on_hard,
        )
        self.attr_encoder_list = attr_encoder_list
        self.attr_use_pos_only = attr_use_pos_only

    def forward(
        self,
        boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        match_pos_flag: torch.Tensor,
        match_gt_id: torch.Tensor,
        ig_flag: Optional[torch.Tensor] = None,
    ):
        matched_gt_boxes = take_row(gt_boxes, match_gt_id)

        cls_label = matched_gt_boxes[..., 4]
        num_attr = matched_gt_boxes.shape[-1]
        attr_label_list = [
            matched_gt_boxes[..., _] for _ in range(5, num_attr)
        ]  # noqa

        # ----------------------- cls label and reg label ---------------------
        # set cls label of negative matchings to 0
        cls_label[match_pos_flag == 0] = 0

        pos_match = match_pos_flag > 0
        # regress on any positive match even when label is negative
        # (hard inst) or only on moderate cases
        reg_label_mask = (
            pos_match if self.reg_on_hard else pos_match * (cls_label > 0)
        )
        reg_label_mask = reg_label_mask[..., None].flatten(
            start_dim=1, end_dim=-2
        )

        out_dict = OrderedDict()

        if self.class_encoder is not None:
            if self.cls_on_hard:
                cls_label.abs_()

            # set cls label of ignored instances to be -abs(label)
            ig_index = match_pos_flag < 0
            cls_label[ig_index] = -cls_label[ig_index].abs()

            if self.cls_use_pos_only:
                cls_label[match_pos_flag == 0] = -1

            cls_label = self.class_encoder(cls_label)

            # optionally set cls_label to ignore
            if ig_flag is not None:
                cls_label[(ig_flag == 1) * (cls_label == 0)] = -1

            cls_label = cls_label.flatten(start_dim=1, end_dim=-2)
            cls_label_mask = cls_label >= 0

            out_dict.update(
                cls_label=cls_label,
                cls_label_mask=cls_label_mask,
            )

        if self.bbox_encoder:
            reg_label = self.bbox_encoder(boxes, matched_gt_boxes).flatten(
                start_dim=1, end_dim=-2
            )
            out_dict.update(
                reg_label=reg_label,
                reg_label_mask=reg_label_mask.expand_as(reg_label),
            )

        # ------------------------- attr label --------------------------------
        type_label, color_label = attr_label_list

        # set attrs label of negative matchings to 0
        type_label[match_pos_flag == 0] = 0
        color_label[match_pos_flag == 0] = 0

        # set attrs label of ignored instances to be -abs(label)
        type_label[ig_index] = -type_label[ig_index].abs()
        color_label[ig_index] = -color_label[ig_index].abs()

        if self.attr_use_pos_only:
            type_label[match_pos_flag == 0] = -1
            color_label[match_pos_flag == 0] = -1

        type_label = self.attr_encoder_list[0][1](type_label)
        color_label = self.attr_encoder_list[1][1](color_label)

        # optionally set cls_label to ignore
        if ig_flag is not None:
            type_label[(ig_flag == 1) * (type_label == 0)] = -1
            color_label[(ig_flag == 1) * (color_label == 0)] = -1

        type_label = type_label.flatten(start_dim=1, end_dim=-2)
        type_label_mask = type_label >= 0
        color_label = color_label.flatten(start_dim=1, end_dim=-2)
        color_label_mask = color_label >= 0

        out_dict.update(
            attr_label_list=[type_label, color_label],
            attr_label_mask_list=[type_label_mask, color_label_mask],
        )

        return out_dict
