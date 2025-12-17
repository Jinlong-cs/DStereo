# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "ANCBEVPSDLoss",
    "ANCBEVPSDLocalLoss",
    "ANCBEVPSDGlobalLoss",
]


@OBJECT_REGISTRY.register
class ANCBEVPSDLoss(nn.Module):
    """Super psd loss.

    Args:
        global_loss: Loss function for global head.
        local_loss: Loss function for local head.
    """

    def __init__(
        self,
        global_loss: nn.Module = None,
        local_loss: nn.Module = None,
        local_loss_near: nn.Module = None,
        gt_name: str = "gt_bev_psd_obj",
    ):
        super().__init__()
        self.global_loss = None
        self.local_loss = None
        self.gt_name = gt_name
        if global_loss is not None:
            self.global_loss = global_loss
        if local_loss is not None:
            self.local_loss = local_loss
        self.local_loss_near = local_loss_near

    def forward(self, preds, targets):

        total_loss = dict()  # noqa
        global_preds, local_preds = preds[0:5], preds[5:9]
        targets_dict = targets[self.gt_name]
        if self.global_loss is not None:
            global_loss = self.global_loss(global_preds, targets_dict)
            total_loss.update(global_loss)
        if self.local_loss is not None:
            local_loss = self.local_loss(local_preds, targets_dict)
            total_loss.update(local_loss)
        if self.local_loss_near is not None:
            local_preds_near = preds[9:]
            local_loss_near = self.local_loss_near(
                local_preds_near, targets_dict
            )
            local_loss_near = {
                k + "_near": v for k, v in local_loss_near.items()
            }
            total_loss.update(local_loss_near)
        return total_loss


@OBJECT_REGISTRY.register
class ANCBEVPSDGlobalLoss(nn.Module):
    """Super psd global loss function.

    Args:
        classification_loss: Slot center point classification loss
            function for global head.
        offset_loss: Coordinate offset loss function for global head.
        occupancy_loss: Loss function used to optimize whether slot
            is occupancy.
        slot_type_loss: Loss function for slot type classification.
        direction_loss: Loss function for slot direction regression.
        loss_weights: Weight list for all five loss function.
    """

    def __init__(
        self,
        classification_loss: nn.Module,
        offset_loss: nn.Module,
        occupancy_loss: nn.Module,
        slot_type_loss: nn.Module,
        direction_loss: nn.Module,
        loss_weights: List[float],  # noqa
    ):
        super().__init__()
        self.loss_weights = loss_weights
        self.classification_loss = classification_loss
        self.offset_loss = offset_loss
        self.occupancy_loss = occupancy_loss
        self.slot_type_loss = slot_type_loss
        self.direction_loss = direction_loss

    def forward(self, global_preds, target_dict):
        classification, offset, occupancy, slot_type, direction = global_preds
        (
            classification_obj,
            offset_obj,
            occupancy_obj,
            slot_type_obj,
            direction_obj,
        ) = (
            target_dict["global_classification_obj"],
            target_dict["global_offset_obj"],
            target_dict["global_occupancy_obj"],
            target_dict["global_slot_type_obj"],
            target_dict["global_direction_obj"],
        )
        (
            classification_grad,
            offset_grad,
            occupancy_grad,
            slot_type_grad,
            direction_grad,
        ) = (
            target_dict["global_classification_grad"],
            target_dict["global_offset_grad"],
            target_dict["global_occupancy_grad"],
            target_dict["global_slot_type_grad"],
            target_dict["global_direction_grad"],
        )
        classification = torch.sigmoid(classification)
        direction = torch.tanh(direction)
        classification_loss = self.classification_loss(
            pred=classification, target=classification_obj
        )

        # Define a small value for epsilon (eps) to avoid sqrt() backward NaN
        _eps = torch.finfo(classification.dtype).eps

        # Calculate the distances between the parking-slot points
        dis_01 = torch.sqrt(
            (offset[:, 0] - offset[:, 2]) ** 2
            + (offset[:, 1] - offset[:, 3]) ** 2
            + _eps
        )
        dis_12 = torch.sqrt(
            (offset[:, 2] - offset[:, 4]) ** 2
            + (offset[:, 3] - offset[:, 5]) ** 2
            + _eps
        )
        dis_23 = torch.sqrt(
            (offset[:, 4] - offset[:, 6]) ** 2
            + (offset[:, 5] - offset[:, 7]) ** 2
            + _eps
        )
        dis_30 = torch.sqrt(
            (offset[:, 6] - offset[:, 0]) ** 2
            + (offset[:, 7] - offset[:, 1]) ** 2
            + _eps
        )

        # Calculate width and height ratios using the min_maximum distances
        width_ratio = torch.min(dis_01, dis_23) / (
            torch.max(dis_01, dis_23) + _eps
        )
        height_ratio = torch.min(dis_12, dis_30) / (
            torch.max(dis_12, dis_30) + _eps
        )

        # Calculate sideness from the product of width and height ratios
        sideness = torch.sqrt(width_ratio * height_ratio + _eps)

        # Calculate the loss for sideness by subtracting sideness from 1 and
        # multiplying by classification objectness score
        loss_sideness = (1 - sideness) * classification_obj

        offset_loss = self.offset_loss(pred=offset, target=offset_obj)
        occupancy_loss = self.occupancy_loss(
            input=occupancy, target=occupancy_obj
        )
        slot_type_loss = self.slot_type_loss(
            input=slot_type, target=slot_type_obj
        )
        direction_loss = self.direction_loss(
            input=direction, target=direction_obj
        )

        classification_loss = (
            classification_grad * classification_loss * self.loss_weights[0]
        )
        classification_loss = torch.sum(classification_loss) / (
            torch.sum(classification_obj) + 1
        )
        offset_loss = offset_grad * offset_loss * self.loss_weights[1]
        offset_loss = torch.sum(offset_loss) / (
            torch.sum(classification_obj) + 1
        )
        occupancy_loss = occupancy_grad * occupancy_loss * self.loss_weights[2]
        occupancy_loss = torch.sum(occupancy_loss) / (
            torch.sum(classification_obj) + 1
        )
        slot_type_loss = slot_type_grad * slot_type_loss * self.loss_weights[3]
        slot_type_loss = torch.sum(slot_type_loss) / (
            torch.sum(classification_obj) + 1
        )
        direction_loss = direction_grad * direction_loss * self.loss_weights[4]
        direction_loss = torch.sum(direction_loss) / (
            torch.sum(classification_obj) + 1
        )
        sideness_loss = torch.sum(loss_sideness) / (
            torch.sum(classification_obj) + 1
        )

        total_loss = dict(  # noqa
            global_classification_loss=classification_loss,
            global_offset_loss=offset_loss,
            global_occupancy_loss=occupancy_loss,
            global_slot_type_loss=slot_type_loss,
            global_direction_loss=direction_loss,
            global_sideness_loss=sideness_loss,
        )
        return total_loss


@OBJECT_REGISTRY.register
class ANCBEVPSDLocalLoss(nn.Module):
    """Super psd local loss function.

    Args:
        classification_loss: Slot corner point classification loss
            function for local head.
        offset_loss: Coordinate offset loss function for local head.
        sline_angle_loss: Sline angle loss function for local head.
        point_type_loss: Loss function for point type classification.
        loss_weights: Weight list for all four loss functions.
        offset_weight: The weight for offset.
        crop_roi: RoI for target heatmap.
        gt_postfix: Target name postfix.
    """

    def __init__(
        self,
        classification_loss: nn.Module,
        offset_loss: nn.Module,
        sline_angle_loss: nn.Module,
        point_type_loss: nn.Module,
        loss_weights: List[int],  # noqa
        offset_weight: List[float],
        crop_roi: List[int] = None,
        gt_postfix: str = "",
    ):
        super().__init__()
        self.classification_loss = classification_loss
        self.offset_loss = offset_loss
        self.sline_angle_loss = sline_angle_loss
        self.point_type_loss = point_type_loss
        self.loss_weights = loss_weights
        self.offset_weight = offset_weight
        self.crop_roi = crop_roi
        self.gt_postfix = gt_postfix

    def crop(self, feat):
        return feat[
            :,
            :,
            self.crop_roi[0] : self.crop_roi[2],
            self.crop_roi[1] : self.crop_roi[3],
        ]

    def forward(self, local_preds, target_dict):
        # local_preds: [batch_size, 24, 112, 112]
        # local_targets: [[[x1, y1, x2, y2, x3, y3, x4, y4,
        #                p1_vec_x, p1_vec_y, p2_vec_x, p2_vec_y,
        #                p3_vec_x, p3_vec_y, p4_vec_x, p4_vec_y,
        #                p1_type, p2_type, p3_type, p4_type], []],[[]]]
        classification, offset, sline_angle, point_type = local_preds
        classification_obj, offset_obj, sline_angle_obj, point_type_obj = (
            target_dict[f"local_classification_obj{self.gt_postfix}"],
            target_dict[f"local_offset_obj{self.gt_postfix}"],
            target_dict[f"local_sline_angle_obj{self.gt_postfix}"],
            target_dict[f"local_point_type_obj{self.gt_postfix}"],
        )
        classification_grad, offset_grad, sline_angle_grad, point_type_grad = (
            target_dict[f"local_classification_grad{self.gt_postfix}"],
            target_dict[f"local_offset_grad{self.gt_postfix}"],
            target_dict[f"local_sline_angle_grad{self.gt_postfix}"],
            target_dict[f"local_point_type_grad{self.gt_postfix}"],
        )
        if self.crop_roi is not None:
            classification_obj, offset_obj, sline_angle_obj, point_type_obj = [
                self.crop(f)
                for f in (
                    classification_obj,
                    offset_obj,
                    sline_angle_obj,
                    point_type_obj,
                )
            ]
            (
                classification_grad,
                offset_grad,
                sline_angle_grad,
                point_type_grad,
            ) = [
                self.crop(f)
                for f in (
                    classification_grad,
                    offset_grad,
                    sline_angle_grad,
                    point_type_grad,
                )
            ]

        classification = torch.sigmoid(classification)
        sline_angle = torch.tanh(sline_angle)

        classification_loss = (
            self.classification_loss(
                logits=classification,
                labels=classification_obj,
                grad_tensor=classification_grad,
            )
            * self.loss_weights[0]
        )
        offset_loss = (
            self.offset_loss(pred=offset, target=offset_obj)
            * self.loss_weights[1]
        )
        sline_angle_loss = (
            self.sline_angle_loss(input=sline_angle, target=sline_angle_obj)
            * self.loss_weights[2]
        )
        point_type_loss = (
            self.point_type_loss(input=point_type, target=point_type_obj)
            * self.loss_weights[3]
        )
        offset_loss *= offset_grad
        # The following numerical 0123 representation
        # Parking-slot four corner points
        number_0123 = (offset_grad > 0).sum()
        offset_loss = torch.sum(offset_loss) / (number_0123 + 1)
        sline_angle_loss *= sline_angle_grad
        sline_angle_loss = torch.sum(sline_angle_loss) / (
            torch.sum(sline_angle_grad) + 1
        )
        point_type_loss *= point_type_grad
        point_type_loss = torch.sum(point_type_loss) / (
            torch.sum(point_type_grad) + 1
        )
        total_loss = dict(  # noqa
            local_classification_loss=classification_loss,
            local_offset_loss=offset_loss,
            local_sline_angle_loss=sline_angle_loss,
            local_point_type_loss=point_type_loss,
        )
        return total_loss
