from typing import Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "ANCBEVParkingRodLoss",
]


@OBJECT_REGISTRY.register
class ANCBEVParkingRodLoss(nn.Module):
    """Bev parkingrod loss function.

    Args:
        classification_loss: Loss function for parkingrod
            center point classification.
        endpoint_offset_loss: Loss function for coordinate offset.
        loss_weight_dict: Loss weight for cls/endpoint_offset/type loss.

            .. code-block:: none

                {
                    cls_loss: 1.0,
                    endpoint_offset_loss: 1.0,
                    rod_type_loss: 1.0,
                    aux_loss: 0.1,
                }

        aux_loss: Loss function for slot branch.
        gt_name: gt name.
    """

    def __init__(
        self,
        classification_loss: nn.Module,
        endpoint_offset_loss: nn.Module,
        loss_weight_dict: dict,
        aux_loss: Optional[nn.Module] = None,
        gt_name: str = "gt_bev_parkingrod_obj",
    ):
        super().__init__()
        self.loss_weight_dict = loss_weight_dict
        self.classification_loss = classification_loss
        self.endpoint_offset_loss = endpoint_offset_loss
        self.aux_loss = aux_loss
        self.gt_name = gt_name

    def forward(self, preds, targets):
        if self.aux_loss is not None:
            assert len(preds) == 3
            classification, endpoint_offset, slot_pred = preds
        else:
            classification, endpoint_offset = preds
        target_dict = targets[self.gt_name]
        (classification_obj, endpoint_offset_obj) = (
            target_dict["classification_obj"],
            target_dict["endpoint_offset_obj"],
        )
        (classification_weight_mask, endpoint_offset_weight_mask,) = (
            target_dict["classification_weight_mask"],
            target_dict["endpoint_offset_weight_mask"],
        )

        classification = torch.sigmoid(classification)
        classification_loss = self.classification_loss(
            pred=classification, target=classification_obj
        )
        endpoint_offset_loss = self.endpoint_offset_loss(
            pred=endpoint_offset, target=endpoint_offset_obj
        )

        classification_loss = (
            classification_weight_mask
            * classification_loss
            * self.loss_weight_dict["cls_loss"]
        )
        classification_loss = torch.sum(classification_loss) / (
            torch.sum(classification_obj) + 1
        )

        endpoint_offset_loss = (
            endpoint_offset_weight_mask
            * endpoint_offset_loss
            * self.loss_weight_dict["endpoint_offset_loss"]
            * classification
        )
        endpoint_offset_loss = torch.sum(endpoint_offset_loss) / (
            torch.sum(classification_obj) + 1
        )

        total_loss = dict(  # noqa
            classification_loss=classification_loss,
            endpoint_offset_loss=endpoint_offset_loss,
        )

        if self.aux_loss is not None:
            # get slot offset loss
            slot_01offset_obj = target_dict["slot_01offset_obj"]
            slot_01offset_weight_mask = target_dict[
                "slot_01offset_weight_mask"
            ]
            slot_01offset_loss = self.aux_loss(
                pred=slot_pred, target=slot_01offset_obj
            )
            slot_01offset_loss = (
                slot_01offset_weight_mask
                * slot_01offset_loss
                * self.loss_weight_dict["aux_loss"]
            )
            slot_01offset_loss = torch.sum(slot_01offset_loss) / (
                torch.sum(classification_obj) + 1
            )
            total_loss["slot_01offset_loss"] = slot_01offset_loss

            # get rod to slot offset loss
            rod_center_offsetx = (
                endpoint_offset[:, 0] + endpoint_offset[:, 2]
            ) / 2
            rod_center_offsety = (
                endpoint_offset[:, 1] + endpoint_offset[:, 3]
            ) / 2
            slot_01center_offsetx = (slot_pred[:, 0] + slot_pred[:, 2]) / 2
            slot_01center_offsety = (slot_pred[:, 1] + slot_pred[:, 3]) / 2
            rod2slot_center_offsetx = (
                rod_center_offsetx - slot_01center_offsetx
            ).unsqueeze(1)
            rod2slot_center_offsety = (
                rod_center_offsety - slot_01center_offsety
            ).unsqueeze(1)
            rod2slot_offset_pred = torch.cat(
                (rod2slot_center_offsetx, rod2slot_center_offsety), dim=1
            )
            rod2slot_offset_obj = target_dict["rod2slot_offset_obj"]
            rod2slot_offset_weight_mask = target_dict[
                "rod2slot_offset_weight_mask"
            ]
            rod2slot_offset_loss = self.aux_loss(
                pred=rod2slot_offset_pred, target=rod2slot_offset_obj
            )
            rod2slot_offset_loss = (
                rod2slot_offset_weight_mask
                * rod2slot_offset_loss
                * self.loss_weight_dict["aux_loss"]
            )
            rod2slot_offset_loss = torch.sum(rod2slot_offset_loss) / (
                torch.sum(classification_obj) + 1
            )
            total_loss["rod2slot_offset_loss"] = rod2slot_offset_loss

        return total_loss
