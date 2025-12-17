import copy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.distributed import get_world_size

from hat.models.task_modules.deform_detr.deformable_criterion import (
    SetCriterion,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import dist_initialized

__all__ = ["TwoStageCriterion", "DINOCriterion"]


class TwoStageCriterion(SetCriterion):
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        losses: List = None,
        eos_coef: float = None,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2,
        two_stage_binary_cls=False,
    ):
        if losses is None:
            losses = ["class", "boxes"]
        super().__init__(
            num_classes,
            matcher,
            weight_dict,
            losses,
            eos_coef,
            loss_class_type,
            alpha,
            gamma,
        )
        self.two_stage_binary_cls = two_stage_binary_cls

    def forward(self, outputs: dict, targets: dict):
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the
        # targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes,
        # for normalization purposes
        num_boxes = sum(len(t) for t in targets["gt_classes"])
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if dist_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes)
            )

        # In case of auxiliary losses, we repeat this process with the output
        # of each intermediate layer.
        weight_dict_all = copy.deepcopy(self.weight_dict)

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                weight_dict_all.update(
                    {k + f"_{i}": v for k, v in self.weight_dict.items()}
                )

        # for two stage
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for i in range(len(bin_targets["gt_classes"])):
                bin_targets["gt_classes"][i] = torch.zeros_like(
                    bin_targets["gt_classes"][i]
                )
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, num_boxes
                )
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)
            weight_dict_all.update(
                {k + "_enc": v for k, v in self.weight_dict.items()}
            )
        for k in losses.keys():
            if k in weight_dict_all:
                losses[k] *= weight_dict_all[k]
        return losses


@OBJECT_REGISTRY.register
class DINOCriterion(TwoStageCriterion):
    """This class computes the loss for DINO.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the
             outputs of the model
        2) we supervise each pair of matched ground-truth / prediction
            (supervise class and box)
    """

    def forward(self, outputs: dict, targets: dict, dn_metas: dict = None):
        losses = super(DINOCriterion, self).forward(outputs, targets)

        num_boxes = sum(len(t) for t in targets["gt_classes"])
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if dist_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses

        aux_num = 0
        if "aux_outputs" in outputs:
            aux_num = len(outputs["aux_outputs"])
        dn_losses = self.compute_dn_loss(dn_metas, targets, aux_num, num_boxes)

        losses.update(dn_losses)

        return losses

    def compute_dn_loss(
        self, dn_metas: dict, targets: dict, aux_num: int, num_boxes: int
    ):
        """Compute dn loss in criterion."""
        losses = {}
        if dn_metas and "output_known_lbs_bboxes" in dn_metas:
            output_known_lbs_bboxes, dn_num, single_padding = (
                dn_metas["output_known_lbs_bboxes"],
                dn_metas["dn_num"],
                dn_metas["single_padding"],
            )
            dn_idx = []
            for i in range(len(targets["gt_classes"])):
                if len(targets["gt_classes"][i]) > 0:
                    t = (
                        torch.arange(0, len(targets["gt_classes"][i]))
                        .long()
                        .cuda()
                    )
                    t = t.unsqueeze(0).repeat(dn_num, 1)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(dn_num)) * single_padding
                    ).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if "labels" in loss:
                    kwargs = {"log": False}
                l_dict.update(
                    self.get_loss(
                        loss,
                        output_known_lbs_bboxes,
                        targets,
                        dn_idx,
                        num_boxes * dn_num,
                        **kwargs,
                    )
                )

            l_dict = {k + "_dn": v for k, v in l_dict.items()}

            for k in l_dict.keys():
                l_dict[k] *= self.weight_dict[k]
            losses.update(l_dict)
        else:
            losses["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")

        for i in range(aux_num):
            # dn aux loss
            l_dict = {}
            if dn_metas and "output_known_lbs_bboxes" in dn_metas:
                output_known_lbs_bboxes_aux = output_known_lbs_bboxes[
                    "aux_outputs"
                ][i]
                for loss in self.losses:
                    kwargs = {}
                    if "labels" in loss:
                        kwargs = {"log": False}
                    l_dict.update(
                        self.get_loss(
                            loss,
                            output_known_lbs_bboxes_aux,
                            targets,
                            dn_idx,
                            num_boxes * dn_num,
                            **kwargs,
                        )
                    )
                l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                for k in l_dict.keys():
                    l_dict[k] *= self.weight_dict[k.strip(f"_{i}")]
            else:
                l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses
