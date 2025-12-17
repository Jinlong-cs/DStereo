import logging

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from hat.core.box_utils import box_center_to_corner
from hat.models.task_modules.e2e_dynamic.e2e_dynamic_util import dict_select
from hat.registry import OBJECT_REGISTRY
from .e2e_box_util import (
    complete_box_iou,
    distance_box_iou,
    generalized_box_iou,
    matched_boxlist_iou,
)

__all__ = ["E2EHungarianMatcher", "E2EClipMatcher"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class E2EHungarianMatcher(nn.Module):
    """Get the assignment matrix between targets and preds.

    For efficiency reasons, the targets don't include the no_object.
    Because of this, in general, there are more predictions than targets.
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Args:
        cls_loss_weight: The weight of class loss in the cost matrix.
        bbox_loss_weight: The weight of bbox regression loss in the cost
            matrix.
        iou_loss_weight: The weight of the iou loss in the cost matrix.
        iou_loss_type: The iou type used in the iou loss. Selected from
            ["ciou", "diou", "giou", None].
        alpha: focal loss param, default:0.25.
        gamma: focal loss param, default:2.0.
        use_psc_rot: whether use psc to encode the heading angle.
    """

    def __init__(
        self,
        cls_loss_weight: float = 1,
        bbox_loss_weight: float = 1,
        iou_loss_weight: float = 1,
        iou_loss_type: str = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        use_psc_rot: bool = False,
    ):
        super().__init__()
        self.cls_loss_weight = cls_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.cost_giou = iou_loss_weight
        self.iou_loss_type = iou_loss_type
        self.alpha = alpha
        self.gamma = gamma
        assert (
            cls_loss_weight != 0
            or bbox_loss_weight != 0
            or iou_loss_weight != 0
        ), "all costs cant be 0 simulatineously!"
        assert iou_loss_type in [
            "ciou",
            "diou",
            "giou",
            None,
        ], "{} is not supported!".format(iou_loss_type)
        self.use_psc_rot = use_psc_rot

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list):
        """Perform the matching.

        Args:
            outputs: This is a dict that contains at least these entries: # noqa

                "pred_logits": Tensor of dim [batch_size, num_queries,
                    num_classes] with the classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4]
                    with the predicted box coordinates
                "pred_yaws": Tensor of dim [batch_size, num_queries,
                    2] with the predicted cos_yaw and sin_yaw

            targets: List of targets, (len(targets) = batch_size),
                where each target is a dict containing:

                "labels": Tensor of dim [num_target_boxes] (where
                    num_target_boxes is the number of ground-truth objects
                    in the target) containing the class labels.
                "boxes": Tensor of dim [num_target_boxes, 4]
                    containing the target box coordinates

        Returns:
            A list containing tuples of indices: [(index_i, index_j), (index_i, index_j)...]: # noqa
                - index_i: indices of the selected predictions (in order),
                    shape: (num_select_objects,) # noqa
                - index_j: the indices of the corresponding selected targets
                    (in order), shape: (num_select_objects,)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(
                    num_queries, num_target_boxes
                    )
        """
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].sigmoid()
        out_bbox = outputs["pred_boxes"]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the focal classification.
        neg_cls_loss_weight = (
            (1 - self.alpha)
            * (out_prob ** self.gamma)
            * (-(1 - out_prob + 1e-8).log())
        )
        pos_cls_loss_weight = (
            self.alpha
            * ((1 - out_prob) ** self.gamma)
            * (-(out_prob + 1e-8).log())
        )
        cls_loss_weight = (
            pos_cls_loss_weight[:, tgt_ids] - neg_cls_loss_weight[:, tgt_ids]
        )

        # Compute the L1 cost between boxes
        bbox_loss_weight = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        if self.iou_loss_type == "giou":
            cost_giou = -generalized_box_iou(
                box_center_to_corner(out_bbox), box_center_to_corner(tgt_bbox)
            )
        elif self.iou_loss_type == "diou":
            cost_giou = -distance_box_iou(
                box_center_to_corner(out_bbox), box_center_to_corner(tgt_bbox)
            )
        elif self.iou_loss_type == "ciou":
            if self.use_psc_rot:
                out_yaws = None
                tgt_yaws = None
            else:
                out_yaws = outputs["pred_yaws"]
                tgt_yaws = torch.cat([v["yaws"] for v in targets])
            cost_giou = -complete_box_iou(
                box_center_to_corner(out_bbox),
                box_center_to_corner(tgt_bbox),
                yaws1=out_yaws,
                yaws2=tgt_yaws,
            )
        else:
            cost_giou = 0

        # Final cost matrix
        C = (
            self.bbox_loss_weight * bbox_loss_weight
            + self.cls_loss_weight * cls_loss_weight
            + self.cost_giou * cost_giou
        )
        C = C.unsqueeze(0).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


@OBJECT_REGISTRY.register
class E2EClipMatcher(nn.Module):
    """
    Compute the match indices between preds and gts.

    Args:
        matcher: E2EHungrainMatcher used to get the match indices between
            det queries and new gt instances.
        match_indices_each_layer: whether each aux layer share the match
            indices with the last output layer, default is False.
        enable_rfs: whether to turn on global matching during matching
            (i.e. matching all predicted outputs with all true values).
        use_psc_rot: whether use psc rot.

    """

    def __init__(
        self,
        matcher: nn.Module,
        match_indices_each_layer: bool = False,
        enable_rfs: bool = False,
        use_psc_rot: bool = False,
    ):
        super().__init__()
        self.matcher = matcher
        self._current_frame_idx = 0
        self.match_indices_each_layer = match_indices_each_layer
        self.use_psc_rot = use_psc_rot
        self.enable_rfs = enable_rfs

    def _step(self):
        """Step the frame_idx in the clip."""
        self._current_frame_idx += 1

    def initialize_for_single_clip(self, clip_targets: dict):
        """Init the parameter for clip forward."""
        assert (
            "bev_tracking" in clip_targets
        ), "`bev_tracking` has to be existed as a key in the input."
        self.gt_instances = clip_targets["bev_tracking"]
        self._current_frame_idx = 0

    def match_for_single_frame(self, outputs: dict, pred_instances: dict):
        """Conduct the matcher and assign the indices."""
        output_for_losses = {}
        gt_instances_cur = self.gt_instances[self._current_frame_idx]
        num_gt_objs = len(gt_instances_cur["labels"])
        aux_outputs = outputs.pop("aux_outputs", None)
        if self.use_psc_rot:
            outputs["pred_yaws"] = 2 * torch.sigmoid(outputs["pred_yaws"]) - 1
        else:
            outputs["pred_yaws"] = F.normalize(outputs["pred_yaws"], dim=-1)
        outputs.update(
            {
                "obj_idxes": pred_instances["obj_idxes"].long(),
                "matched_gt_idxes": -torch.ones_like(
                    pred_instances["matched_gt_idxes"]
                ).long(),
            }
        )

        device = outputs["pred_boxes"].device
        num_pred_trks = len(outputs["pred_logits"])

        gt_trk_id_list = (
            gt_instances_cur["obj_idxes"].detach().cpu().numpy().tolist()
        )
        trk_id_to_gt_idx = {
            trk_id: gt_idx for gt_idx, trk_id in enumerate(gt_trk_id_list)
        }

        # step 1. inherit and update the match indices.
        num_disappear_track = 0
        ious = torch.ones(num_pred_trks, device=device)
        for i, pred_trk_id in enumerate(outputs["obj_idxes"]):
            pred_trk_id = pred_trk_id.item()
            if pred_trk_id >= 0:
                if pred_trk_id in trk_id_to_gt_idx:
                    outputs["matched_gt_idxes"][i] = trk_id_to_gt_idx[
                        pred_trk_id
                    ]
                    gt_boxes = box_center_to_corner(
                        gt_instances_cur["boxes"][
                            trk_id_to_gt_idx[pred_trk_id]
                        ].unsqueeze(0)
                    )
                    pred_boxes = box_center_to_corner(
                        outputs["pred_boxes"][i].unsqueeze(0)
                    )
                    iou = matched_boxlist_iou(pred_boxes, gt_boxes)[0]
                    ious[i] = iou
                else:
                    num_disappear_track += 1
                    # TODO: set disappear track's flag as -2, prevent them
                    # from participating in the match.
                    # disappear trk doesnot particate in the next match.
                    outputs["matched_gt_idxes"][i] = -1
        # alive in successive frames.
        pre_matched_indices_mask = outputs["matched_gt_idxes"] >= 0
        pre_matched_indices = torch.stack(
            (
                torch.arange(num_pred_trks).to(device),
                outputs["matched_gt_idxes"],
            ),
            dim=1,
        )[pre_matched_indices_mask]

        # step 2. assign the untracked query to new gt instances.

        # queries mask that never be assigned to any gt in current clip
        unmatch_pred_trk_ids_mask = outputs["obj_idxes"] == -1
        unmatch_pred_trk_idxs = torch.arange(num_pred_trks).to(device)[
            unmatch_pred_trk_ids_mask
        ]
        unmatched_preds = {
            "pred_logits": (outputs["pred_logits"][unmatch_pred_trk_idxs]),
            "pred_boxes": (outputs["pred_boxes"][unmatch_pred_trk_idxs]),
            "pred_yaws": (outputs["pred_yaws"][unmatch_pred_trk_idxs]),
        }
        # gt idxes that not be assinged to any pred trk
        matched_gt_idxes = outputs["matched_gt_idxes"][
            pre_matched_indices_mask
        ]
        unmatched_gt_mask = torch.ones(num_gt_objs).to(device)
        unmatched_gt_mask[matched_gt_idxes] = 0
        unmatch_gt_trk_idxs = torch.arange(num_gt_objs).to(device)[
            unmatched_gt_mask == 1
        ]
        unmatched_gt_instance = dict_select(
            gt_instances_cur, unmatch_gt_trk_idxs
        )

        def match_for_single_decoder_layer(
            unmatched_pred, aux_supervise
        ):  # aux_supervise on represents the matching of the middle layer
            if len(outputs["pred_logits"]) == 0:
                return torch.zeros((0, 2)).to(device)
            if aux_supervise:
                indices = self.matcher(
                    unmatched_pred, [gt_instances_cur]
                )  # list[tuple(src_idx, tgt_idx)]
                new_matched_indices = torch.stack(
                    [
                        indices[0][0],
                        indices[0][1],
                    ],
                    dim=1,
                ).to(device)
            else:
                indices = self.matcher(
                    unmatched_pred, [unmatched_gt_instance]
                )  # list[tuple(src_idx, tgt_idx)]
                new_matched_indices = torch.stack(
                    [
                        unmatch_pred_trk_idxs[indices[0][0]],
                        unmatch_gt_trk_idxs[indices[0][1]],
                    ],
                    dim=1,
                )
            return new_matched_indices

        new_matched_indices = match_for_single_decoder_layer(
            unmatched_preds, aux_supervise=False
        )
        new_matched_indices = new_matched_indices.long()

        # step3. update the `obj_idxes` of the new matched queries
        outputs["obj_idxes"][new_matched_indices[:, 0]] = gt_instances_cur[
            "obj_idxes"
        ][new_matched_indices[:, 1]]
        outputs["matched_gt_idxes"][
            new_matched_indices[:, 0]
        ] = new_matched_indices[:, 1]

        all_matched_indices = torch.cat(
            [new_matched_indices, pre_matched_indices], dim=0
        )

        # step4. format the output vars
        num_samples = len(gt_instances_cur["obj_idxes"]) + num_disappear_track
        output_loss_key = f"frame_{self._current_frame_idx}_output_layer"
        output_for_losses[output_loss_key] = {
            "pred_output": outputs,
            "matched_ious": ious,
            "gt_instances": gt_instances_cur,
            "num_samples": num_samples,
            "match_indices": (
                all_matched_indices[:, 0],
                all_matched_indices[:, 1],
            ),
            "pre_matched_indices": (
                pre_matched_indices[:, 0],
                pre_matched_indices[:, 1],
            ),
        }
        pred_instances.update(
            {
                "obj_idxes": outputs["obj_idxes"],
                "matched_gt_idxes": outputs["matched_gt_idxes"],
            }
        )

        if aux_outputs is not None:
            output_for_losses["aux_outputs_for_loss"] = {}

            for i, aux_output in enumerate(aux_outputs):
                # update the output yaws
                if self.use_psc_rot:
                    aux_output["pred_yaws"] = (
                        2 * torch.sigmoid(aux_output["pred_yaws"]) - 1
                    )
                else:
                    aux_output["pred_yaws"] = F.normalize(
                        aux_output["pred_yaws"], dim=-1
                    )
                if self.match_indices_each_layer:
                    if self.enable_rfs:
                        all_aux_preds = {
                            "pred_logits": aux_output["pred_logits"],
                            "pred_boxes": aux_output["pred_boxes"],
                            "pred_yaws": aux_output["pred_yaws"],
                        }
                    else:
                        all_aux_preds = {
                            "pred_logits": (
                                aux_output["pred_logits"][
                                    unmatch_pred_trk_idxs
                                ]
                            ),
                            "pred_boxes": (
                                aux_output["pred_boxes"][unmatch_pred_trk_idxs]
                            ),
                            "pred_yaws": (
                                aux_output["pred_yaws"][unmatch_pred_trk_idxs]
                            ),
                        }
                    aux_new_matched_indices = match_for_single_decoder_layer(
                        all_aux_preds,
                        aux_supervise=self.enable_rfs,
                    )
                    all_matched_indices = torch.cat(
                        [aux_new_matched_indices, pre_matched_indices], dim=0
                    )

                out_loss_key = (
                    f"frame_{self._current_frame_idx}_aux_output_layer_{i}"
                )
                aux_output = {
                    "pred_logits": aux_output["pred_logits"],
                    # num_queries x num_classes
                    "pred_boxes": aux_output["pred_boxes"],
                    "pred_yaws": aux_output["pred_yaws"],
                    "pred_zheights": aux_output["pred_zheights"],
                    "matched_gt_idxes": outputs["matched_gt_idxes"],
                }
                output_for_losses["aux_outputs_for_loss"][out_loss_key] = {
                    "pred_output": aux_output,
                    "matched_ious": ious,
                    "gt_instances": gt_instances_cur,
                    "match_indices": (
                        all_matched_indices[:, 0],
                        all_matched_indices[:, 1],
                    ),
                }
        return pred_instances, output_for_losses
