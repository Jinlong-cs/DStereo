# Copyright (c) Horizon Robotics. All rights reserved.
import copy
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import reduce_mean
from .target import COS_YAW, SIN_YAW

__all__ = ["Sparse4DHead"]


@OBJECT_REGISTRY.register
class Sparse4DHead(nn.Module):
    def __init__(
        self,
        instance_bank: nn.Module,
        anchor_encoder: nn.Module,
        instance_interaction: nn.Module,
        norm_layer: nn.Module,
        ffn: nn.Module,
        deformable_model: nn.Module,
        refine_layer: nn.Module,
        temp_instance_interaction: Optional[nn.Module] = None,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        loss_cls: Optional[nn.Module] = None,
        loss_reg: Optional[nn.Module] = None,
        loss_cns: Optional[nn.Module] = None,
        loss_yns: Optional[nn.Module] = None,
        decoder: Optional[Callable] = None,
        target: Optional[Callable] = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        reg_weights: Optional[List[float]] = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1.0,
        dn_loss_weight: float = 5.0,
    ):
        super(Sparse4DHead, self).__init__()
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.reg_weights = [1.0] * 10 if reg_weights is None else reg_weights
        if operation_order is None:
            operation_order = [
                "interaction",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        self.op_map = {
            "temp_interaction": [temp_instance_interaction, 0],
            "interaction": [instance_interaction, 0],
            "norm": [norm_layer, 0],
            "ffn": [ffn, 0],
            "deformable": [deformable_model, 0],
            "refine": [refine_layer, 0],
        }
        self.layers = nn.ModuleList()
        for op in self.operation_order:
            if op not in self.op_map:
                self.layers.append(None)
            elif self.op_map[op][1] == 0:
                self.layers.append(self.op_map[op][0])
            else:
                self.layers.append(copy.deepcopy(self.op_map[op][0]))
            self.op_map[op][1] += 1
        self.instance_bank = instance_bank
        self.anchor_encoder = anchor_encoder
        self.target = target
        self.decoder = decoder
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.loss_cns = loss_cns
        self.loss_yns = loss_yns
        self.init_weights()

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    @autocast(enabled=False)
    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
        feature_queue: Optional[List[List[torch.Tensor]]] = None,
        meta_queue: Optional[List[dict]] = None,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        if isinstance(feature_maps[0], Sequence):
            batch_size = feature_maps[0][0].shape[0]
        else:
            batch_size = feature_maps[0].shape[0]
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(batch_size, metas)

        # prepare for denosing training
        attn_mask = None
        dn_metas = None
        if self.training and hasattr(self.target, "get_dn_anchors"):
            dn_metas = self.target.get_dn_anchors(*self.parse_labels(metas))
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        # generate anchor embed
        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # prepare feature and meta queue for temporal fusion of sparse4dv1
        _feature_queue = self.instance_bank.feature_queue
        _meta_queue = self.instance_bank.meta_queue
        if feature_queue is not None and _feature_queue is not None:
            feature_queue = feature_queue + _feature_queue
            meta_queue = meta_queue + _meta_queue
        elif feature_queue is None:
            feature_queue = _feature_queue
            meta_queue = _meta_queue

        # forward layers
        prediction = []
        classification = []
        centerness = []
        yawness = []
        for i, op in enumerate(self.operation_order):
            if op == "temp_interaction":
                instance_feature = self.layers[i](
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask
                    if temp_instance_feature is None
                    else None,
                )
            elif op == "interaction":
                instance_feature = self.layers[i](
                    instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                    feature_queue=feature_queue,
                    meta_queue=meta_queue,
                    anchor_encoder=self.anchor_encoder,
                )
            elif op == "refine":
                anchor, cls, cn, yn = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                centerness.append(cn)
                yawness.append(yn)
                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}
        if dn_metas is not None:
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            centerness = [
                x[:, :num_free_instance] if x is not None else None
                for x in centerness
            ]
            yawness = [
                x[:, :num_free_instance] if x is not None else None
                for x in yawness
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "centerness": centerness,
                "yawness": yawness,
            }
        )

        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        return output

    def parse_labels(self, data):
        gt_cls = []
        gt_reg = []
        if "gt_ignore" in data:
            gt_ignore = []
        else:
            gt_ignore = None
        for i, (cls, reg) in enumerate(
            zip(data[self.gt_cls_key], data[self.gt_reg_key])
        ):
            valid_mask = cls >= 0
            gt_cls.append(cls[valid_mask])
            gt_reg.append(reg[valid_mask])
            if gt_ignore is not None:
                gt_ignore.append(data["gt_ignore"][i][valid_mask])
        return gt_cls, gt_reg, gt_ignore

    @autocast(enabled=False)
    def loss(self, model_outs, data, feature_maps=None):
        gt_cls, gt_reg, gt_ignore = self.parse_labels(data)

        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        centerness = model_outs["centerness"]
        yawness = model_outs["yawness"]
        output = {}
        for decoder_idx, (cls, reg, cns, yns) in enumerate(
            zip(cls_scores, reg_preds, centerness, yawness)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, cls_weights, reg_weights = self.target(
                cls,
                reg,
                gt_cls,
                gt_reg,
                gt_ignore,
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask = torch.logical_and(mask, cls_weights != 0)

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weights = cls_weights.flatten(end_dim=1)
            cls_loss = self.loss_cls(
                cls, cls_target, avg_factor=num_pos, weight=cls_weights
            )

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            reg_loss = self.loss_reg(
                reg, reg_target, weight=reg_weights, avg_factor=num_pos
            )

            cls_loss = (
                sum(cls_loss.values())
                if isinstance(cls_loss, dict)
                else cls_loss
            )
            reg_loss = (
                sum(reg_loss.values())
                if isinstance(reg_loss, dict)
                else reg_loss
            )
            output.update(
                {
                    f"loss_cls_{decoder_idx}": cls_loss,
                    f"loss_reg_{decoder_idx}": reg_loss,
                }
            )
            if cns is not None:
                cns = cns.flatten()[mask]
                cns_target = torch.norm(
                    reg_target[..., :3] - reg[..., :3], p=2, dim=-1
                )
                cns_target = torch.exp(-cns_target)
                cns_loss = self.loss_cns(cns, cns_target, avg_factor=num_pos)
                output[f"loss_cns_{decoder_idx}"] = cns_loss

            if yns is not None:
                yns = yns.flatten()[mask].sigmoid()
                yns_target = (
                    torch.nn.functional.cosine_similarity(
                        reg_target[..., [SIN_YAW, COS_YAW]],
                        reg[..., [SIN_YAW, COS_YAW]],
                        dim=-1,
                    )
                    > 0
                )
                yns_target = yns_target.float()
                yns_loss = self.loss_yns(yns, yns_target)
                output[f"loss_yns_{decoder_idx}"] = yns_loss

        if "dn_prediction" not in model_outs:
            return output

        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]
        dn_valid_mask = model_outs["dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs["dn_cls_target"].flatten(end_dim=1)[
            dn_valid_mask
        ]
        dn_reg_target = model_outs["dn_reg_target"].flatten(end_dim=1)[
            dn_valid_mask
        ][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg.dtype)), 1.0
        )
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
            )
            cls_loss = (
                sum(cls_loss.values())
                if isinstance(cls_loss, dict)
                else cls_loss
            ) * self.dn_loss_weight
            reg_loss = (
                sum(reg_loss.values())
                if isinstance(reg_loss, dict)
                else reg_loss
            ) * self.dn_loss_weight

            output.update(
                {
                    f"dn_loss_cls_{decoder_idx}": cls_loss,
                    f"dn_loss_reg_{decoder_idx}": reg_loss,
                }
            )
        return output

    def post_process(self, model_outs):
        return self.decoder(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("centerness"),
        )
