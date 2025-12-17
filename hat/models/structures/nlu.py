# Copyright (c) Horizon Robotics. All rights reserved.

import logging

import torch
import torch.nn as nn

from hat.core.nlu.nlu_utils import get_batch_ner_label
from hat.registry import OBJECT_REGISTRY

__all__ = ["NluModel"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class NluModel(nn.Module):
    def __init__(self, backbone, losses=None):
        super(NluModel, self).__init__()
        self.backbone = backbone
        self.losses = losses

    def forward(self, data):
        batch_query = data.get("batch_query", None)
        batch_ids = data.get("batch_ids", None)
        batch_pad_masks = data.get("batch_pad_masks", None)
        batch_domain_labs = data.get("batch_domain_labs", None)
        batch_intent_labs = data.get("batch_intent_labs", None)
        batch_ner_labs = data.get("batch_ner_labs", None)
        batch_sample_weights = data.get("batch_sample_weights", None)

        if batch_ids is None:  # deploy and compile
            input_emb = data.get("input_emb", None)
            assert input_emb is not None
        else:
            input_emb = self.backbone.embedding(batch_ids)  # (N,W,C)
            input_emb = (
                input_emb.permute(0, 2, 1).contiguous().unsqueeze(2)
            )  # (N,C,1,W)
        domain_out, intent_out, slots_out = self.backbone(input_emb)

        if batch_ids is None:  # deploy and compile
            return (domain_out, intent_out, slots_out)

        domain_pred_id = torch.argmax(domain_out, dim=1)  # torch.tensor (N)
        intent_pred_id = torch.argmax(intent_out, dim=1)  # torch.tensor (N)
        # slots decode
        ner_tag_all_id = self.backbone.crf.decode(slots_out)  # list (N,W-2)
        valid_lengths = batch_pad_masks.sum(dim=1).tolist()
        ner_pred_id = [
            ner_tag_all_id[i][: valid_lengths[i]]
            for i in range(len(ner_tag_all_id))
        ]

        # compute loss
        domain_loss = self.losses(domain_out, batch_domain_labs)
        weighted_domain_loss = torch.sum(
            domain_loss * batch_sample_weights
        ) / torch.sum(batch_sample_weights)
        intent_loss = self.losses(intent_out, batch_intent_labs)
        weighted_intent_loss = torch.sum(
            intent_loss * batch_sample_weights
        ) / torch.sum(batch_sample_weights)
        ner_loss = -self.backbone.crf(
            slots_out,
            batch_ner_labs,
            mask=batch_pad_masks,
            reduction="none",
        )
        weighted_ner_loss = torch.sum(
            ner_loss * batch_sample_weights / torch.sum(batch_pad_masks, dim=1)
        ) / torch.sum(batch_sample_weights)
        total_loss = (
            weighted_domain_loss + weighted_intent_loss + weighted_ner_loss
        )

        # reformat tag for metrics computing
        cur_domain_pred = [
            self.backbone.label_processor.index_2_domain[i]
            for i in domain_pred_id.tolist()
        ]
        cur_intent_pred = [
            self.backbone.label_processor.index_2_intent[i]
            for i in intent_pred_id.tolist()
        ]
        cur_domain_label = [
            self.backbone.label_processor.index_2_domain[i]
            for i in batch_domain_labs.tolist()
        ]
        cur_intent_label = [
            self.backbone.label_processor.index_2_intent[i]
            for i in batch_intent_labs.tolist()
        ]
        cur_slots_pred = [
            [self.backbone.label_processor.index_2_slots[item] for item in seq]
            for seq in ner_pred_id
        ]
        cur_slots_label = get_batch_ner_label(
            batch_ner_labs.tolist(),
            batch_pad_masks.sum(dim=1).tolist(),
            self.backbone.label_processor.index_2_slots,
        )

        preds = (
            cur_domain_pred,
            cur_intent_pred,
            cur_slots_pred,
            batch_query,
            total_loss,
        )
        targets = (cur_domain_label, cur_intent_label, cur_slots_label)

        return preds, targets, total_loss

    def fuse_model(self):
        for module in [self.backbone, self.losses]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    # 需要父模块实现 set_qconfig 方法
    def set_qconfig(self):
        from hat.utils import qconfig_manager

        # 1. 首先指定父模块的 qconfig,
        # 如果未对子模块设置 qconfig，子模块会自动使用父模块的 qconfig
        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        # 2. 如果有某子模块有特殊 layer，实现了 set_qconfig 方法，调用
        if self.backbone is not None:
            if hasattr(self.backbone, "set_qconfig"):
                self.backbone.set_qconfig()

        # 3. 如果有子模块不需要设置 Qconfig，需要设置 Qconfig 为 None
        if self.losses is not None:
            self.losses.qconfig = None

    def set_calibration_qconfig(self):
        # 设置 Loss 的 qconfig 为 None，就会不再对 Loss 做 Calibration，
        # 可以一定程度减少统计量，提升 Calibration 速度，降低显存占用
        if self.losses is not None:
            self.losses.qconfig = None
