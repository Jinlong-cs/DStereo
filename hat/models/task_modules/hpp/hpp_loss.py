import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class HppLoss(nn.Module):
    def __init__(
        self,
        ins_embedding_channel: int,
        weight_offset: float,
        weight_exist: float,
        weight_nonexist: float,
        weight_attention: float,
        weight_sisc: float,
        eps: float = 0.01,
    ):
        """Compute hpp loss.

        Args:
            ins_embedding_channel (int): the channel of instance embedding
            weight_offset (float): the weight offset loss
            weight_exist (float): the weight exist loss
            weight_nonexist (float): the weight nonexist loss
            weight_attention (float): the weight attention loss
            weight_sisc (float): the weight sisc loss
            eps (float, optional): the balance ratio between positive
                and negative samples. Defaults to 0.01.
        """
        super(HppLoss, self).__init__()
        self.ins_embedding_channel = ins_embedding_channel
        self.weight_offset = weight_offset
        self.weight_exist = weight_exist
        self.weight_nonexist = weight_nonexist
        self.weight_attention = weight_attention
        self.weight_sisc = weight_sisc
        self.eps = eps

    def forward(self, preds, targets):
        if not isinstance(targets, dict):
            raise TypeError("targets should be dict type.")
        if not isinstance(preds, dict):
            raise TypeError("preds should be dict type.")
        ground_truth_point, ground_truth_instance = targets["labels"]
        grid_h, grid_w = ground_truth_point.shape[-2:]
        confidences = preds["confidences"]
        offsets = preds["offsets"]
        inst_embeddings = preds["instances"]
        batch_size = len(ground_truth_point)
        exist_condidence_loss = 0
        nonexist_confidence_loss = 0
        offset_loss = 0
        sisc_loss = 0

        for (confidence, offset, ins_embedding) in zip(
            confidences, offsets, inst_embeddings
        ):
            confidence_gt = ground_truth_point[:, 0:1, :, :]
            # exist confidence loss
            # 1: foreground/0: background
            exist_condidence_loss = exist_condidence_loss + torch.sum(
                (1 - confidence[confidence_gt == 1]) ** 2
            ) / torch.sum(confidence_gt == 1)
            # non exist confidence loss
            neg_cls_target = confidence[confidence_gt == 0]
            # to avoid imbalance between neg samples and pos samples,
            # nonexist_confidence_loss just compute conf that higher than eps
            nonexist_confidence_loss = nonexist_confidence_loss + torch.sum(
                (neg_cls_target[neg_cls_target > self.eps]) ** 2
            ) / (torch.sum(neg_cls_target > self.eps) + 1)
            # offset loss
            offset_x_gt = ground_truth_point[:, 1:2, :, :]
            offset_y_gt = ground_truth_point[:, 2:3, :, :]
            offset_x_pred = offset[:, 0:1, :, :]
            offset_y_pred = offset[:, 1:2, :, :]
            offset_loss = (
                offset_loss
                + torch.sum(
                    (
                        offset_x_gt[confidence_gt == 1]
                        - offset_x_pred[confidence_gt == 1]
                    )
                    ** 2
                )
                / torch.sum(confidence_gt == 1)
                + torch.sum(
                    (
                        offset_y_gt[confidence_gt == 1]
                        - offset_y_pred[confidence_gt == 1]
                    )
                    ** 2
                )
                / torch.sum(confidence_gt == 1)
            )
            # compute loss for similarity
            # 1. give the instance embedding:[[a, b], [c, d]]
            # 2. reshape to [a, b, c, d]
            # 3. stack reshape embedding at row and column, e.g.
            # ins_embedding_row: [[a, b, c, d], [a, b, c, d], [a, b, c, d], ...]  # noqa
            # ins_embedding_col: [[a, a, a, a], [b, b, b, b], [c, c, c, c], ...]  # noqa
            ins_embedding_row_temp = ins_embedding.view(
                batch_size,
                self.ins_embedding_channel,
                1,
                grid_h * grid_w,
            )
            ins_embedding_row = ins_embedding_row_temp.expand(
                batch_size,
                self.ins_embedding_channel,
                grid_h * grid_w,
                grid_h * grid_w,
            )
            ins_embedding_col_temp = ins_embedding.view(
                batch_size,
                self.ins_embedding_channel,
                grid_h * grid_w,
                1,
            )
            ins_embedding_col = ins_embedding_col_temp.expand(
                batch_size,
                self.ins_embedding_channel,
                grid_h * grid_w,
                grid_h * grid_w,
            )
            distance_map = (ins_embedding_row - ins_embedding_col) ** 2
            distance_map = torch.sum(distance_map, dim=1).view(
                batch_size,
                1,
                grid_h * grid_w,
                grid_h * grid_w,
            )
            sisc_loss = sisc_loss + torch.sum(
                distance_map[ground_truth_instance == 1]
            ) / torch.sum(ground_truth_instance == 1)

        # attention loss
        att_feats = preds["att_feats"]
        attention_loss = 0
        att_feat_first_blocks = att_feats[:-1]
        for i in range(batch_size):
            # original code use att_feats[-1][i].data,
            # which is dangerous and can make wrong computation, following by
            # https://discuss.pytorch.org/t/the-difference-between-torch-tensor-data-and-torch-tensor/25995/4  # noqa
            with torch.no_grad():
                # the deepest hourglass module can be a teacher network
                att_target = torch.sum((att_feats[-1][i]) ** 2, dim=0).view(-1)
            att_target = F.softmax(att_target, dim=0)
            for att_feat_first in att_feat_first_blocks:
                att_feat_first_temp = torch.sum(
                    att_feat_first[i] ** 2, dim=0
                ).view(-1)
                att_feat_first_soft = F.softmax(att_feat_first_temp, dim=0)
                attention_loss = attention_loss + torch.sum(
                    (att_feat_first_soft - att_target) ** 2
                ) / (len(att_target) * batch_size)

        result_dict = {}
        result_dict["hpp_exist_loss"] = (
            self.weight_exist * exist_condidence_loss
        )
        result_dict["hpp_nonexist_loss"] = (
            self.weight_nonexist * nonexist_confidence_loss
        )
        result_dict["hpp_offset_loss"] = self.weight_offset * offset_loss
        result_dict["hpp_sisc_loss"] = self.weight_sisc * sisc_loss
        result_dict["hpp_attention_loss"] = (
            self.weight_attention * attention_loss
        )

        return result_dict
