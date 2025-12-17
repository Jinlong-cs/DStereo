from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from hat.core.box_utils import zoom_boxes
from hat.registry import OBJECT_REGISTRY
from hat.utils.tensor_func import take_row


@OBJECT_REGISTRY.register
class HumanPoseLabelFromMatch(nn.Module):
    """Human landmark detection label encoder.

    This class is used to generate target based on gt
    landmark from the matching box. Generally
    used in anchor-based method like frcnn.

    Args:
        target_h_out: Height of the output target.
        target_w_out: Width of the output target.
        ldmk_num: Number of landmark.
        pos_distance: Distance ratio.
        ignore_labels: GT labels of landmark which need to be ignored.
        {'invisible': 0, 'occluded': 1, 'full_visible': 2, 'ignore': 3}
        ldmk_loss_type: Landmark loss type.
        Now it has two types. "cross_entropy" and "smooth_L1".
        pos_distance_scale: Distance scale. Defaults to 1.0.
        unbiased_encode: Whether use unbiased encode. Defaults to True.
        roi_expand_param: Expand ratio of roi . Defaults to 1.0.
        gauss_threshold: Sigma value of gauss function. Defaults to None.
        If it is None, that means use GRMI encoder instead of gauss encoder.
        keep_outside_box: Whether keep landmark outside box. Defaults to False.
        train_on_unvis: Whether train unvisiable landmark. Defaults to False.
    """

    def __init__(
        self,
        target_h_out: int,
        target_w_out: int,
        ldmk_num: int,
        pos_distance: float,
        ignore_labels: Tuple[int],
        ldmk_loss_type: str,
        pos_distance_scale: float = 1.0,
        unbiased_encode: bool = True,
        roi_expand_param: Optional[float] = 1.0,
        gauss_threshold: Optional[float] = None,
        keep_outside_box: bool = False,
        train_on_unvis: bool = False,
    ):
        assert isinstance(ignore_labels, tuple)
        super().__init__()
        self.target_h_out = target_h_out
        self.target_w_out = target_w_out
        self.ldmk_num = ldmk_num
        self.keep_outside_box = keep_outside_box
        self.unbiased_encode = unbiased_encode

        # init bin map
        self.bin_x_int = torch.arange(0, self.target_h_out)
        self.bin_y_int = torch.arange(0, self.target_w_out)
        self.bin_y_int, self.bin_x_int = torch.meshgrid(
            self.bin_y_int, self.bin_x_int
        )
        self.bin_x_int = self.bin_x_int.flatten()
        self.bin_y_int = self.bin_y_int.flatten()
        self.pos_distance = pos_distance * target_w_out
        self.pos_distance_scale = pos_distance_scale
        self.train_on_unvis = train_on_unvis
        self.ldmk_loss_type = ldmk_loss_type

        self.expand_param = roi_expand_param
        self.gauss_threshold = gauss_threshold
        self.ignore_labels = ignore_labels

    def filter_ldmk(self, landmark, ldmk_xy_int):
        # landmark's shape (:, 3)
        # {'invisible': 0, 'occluded': 1, 'full_visible': 2, 'ignore': 3}
        valid_mask = torch.ones(landmark[:, 2].shape).to(
            device=landmark.device
        )
        for ignore_label in self.ignore_labels:
            valid_mask = torch.logical_and(
                valid_mask, landmark[:, 2] != ignore_label
            )

        if not self.keep_outside_box:
            within_box = torch.logical_and(
                torch.logical_and(
                    ldmk_xy_int[:, 0] >= 0, ldmk_xy_int[:, 1] >= 0
                ),
                torch.logical_and(
                    ldmk_xy_int[:, 0] < self.target_w_out,
                    ldmk_xy_int[:, 1] < self.target_h_out,
                ),
            )
            valid_mask = torch.logical_and(within_box, valid_mask)
        return valid_mask

    def forward(
        self,
        boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        match_pos_flag: torch.Tensor,
        match_gt_id: torch.Tensor,
        **kwargs
    ):
        """Forward.

        The idea of top-down landmark detection approach is
        adopted here.

        Args:
            boxes: (B, N, 4), batched predicted boxes
            gt_boxes: (B, M, 5+), batched ground truth boxes,
                might be padded if gt_box is different in each sample.
            match_pos_flag: (B, N), matched result of each predicted
                box, Entries with value 1 represents positive
                in matching, 0 for neg and -1 for ignore.
            match_gt_id: (B, N), matched gt box index
                of each predicted box
        """
        self.bin_x_int = self.bin_x_int.to(device=boxes.device)
        self.bin_y_int = self.bin_y_int.to(device=boxes.device)

        pos_num_boxes = torch.sum(match_pos_flag > 0, dim=1)

        matched_gt_boxes = take_row(gt_boxes, match_gt_id)

        batch_size_per_ctx, proposal_num = boxes.shape[:2]

        pos_match_label = (match_pos_flag > 0).float()
        pos_match_label = torch.repeat_interleave(
            pos_match_label, repeats=self.ldmk_num, axis=1
        )

        cls_labels = boxes.new_full(
            (
                batch_size_per_ctx,
                proposal_num,
                self.ldmk_num,
                self.target_h_out,
                self.target_w_out,
            ),
            fill_value=-1,
        )
        cls_label_weight = torch.zeros_like(cls_labels)

        reg_offset = boxes.new_zeros(
            (
                batch_size_per_ctx,
                proposal_num,
                self.ldmk_num * 2,
                self.target_h_out,
                self.target_w_out,
            )
        )
        reg_offset_weight = torch.zeros_like(reg_offset)

        for i in range(batch_size_per_ctx):
            pos_indices = torch.where(pos_match_label[i] == 1)[0]
            if len(pos_indices) == 0:
                continue

            # pos_num_boxes, 4
            box = zoom_boxes(boxes[i], (self.expand_param, self.expand_param))
            pos_gt_roi_boxes = matched_gt_boxes[i]

            # pos_num_boxes, self.ldmk_num * 3
            landmark = pos_gt_roi_boxes[:, 5 : 5 + self.ldmk_num * 3]

            num_boxes = box.shape[0]
            landmark = landmark.reshape((num_boxes * self.ldmk_num, 3))

            scales_xy = box.new_zeros((num_boxes, 2))
            scales_xy[:, 0] = self.target_w_out / (box[:, 2] - box[:, 0] + 1)
            scales_xy[:, 1] = self.target_h_out / (box[:, 3] - box[:, 1] + 1)
            # (num_boxes， num_ldmk*2)
            scales_xy = torch.tile(scales_xy, (1, self.ldmk_num))
            # (num_boxes * num_ldmk, 2)
            scales_xy = scales_xy.reshape((num_boxes * self.ldmk_num, 2))
            # (num_boxes, 2)
            offsets_xy = box[:, :2]
            # (num_boxes, num_ldmk*2)
            offsets_xy = torch.tile(offsets_xy, (1, self.ldmk_num))
            # (num_boxes * num_ldmk, 2)
            offsets_xy = offsets_xy.reshape((num_boxes * self.ldmk_num, 2))
            # (num_boxes * num_ldmk, 2)
            ldmk_xy = (landmark[:, :2] - offsets_xy) * scales_xy
            ldmk_xy_int = torch.floor(ldmk_xy)
            # (num_boxes * num_ldmk, 2)
            vis = self.filter_ldmk(landmark, ldmk_xy_int)
            keep = torch.where(vis == 1)[0]
            keep_copy = keep.cpu().detach().numpy()
            pos_indices_copy = pos_indices.cpu().detach().numpy()
            keep = list(set(keep_copy) & set(pos_indices_copy))

            keep = [box.new_tensor(i).long() for i in keep]
            ldmk_label = box.new_full(
                (
                    num_boxes * self.ldmk_num,
                    self.target_h_out * self.target_w_out,
                ),
                fill_value=-1,
            )  # noqa

            ldmk_label_weight = torch.zeros_like(ldmk_label)
            ldmk_pos_offset = box.new_zeros(
                (
                    num_boxes * self.ldmk_num,
                    2,
                    self.target_h_out * self.target_w_out,
                ),
            )
            ldmk_pos_offset_weight = torch.zeros_like(ldmk_pos_offset)

            if self.train_on_unvis:
                unvis = torch.where(vis == 0)[0]
                unvis_copy = unvis.cpu().detach().numpy()
                unvis = list(set(unvis_copy) & set(pos_indices_copy))
                for keep_i in unvis:
                    ldmk_label_weight[keep_i] = 1.0
                    ldmk_pos_offset_weight[keep_i] = 0.0

            if len(keep) > 0:
                num_keep_per_box = float(len(keep)) / pos_num_boxes[i].item()
                num_keep_pos = 0
                for keep_i in keep:
                    if self.gauss_threshold is not None:
                        if self.unbiased_encode:
                            pos_offset_x = ldmk_xy[keep_i, 0] - self.bin_x_int
                            pos_offset_y = ldmk_xy[keep_i, 1] - self.bin_y_int
                        else:
                            pos_offset_x = (
                                ldmk_xy_int[keep_i, 0] - self.bin_x_int
                            )
                            pos_offset_y = (
                                ldmk_xy_int[keep_i, 1] - self.bin_y_int
                            )
                        sigma = self.gauss_threshold
                        thre = (3 * sigma) ** 2
                        dis = pos_offset_x ** 2 + pos_offset_y ** 2
                        keep_pos = torch.where((dis <= thre) & (dis >= 0))[0]
                    else:
                        pos_offset_x = ldmk_xy[keep_i, 0] - self.bin_x_int
                        pos_offset_y = ldmk_xy[keep_i, 1] - self.bin_y_int
                        pos_offset_x /= self.pos_distance
                        pos_offset_y /= self.pos_distance
                        dis = pos_offset_x ** 2 + pos_offset_y ** 2
                        keep_pos = torch.where((dis <= 1) & (dis >= 0))[0]
                    if len(keep_pos) > 0:
                        ldmk_label[keep_i] = 0
                        ldmk_label_weight[keep_i] = 1.0
                        if self.gauss_threshold is not None:
                            ldmk_label[keep_i, keep_pos] = torch.exp(
                                dis[keep_pos] / (-2.0 * sigma * sigma)
                            )
                            if not self.unbiased_encode:
                                ldmk_label[
                                    keep_i, torch.where(dis == 0)[0]
                                ] += 1
                        else:
                            ldmk_label[keep_i, keep_pos] = 1
                        ldmk_pos_offset[keep_i, 0, keep_pos] = (
                            pos_offset_x[keep_pos] * self.pos_distance_scale
                        )
                        ldmk_pos_offset[keep_i, 1, keep_pos] = (
                            pos_offset_y[keep_pos] * self.pos_distance_scale
                        )
                        ldmk_pos_offset_weight[keep_i, 0, keep_pos] = 1.0
                        ldmk_pos_offset_weight[keep_i, 1, keep_pos] = 1.0
                        num_keep_pos += len(keep_pos)

                if num_keep_pos > 0:
                    if self.ldmk_loss_type == "cross_entropy":
                        ldmk_label_weight[ldmk_label_weight > 0] = (
                            1.0 / num_keep_pos
                        )
                    elif self.ldmk_loss_type == "smooth_L1":
                        ldmk_label_weight[ldmk_label_weight > 0] = (
                            num_keep_per_box / num_keep_pos
                        )
                    else:
                        raise ValueError(
                            "unknown kps loss type {}".format(
                                self.ldmk_loss_type
                            )
                        )
                    ldmk_pos_offset_weight[ldmk_pos_offset_weight > 0] = (
                        num_keep_per_box / num_keep_pos
                    )

            ldmk_label = ldmk_label.view(
                (
                    num_boxes,
                    self.ldmk_num,
                    self.target_h_out,
                    self.target_w_out,
                )
            )
            ldmk_label_weight = ldmk_label_weight.view_as(ldmk_label)

            ldmk_pos_offset = ldmk_pos_offset.view(
                (
                    num_boxes,
                    self.ldmk_num * 2,
                    self.target_h_out,
                    self.target_w_out,
                )
            )
            ldmk_pos_offset_weight = ldmk_pos_offset_weight.view_as(
                ldmk_pos_offset
            )

            cls_labels[i] = ldmk_label
            cls_label_weight[i] = ldmk_label_weight
            reg_offset[i] = ldmk_pos_offset
            reg_offset_weight[i] = ldmk_pos_offset_weight

        return OrderedDict(
            ldmk_cls_label=cls_labels,
            ldmk_cls_label_weight=cls_label_weight,
            ldmk_reg_label=reg_offset,
            ldmk_reg_label_weight=reg_offset_weight,
        )
