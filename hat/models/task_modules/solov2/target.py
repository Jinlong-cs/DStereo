# Copyright (c) Horizon Robotics. All rights reserved.
# Source code reference to mmdetection

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import multi_apply
from .utils import center_of_mass, dynamic_upsample


@OBJECT_REGISTRY.register
class SOLOV2Target(nn.Module):
    """Generate target for solov2 head.

    Args:
        num_classes: Number of categories excluding the background category.
        scale_ranges: Area range of multiple level masks, in the format
            [(min1, max1), (min2, max2), ...]. A range of (16, 64) means the
            area range between (16, 64).
        pos_scale: Constant scale factor to control the center region.
            Default: 0.2.
        num_grids: Divided image into a uniform grids, each feature map has
            a different grid value. The number of output channels is grid ** 2.
            Default: [40, 36, 24, 16, 12].
        mask_stride: Downsample factor of the mask feature map output.
            Default: 4.
        upsample_mask_logit: Learning the sampling factors for upsampling the
            mask logits to stride = 1.
        dynamic_conv_size: Dynamic Conv kernel size. Default: 1.
        ignore_index: Ignore index for attribute classification.
        attr_name2num: Instance attributes to be predicted in parallel with
            classification.
    """

    def __init__(
        self,
        num_classes: int,
        scale_ranges: Tuple[Tuple[int, int]] = (
            (1, 96),
            (48, 192),
            (96, 384),
            (192, 768),
            (384, 2048),
        ),
        pos_scale: float = 0.2,
        num_grids: Sequence[int] = (40, 36, 24, 16, 12),
        mask_stride: int = 4,
        upsample_mask_logit: bool = False,
        dynamic_conv_size: int = 1,
        ignore_index: int = 255,
        attr_name2num: Optional[Dict] = None,
        max_pos_num: Optional[int] = None,
        use_ignore_region: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.scale_ranges = scale_ranges
        self.num_grids = num_grids
        self.pos_scale = pos_scale
        self.mask_stride = mask_stride
        self.upsample_mask_logit = upsample_mask_logit
        self.dynamic_conv_size = dynamic_conv_size
        self.ignore_index = ignore_index
        self.attr_name2num = attr_name2num if attr_name2num else {}
        self.attr_names = list(attr_name2num.keys())
        self.attribute_num = len(attr_name2num)
        self.max_pos_num = max_pos_num
        self.use_ignore_region = use_ignore_region

    def _get_targets_single(
        self,
        gt_bboxes,
        gt_labels,
        gt_masks,
        gt_attrs,
        gt_ignore_region,
        featmap_size=None,
    ):
        """Compute targets for predictions of single image.

        Args:
            gt_bboxes (Tensor): Ground truth bbox of each instance,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth label of each instance,
                shape (num_gts,).
            gt_masks (Tensor): Ground truth mask of each instance,
                shape (num_gts, h, w).
            gt_attrs (Tensor): Ground truth attributeds of each instance,
                shape (num_gts, attribute_number).
            featmap_sizes (:obj:`torch.size`): Size of UNified mask
                feature map used to generate instance segmentation
                masks by dynamic convolution, each element means
                (feat_h, feat_w). Default: None.

        Returns:
            Tuple: Usually returns a tuple containing targets for predictions.
                - mlvl_pos_mask_targets (list[Tensor]): Each element represent
                  the binary mask targets for positive points in this
                  level, has shape (pos_num, out_h, out_w).
                - mlvl_labels (list[Tensor]): Each element is one-hot
                  classification labels for all points in this level,
                  has shape (num_grid, num_grid).
                - mlvl_attr_labels (list[Tensor]): Each element is
                  attribute labels for all points in this level, has shape
                  (num_grid, num_grid, attribute_number).
                - mlvl_pos_indexes  (list[list]): Each element
                  in the list contains the positive index in
                  corresponding level, has shape (pos_num).
        """
        num_inst = len(gt_labels)
        if num_inst == 0:
            logging.warning("num_inst = 0")
            gt_areas = torch.zeros(1, device=self.device)
        else:
            gt_areas = torch.sqrt(
                (gt_bboxes[:, 2] - gt_bboxes[:, 0])
                * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            )

        mlvl_pos_mask_targets = []
        mlvl_pos_indexes = []
        mlvl_cls_labels = []
        mlvl_attr_labels = []

        if len(gt_masks) == 0:
            scale_range_factor = 1.0
        else:
            scale_range_factor = np.sqrt(
                (gt_masks.shape[1] * gt_masks.shape[2]) / (640.0 * 1024.0)
            )
        for (_lower_bound, _upper_bound), num_grid in zip(
            self.scale_ranges, self.num_grids
        ):
            lower_bound = int(_lower_bound * scale_range_factor)
            upper_bound = int(_upper_bound * scale_range_factor)
            mask_target = []

            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            pos_index = []
            cls_labels = torch.zeros(
                [num_grid, num_grid, self.num_classes],
                dtype=torch.int64,
                device=self.device,
            )
            if self.use_ignore_region:
                gt_ignore_region_resized = (
                    F.interpolate(
                        gt_ignore_region.unsqueeze(0).unsqueeze(0),
                        size=(num_grid, num_grid),
                        mode="bilinear",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
                cls_labels[gt_ignore_region_resized > 0.5] = self.ignore_index

            attr_labels = (
                torch.ones(
                    [num_grid, num_grid, self.attribute_num],
                    dtype=torch.int64,
                    device=self.device,
                )
                * self.ignore_index
            )
            pos_flag = torch.zeros(
                [num_grid, num_grid],
                dtype=torch.int64,
                device=self.device,
            )

            gt_inds = (
                ((gt_areas >= lower_bound) & (gt_areas <= upper_bound))
                .nonzero()
                .flatten()
            )
            if len(gt_inds) == 0:
                mlvl_pos_mask_targets.append(
                    torch.zeros(
                        [
                            0,
                            featmap_size[0] * self.mask_stride,
                            featmap_size[1] * self.mask_stride,
                        ],
                        dtype=torch.uint8,
                        device=self.device,
                    )
                )
                mlvl_cls_labels.append(cls_labels)
                mlvl_attr_labels.append(attr_labels)
                mlvl_pos_indexes.append([])
                continue
            hit_gt_bboxes = gt_bboxes[gt_inds]
            hit_gt_labels = gt_labels[gt_inds]
            hit_gt_masks = gt_masks[gt_inds, ...]
            hit_gt_attrs = gt_attrs[gt_inds, ...]

            pos_w_ranges = (
                0.5
                * (hit_gt_bboxes[:, 2] - hit_gt_bboxes[:, 0])
                * self.pos_scale
            )
            pos_h_ranges = (
                0.5
                * (hit_gt_bboxes[:, 3] - hit_gt_bboxes[:, 1])
                * self.pos_scale
            )

            # Make sure hit_gt_masks has a value
            valid_mask_flags = hit_gt_masks.sum(dim=-1).sum(dim=-1) > 0

            for (
                gt_mask,
                gt_attr,
                gt_label,
                pos_h_range,
                pos_w_range,
                valid_mask_flag,
            ) in zip(
                hit_gt_masks,
                hit_gt_attrs,
                hit_gt_labels,
                pos_h_ranges,
                pos_w_ranges,
                valid_mask_flags,
            ):
                if not valid_mask_flag:
                    continue
                upsampled_size = (
                    featmap_size[0] * self.mask_stride,
                    featmap_size[1] * self.mask_stride,
                )
                center_h, center_w = center_of_mass(gt_mask)

                coord_h = int(float(center_h) / upsampled_size[0] * num_grid)
                coord_w = int(float(center_w) / upsampled_size[1] * num_grid)

                # left, top, right, down
                top_box = max(
                    0,
                    int(
                        float(center_h - pos_h_range)
                        / upsampled_size[0]
                        * num_grid
                    ),
                )
                down_box = min(
                    num_grid - 1,
                    int(
                        float(center_h + pos_h_range)
                        / upsampled_size[0]
                        * num_grid
                    ),
                )

                left_box = max(
                    0,
                    int(
                        float(center_w - pos_w_range)
                        / upsampled_size[1]
                        * num_grid
                    ),
                )
                right_box = min(
                    num_grid - 1,
                    int(
                        float(center_w + pos_w_range)
                        / upsampled_size[1]
                        * num_grid
                    ),
                )

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                cls_labels[
                    top : (down + 1), left : (right + 1), int(gt_label)
                ] = 1
                attr_labels[top : (down + 1), left : (right + 1)] = gt_attr
                pos_flag[top : (down + 1), left : (right + 1)] += 1

                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        index = int(i * num_grid + j)
                        mask_target.append(gt_mask)
                        pos_index.append(index)

            if len(mask_target) > 0:
                mask_target = torch.stack(mask_target, 0)
                pos_index = np.array(pos_index)
                overlap_flag = (
                    pos_index.reshape(-1, 1) == pos_index.reshape(1, -1)
                ).sum(0) > 1
                pos_index = pos_index[~overlap_flag].tolist()
                mask_target = mask_target[~overlap_flag]

            if len(mask_target) == 0:
                mask_target = torch.zeros(
                    [
                        0,
                        featmap_size[0] * self.mask_stride,
                        featmap_size[1] * self.mask_stride,
                    ],
                    dtype=torch.uint8,
                    device=self.device,
                )

            # overlap
            cls_labels[pos_flag > 1] = 0
            attr_labels[pos_flag > 1] = self.ignore_index

            mlvl_cls_labels.append(cls_labels)
            mlvl_attr_labels.append(attr_labels)
            mlvl_pos_mask_targets.append(mask_target)
            mlvl_pos_indexes.append(pos_index)

        return (
            mlvl_pos_mask_targets,
            mlvl_cls_labels,
            mlvl_attr_labels,
            mlvl_pos_indexes,
        )

    def mask2box(self, gt_mask):
        if gt_mask.sum() == 0:
            return None, False
        inds = torch.nonzero(gt_mask == 1)
        ymin, ymax = inds[:, 0].min(), inds[:, 0].max()
        xmin, xmax = inds[:, 1].min(), inds[:, 1].max()
        gt_bbox = torch.stack([xmin, ymin, xmax, ymax])
        valid_flag = xmax > xmin and ymax > ymin
        return gt_bbox, valid_flag

    def prepare_ground_truth(self, target):
        if "gt_masks" in target:
            gt_masks = target["gt_masks"]
            bs = len(gt_masks)
            gt_classes = target["gt_classes"]
            gt_bboxes = target["gt_bboxes"]
            gt_attrs = target.get("gt_attrs", [None] * bs)
            gt_ignore_regions = target.get("gt_ignore_regions", [None] * bs)
            return gt_classes, gt_masks, gt_bboxes, gt_attrs, gt_ignore_regions

        gt_labels, gt_id_maps = target["gt_labels"], target["gt_seg"]
        gt_classes, gt_masks, gt_bboxes = [], [], []
        gt_attrs, gt_ignore_regions = [], []
        for gt_label, gt_id_map in zip(gt_labels, gt_id_maps):
            gt_classes_i = []
            gt_masks_i = []
            gt_bboxes_i = []
            gt_attrs_i = []

            inst_num = 0
            for lab in gt_label:
                index, classes_id, *attr_ids = lab
                if index == self.ignore_index:
                    continue
                gt_mask = (gt_id_map == index).float()
                gt_bbox, valid_flag = self.mask2box(gt_mask)
                if valid_flag:
                    gt_classes_i.append(classes_id)
                    gt_masks_i.append(gt_mask)
                    gt_bboxes_i.append(gt_bbox)
                    gt_attrs_i.append(attr_ids)
                    inst_num += 1
            if inst_num > 0:
                gt_classes_i = gt_id_map.new_tensor(gt_classes_i)
                gt_masks_i = torch.stack(gt_masks_i).to(self.device)
                gt_bboxes_i = torch.stack(gt_bboxes_i).to(self.device)
                gt_attrs_i = gt_id_map.new_tensor(gt_attrs_i)
            if self.use_ignore_region:
                gt_ignore_regions_i = (gt_id_map == self.ignore_index).float()
            else:
                gt_ignore_regions_i = None
            gt_classes.append(gt_classes_i)
            gt_masks.append(gt_masks_i)
            gt_bboxes.append(gt_bboxes_i)
            gt_attrs.append(gt_attrs_i)
            gt_ignore_regions.append(gt_ignore_regions_i)
        return gt_classes, gt_masks, gt_bboxes, gt_attrs, gt_ignore_regions

    def forward(self, target, pred):
        (
            mlvl_kernel_preds,
            mlvl_cls_preds,
            mlvl_attr_preds,
            mask_feats,
            upsample_factors,
        ) = pred
        self.device = mask_feats.device
        featmap_size = mask_feats.size()[-2:]

        with torch.no_grad():
            (
                gt_labels,
                gt_masks,
                gt_bboxes,
                gt_attrs,
                gt_ignore_regions,
            ) = self.prepare_ground_truth(target)

            (
                pos_mask_targets,
                cls_labels,
                attr_labels,
                pos_indexes,
            ) = multi_apply(
                self._get_targets_single,
                gt_bboxes,
                gt_labels,
                gt_masks,
                gt_attrs,
                gt_ignore_regions,
                featmap_size=featmap_size,
            )

        pos_mask_targets = [
            torch.cat(img_mask_targets, 0)
            for img_mask_targets in pos_mask_targets
        ]

        mlvl_pos_kernel_preds = []
        for lvl_kernel_preds, lvl_pos_indexes in zip(
            mlvl_kernel_preds, zip(*pos_indexes)
        ):
            lvl_pos_kernel_preds = []
            for img_lvl_kernel_preds, img_lvl_pos_indexes in zip(
                lvl_kernel_preds, lvl_pos_indexes
            ):
                img_lvl_pos_kernel_preds = img_lvl_kernel_preds.view(
                    img_lvl_kernel_preds.shape[0], -1
                )[:, img_lvl_pos_indexes]
                lvl_pos_kernel_preds.append(img_lvl_pos_kernel_preds)
            mlvl_pos_kernel_preds.append(lvl_pos_kernel_preds)

        pos_kernel_preds = [
            torch.cat(img_pos_kernel_preds, 1)
            for img_pos_kernel_preds in zip(*mlvl_pos_kernel_preds)
        ]

        pos_num = sum(
            [
                img_pos_kernel_preds.size()[-1]
                for img_pos_kernel_preds in pos_kernel_preds
            ]
        )
        pos_mask_num = pos_num
        if self.max_pos_num is not None and pos_mask_num > self.max_pos_num:
            remained_index = torch.randperm(pos_mask_num)[: self.max_pos_num]
            remained_flag = torch.zeros(pos_mask_num, dtype=torch.bool)
            remained_flag[remained_index] = True
            start_index = 0
            for img_id, img_pos_kernel_preds in enumerate(pos_kernel_preds):
                img_pos_num = img_pos_kernel_preds.size()[-1]
                if img_pos_num == 0:
                    continue
                img_remained_flag = remained_flag[
                    start_index : start_index + img_pos_num
                ]
                pos_kernel_preds[img_id] = pos_kernel_preds[img_id][
                    :, img_remained_flag
                ]
                pos_mask_targets[img_id] = pos_mask_targets[img_id][
                    img_remained_flag
                ]
                start_index += img_pos_num
            pos_mask_num = self.max_pos_num

        mask_preds = []
        for img_pos_kernel_preds, img_mask_feats, img_upsample_factor in zip(
            pos_kernel_preds, mask_feats, upsample_factors
        ):
            if img_pos_kernel_preds.size()[-1] == 0:
                img_mask_pred = None
            else:
                h, w = img_mask_feats.shape[-2:]
                num_kernel = img_pos_kernel_preds.shape[1]
                with autocast(enabled=False):
                    img_mask_pred = F.conv2d(
                        img_mask_feats.unsqueeze(0).float(),
                        img_pos_kernel_preds.permute(1, 0)
                        .view(
                            num_kernel,
                            -1,
                            self.dynamic_conv_size,
                            self.dynamic_conv_size,
                        )
                        .float(),
                        stride=1,
                    )
                    if self.upsample_mask_logit:
                        img_mask_pred = dynamic_upsample(
                            img_mask_pred,
                            img_upsample_factor.float(),
                            self.mask_stride,
                        )
                    else:
                        img_mask_pred = F.interpolate(
                            img_mask_pred,
                            scale_factor=self.mask_stride,
                            mode="bilinear",
                            align_corners=False,
                        )
                img_mask_pred = img_mask_pred.view(
                    -1, h * self.mask_stride, w * self.mask_stride
                )
            mask_preds.append(img_mask_pred)

        mask_target = {
            "pred": mask_preds,
            "target": pos_mask_targets,
            "avg_factor": pos_mask_num,
        }
        if pos_mask_num == 0:
            mask_target.update(
                {
                    "pred_mean": mask_feats[0].mean()
                    + mlvl_kernel_preds[-1].mean()
                }
            )
            if self.upsample_mask_logit:
                mask_target["pred_mean"] = (
                    mask_target["pred_mean"] + upsample_factors[0].mean()
                )

        # cate loss
        flatten_cls_labels = [
            torch.cat(
                [
                    img_lvl_cls_labels.reshape(-1, self.num_classes)
                    for img_lvl_cls_labels in lvl_cls_labels
                ]
            )
            for lvl_cls_labels in zip(*cls_labels)
        ]
        flatten_cls_labels = torch.cat(flatten_cls_labels)

        flatten_cls_preds = [
            lvl_cls_preds.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for lvl_cls_preds in mlvl_cls_preds
        ]
        flatten_cls_preds = torch.cat(flatten_cls_preds)
        if self.use_ignore_region:
            valid_flag = (flatten_cls_labels != self.ignore_index).all(1)
            flatten_cls_preds = flatten_cls_preds[valid_flag].contiguous()
            flatten_cls_labels = flatten_cls_labels[valid_flag].contiguous()
        cls_target = {
            "pred": flatten_cls_preds,
            "target": flatten_cls_labels,
            "avg_factor": max(1, pos_num),
        }

        # attr loss
        # attr_labels: [[num_grid, num_grid, num_attr] * level_num] * img_num
        # mlvl_attr_preds:
        #     [[N, C_attr, num_grid, num_grid]] * attr_num] * level_num
        attr_targets = []
        for i, k in enumerate(self.attr_names):
            attr_k_labels = [
                torch.cat(
                    [
                        lvl_img_attr_labels[..., i].reshape(-1)
                        for lvl_img_attr_labels in lvl_attr_labels
                    ],
                    0,
                )
                for lvl_attr_labels in zip(*attr_labels)
            ]
            attr_k_labels = torch.cat(attr_k_labels, 0)

            attr_k_preds = [
                lvl_attr_preds[i]
                .permute(0, 2, 3, 1)
                .reshape(-1, self.attr_name2num[k])
                for lvl_attr_preds in mlvl_attr_preds
            ]
            attr_k_preds = torch.cat(attr_k_preds, 0)

            attr_targets.append(
                {
                    "pred": attr_k_preds,
                    "target": attr_k_labels,
                    "avg_factor": max(1, pos_num),
                }
            )

        return cls_target, mask_target, attr_targets
