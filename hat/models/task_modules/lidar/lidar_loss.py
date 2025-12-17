# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Any, Dict, List, Optional

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.core import box_torch_ops
from hat.core.center_utils import _transpose_and_gather_feat
from hat.models.losses.lidar_losses import (
    AfdetFocalLoss,
    BoundaryLoss,
    LidarFastFocalLoss,
    LidarRegLoss,
    LidarSmoothL1RegLoss,
)
from hat.models.losses.mse_loss import MSELoss
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    from torchvision.transforms.functional import InterpolationMode, resize
except ImportError:
    InterpolationMode = None
    resize = None

logger = logging.getLogger(__file__)

eps = 1e-6

__all__ = ["AfdetLidarLoss", "AfsegLidarLoss"]


class _Sigmoid_clamp(nn.Module):
    def __init__(self, min: float = 1e-4, max: float = 1 - 1e-4):
        """Calculate input with sigmoid and clamp.

        Args:
            min:min value of clamp.
            max:max value of clamp.
        Returns:
            y:output after sigmoid and clamp.
        """
        super().__init__()

        self.min = min
        self.max = max

    def forward(self, x):
        temp = torch.sigmoid(x)
        y = torch.clamp(temp, min=self.min, max=self.max)
        return y


@OBJECT_REGISTRY.register
class AfdetLidarLoss(nn.Module):
    def __init__(
        self,
        weight: float = 0.25,
        weight_iou: List[float] = None,
        code_weights: List[float] = None,
        hm_key: str = "hm",
        fb_head_cfg: Optional[Dict] = None,
        split_hm_head: bool = False,
        use_gaussian_reg_loss: bool = False,
    ):
        """Calculate lidar det obj heatmap loss and dimension loss.

        Classification uses focal loss by default.
        Regression (dimension, rotation, center_offset
        etc.) use L1 loss.

        Args:
            weight:Global loss weight for each sub-loss. Default
                weight is 0.25.
            weight_iou: speical iou loss weight.
            code_weights: code weight.
            hm_key:heat map feature key name.
            fb_head_cfg:foreground and background head config.
            split_hm_head:heat map head slipt or not.
            use_gaussian_reg_loss: whether to use gaussian kernel for
                regression loss.
        """
        super(AfdetLidarLoss, self).__init__()
        self.code_weights = code_weights
        self.weight = weight  # weight between hm loss and loc loss
        self.weight_iou = weight_iou

        self.hm_key = hm_key
        if hm_key == "hm":
            self.crit = LidarFastFocalLoss()
        else:
            self.crit = AfdetFocalLoss()
            self.bd_loss = BoundaryLoss()
        self.crit_reg = LidarRegLoss()
        self.fb_head_cfg = fb_head_cfg
        if self.fb_head_cfg and self.fb_head_cfg["with_bg"]:
            self.crit_fb = AfdetFocalLoss()
        else:
            self.crit_fb = nn.BCELoss()

        self.crit_iou = LidarSmoothL1RegLoss()
        self.crit_kps = AfdetFocalLoss()

        self.split_hm_head = split_hm_head and hm_key == "hm"

        self.sigmoid = _Sigmoid_clamp()

        if self.fb_head_cfg is None or not self.fb_head_cfg["split"]:
            self.fb_head_split = False
        else:
            self.fb_head_split = True

        self.use_gaussian_reg_loss = use_gaussian_reg_loss

    @autocast(enabled=False)
    def forward(self, example: Dict, preds_dict: Dict, box_preds_all: List):
        """Calculate loss between pred and target items.

        Args:
            preds_dict (dict): Predict bev discrete obj output (e.g.,
                pred_bev_discobj_hm, pred_bev_discobj_wh, etc).
            example (dict): Target contains bev discrete obj ground truth
        """
        result_dict = {}

        if self.split_hm_head:
            hms = [preds_dict[k] for k in self.hm_keys]
            preds_dict_hm = torch.cat(hms, dim=1)
        else:
            preds_dict_hm = preds_dict["hm"]
        preds_dict_hm = self.sigmoid(preds_dict_hm)

        hm_loss = self.crit(
            preds_dict_hm,
            example["hm"],
            example["ind"],
            example["mask"],
            example["cat"],
        )

        if self.use_gaussian_reg_loss:
            target_box = example["anno_box_reg"]
        else:
            target_box = example["anno_box"]

        pred_list = [
            preds_dict["reg"],
            preds_dict["height"],
            preds_dict["dim"],
            preds_dict["rot"],
        ]
        if "vel" in preds_dict:
            pred_list.insert(-2, preds_dict["vel"])

        preds_dict_anno_box = torch.cat(pred_list, dim=1)

        # Regression loss for dimension, offset, height, rotation
        if self.use_gaussian_reg_loss:
            box_loss = self.crit_reg(
                preds_dict_anno_box,
                example["mask_reg"],
                example["ind_reg"],
                target_box,
            )
        else:
            box_loss = self.crit_reg(
                preds_dict_anno_box,
                example["mask"],
                example["ind"],
                target_box,
            )

        loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()

        loss = hm_loss + self.weight * loc_loss

        # IOU prediction loss
        # TODO (2023/05/10 Xiangyu Wei): check iou
        if "iou" in preds_dict and box_preds_all is not None:
            box_pred_all = box_preds_all[0]
            # iou_preds = preds_dict["iou"]
            B, C, H, W = preds_dict["reg"].shape
            if "vel" in preds_dict:
                preds_boxes_task = box_pred_all.view((B, H, W, 9))[
                    ..., [0, 1, 2, 3, 4, 5, -1]
                ]
            else:
                preds_boxes_task = box_pred_all.view((B, H, W, 7))
            preds_boxes_task = preds_boxes_task.permute([0, 3, 1, 2])
            qboxes = _transpose_and_gather_feat(
                preds_boxes_task, example["ind"]
            )
            gboxes = example["gt_boxes_tasks"][..., [0, 1, 2, 3, 4, 5, -1]]

            gboxes = gboxes.to(qboxes.device)
            iou_pos_targets = torch.diag(
                box_torch_ops.boxes_iou3d_gpu_horizon(
                    qboxes.view(-1, 7), gboxes.view(-1, 7)
                )[0].cuda()
            ).detach()

            iou_pos_targets = iou_pos_targets.view((B, -1, 1))
            iou_pos_targets = 2 * iou_pos_targets - 1

            iou_pred_loss = self.crit_iou(
                preds_dict["iou"],
                example["mask"],
                example["ind"],
                iou_pos_targets,
            )
            loss += self.weight_iou[0] * iou_pred_loss.sum()
            loss_iou = iou_pred_loss
            result_dict["loss_iou"] = loss_iou

        result_dict["loss"] = loss
        result_dict["loss_loc"] = loc_loss
        result_dict["loss_hm"] = hm_loss

        return result_dict

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class AfsegLidarLoss(nn.Module):
    @require_packages("torchvision")
    def __init__(
        self,
        tasks: List[Dict[str, Any]] = None,
        weight: float = 0.25,
        hm_key: str = "map_seg_hm",
        use_bd: bool = False,
        use_pixelwise_weight: bool = False,
        pixelwise_weight_size: int = 3,
    ):
        """Calculate lidar seg heatmap loss and dimension loss.

        Classification uses focal loss by default.
        Regression (dimension, rotation, center_offset
        etc.) use L1 loss.

        Args:
            tasks: head be responsible for tasks.
            weights:Global loss weight for each sub-loss. Default
                weight is 0.25.
            hm_key:heat map feature key name.
            use_bd: use boundary loss.
            use_pixelwise_weight: use pixel wise weight.
            pixelwise_weight_size: kernal size of pixel wise.
        """
        super(AfsegLidarLoss, self).__init__()
        num_classes = [(t["num_class"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.weight = weight  # weight between hm loss and loc loss

        self._hm_key = hm_key

        self.crit = MSELoss(reduction="mean")
        self.high_loss = nn.MSELoss()
        self.bd_loss = BoundaryLoss()

        self.sigmoid = _Sigmoid_clamp()

        self.use_bd = use_bd

        self.use_pixelwise_weight = use_pixelwise_weight
        if use_pixelwise_weight:
            self.pad_op = nn.ReflectionPad2d(
                padding=[pixelwise_weight_size // 2] * 4
            )
            self.unfold_op = nn.Unfold(kernel_size=pixelwise_weight_size)
        # change this if your box is different
        logger.info(f"num_classes: {num_classes}")

    def generate_pixelwise_weight(
        self, target: torch.Tensor, downsample_stride: int = 2
    ):
        """Generate more weight for boundary and small object.

        Args:
            target: target mask with 0, 1, 255 values,
                shape (b,c, h, w).
            downsample_stride: to reduce memory.
        """
        resized_target_all = target
        if downsample_stride > 1:
            h_ori, w_ori = target.shape[-2:]
            assert (
                h_ori % downsample_stride == 0
                and w_ori % downsample_stride == 0
            )
            size = (h_ori // downsample_stride, w_ori // downsample_stride)
            resized_target_all = resize(
                target, size, interpolation=InterpolationMode.NEAREST
            )
        b, c, h, w = resized_target_all.shape
        weight_list = []
        for i in range(c):
            resized_target = torch.reshape(
                resized_target_all[:, i, :, :], (b, 1, h, w)
            )
            padded_target = self.pad_op(resized_target.float())
            data = self.unfold_op(padded_target)
            data = torch.reshape(data, (b, -1, h, w))
            weight = torch.sum(
                1 - (data == resized_target).float(), dim=1, keepdim=True
            )  # (b, 1, h, w)
            normed_weight = torch.log2(weight + 2)  # greater than 1
            if downsample_stride > 1:
                normed_weight = resize(
                    normed_weight,
                    (h_ori, w_ori),
                    interpolation=InterpolationMode.NEAREST,
                )
            weight_list.append(normed_weight)
        return weight_list

    @autocast(enabled=False)
    def forward(
        self,
        example: Dict,
        preds_dicts: Dict,
        grid: Optional[torch.Tensor] = None,
    ):
        """Calculate loss between pred and target items.

        Args:
            preds_dicts: Predict bev segmentation output (e.g.,
                pred_bev_discseg_hm, , etc).
            example : Target contains bev segmetaion ground truth.
            grid: Grid mask of output.
        """
        loss = 0
        for task_id, preds_dict in enumerate(preds_dicts):

            preds_dict[self._hm_key] = self.sigmoid(preds_dict[self._hm_key])

            if "high_point" in preds_dict and grid is not None:
                preds_dict_high_point = self.sigmoid(preds_dict["high_point"])
                high_map = torch.pow(
                    example[self._hm_key][task_id][:, 0, :, :]
                    - preds_dict_high_point[:, 0, :, :],
                    2,
                )
                high_loss = (grid.detach() * high_map).mean() * 100
                loss += high_loss

            pixelwise_weight = None
            if self.use_pixelwise_weight:
                weight_list = self.generate_pixelwise_weight(
                    example[self._hm_key][task_id]
                )
                pixelwise_weight = torch.cat(weight_list, dim=1)

            valid_mask = example["seg_loss_mask"][task_id] == 1
            hm_loss = self.crit(
                preds_dict[self._hm_key],
                example[self._hm_key][task_id],
                valid_mask=valid_mask,
                weight=pixelwise_weight,
            )
            if self.use_bd:
                bd_loss = self.bd_loss(
                    preds_dict[self._hm_key], example[self._hm_key][task_id]
                )

                loss += self.weight * (hm_loss) + bd_loss
            else:
                loss += self.weight * (hm_loss)

        result_dict = {}
        result_dict["loss"] = loss
        result_dict["loss_hm"] = hm_loss
        if "high_point" in preds_dict and grid is not None:
            result_dict["loss_high"] = high_loss
        if self.use_bd:
            result_dict["loss_bd"] = bd_loss

        return result_dict

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig()
