# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, resize

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["ANCBEVFreespaceLoss"]


@OBJECT_REGISTRY.register
class ANCBEVFreespaceLoss(nn.Module):
    """Calculate bev freespace loss.

    Args:
        cls_loss_name: freespace loss name.
        cls_loss_cfg: the freespace loss config,
            support classification and regression loss.
        use_pixelwise_weight: use enhanched pixelwise weight for
            boundary and small object.
        pixelwise_weight_size: the kernel size for compute pixelwise
            weight. On gt label map, for a pixel (i, j) with label A,
            then compute the number of pixels with label not A, in
            the kernel size region around pixel (i, j). usually,
            the larger the number, the pixel (i, j) is more likely
            in small object or object boundary.
        boundary_loss_cfg: config dict for boundary loss, e.g.

            .. code-block:: none

                {
                    "use_boundary_loss": True,
                    "boundary_winsize_half": 3,
                    "boundary_loss_weight": 1.0,
                }

    """

    def __init__(
        self,
        cls_loss_name: str = "loss_bev_freespace",
        cls_loss_cfg: Optional[torch.nn.Module] = None,
        use_pixelwise_weight: bool = True,
        pixelwise_weight_size: int = 17,
        boundary_loss_cfg: Optional[dict] = None,
        pred_name: str = "pred_bev_freespace_frame0",
        gt_name: str = "gt_bev_freespace",
    ):
        super(ANCBEVFreespaceLoss, self).__init__()
        self.cls_loss_name = cls_loss_name
        self.pred_name = pred_name
        self.gt_name = gt_name

        self.cls_loss = cls_loss_cfg
        self.use_pixelwise_weight = use_pixelwise_weight

        if use_pixelwise_weight:
            self.pad_op = nn.ReflectionPad2d(
                padding=[pixelwise_weight_size // 2] * 4
            )
            self.unfold_op = nn.Unfold(kernel_size=pixelwise_weight_size)

        self.use_boundary_loss = False
        if boundary_loss_cfg is not None:
            self.use_boundary_loss = boundary_loss_cfg["use_boundary_loss"]
            self.boundary_winsize_half = boundary_loss_cfg[
                "boundary_winsize_half"
            ]
            self.boundary_loss_weight = boundary_loss_cfg[
                "boundary_loss_weight"
            ]

    def generate_pixelwise_weight(
        self, target: torch.Tensor, downsample_stride: int = 2
    ):
        """Generate more weight for boundary and small object.

        Args:
            target: target mask with 0, 1, 255 values,
                shape (b, h, w).
            downsample_stride: to reduce memory.
        """
        resized_target = target
        if downsample_stride > 1:
            h_ori, w_ori = target.shape[-2:]
            assert (
                h_ori % downsample_stride == 0
                and w_ori % downsample_stride == 0
            )
            size = (h_ori // downsample_stride, w_ori // downsample_stride)
            resized_target = resize(
                target, size, interpolation=InterpolationMode.NEAREST
            )
        b, h, w = resized_target.shape
        resized_target = torch.reshape(resized_target, (b, 1, h, w))
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

        return normed_weight

    def get_boundary_mask(
        self,
        label: torch.tensor,
        padding: Optional[int] = 1,
        boundary_type: str = "freespace_no",
        ignore_thresh_scale: Optional[float] = 0.5,
        extra_obstacle: Optional[torch.tensor] = None,
    ):
        """Get freespace boundary mask for autolabel and mannual label.

        If a ksize * ksize window center at (i,j) contains
        both freespace and non_freespace pixels, also ignore pixel num
        less than num_thresh, then (i, j) is defined as a boundary point.
        Specifically, only mark boundary point with boundary_type.

        Args:
            label: (b,h,w)
            padding: pad size. Defaults to 1.
            boundary_type: specify the boundary point type.
                take values in ["freespace_no", "freespace_yes"].
                if boundary type is "freespace_no", in the boundary
                region, only consider the pixel with freespace_no label
                as boundary point. Similar for "freespace_yes".
            ignore_thresh_scale: the scale factor
                to ignore pixel num. Defaults to 0.5.
            extra_obstacle: extra fine-grained obstacle gt.
                 Shape (b, h, w), default is None.

        Returns:
            tensor: mask for boundary point, shape (b, h, w).
        """
        assert boundary_type in ["freespace_no", "freespace_yes"]

        data = label.clone()
        if extra_obstacle is not None:
            data[extra_obstacle == 1] = 1
        b, h, w = data.shape
        pad = (padding, padding, padding, padding)  # pad last two dimension
        kernel_size = 2 * padding + 1
        data = F.pad(data.float()[:, None, :, :], pad, mode="replicate")
        data = F.unfold(data, kernel_size)
        data = torch.reshape(data, (b, -1, h, w))  # (b, kernel_size**2, h, w)

        num_ignore = torch.sum(
            (data == self.cls_loss.ignore_index).float(),
            dim=1,
            keepdim=False,
        )  # (b, h, w)
        num_ignore_thresh = ignore_thresh_scale * kernel_size * kernel_size

        if boundary_type == "freespace_no":
            anchor_mask = label == 1
            not_anchor_num = torch.sum(
                (data == 0).float(), dim=1, keepdim=False
            )
        else:
            anchor_mask = label == 0
            not_anchor_num = torch.sum(
                (data == 1).float(), dim=1, keepdim=False
            )

        bdry_mask = torch.where(
            torch.logical_and(
                torch.logical_and(num_ignore < num_ignore_thresh, anchor_mask),
                not_anchor_num > 0,
            ),
            torch.ones_like(label),
            torch.zeros_like(label),
        )
        return bdry_mask

    def classification_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """Compute class loss for use sigmoid and not use sigmoid.

        Args:
            pred: logits pred output, (b, 2, h, w).
            gt: gt label, (b, h, w)
            weight: pixelwise weight, (b, 2, h, w).
                Defaults to None.
            mask: mask region as 1,
                for computing loss, (b, h, w). Defaults to None.

        Returns:
            tensor: classification loss value.
        """

        target = gt.clone()
        num_classes = pred.shape[1]
        if self.cls_loss.use_sigmoid:
            ignore_mask = target == self.cls_loss.ignore_index
            if mask is not None:
                ignore_mask = torch.logical_or(
                    ignore_mask,
                    (1 - mask) > 0,
                )

            target[ignore_mask] = 0
            ignore_mask = ignore_mask.unsqueeze(1)
            target = F.one_hot(target, num_classes=num_classes).permute(
                0, 3, 1, 2
            )
            avg_factor = (
                (1 - ignore_mask.float()) * torch.ones_like(target)
            ).sum()
            avg_factor = torch.maximum(avg_factor, torch.ones_like(avg_factor))
            if weight is not None:
                weight = weight * (1 - ignore_mask.float())
            else:
                weight = 1 - ignore_mask.float()
            cls_loss = self.cls_loss(
                pred, target, weight=weight, avg_factor=avg_factor
            )
        else:
            cls_loss = self.cls_loss(pred, target, weight=weight)
        return cls_loss

    def forward(
        self,
        pred_dict: Union[torch.Tensor, Sequence[torch.Tensor]],
        target_dict: torch.Tensor,
    ) -> torch.Tensor:

        assert self.pred_name in pred_dict
        assert self.gt_name in target_dict
        result_dict = {}

        target = target_dict[self.gt_name]
        pred = _as_list(pred_dict[self.pred_name])[0]

        gt_bev_freespace = target[self.gt_name]
        gt_bev_extra_obstacle = (
            target["gt_bev_extra_obstacle"]
            if "gt_bev_extra_obstacle" in target.keys()
            else None
        )
        gt_lidardet_veh_mask = (
            target["gt_lidardet_veh_mask"]
            if "gt_lidardet_veh_mask" in target.keys()
            else None
        )
        # there can upsample predict to high resolution when gt resolution
        # greater than pred resolution, which is useful for small object
        # segmentation for loss supervision on high resolution,
        # e.g. pred is 512x512, gt is 1024x1024.
        # always ensure target shape > pred shape
        pixelwise_weight_down_stride = 2
        if pred.shape[-2:] != gt_bev_freespace.shape[-2:]:
            pred = resize(
                pred,
                gt_bev_freespace.shape[-2:],
                interpolation=InterpolationMode.NEAREST,
            )
            pixelwise_weight_down_stride *= 2  # reduce cuda memory

        pixelwise_weight = 1
        if self.use_pixelwise_weight:
            pixelwise_weight = self.generate_pixelwise_weight(
                gt_bev_freespace, pixelwise_weight_down_stride
            )

        bev_weight_map = (
            target["bev_weight_map"].unsqueeze(axis=1)
            if "bev_weight_map" in target.keys()
            else torch.ones_like(pixelwise_weight)
        )

        assert pixelwise_weight.shape == bev_weight_map.shape

        cls_loss = self.classification_loss(
            pred,
            gt_bev_freespace,
            weight=torch.mul(pixelwise_weight, bev_weight_map),
        )
        result_dict[self.cls_loss_name] = cls_loss
        # boundary loss
        if self.use_boundary_loss:
            # usually freespace enclose small object
            # to balance boundary pixels, use larger padding for freespace_no
            freespace_no_bdry_mask = self.get_boundary_mask(
                gt_bev_freespace,
                padding=self.boundary_winsize_half,
                boundary_type="freespace_no",
                extra_obstacle=gt_bev_extra_obstacle,
            )
            freespace_yes_bdry_mask = self.get_boundary_mask(
                gt_bev_freespace,
                padding=self.boundary_winsize_half // 2,
                boundary_type="freespace_yes",
            )
            if gt_lidardet_veh_mask is not None:
                freespace_veh_bdry_mask = self.get_boundary_mask(
                    gt_lidardet_veh_mask,
                    padding=self.boundary_winsize_half // 2,
                    boundary_type="freespace_no",
                )
            else:
                freespace_veh_bdry_mask = torch.zeros_like(
                    freespace_yes_bdry_mask
                )

            target_bdry_mask = gt_bev_freespace.new_tensor(
                torch.logical_or(
                    torch.logical_or(
                        freespace_no_bdry_mask, freespace_yes_bdry_mask
                    ),
                    freespace_veh_bdry_mask,
                )
            )
            boundary_cls_loss = self.classification_loss(
                pred,
                gt_bev_freespace,
                weight=bev_weight_map,
                mask=target_bdry_mask,
            )
            result_dict["loss_freespace_boundary"] = (
                boundary_cls_loss * self.boundary_loss_weight
            )

        return result_dict
