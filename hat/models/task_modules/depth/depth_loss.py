from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = [
    "MultiStrideDepthLoss",
    "DepthConfidenceCombinedLoss",
    "DepthVNLLoss",
    "DepthL1Loss",
    "ConfL1Loss",
    "SmoothDepthLoss",
]


@OBJECT_REGISTRY.register
class MultiStrideDepthLoss(nn.Module):
    """Multi-Stride Depth Loss.

    Args:
        multi_scale_weight: Weight of each scale prediction.
        depth_scale: Scale of depth.
        max_depth: Maximum limit of depth.
        min_depth: Minimum limit of depth.
        avoid_zero: Epsilon.
        out_strides: Stride of output feature map.
        output_confidence: Whether the outputs contains confidence predictions.
        confidence_scale_weight: Confidence weight of each scale prediction.
    """

    def __init__(
        self,
        multi_scale_weight: List[float],
        depth_scale: int,
        max_depth: float,
        min_depth: float,
        avoid_zero: float,
        out_strides: List[int],
        output_confidence: bool = True,
        confidence_scale_weight: List[float] = None,
    ):

        super().__init__()
        if output_confidence:
            if confidence_scale_weight is None:
                self.confidence_scale_weight = multi_scale_weight
            else:
                assert len(confidence_scale_weight) == len(
                    multi_scale_weight
                ), "%d vs. %d" % (
                    len(confidence_scale_weight),
                    len(multi_scale_weight),
                )
                self.confidence_scale_weight = confidence_scale_weight
        else:
            self.confidence_scale_weight = [None] * len(multi_scale_weight)

        self.multi_scale_weight = multi_scale_weight
        self.max_depth = max_depth / depth_scale
        self.min_depth = min_depth / depth_scale
        self.avoid_zero = avoid_zero / depth_scale
        self.out_strides = out_strides
        self.output_confidence = output_confidence

    @autocast(enabled=False)
    def forward(self, preds: List[torch.Tensor], label: torch.Tensor):

        assert len(preds) == len(self.multi_scale_weight), "%d vs. %d" % (
            len(preds),
            len(self.multi_scale_weight),
        )

        # convert to float32 while using amp
        preds = [pred.float() for pred in preds]

        losses = OrderedDict()
        for i, (pred, scale, conf_scale) in enumerate(
            zip(preds, self.multi_scale_weight, self.confidence_scale_weight)
        ):  # noqa
            pred = F.interpolate(
                pred,
                scale_factor=label.shape[2] / pred.shape[2],
                mode="bilinear",
                align_corners=False,
            )
            if self.output_confidence:
                depth, conf = torch.split(pred, 1, dim=1)
            else:
                depth = pred
                conf = None

            # 1. depth loss
            depth = depth.clamp(self.min_depth, self.max_depth)

            label_mask_bigger_min = label >= self.min_depth
            label_mask_smaller_max = label <= self.max_depth
            label_valid_mask = label_mask_bigger_min * label_mask_smaller_max
            pred_valid = depth * label_valid_mask
            label_valid = label * label_valid_mask

            valid_element = label_valid_mask.sum((1, 2, 3))
            l1_depth_error = (
                pred_valid - label_valid
            ).abs_() * label_valid_mask
            rel_norm = (
                1
                + 1.2 * ((self.max_depth - label_valid) / self.max_depth) ** 2
            )

            quad_rel_depth_error = rel_norm * l1_depth_error
            loss_quadl1 = (
                quad_rel_depth_error.sum((1, 2, 3))
                / (valid_element + self.avoid_zero)
                * (valid_element > 0)
            )

            losses[f"loss_stride_{self.out_strides[i]}"] = loss_quadl1 * scale

            # 2. confidence loss
            if self.output_confidence:
                assert conf_scale is not None
                conf_valid = conf * label_valid_mask
                l1_depth_error_block = l1_depth_error.detach() / (
                    label_valid + self.avoid_zero
                )
                l1_conf_error = (
                    conf_valid - torch.exp(-l1_depth_error_block)
                ).abs_() * label_valid_mask
                loss_conf = (
                    l1_conf_error.sum((1, 2, 3))
                    / (valid_element + self.avoid_zero)
                    * (valid_element > 0)
                )

                losses[f"conf_loss_stride_{self.out_strides[i]}"] = (
                    loss_conf * conf_scale
                )

        return losses


@OBJECT_REGISTRY.register
class DepthVNLLoss(nn.Module):
    """Using geometry property normal vector as a loss item.

    More detail in `<https://arxiv.org/pdf/1907.12209.pdf>`_.

    Args:
        focal_x: Focal x of camera.
        focal_y: Focal y of camera.
        input_size: Input size of image.
        delta_diff_x: Threshold for x to avoid sampled points too close.
        delta_diff_y: Threshold for y to avoid sampled points too close.
        delta_diff_z: Threshold for z to avoid sampled points too close.
        delta_z: Threshold for z coordinate to filter near point.
        sample_ratio: Percentage that to be sampled
        gt_scale: Value for gt scale.
        scale_pred: Flag indicate if the prediction is scaled
        depth_type: Depth coordinate.
        loss_name: Loss name
    """

    def __init__(
        self,
        focal_x: float,
        focal_y: float,
        input_size: Tuple[int, int],
        delta_cos: float = 0.867,
        delta_diff_x: float = 0.01,
        delta_diff_y: float = 0.01,
        delta_diff_z: float = 0.01,
        delta_z: float = 0.00001,
        sample_ratio: float = 0.15,
        gt_scale: float = 1.0,
        scale_pred: bool = False,
        depth_type: str = "Cylindrical",
        loss_name: str = "vnl_loss",
    ):
        super(DepthVNLLoss, self).__init__()
        self.fx = torch.tensor([focal_x], dtype=torch.float32)
        self.fy = torch.tensor([focal_y], dtype=torch.float32)
        self.input_size = input_size
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32)
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32)
        self.init_image_coor()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio
        self.gt_scale = gt_scale
        self.scale_pred = scale_pred
        self.depth_type = depth_type
        self.loss_name = loss_name

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy())
        self.u_u0 = x - self.u0

        y_col = np.arange(
            0, self.input_size[0]
        )  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy())
        self.v_v0 = y - self.v0

    def transfer_xyz(self, depth):
        self.u_u0 = self.u_u0.to(depth.device)
        self.v_v0 = self.v_v0.to(depth.device)
        theta = self.u_u0 / self.fx.to(depth.device)
        if self.depth_type == "Cylindrical":
            rho = depth
            depth_z = torch.abs(rho * torch.cos(theta))
        if self.depth_type != "Cylindrical":
            rho = torch.abs(depth / torch.cos(theta))
            depth_z = depth
        x = rho * torch.sin(theta)
        y = rho * self.v_v0 / self.fy.to(depth.device)
        z = depth_z
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1)  # [b, h, w, c]
        return pw

    def select_index(self):
        valid_width = self.input_size[1]
        valid_height = self.input_size[0]
        num = valid_width * valid_height
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)

        # 求 图像坐标 x y
        p1_x = p1 % self.input_size[1]
        p1_y = (p1 / self.input_size[1]).astype(np.int64)

        p2_x = p2 % self.input_size[1]
        p2_y = (p2 / self.input_size[1]).astype(np.int64)

        p3_x = p3 % self.input_size[1]
        p3_y = (p3 / self.input_size[1]).astype(np.int64)
        p123 = {
            "p1_x": p1_x,
            "p1_y": p1_y,
            "p2_x": p2_x,
            "p2_y": p2_y,
            "p3_x": p3_x,
            "p3_y": p3_y,
        }
        return p123

    def form_pw_groups(self, p123, pw):
        """Form 3D points groups, with 3 points in each group.

        Args:
            p123: points index.
            pw: 3D points.
        """
        p1_x = p123["p1_x"]
        p1_y = p123["p1_y"]
        p2_x = p123["p2_x"]
        p2_y = p123["p2_y"]
        p3_x = p123["p3_x"]
        p3_y = p123["p3_y"]

        pw1 = pw[:, p1_y, p1_x, :]
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.cat(
            [
                pw1[:, :, :, np.newaxis],
                pw2[:, :, :, np.newaxis],
                pw3[:, :, :, np.newaxis],
            ],
            3,
        )
        return pw_groups

    def filter_mask(
        self,
        p123,
        gt_xyz,
        delta_cos=0.867,
        delta_diff_x=0.005,
        delta_diff_y=0.005,
        delta_diff_z=0.005,
    ):
        # 选择同样三个点
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        # ignore linear
        pw_diff = torch.cat(
            [
                pw12[:, :, :, np.newaxis],
                pw13[:, :, :, np.newaxis],
                pw23[:, :, :, np.newaxis],
            ],
            3,
        )  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(
            0, 2, 1
        )  # (B* X CX(3)) [bn, 3(p123), 3(xyz)]
        proj_key = pw_diff.view(
            m_batchsize * groups, -1, index
        )  # B X  (3)*C [bn, 3(xyz), 3(p123)]
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(
            q_norm.view(m_batchsize * groups, index, 1),
            q_norm.view(m_batchsize * groups, 1, index),
        )  # []
        energy = torch.bmm(
            proj_query, proj_key
        )  # transpose check [bn, 3(p123), 3(p123)]
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize * groups, -1)
        mask_cos = (
            torch.sum(
                (norm_energy > delta_cos) + (norm_energy < -delta_cos), 1
            )
            > 3
        )  # igonre
        mask_cos = mask_cos.view(m_batchsize, groups)
        # ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3

        # ignore near
        mask_x = (
            torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        )
        mask_y = (
            torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        )
        mask_z = (
            torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0
        )

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near

        return mask, pw

    def image_mask2pts_mask(self, img_mask, p123):
        mask_p1 = img_mask[:, :, p123["p1_y"], p123["p1_x"]]
        mask_p2 = img_mask[:, :, p123["p2_y"], p123["p2_x"]]
        mask_p3 = img_mask[:, :, p123["p3_y"], p123["p3_x"]]
        pts_mask = mask_p1 & mask_p2 & mask_p3
        return pts_mask

    def select_points_groups(self, gt_depth, pred_depth, img_mask=None):
        # 2d -> 3d using depth
        pw_gt = self.transfer_xyz(gt_depth)
        pw_pred = self.transfer_xyz(pred_depth)
        B, C, H, W = gt_depth.shape
        # 选择点
        p123 = self.select_index()
        # mask:[b, n], pw_groups_gt: [b, n, 3(x,y,z), 3(p1,p2,p3)]
        mask, pw_groups_gt = self.filter_mask(
            p123,
            pw_gt,
            self.delta_cos,
            self.delta_diff_x,
            self.delta_diff_y,
            self.delta_diff_z,
        )
        if img_mask is not None:
            mask_from_img = self.image_mask2pts_mask(img_mask, p123)
            mask = mask & mask_from_img[:, 0]
        # [b, n, 3, 3]
        pw_groups_pred = self.form_pw_groups(p123, pw_pred)
        pw_groups_pred[
            (pw_groups_pred[:, :, 2:3, :] == 0).tile(1, 1, 3, 1)
        ] = 0.0001
        mask_broadcast = (
            mask.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2)
        )
        pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(
            1, -1, 3, 3
        )
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(
            1, -1, 3, 3
        )

        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    def forward(self, pred, target, select=True):
        gt_depth = target["gt_depth"] * self.gt_scale
        pred_depth = pred[:, :1]
        if self.scale_pred:
            pred_depth = pred_depth * self.gt_scale
        B, C, H, W = gt_depth.shape
        if "mask" in target:
            mask = target["mask"]
        else:
            mask = None
        gt_points, dt_points = self.select_points_groups(
            gt_depth, pred_depth, img_mask=mask
        )

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        dt_p12 = dt_points[:, :, :, 1] - dt_points[:, :, :, 0]
        dt_p13 = dt_points[:, :, :, 2] - dt_points[:, :, :, 0]

        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        dt_normal = torch.cross(dt_p12, dt_p13, dim=2)
        dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        dt_mask = dt_norm == 0.0
        gt_mask = gt_norm == 0.0
        dt_mask = dt_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        dt_mask *= 0.01
        gt_mask *= 0.01
        gt_norm = gt_norm + gt_mask
        dt_norm = dt_norm + dt_mask
        gt_normal = gt_normal / gt_norm
        dt_normal = dt_normal / dt_norm
        loss = torch.abs(gt_normal - dt_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25) :]
        loss = torch.mean(loss)
        ret = {self.loss_name: loss}
        return ret


def depth_l1_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_gt_depth: int = 150,
    weighted: bool = False,
    eps: float = 1e-6,
    beta: float = 1e-6,
):
    if mask is not None:
        valid_element = mask.sum(dim=(1, 2, 3), keepdim=True) + eps
    diff = (pred_depth - gt_depth).abs()
    if weighted:
        weight_map = torch.exp(1 - gt_depth / max_gt_depth)
        loss = weight_map * diff

    # smooth_l1_loss
    if beta >= 1e-5:
        loss = torch.where(
            loss < beta, 0.5 * loss ** 2 / beta, loss - 0.5 * beta
        )

    if mask is not None:
        loss = loss * mask
        loss = loss.sum(dim=(1, 2, 3), keepdim=True) / valid_element
        return loss.mean()
    else:
        return diff.mean()


def confidence_l1_loss(
    pred_conf,
    gt_conf,
    gt_depth,
    max_gt_depth=150,
    mask=None,
    weighted=False,
    eps=1e-6,
):
    if mask is not None:
        valid_element = mask.sum(dim=(1, 2, 3), keepdim=True) + eps
    conf_l1_loss = torch.abs(
        pred_conf - torch.exp(-gt_conf / (gt_depth + eps))
    )
    if weighted:
        weight_map = 1 + 1.2 * ((max_gt_depth - gt_depth) / max_gt_depth) ** 2
        conf_l1_loss = weight_map * conf_l1_loss
    if mask is not None:
        conf_l1_loss = conf_l1_loss * mask
        conf_l1_loss = (
            conf_l1_loss.sum(dim=(1, 2, 3), keepdim=True) / valid_element
        )

    return conf_l1_loss.mean()


def smooth_depth_loss(
    pred, target, do_noramlization=True, mask=None, eps=1e-6
):
    """Compute the smoothness loss for a depth image."""
    if do_noramlization:
        max_gt_val = torch.max(target)
        min_gt_val = torch.min(target)
        target = (target - min_gt_val) / (max_gt_val - min_gt_val + eps)
        max_pred_val = torch.max(pred)
        min_pred_val = torch.min(pred)
        pred = (pred - min_pred_val) / (max_pred_val - min_pred_val + eps)

    grad_disp_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    grad_disp_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

    grad_img_x = torch.mean(
        torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), 1, keepdim=True
    )
    grad_img_y = torch.mean(
        torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), 1, keepdim=True
    )
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    if mask is not None:
        grad_disp_x = F.pad(grad_disp_x, pad=(0, 1, 0, 0), mode="replicate")
        grad_disp_y = F.pad(grad_disp_y, pad=(0, 0, 0, 1), mode="replicate")
        grad_disp_x *= mask
        grad_disp_y *= mask
        valid_element = mask.sum(dim=(1, 2, 3), keepdim=True) + eps
        grad_disp_x = (
            grad_disp_x.sum(dim=(1, 2, 3), keepdim=True) / valid_element
        )
        grad_disp_y = (
            grad_disp_y.sum(dim=(1, 2, 3), keepdim=True) / valid_element
        )

    return grad_disp_x.mean() + grad_disp_y.mean()


@OBJECT_REGISTRY.register
class DepthL1Loss(nn.Module):
    """L1 loss for depth regression tasks.

    Args:
        loss_weight: Loss weight
        max_gt_depth: Max distance limit to calculate depth loss.
        beta: A smooth l1 loss hyperparam.
        loss_weight: Loss weight
        use_weight_map: Whether use weight_map for near distance.
        loss_name: Loss name.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        max_gt_depth: int = 150,
        beta: float = 1e-6,
        use_weight_map: bool = False,
        loss_name: str = "depth_l1_loss",
    ):
        super(DepthL1Loss, self).__init__()
        self.beta = beta
        self.max_gt_depth = max_gt_depth
        self.use_weight_map = use_weight_map
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, pred, target):
        if "mask" in target:
            mask = target["mask"]
        else:
            mask = None
        gt_depth = target["gt_depth"]

        loss_l1 = depth_l1_loss(
            pred,
            gt_depth,
            mask=mask,
            beta=self.beta,
            weighted=self.use_weight_map,
            max_gt_depth=self.max_gt_depth,
        )
        result_dict = {}
        result_dict[self.loss_name] = loss_l1 * self.loss_weight
        return result_dict


@OBJECT_REGISTRY.register
class ConfL1Loss(nn.Module):
    """L1 loss for depth confidence regression tasks.

    Args:
        max_gt_depth: Max distance limit to calculate depth loss.
        use_weight_map: Whether to use weight map.
        loss_weight: Loss weight.
        loss_name: Loss name.
    """

    def __init__(
        self,
        max_gt_depth: float = 150.0,
        use_weight_map: bool = False,
        loss_weight: float = 1.0,
        loss_name: str = "conf_l1_loss",
    ):
        super(ConfL1Loss, self).__init__()
        self.max_gt_depth = max_gt_depth
        self.use_weight_map = use_weight_map
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, pred_conf, pred_depth, target):

        if "mask" in target:
            mask = target["mask"]
        else:
            mask = None
        gt_depth = target["gt_depth"]
        gt_conf = (pred_depth.detach() - gt_depth).abs()

        loss_l1 = confidence_l1_loss(
            pred_conf,
            gt_conf,
            gt_depth,
            self.max_gt_depth,
            mask=mask,
            weighted=self.use_weight_map,
        )
        result_dict = {}
        result_dict[self.loss_name] = loss_l1 * self.loss_weight
        return result_dict


@OBJECT_REGISTRY.register
class SmoothDepthLoss(nn.Module):
    """Compute the smoothness loss for a disparity image.

    Args:
        do_normalization: Whether normalize input tensor when computing loss.
            Default is True.
        loss_weight: Loss weight.
        loss_name: Loss name.
    """

    def __init__(
        self,
        do_normalization: bool = True,
        loss_weight: float = 1.0,
        loss_name: str = "smooth_loss",
    ):
        super(SmoothDepthLoss, self).__init__()
        self.do_normalization = do_normalization
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, pred, target):
        pred_depth = pred[:, :1]

        gt_depth = target["origin_img"]

        if "mask" in target:
            mask = target["mask"]
            mask = ~mask
        else:
            mask = None

        loss_smooth = smooth_depth_loss(
            pred_depth,
            gt_depth,
            self.do_normalization,
            mask=mask,
        )
        loss_smooth = loss_smooth * self.loss_weight
        result_dict = {}
        result_dict[self.loss_name] = loss_smooth
        return result_dict


@OBJECT_REGISTRY.register
class DepthConfidenceCombinedLoss(nn.Module):
    """Combining depth loss and confidence loss.

    Args:
        depth_loss: Depth loss module.
        conf_loss: Confidence loss module.
        depth_loss_name: Depth loss name.
        conf_loss_name: Confidence loss name.
        loss_weight: Loss weight for the two losses.
    """

    def __init__(
        self,
        depth_loss: nn.Module = None,
        conf_loss: nn.Module = None,
        depth_loss_name: str = "depth_loss",
        conf_loss_name: str = "conf_loss",
        loss_weight: Tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__()
        self.depth_loss = depth_loss
        self.conf_loss = conf_loss
        self.depth_loss_name = depth_loss_name
        self.conf_loss_name = conf_loss_name
        loss_weight = _as_list(loss_weight)
        if len(loss_weight) == 1:
            loss_weight.append(loss_weight[0])
        self.loss_weight = {
            self.depth_loss_name: loss_weight[0],
            self.conf_loss_name: loss_weight[1],
        }
        self.loss_name = "depth_conf_loss_v2"

    def forward(
        self,
        pred: torch.Tensor,
        target: Dict,
    ) -> torch.Tensor:
        c = pred.size()[1]
        pred_depth_split = torch.split(pred, 1, dim=1)
        if c == 2:
            pred_depth, pred_conf = pred_depth_split
        else:
            pred_depth = pred_depth_split[0]
            pred_conf = None
        result = OrderedDict()
        if self.depth_loss:
            depth_loss_name = self.depth_loss.loss_name
            depth_loss = self.depth_loss(pred_depth, target)[depth_loss_name]
            result[self.depth_loss_name] = (
                depth_loss * self.loss_weight[self.depth_loss_name]
            )
        if self.conf_loss and pred_conf is not None:
            conf_loss_name = self.conf_loss.loss_name
            conf_loss = self.conf_loss(pred_conf, pred_depth, target)[
                conf_loss_name
            ]
            result[self.conf_loss_name] = (
                conf_loss * self.loss_weight[self.conf_loss_name]
            )

        return result
