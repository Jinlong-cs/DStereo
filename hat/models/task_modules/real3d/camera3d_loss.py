from collections import OrderedDict, defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY


def sigmoid_and_clip(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


@OBJECT_REGISTRY.register
class Camera3DLoss(nn.Module):
    """Camera3DLoss module for real3d task.

    Args:
        hm_loss: To define heatmap loss.
        box2d_wh_loss: To define box2d width and height loss.
        dimensions_loss: To define dimensions loss.
        location_offset_loss: To define location offset loss
        depth_loss: To define depth loss.
        loss_weights: Loss weights for calculating losses.
        output_head_with_2d_wh:  Whether to include wh in output items.
        use_depth_rotation_multiscale: Whether to use depth rotation
            multiscale.
        undistort_depth_uv: Whether to undistort depth branch into depth_u/v.
        max_depth: Define maximum depth,default 80.
    """

    def __init__(
        self,
        hm_loss: nn.Module,
        box2d_wh_loss: nn.Module,
        dimensions_loss: nn.Module,
        location_offset_loss: nn.Module,
        depth_loss: nn.Module,
        loss_weights: dict,
        loss_custom_weight: dict = None,
        output_head_with_2d_wh: bool = True,
        use_depth_rotation_multiscale: bool = False,
        undistort_depth_uv: bool = False,
        max_depth: int = 80,
    ):
        super(Camera3DLoss, self).__init__()
        self.loss_weights = defaultdict(lambda: 1.0, loss_weights)

        self.hm_lossF = hm_loss
        self.box2d_wh_lossF = box2d_wh_loss
        self.dimensions_lossF = dimensions_loss
        self.location_offset_lossF = location_offset_loss
        self.depth_lossF = depth_loss

        self.output_head_with_2d_wh = output_head_with_2d_wh
        self.use_depth_rotation_multiscale = use_depth_rotation_multiscale
        self.undistort_depth_uv = undistort_depth_uv
        self.max_depth = max_depth
        self.loss_custom_weight = loss_custom_weight

    @autocast(enabled=False)
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        dimensions = data["dimensions"]
        location_offset = data["location_offset"]
        heatmap = data["heatmap"]
        heatmap_weight = data["heatmap_weight"]
        depth = data["depth"]
        box2d_wh = data["box2d_wh"]
        ignore_mask = data["ignore_mask"]
        index_mask = data["index_mask"]
        index = data["index"]
        location = data["location"]
        dimensions_ = data["dimensions_"]
        rotation_y = data["rotation_y"]

        if self.loss_custom_weight is not None:
            loss_custom_weight = data["loss_custom_weight"]
            if isinstance(loss_custom_weight, list):
                loss_custom_weight = loss_custom_weight[0]
                for key in loss_custom_weight:
                    batch_size = loss_custom_weight[key].shape[0]
                    loss_custom_weight[key] = torch.reshape(
                        loss_custom_weight[key], (batch_size, 1, 1, 1)
                    )

        total_loss = []
        # convert to float32 while using amp
        for k, v in pred.items():
            pred[k] = v.float()

        if ignore_mask is not None:
            ignore_mask = ignore_mask.float()

        if self.undistort_depth_uv:
            dep = torch.cat([pred["dep_u"], pred["dep_v"]], dim=1)
        else:
            dep = pred["dep"]

        hm_weight = self.loss_weights["heatmap"]
        hm = sigmoid_and_clip(pred["hm"])
        dense_heatmap_weight = (
            loss_custom_weight["dense_heatmap"]
            if self.loss_custom_weight
            else None
        )
        hm_loss = (
            self.hm_lossF(
                hm,
                heatmap,
                ignore_mask=ignore_mask,
                weight=dense_heatmap_weight,
            )
            * hm_weight
        )

        rot_weight = self.loss_weights["rotation"]
        dense_rotation_weight = (
            loss_custom_weight["dense_rotation"]
            if self.loss_custom_weight
            else None
        )
        rot_loss = (
            rot_corner_loss(
                pred["rot"],
                index_mask,
                index,
                location,
                dimensions_,
                rotation_y,
                weight=dense_rotation_weight,
            )
            * rot_weight
        )

        dep_weight = self.loss_weights["depth"]
        dep = 1.0 / (dep.sigmoid() + 1e-6) - 1.0
        dense_depth_weight = (
            loss_custom_weight["dense_depth"]
            if self.loss_custom_weight
            else None
        )
        dep_loss = (
            self.depth_lossF(
                dep,
                depth,
                heatmap_weight,
                ignore_mask,
                weight=dense_depth_weight,
            )
            * dep_weight
        )

        dense_dimensions_weight = (
            loss_custom_weight["dense_dimensions"]
            if self.loss_custom_weight
            else None
        )
        dense_location_offset_weight = (
            loss_custom_weight["dense_location_offset"]
            if self.loss_custom_weight
            else None
        )
        dense_box2d_wh_weight = (
            loss_custom_weight["dense_box2d_wh"]
            if self.loss_custom_weight
            else None
        )
        reg_out = [
            ("dimensions", pred["dim"], dimensions, dense_dimensions_weight),
            (
                "location_offset",
                pred["loc_offset"],
                location_offset,
                dense_location_offset_weight,
            ),
        ]
        if self.output_head_with_2d_wh:
            reg_out.append(
                ("box2d_wh", pred["wh"], box2d_wh, dense_box2d_wh_weight)
            )

        for key, out, target, loss_tmp_weight in reg_out:
            weight = self.loss_weights[key]
            loss = (
                getattr(self, f"{key}_lossF")(
                    pred=out,
                    target=target,
                    weight_mask=heatmap_weight,
                    ignore_mask=ignore_mask,
                    weight=loss_tmp_weight,
                )
                * weight
            )
            total_loss.append(loss)

        dict_loss = OrderedDict(
            hm_loss=hm_loss,
            rot_loss=rot_loss,
            dep_loss=dep_loss,
            dim_loss=total_loss[0],
            loc_offset_loss=total_loss[1],
        )

        if self.output_head_with_2d_wh:
            dict_loss["wh_loss"] = total_loss[2]

        return dict_loss


def gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind.long())
    return feat


def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def alpha2rot_y_simplified(alpha, location):
    """Convert alpha to rotation_y.

    Args:
        alpha: Observation angle of object, ranging [-pi..pi].
        location: Object's 3D location.
    """
    rays = torch.arctan(location[:, :, [0]] / (location[:, :, [2]] + 1e-9))
    rotation_y = alpha + rays
    mask_0 = rotation_y > np.pi
    mask_1 = rotation_y < -np.pi
    rotation_y = rotation_y + (mask_1.float() - mask_0.float()) * 2 * np.pi

    return rotation_y


def compute_box_3d(dimensions, location, rotation_y):
    """Compute 3d box by dimensions location and rotation_y.

    Args:
        dimensions: Object's length width hight.
        location: Object's 3D location.
        rotation_y: Object's rotation_y. The angle between the driving
            direction of thevehicle and the imaging plane of the camera.
    """
    c, s = torch.cos(rotation_y), torch.sin(rotation_y)
    zeros, ones = torch.zeros_like(c), torch.ones_like(c)

    R = torch.cat(
        [c, zeros, s, zeros, ones, zeros, -s, zeros, c], dim=-1
    ).unsqueeze(2)
    l, w, h = (
        dimensions[:, :, [2]],
        dimensions[:, :, [1]],
        dimensions[:, :, [0]],
    )  # noqa

    x_corners = torch.cat(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=-1
    ).unsqueeze(3)
    y_corners = torch.cat(
        [zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=-1
    ).unsqueeze(3)
    z_corners = torch.cat(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=-1
    ).unsqueeze(3)

    corners = torch.cat([x_corners, y_corners, z_corners], dim=-1)

    corners_3d = R.expand(corners.shape[:3] + (9,)) * corners.repeat(
        1, 1, 1, 3
    )

    corners_3d = torch.cat(
        [
            torch.sum(corners_3d[:, :, :, :3], axis=-1, keepdims=True),
            torch.sum(corners_3d[:, :, :, 3:6], axis=-1, keepdims=True),
            torch.sum(corners_3d[:, :, :, 6:], axis=-1, keepdims=True),
        ],
        dim=-1,
    )
    corners_3d = corners_3d + location.unsqueeze(2).expand(
        corners_3d.shape
    )  # noqa

    return corners_3d


def rot_corner_loss(
    output_rot,
    batch_rot_mask,
    batch_ind,
    batch_loc,
    batch_dim,
    batch_rot_y,
    weight=None,
):
    """Compute 3d box corner loss.

    Args:
        output_rot: shape (batch, 2, H, W).
        batch_ind: shape (batch, X).
        batch_loc: shape (batch, X, 3).
        batch_dim: shape (batch, X, 3).
        batch_rot_mask: shape (batch, X).
        batch_rot_y: shape (batch, X, 1).
    """
    # Fetch features by indices
    pred_rot = transpose_and_gather_feat(output_rot, batch_ind)

    # Get alpha and rotation_y
    pred_alpha = get_alpha_simplified(pred_rot)
    mask = batch_rot_mask.unsqueeze(2).expand(pred_alpha.shape)
    pred_alpha = pred_alpha * mask

    pred_rot_y = alpha2rot_y_simplified(pred_alpha, batch_loc)
    pred_rot_y = mask * pred_rot_y
    batch_size = batch_loc.shape[0]
    loc_ = batch_loc.repeat(2, 1, 1)
    rot_ = torch.cat([pred_rot_y, batch_rot_y], dim=0)
    dim_ = batch_dim.repeat(2, 1, 1)
    corners3d = compute_box_3d(dim_, loc_, rot_)
    pred_corners_3d_rot = corners3d[:batch_size, :]
    gt_corners_3d = corners3d[batch_size:, :]

    mask = (
        batch_rot_mask.view(
            batch_rot_mask.shape
            + (
                1,
                1,
            )
        )
        .expand(gt_corners_3d.shape)
        .float()
    )
    rot_loss = torch.abs(pred_corners_3d_rot * mask - gt_corners_3d * mask)
    if weight is not None:
        rot_loss *= weight
    rot_loss = rot_loss.sum() / torch.max(mask.sum(), torch.tensor(1.0))

    return rot_loss


def get_alpha_simplified(rot, normalize=False):
    if normalize:
        sq_sum = torch.sqrt(rot[:, :, [0]] ** 2 + rot[:, :, [1]] ** 2) + 1e-9
        sin = rot[:, :, [0]] / sq_sum
        cos = rot[:, :, [1]] / sq_sum
    else:
        sin = rot[:, :, [0]]
        cos = rot[:, :, [1]]
    alpha = torch.arctan(sin / (1e-9 + cos))
    cos_pos_idx = cos >= 0
    cos_neg_idx = cos < 0
    alpha = alpha + (cos_neg_idx.float() - cos_pos_idx.float()) * np.pi / 2.0

    return alpha
