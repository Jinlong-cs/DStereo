# Copyright (c) Horizon Robotics. All rights reserved.
# Source code reference to mmdetection3d

import itertools
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from shapely.geometry import Polygon
from torch import Tensor

from hat.core.box_utils import bbox_overlaps
from hat.core.virtual_camera import CylindricalCamera
from hat.core.virtual_camera.camera_base import CameraBase
from hat.models.task_modules.fcos.target import distance2bbox, get_points
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import multi_apply
from .decoder import MTFCOS3DDecoder, limit_period

__all__ = ["MTFCOS3DTarget"]

INF = 1e8


@OBJECT_REGISTRY.register
class MTFCOS3DTarget(nn.Module):
    """Generate 3d and 2d cls and reg targets for Multitask-Fused-FCOS.

    Args:
        num_classes: Number of categories excluding the background category.
        strides: Strides of points in multiple feature levels.
        regress_ranges: Regress range of multiple level points.
        norm_on_bbox: If true, normalize the regression targets with
            FPN strides.
        center_sampling: If true, use center sampling.
        center_sample_radius: Radius of center sampling. Default is 1.5.
        centerness_alpha: An scalar to centerness.
        head_channels: A dict to tell which head channel should be outputed.
        use_direction_classifier: Default is True.
        dir_offset: Default is pi / 4.
        diff_rad_by_sin: Default is True.
        loss_3d_reg_weights: 3d reg head channel weights
        loss_2d_2dBatch_weight: 2d detection loss weight for 2d task
        loss_2d_3dBatch_weight: 2d detection loss weight for 3d task
        task_batch_list: A list to indicate class id of each sample in one
            batch, WIP.
        disentangled_corner3D: whether decode preds for disentangled_corner3D.
        disentangled_IOU3D: Default is False.
        is_train_3d_branch: Whether to train 3d info branch.
        use_2d_ctr_pos: definition of pos/neg whether use 2d bbox ctr.
        use_iou_replace_ctrness: use iou replace ctrness.
        depth_type: The of representing depth. Choices are "Cartesian" and
            "Cylindrical".
        use_multibin: Whether to use multibin strategy.
        multibin_centers: The centers of bins.
        multibin_margin: The margin of bins.
    """

    def __init__(
        self,
        num_classes: int,
        strides: List[int],
        regress_ranges: Tuple[Tuple[int, int]],
        norm_on_bbox: bool = True,
        center_sampling: bool = True,
        center_sample_radius: float = 1.5,
        centerness_alpha: float = 2.5,
        head_channels: Optional[OrderedDict] = None,
        use_direction_classifier: bool = True,
        dir_offset: float = 0.7854,  # pi / 4
        diff_rad_by_sin: bool = True,
        loss_3d_reg_weights: Optional[List[float]] = None,
        loss_2d_2dBatch_weight: float = 1.0,
        loss_2d_3dBatch_weight: float = 1.0,
        task_batch_list: Optional[List[int]] = None,
        disentangled_corner3D: bool = False,
        disentangled_IOU3D: bool = False,
        is_train_3d_branch: bool = True,
        use_2d_ctr_pos: bool = True,
        use_iou_replace_ctrness: bool = False,
        depth_type: str = "Cartesian",
        use_multibin: bool = False,
        multibin_centers: Optional[List] = None,
        multibin_margin: float = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.cls_out_channels = num_classes
        self.background_label = num_classes
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.head_channels = head_channels
        self.bbox_code_size = 7  # (x, y, z, x_size, y_size, z_size, yaw)
        self.centerness_alpha = centerness_alpha
        self.group_reg_dims = []
        self.use_direction_classifier = use_direction_classifier
        self.dir_offset = dir_offset
        self.diff_rad_by_sin = diff_rad_by_sin
        self.loss_3d_reg_weights = loss_3d_reg_weights
        self.loss_2d_2dBatch_weight = loss_2d_2dBatch_weight
        self.loss_2d_3dBatch_weight = loss_2d_3dBatch_weight
        self.task_batch_list = task_batch_list
        self.use_2d_ctr_pos = use_2d_ctr_pos
        self.use_iou_replace_ctrness = use_iou_replace_ctrness
        self.use_multibin = use_multibin
        self.multibin_centers = multibin_centers
        self.multibin_margin = multibin_margin
        # whether use disentangled_corner3D Loss
        self.disentangled_corner3D = disentangled_corner3D
        self.disentangled_IOU3D = disentangled_IOU3D
        # get bbox3d channels
        if self.use_multibin:
            assert (
                self.multibin_centers is not None
            ), "to use multibin, but multibin_centers is None"
            self.multibin_centers = np.array(multibin_centers)
            self.multibin_centers = torch.from_numpy(self.multibin_centers)
        for name, chanels in head_channels.items():
            if "_3d_group" in name:
                self.group_reg_dims.append(chanels[-1])
        assert use_direction_classifier, "only suport dir_cls is True"
        assert diff_rad_by_sin, "only suport diff_rad_by_sin is True"

        # is train 3d info flag
        self.is_train_3d_branch = is_train_3d_branch
        assert depth_type in [
            "Cartesian",
            "Cylindrical",
        ], f"depth encode type must in [Cartesian, Cylindrical], \
                but got {depth_type}"
        self.depth_type = depth_type

    @staticmethod
    def _get_ignore_bboxes(gt_bboxes_list, gt_labels_list):
        # Currently, the box corresponding to label <0 indicates that it needs
        # to be ignored
        gt_bboxes_ignore_list = [None] * len(gt_bboxes_list)
        for ii, (gt_bboxes, gt_labels) in enumerate(
            zip(gt_bboxes_list, gt_labels_list)
        ):
            if gt_bboxes.shape[0] > 0:
                gt_bboxes_ignore = gt_bboxes[gt_labels < 0]
                if len(gt_bboxes_ignore.shape) == 1:
                    gt_bboxes_ignore = gt_bboxes_ignore.unsqueeze(0)
                gt_bboxes_ignore_list[ii] = gt_bboxes_ignore
        return gt_bboxes_ignore_list

    def get_direction_target_multibin(
        self, reg_targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Get multibin cls target according to the angle.

        Args:
            reg_target:  3D bbox regression target, the angle index is 6
                in the last axis.

        Returns:
            dir_cls_target: bin class [0, len(multibin_centers) - 1].
            dir_offset_target: offset relative to nearest bin center.

        """
        rot_gt = reg_targets[..., 6:7]
        multibin_centers = self.multibin_centers.to(rot_gt.device)
        angle_offset = rot_gt - multibin_centers
        angle_offset = limit_period(angle_offset, 0.5, 2 * np.pi)
        abs_angle_offset = angle_offset.abs()
        if self.multibin_margin > 0:
            bin_size = 2 * np.pi / len(multibin_centers)
            dir_cls_target = (
                abs_angle_offset < bin_size / 2 + self.multibin_margin
            ).float()
            return dir_cls_target, angle_offset
        else:
            _, dir_cls_target = abs_angle_offset.min(dim=-1)
            dir_offset_target = angle_offset.gather(
                -1, dir_cls_target[..., None]
            ).squeeze(dim=-1)
            return dir_cls_target, dir_offset_target

    def get_direction_target(
        self,
        reg_targets: Tensor,
        dir_offset: int = 0,
        num_bins: int = 2,
        one_hot: bool = True,
    ) -> Tensor:
        """Encode direction to 0 ~ num_bins-1.

        Args:
            reg_targets: Bbox regression targets.
            dir_offset: Direction offset.
            num_bins: Number of bins to divide 2*PI.
            one_hot: Whether to encode as one hot.

        Returns:
            Encoded direction targets.
        """
        rot_gt = reg_targets[..., 6]
        offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(
            offset_rot / (2 * np.pi / num_bins)
        ).long()

        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(
                *list(dir_cls_targets.shape),
                num_bins,
                dtype=reg_targets.dtype,
                device=dir_cls_targets.device,
            )
            dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    @staticmethod
    def add_sin_difference(
        boxes1: Tensor, boxes2: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1: Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2: Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            `boxes1`` and ``boxes2`` whose 7th dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7]
        )
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(
            boxes2[..., 6:7]
        )
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1
        )
        boxes2 = torch.cat(
            [boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]], dim=-1
        )
        return boxes1, boxes2

    def decode_depth(
        self, bboxes_3d: Tensor, grid_points: Tensor, cameras: CameraBase
    ) -> Tensor:
        """Decode z coord.

        Args:
            bboxes_3d: Pred 3d bbox, [offset_x, offset_y, depth,...].
            grid_points: Location of grid.
            cameras: Camera list for every image.
        Returns:
            bboxes_3d: Location in cartisen coord, [x, y, z,...].
        """
        bboxes_3d[:, :2] = grid_points - bboxes_3d[:, :2]
        num_bboxes = len(bboxes_3d)
        fx = bboxes_3d.new_ones((num_bboxes))
        fy = bboxes_3d.new_ones((num_bboxes))
        cu = bboxes_3d.new_ones((num_bboxes))
        cv = bboxes_3d.new_ones((num_bboxes))
        # device = bboxes_3d.device
        for idx, camera in enumerate(cameras):
            # transform intrinsic to virtual camera
            camera_matrix = torch.tensor(camera.camera_matrix)
            fx[idx] = camera_matrix[0, 0]
            fy[idx] = camera_matrix[1, 1]
            cu[idx] = camera_matrix[0, 2]
            cv[idx] = camera_matrix[1, 2]

        if self.depth_type == "Cartesian":
            # notice: no grad_fn.
            points2D = bboxes_3d[:, :2].clone().detach().cpu().numpy()
            depths = bboxes_3d[:, 2].view(-1, 1).clone().detach().cpu().numpy()

            # 2.5D loc decode to 3D loc in virtual camera
            points3D = torch.from_numpy(
                camera.project_pixel2cam(points2D, depth=depths)
            )
            bboxes_3d[:, :3] = points3D
        elif self.depth_type == "Cylindrical":
            theta = (bboxes_3d[:, 0] - cu) / fx
            rho = bboxes_3d[:, 2].clone()
            # z
            bboxes_3d[:, 2] = torch.mul(rho, torch.cos(theta))
            # x
            bboxes_3d[:, 0] = torch.mul(rho, torch.sin(theta))
            # y
            bboxes_3d[:, 1] = torch.mul(
                rho, torch.divide((bboxes_3d[:, 1] - cv), fy)
            )

        else:
            raise NotImplementedError(
                "depth_type has supported [Cartesian, Cylindrical]"
            )

        return bboxes_3d

    def _compute_box_3d(self, dim, loc, rot_y):
        """Batch transform bbox3d to corners.

        Args:
            dim: size of objs, order hwl. Shape: B*N*3.
            loc: ctr of objs, order xyz. Shape: B*N*3.
            rot_y: global yaw of objs.
        """

        rot_y = rot_y.unsqueeze(2)
        c, s = torch.cos(rot_y), torch.sin(rot_y)
        zeros, ones = torch.zeros_like(c), torch.ones_like(c)
        R = torch.cat(
            [c, zeros, s, zeros, ones, zeros, -s, zeros, c], dim=-1
        ).unsqueeze(2)

        l, w, h = dim[:, :, [2]], dim[:, :, [1]], dim[:, :, [0]]
        x_corners = torch.cat(
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            dim=-1,
        ).unsqueeze(3)
        y_corners = torch.cat(
            [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
            dim=-1,
        ).unsqueeze(3)
        z_corners = torch.cat(
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            dim=-1,
        ).unsqueeze(3)
        corners = torch.cat([x_corners, y_corners, z_corners], dim=-1)

        corners_3d = R * corners.repeat(1, 1, 1, 3)
        corners_3d = torch.cat(
            [
                torch.sum(corners_3d[:, :, :, :3], dim=-1, keepdim=True),
                torch.sum(corners_3d[:, :, :, 3:6], dim=-1, keepdim=True),
                torch.sum(corners_3d[:, :, :, 6:], dim=-1, keepdim=True),
            ],
            dim=-1,
        )

        corners_3d = corners_3d + loc.unsqueeze(2)

        return corners_3d

    @staticmethod
    def cal_iou_by_polygon(g, p):

        g = tuple(tuple(x) for x in g)
        p = tuple(tuple(x) for x in p)

        g = Polygon(g)
        p = Polygon(p)
        if not g.is_valid or not p.is_valid:
            return 0
        inter = Polygon(g).intersection(Polygon(p)).area
        union = g.area + p.area - inter
        if union == 0:
            return 0
        else:
            return inter / union

    @staticmethod
    def cal_bev_iou(pred_bev, gt_bev):
        gt_bev_arr = gt_bev.cpu().detach().numpy()
        pred_bev_arr = pred_bev.cpu().detach().numpy()
        prop_num, _, _ = gt_bev_arr.shape
        iou_matrix = torch.zeros((prop_num), dtype=torch.float32).to(
            gt_bev.device
        )
        for pi in range(prop_num):
            iou_matrix[pi] = MTFCOS3DTarget.cal_iou_by_polygon(
                gt_bev_arr[pi, :, :], pred_bev_arr[pi, :, :]
            )

        return iou_matrix

    def _get_target_single(
        self,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore,
        gt_bboxes_3d,
        gt_labels_3d,
        centers2d,
        depths,
        points,
        regress_ranges,
        num_points_per_lvl,
    ):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        # If the gt label is full of -1, do not calculate the image loss
        num_pos_gts = sum(gt_labels != -1)
        num_ignore = 0
        if gt_bboxes_ignore is not None:
            num_ignore = gt_bboxes_ignore.size(0)

        if not isinstance(gt_bboxes_3d, torch.Tensor):
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device)

        if num_pos_gts == 0:
            return (
                gt_labels.new_full((num_points,), self.background_label),
                gt_bboxes.new_zeros((num_points, 4)),
                gt_labels_3d.new_full((num_points,), self.background_label),
                gt_bboxes_3d.new_zeros((num_points, self.bbox_code_size)),
                gt_bboxes_3d.new_zeros((num_points,)),
                gt_bboxes_3d.new_zeros((num_points, self.bbox_code_size)),
                points,
            )

        ##########################################################
        # change orientation to local yaw @hxd need check marker
        # gt_bboxes_3d (x, y, z, h, w, l, rot_y)  to
        # (x, y, z, h, w, l, alpha_x) independent with depth_type
        alpha_x = (
            -torch.atan2(gt_bboxes_3d[..., 0], gt_bboxes_3d[..., 2])
            + gt_bboxes_3d[..., 6]
        )
        # limit alpha_x to (-pi, pi)
        alpha_x[alpha_x > np.pi] -= 2 * np.pi
        alpha_x[alpha_x < -np.pi] += 2 * np.pi
        gt_bboxes_3d[:, 6] = alpha_x.to(gt_bboxes.device)
        ##########################################################

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1]
        )
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2
        )
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        centers2d = centers2d[None].expand(num_points, num_gts, 2)
        gt_bboxes_3d = gt_bboxes_3d[None].expand(
            num_points, num_gts, self.bbox_code_size
        )
        # default: Dartesian coord, also can be Cylindrical or Spherical
        depths = depths[None, :, None].expand(num_points, num_gts, 1)
        # grid pos
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        # 2.5D ctr point offset with grid pos
        # delta_xs shpae = (num_points, num_gts)
        delta_xs = (xs - centers2d[..., 0])[..., None]
        delta_ys = (ys - centers2d[..., 1])[..., None]
        # Notice:
        # gt_bboxes_3d[2] is z coord in camera
        # depths is encoded "depth"
        bbox_targets_3d = torch.cat(
            (delta_xs, delta_ys, depths, gt_bboxes_3d[..., 3:]), dim=-1
        )

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if num_ignore != 0:
            gt_bboxes_ignore = gt_bboxes_ignore[None].expand(
                num_points, num_ignore, 4
            )
            xs_ignore = points[:, 0][:, None].expand(num_points, num_ignore)
            ys_ignore = points[:, 1][:, None].expand(num_points, num_ignore)

            left = xs_ignore - gt_bboxes_ignore[..., 0]
            right = gt_bboxes_ignore[..., 2] - xs_ignore
            top = ys_ignore - gt_bboxes_ignore[..., 1]
            bottom = gt_bboxes_ignore[..., 3] - ys_ignore
            ignore_bbox_targets = torch.stack((left, top, right, bottom), -1)

        assert self.center_sampling is True, (
            "Setting center_sampling to "
            "False has not been implemented for FCOS3D."
        )
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius

        if self.use_2d_ctr_pos:
            # ctr bbox from 2d bbox
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
        else:
            # ctr bbox from 3d prjed ctr points
            center_xs = centers2d[..., 0]
            center_ys = centers2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        x_mins = center_xs - stride
        y_mins = center_ys - stride
        x_maxs = center_xs + stride
        y_maxs = center_ys + stride

        # Assined with FCOS(2D)Target
        center_gts[..., 0] = torch.where(
            x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0]
        )
        center_gts[..., 1] = torch.where(
            y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1]
        )
        center_gts[..., 2] = torch.where(
            x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs
        )
        center_gts[..., 3] = torch.where(
            y_maxs > gt_bboxes[..., 3], gt_bboxes[..., 3], y_maxs
        )

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1
        )
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]
        ) & (max_regress_distance <= regress_ranges[..., 1])

        # If gt_bboxes_ignore is not none, condition: limit the regression
        # range out of gt_bboxes_ignore
        if num_ignore != 0:
            inside_ignore_bbox_mask = ignore_bbox_targets.min(-1)[0] > 0

        if self.use_2d_ctr_pos:
            # FCOS(2D)
            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            areas[inside_gt_bbox_mask == 0] = INF
            areas[inside_regress_range == 0] = INF
            min_dist, min_dist_inds = areas.min(dim=1)
            # min_area, min_area_inds = areas.min(dim=1)
        else:
            # FCOS3D
            # center-based criterion to deal with ambiguity
            # if matched many target
            dists = torch.sqrt(
                torch.sum(bbox_targets_3d[..., :2] ** 2, dim=-1)
            )
            dists[inside_gt_bbox_mask == 0] = INF
            dists[inside_regress_range == 0] = INF
            min_dist, min_dist_inds = dists.min(dim=1)

        labels: Tensor = gt_labels[min_dist_inds]
        labels_3d = gt_labels_3d[min_dist_inds]
        labels[min_dist == INF] = self.background_label  # set as BG
        labels_3d[min_dist == INF] = self.background_label  # set as BG
        label_weights = labels.new_ones(labels.shape[0], dtype=torch.float)
        if num_ignore != 0:
            inside_ignore, _ = inside_ignore_bbox_mask.max(dim=1)
            label_weights[inside_ignore == 1] = 0.0

        bbox_targets = bbox_targets[range(num_points), min_dist_inds]
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_dist_inds]
        # "original" bbox3d callback by DisentangledCorner3DLoss
        points_gt_bboxes_3d = gt_bboxes_3d[range(num_points), min_dist_inds]
        grid_points = points[:, None, :].expand(num_points, num_gts, 2)
        grid_points = grid_points[range(num_points), min_dist_inds]

        return (
            labels,
            bbox_targets,
            labels_3d,
            bbox_targets_3d,
            label_weights,
            points_gt_bboxes_3d,
            grid_points,
        )

    def get_targets(
        self,
        points: List[Tensor],
        gt_bboxes_list: List[Tensor],
        gt_labels_list: List[Tensor],
        gt_bboxes_3d_list: List[Tensor],
        gt_labels_3d_list: List[Tensor],
        centers2d_list: List[Tensor],
        depths_list: List[Tensor],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Compute reg, cls and ctr_ness targets for points in multiple imgs.

        Args:
            points: Points of each fpn level, each has shape B*(num_points,2).
            gt_bboxes_list: Ground truth bboxes of each image, each has shape
                B * (num_gt, 4).
            gt_labels_list: Ground truth labels of each box, each has shape
                B * (num_gt,).
            gt_bboxes_3d_list: 3D Ground truth bboxes of each image, each has
                shape B * (num_gt, bbox_code_size).
            gt_labels_3d_list: 3D Ground truth labels of each box, each has
                shape B * (num_gt,).
            centers2d_list: Projected 3D centers onto 2D image, each has shape
                B * (num_gt, 2).
            depths_list: Depth of projected 3D centers onto 2D image, each has
                shape B * (num_gt, 1).

        Returns:
            concat_lvl_labels: Labels of each level.
            concat_lvl_bbox_targets: BBox targets of each level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i]
            .new_tensor(self.regress_ranges[i])[None]
            .expand_as(points[i])
            for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # should be aligned with real3d
        gt_bboxes_ignore_list = self._get_ignore_bboxes(
            gt_bboxes_list, gt_labels_list
        )

        # get labels and bbox_targets of each image
        (
            labels_list,
            bbox_targets_list,
            labels_3d_list,
            bbox_targets_3d_list,
            label_weights_list,
            points_gt_bboxes_3d_list,
            grid_points_list,
        ) = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
            gt_bboxes_3d_list,
            gt_labels_3d_list,
            centers2d_list,
            depths_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points,
        )

        device = labels_list[0].device
        # For decode depth
        image_idx_list = [
            torch.Tensor([idx] * len(labels))
            for idx, labels in enumerate(labels_list)
        ]
        image_idx_list = [
            image_idxes.split(num_points, 0) for image_idxes in image_idx_list
        ]
        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        labels_3d_list = [
            labels_3d.split(num_points, 0) for labels_3d in labels_3d_list
        ]
        bbox_targets_3d_list = [
            bbox_targets_3d.split(num_points, 0)
            for bbox_targets_3d in bbox_targets_3d_list
        ]

        # for DisentangledCorner3DLoss
        points_gt_bboxes_3d_list = [
            points_gt_bbox_3d.split(num_points, 0)
            for points_gt_bbox_3d in points_gt_bboxes_3d_list
        ]
        label_weights_list = [
            label_weights.split(num_points, 0)
            for label_weights in label_weights_list
        ]
        # grid_points_list = [points, points]
        grid_points_list = [
            grid_points.split(num_points, 0)
            for grid_points in grid_points_list
        ]

        # concat per level image
        concat_lvl_image_idxes = []
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_labels_3d = []
        concat_lvl_bbox_targets_3d = []
        concat_lvl_points_gt_bboxes_3d = []
        concat_lvl_label_weights = []
        concat_lvl_grid_points = []
        for i in range(num_levels):
            concat_lvl_image_idxes.append(
                torch.cat(
                    [image_idxes[i] for image_idxes in image_idx_list]
                ).to(device)
            )
            ########################### 2D BBOX ########################### # noqa
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list])
            )
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list]
            )
            # notice: scale op.
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            ################################################################

            concat_lvl_labels_3d.append(
                torch.cat([labels[i] for labels in labels_3d_list])
            )
            concat_lvl_grid_points.append(
                torch.cat([points[i] for points in grid_points_list])
            )
            bbox_targets_3d = torch.cat(
                [
                    bbox_targets_3d[i]
                    for bbox_targets_3d in bbox_targets_3d_list
                ]
            )

            # for DisentangledCorner3DLoss
            points_gt_bboxes_3d = torch.cat(
                [
                    points_gt_bboxes_3d[i]
                    for points_gt_bboxes_3d in points_gt_bboxes_3d_list
                ]
            )
            concat_lvl_points_gt_bboxes_3d.append(points_gt_bboxes_3d)

            concat_lvl_label_weights.append(
                torch.cat(
                    [label_weights[i] for label_weights in label_weights_list]
                )
            )
            # notice: scale op.
            if self.norm_on_bbox:
                bbox_targets_3d[:, :2] = (
                    bbox_targets_3d[:, :2] / self.strides[i]
                )
            concat_lvl_bbox_targets_3d.append(bbox_targets_3d)

        return (
            concat_lvl_image_idxes,
            concat_lvl_labels,
            concat_lvl_bbox_targets,
            concat_lvl_labels_3d,
            concat_lvl_bbox_targets_3d,
            concat_lvl_label_weights,
            concat_lvl_points_gt_bboxes_3d,
            concat_lvl_grid_points,
        )

    @staticmethod
    def _centerness_target(pos_bbox_targets: torch.Tensor):
        """Compute centerness targets.

        Args:
            pos_bbox_targets: BBox targets of positive bboxes, with shape
                (num_pos, 4).

        Returns:
            torch.Tensor.
        """
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]
        ) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def forward(self, gt, pred: OrderedDict, *args):
        if isinstance(pred, dict):
            assert isinstance(pred, OrderedDict)
        else:
            raise NotImplementedError

        assert len(pred) == len(self.head_channels)

        dir_cls_preds = []
        cls_scores = []
        bbox_preds = []
        bbox_preds_3d = []
        ctrnesses_2d = []
        ctrnesses_3d = []
        for name, value_list in pred.items():
            # dir_cls need read before cls task
            if "dir" in name:
                dir_cls_preds = value_list
            elif "cls" == name:
                cls_scores = value_list
            elif "ctrness_2d" in name:
                ctrnesses_2d = value_list
            elif "ctrness_3d" in name:
                ctrnesses_3d = value_list
            elif "offset_2d_reg" in name:
                bbox_preds = value_list
            else:
                assert (
                    "_group_reg" in name
                ), f"unkown head name:{name} as pre-defined!"
                if bbox_preds_3d == []:
                    bbox_preds_3d = value_list
                else:
                    for i, lvl_preds in enumerate(value_list):
                        bbox_preds_3d[i] = torch.cat(
                            [bbox_preds_3d[i], lvl_preds], dim=1
                        )

        assert (
            sum(self.group_reg_dims) == bbox_preds_3d[0].size()[1]
        ), "check cat channels with group_reg_dims"
        assert (
            len(cls_scores)
            == len(bbox_preds_3d)
            == len(ctrnesses_2d)
            == len(ctrnesses_3d)
            == len(dir_cls_preds)
        )
        # generate predection targets
        # flatten cls_scores, bbox_preds, dir_cls_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        # bbox 2d
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds_3d = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
            for bbox_pred in bbox_preds_3d
        ]
        if self.use_multibin:
            flatten_dir_cls_preds = [
                dir_cls_pred.permute(0, 2, 3, 1).reshape(
                    -1, len(self.multibin_centers)
                )
                for dir_cls_pred in dir_cls_preds
            ]
        else:
            flatten_dir_cls_preds = [
                dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
                for dir_cls_pred in dir_cls_preds
            ]
        flatten_centerness_2d = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in ctrnesses_2d
        ]
        flatten_centerness_3d = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in ctrnesses_3d
        ]

        # refer init param
        # default must train 3d branch,when is_train_3d_branch False.
        # 3d info loss weight set 0.
        if self.is_train_3d_branch or True:
            flatten_bbox_preds_enc = []
            for i, scale in enumerate(self.strides):
                flatten_bbox_preds_enc.append(flatten_bbox_preds_3d[i].clone())
                flatten_bbox_preds_enc[i][:, :2] *= scale

            flatten_bbox_preds_enc = torch.cat(flatten_bbox_preds_enc)

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)  # bbox 2d
        flatten_bbox_preds_3d = torch.cat(flatten_bbox_preds_3d)
        flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds)
        flatten_centerness_2d = torch.cat(flatten_centerness_2d)
        flatten_centerness_3d = torch.cat(flatten_centerness_3d)

        # get gt target
        num_imgs = cls_scores[0].size(0)
        device = bbox_preds_3d[0].device
        dtype = bbox_preds_3d[0].dtype

        gt_bboxes = gt["gt_bboxes"]
        gt_labels = gt["gt_classes"]
        if "gt_bboxes_3d" not in gt:
            self.is_train_3d_branch = False
            # generate fake data stream for train
            gt_bboxes_3d = [
                torch.zeros((len(gt_bboxes[idx]), 7), device=device)
                for idx in list(range(num_imgs))
            ]
            gt_labels_3d = [
                torch.tensor([-1] * len(gt_bboxes[idx]), device=device)
                for idx in list(range(num_imgs))
            ]
            centers2d = [
                torch.zeros((len(gt_bboxes[idx]), 2), device=device)
                for idx in list(range(num_imgs))
            ]
            depths = [
                torch.ones((len(gt_bboxes[idx]),), device=device)
                for idx in list(range(num_imgs))
            ]
        else:
            gt_bboxes_3d = gt["gt_bboxes_3d"]
            gt_labels_3d = gt["gt_classes_3d"]
            centers2d = gt["centers2d_prj"]
            depths = gt["depths"]
            # must update cause reuse 2d block updated.
            self.is_train_3d_branch = True
        feat_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        all_level_points = get_points(feat_sizes, self.strides, dtype, device)

        # ignore targets label_weights is 0 else 1
        # labels_3d is not used yet
        (
            image_idxes,
            labels,
            bbox_targets,
            labels_3d,
            bbox_targets_3d,
            label_weights,
            points_gt_bboxes_3d,
            grid_points,
        ) = self.get_targets(
            all_level_points,
            gt_bboxes,
            gt_labels,
            gt_bboxes_3d,
            gt_labels_3d,
            centers2d,
            depths,
        )

        # get match image idxes.
        flatten_image_idxes = torch.cat(image_idxes)
        ############################ 2D BOX ############################# # noqa
        flatten_labels = torch.cat(labels)
        # equal to flatten_labels and need tobe replace
        # #  Deprecated and replace by flatten_labels
        # flatten_labels_3d = torch.cat(labels_3d)
        flatten_bbox_targets = torch.cat(bbox_targets)

        # TODO(xudong.he): remove
        # the same as flatten_all_level_points
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points]
        )
        #################################################################
        flatten_bbox_targets_3d = torch.cat(bbox_targets_3d)
        flatten_points_gt_bboxes_3d = torch.cat(points_gt_bboxes_3d)
        flatten_label_weights = torch.cat(label_weights)
        flatten_all_level_points = torch.cat(grid_points)
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = (
            (
                (flatten_labels >= 0)
                & (flatten_labels != self.background_label)
                & (flatten_label_weights > 0)
            )
            .nonzero()
            .reshape(-1)
        )

        num_pos = len(pos_inds)
        cls_avg_factor = max(num_pos, 1.0)

        self.pos_weight = torch.tensor([1.0], device=device)
        if num_pos == 0:
            # generate fake pos_inds
            pos_inds = flatten_labels.new_zeros((1,), dtype=torch.int64)
            self.pos_weight = torch.tensor([0.0], device=device)

        ############################ WIP:MTL ############################ # noqa
        # TODO(xudong.he): Deprecated:Multi task should reused graphmodel
        points_per_strides = []
        for feat_size in feat_sizes:
            points_per_strides.append(feat_size[0] * feat_size[1])

        valid_classes_list = None
        if self.task_batch_list is not None:
            valid_classes_list = []
            accumulate_list = list(itertools.accumulate(self.task_batch_list))
            task_id = 0
            for ii in range(num_imgs):
                if ii >= accumulate_list[task_id]:
                    task_id += 1
                valid_classes = []
                for cls in gt["gt_classes"][ii].unique():
                    if cls >= 0:
                        valid_classes.append(cls.item())
                if len(valid_classes) == 0:
                    valid_classes.append(task_id)
                valid_classes_list.append(valid_classes)
        #################################################################

        loss_2d_weight = (
            self.loss_2d_3dBatch_weight
            if self.is_train_3d_branch
            else self.loss_2d_2dBatch_weight
        )
        cls_target = OrderedDict(
            pred=flatten_cls_scores,
            target=flatten_labels.long(),
            weight=flatten_label_weights * loss_2d_weight,
            avg_factor=cls_avg_factor,
            points_per_strides=points_per_strides,
            valid_classes_list=valid_classes_list,
        )

        ###################### 2D bbox block START ###################### # noqa
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness_2d = flatten_centerness_2d[pos_inds]

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_points = flatten_points[pos_inds]
        # decode pos_bbox_targets to bbox for calculating IOU-like loss
        pos_decoded_targets = distance2bbox(pos_points, pos_bbox_targets)
        eps = 1e-3  # to avoid iou loss not converge bug
        pos_decoded_bbox_preds = distance2bbox(
            pos_points, pos_bbox_preds.relu() + eps
        )

        if num_pos == 0:
            pos_ctrness_2d_targets = torch.tensor([0.0], device=device)
            centerness_weight = torch.tensor([0.0], device=device)  # noqa
            bbox_weight = torch.tensor([0.0], device=device)
            bbox_avg_factor = None
        else:
            if not self.use_iou_replace_ctrness:
                pos_ctrness_2d_targets = self._centerness_target(
                    pos_bbox_targets
                )  # noqa
                bbox_weight = pos_ctrness_2d_targets
                bbox_avg_factor = pos_ctrness_2d_targets.sum()
            else:
                pos_ctrness_2d_targets = bbox_overlaps(
                    pos_decoded_bbox_preds.clone().detach(),  # noqa
                    pos_decoded_targets,
                    is_aligned=True,
                )
                bbox_weight = pos_ctrness_2d_targets.new_ones(
                    pos_ctrness_2d_targets.shape
                )
                bbox_avg_factor = None
            centerness_weight = pos_ctrness_2d_targets.new_ones(
                pos_ctrness_2d_targets.shape
            )
        giou_target = {
            "pred": pos_decoded_bbox_preds,
            "target": pos_decoded_targets,
            "weight": bbox_weight * loss_2d_weight,
            "avg_factor": bbox_avg_factor,
        }

        ctrness_2d_target = OrderedDict(
            pred=pos_centerness_2d,
            target=pos_ctrness_2d_targets,
            weight=centerness_weight * loss_2d_weight,
        )
        output = OrderedDict(
            cls=cls_target,
            offset_2d_reg=giou_target,
            ctrness_2d_reg=ctrness_2d_target,
        )
        self.group_3d_info_weight = torch.tensor([1.0], device=device)
        if not self.is_train_3d_branch:
            self.group_3d_info_weight = torch.tensor([0.0], device=device)
            # return output
        ###################### 2D bbox block END ###################### # noqa
        # bbox3d block
        pos_images_idxes = (
            flatten_image_idxes[pos_inds].cpu().numpy().astype(int).tolist()
        )
        pos_bbox_preds_3d = flatten_bbox_preds_3d[pos_inds]
        pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
        pos_ctrness_3d_preds = flatten_centerness_3d[pos_inds]
        pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds]

        group_reg_object_num = 0
        if self.use_multibin:
            if self.multibin_margin > 0:
                group_reg_object_num = (
                    sum(self.group_reg_dims) - len(self.multibin_centers) + 1
                )
            else:
                group_reg_object_num = sum(self.group_reg_dims)
        else:
            group_reg_object_num = sum(self.group_reg_dims)
        # loss weight block
        bbox_weights = pos_ctrness_3d_preds.new_ones(
            len(pos_ctrness_3d_preds), group_reg_object_num
        )
        equal_weights = pos_ctrness_3d_preds.new_ones(
            pos_ctrness_3d_preds.shape
        )

        # group 3d info loss weights
        loss_3d_reg_weights = self.loss_3d_reg_weights

        # decode for ctrness_3d
        if self.is_train_3d_branch or True:
            pos_bbox_preds_enc = flatten_bbox_preds_enc[pos_inds]
            # get gt_bbox_3d and mapped grid points
            pos_points_gt_bboxes_3d = flatten_points_gt_bboxes_3d[pos_inds]
            pos_all_level_points = flatten_all_level_points[pos_inds]

            # get camera list
            camera_list = []
            for idx in pos_images_idxes:
                try:
                    camera_list.append(gt["virtual_cam"][idx])
                except KeyError:
                    # hard code anyway.
                    camera_list.append(
                        CylindricalCamera.init_cam_param_by_matrix(
                            image_size=[704, 576],  # W * H
                            camera_matrix=np.array(
                                [[220, 0, 352], [0, 220, 288], [0, 0, 1]]
                            ),
                            is_virtual=True,
                        )
                    )

            # decode location
            pos_bbox_pred_3d_dec = self.decode_depth(
                pos_bbox_preds_enc.clone(), pos_all_level_points, camera_list
            )
            # decode alphax
            if self.use_multibin:
                self.multibin_centers = self.multibin_centers.to(
                    pos_dir_cls_preds.device
                )
                if self.multibin_margin > 0:
                    pos_bbox_pred_3d_dec = (
                        MTFCOS3DDecoder.decode_alphax_multibin_margin(
                            pos_bbox_pred_3d_dec.clone(),
                            pos_dir_cls_preds,
                            self.multibin_centers,
                            with_max=True,
                        )
                    )
                else:
                    pos_bbox_pred_3d_dec = (
                        MTFCOS3DDecoder.decode_alphax_multibin(
                            pos_bbox_pred_3d_dec.clone(),
                            pos_dir_cls_preds,
                            self.multibin_centers,
                            with_max=True,
                        )
                    )
            else:
                pos_bbox_pred_3d_dec = (
                    MTFCOS3DDecoder.decode_alphax_2bin_sin_differece(
                        pos_bbox_pred_3d_dec.clone(),
                        torch.max(pos_dir_cls_preds, dim=-1)[1],
                        self.dir_offset,
                    )
                )
            # decode pred rot_y
            pos_bbox_pred_3d_dec[:, 6] = MTFCOS3DDecoder.decode_roty(
                pos_bbox_pred_3d_dec[:, :3].clone(),
                pos_bbox_pred_3d_dec[:, 6].clone(),
            )
            # decode gt rot_y
            pos_points_gt_bboxes_3d[:, 6] = MTFCOS3DDecoder.decode_roty(
                pos_points_gt_bboxes_3d[:, :3].clone(),
                pos_points_gt_bboxes_3d[:, 6].clone(),
            )

            pos_preds_corners_3d = self._compute_box_3d(
                pos_bbox_pred_3d_dec[None, :, 3:6],
                pos_bbox_pred_3d_dec[None, :, :3],
                pos_bbox_pred_3d_dec[None, :, 6],
            )
            pos_target_corners_3d = self._compute_box_3d(
                pos_points_gt_bboxes_3d[None, :, 3:6],
                pos_points_gt_bboxes_3d[None, :, :3],
                pos_points_gt_bboxes_3d[None, :, 6],
            )

            if self.disentangled_IOU3D:
                pred_bev = pos_preds_corners_3d[0, :, :4, [0, 2]]
                gt_bev = pos_target_corners_3d[0, :, :4, [0, 2]]
                if self.is_train_3d_branch:
                    pos_ctrness_3d_targets = self.cal_bev_iou(pred_bev, gt_bev)
                else:
                    pos_ctrness_3d_targets = torch.tensor([0.0], device=device)
                # centerness_weight = None
                centerness_weight = pos_ctrness_2d_targets.clone().detach()
                centerness_weight = (
                    centerness_weight * self.group_3d_info_weight
                )

                depth_entangled_l1_dist = (
                    (pos_bbox_preds_3d[:, 2] - pos_bbox_targets_3d[:, 2])
                    .clone()
                    .detach()
                    .abs()
                )
                pos_ctrness_3d_targets = (
                    torch.exp(-1.0 * depth_entangled_l1_dist)
                    * pos_ctrness_3d_targets
                )

                ctrness_3d_reg = OrderedDict(
                    pred=pos_ctrness_3d_preds,
                    target=pos_ctrness_3d_targets,
                    weight=centerness_weight * self.pos_weight,
                )

            else:
                pos_preds_corners_3d = pos_preds_corners_3d.reshape(-1, 24)
                pos_target_corners_3d = pos_target_corners_3d.reshape(-1, 24)

                bbox3d_entangled_l1_dist = (
                    (pos_preds_corners_3d - pos_target_corners_3d)
                    .clone()
                    .detach()
                    .abs()
                    .mean(dim=1)
                )
                # range (0,1]
                pos_ctrness_3d_targets = torch.exp(
                    -1.0 * bbox3d_entangled_l1_dist
                )

                # change 3d score
                # centerness_weight = pos_ctrness_3d_preds.new_zeros(len(pos_ctrness_3d_preds)) # noqa
                centerness_weight = pos_ctrness_2d_targets.clone().detach()
                centerness_weight = (
                    centerness_weight * self.group_3d_info_weight
                )
                # centerness_weight = None
                ctrness_3d_reg = OrderedDict(
                    pred=pos_ctrness_3d_preds,
                    target=pos_ctrness_3d_targets,
                    weight=centerness_weight * self.pos_weight,
                )
                if self.disentangled_corner3D:
                    _weight = (
                        (1 - pos_ctrness_3d_targets[..., None])
                        * bbox_weights[:, 6][:, None].expand(
                            len(pos_target_corners_3d), 24
                        )
                        / 24
                    )
                    corners = OrderedDict(
                        pred=pos_preds_corners_3d,
                        target=pos_target_corners_3d,
                        weight=_weight
                        * self.group_3d_info_weight
                        * self.pos_weight,
                        avg_factor=equal_weights.sum(),
                    )

        if loss_3d_reg_weights:
            assert len(loss_3d_reg_weights) == group_reg_object_num
            bbox_weights = bbox_weights * bbox_weights.new_tensor(
                loss_3d_reg_weights
            )

        if self.use_multibin:
            (
                pos_dir_cls_targets,
                pos_dir_offset_targets,
            ) = self.get_direction_target_multibin(pos_bbox_targets_3d)
            # pos_bbox_targets_3d[..., 6] = pos_dir_offset_targets
        else:
            if self.use_direction_classifier:
                pos_dir_cls_targets = self.get_direction_target(
                    pos_bbox_targets_3d,
                    self.dir_offset,
                    num_bins=2,
                    one_hot=False,
                )

            if self.diff_rad_by_sin:
                (
                    pos_bbox_preds_3d,
                    pos_bbox_targets_3d,
                ) = self.add_sin_difference(
                    pos_bbox_preds_3d, pos_bbox_targets_3d
                )

        # set neg target weight be 0
        if num_pos <= 0:
            bbox_weights = bbox_weights.new_zeros(bbox_weights.shape)
            # equal_weights = equal_weights.new_zeros(equal_weights.shape)

        offset_3d_group_reg = OrderedDict(
            pred=pos_bbox_preds_3d[:, :2],
            target=pos_bbox_targets_3d[:, :2],
            weight=bbox_weights[:, :2] * self.group_3d_info_weight,
            avg_factor=equal_weights.sum(),
        )
        depth_3d_group_reg = OrderedDict(
            pred=pos_bbox_preds_3d[:, 2],
            target=pos_bbox_targets_3d[:, 2],
            weight=bbox_weights[:, 2] * self.group_3d_info_weight,
            avg_factor=equal_weights.sum(),
        )
        dim_3d_group_reg = OrderedDict(
            pred=pos_bbox_preds_3d[:, 3:6].relu(),
            target=pos_bbox_targets_3d[:, 3:6],
            weight=bbox_weights[:, 3:6] * self.group_3d_info_weight,
            avg_factor=equal_weights.sum(),
        )

        if self.use_multibin:
            pos_bbox_preds_3d_angle_info = pos_bbox_preds_3d[:, 6:]
            if self.multibin_margin > 0:
                pos_bbox_preds_3d_angle_info = (
                    pos_bbox_preds_3d_angle_info.reshape(
                        (-1, len(self.multibin_centers))
                    )
                )
                rot_mask_l1 = OrderedDict(
                    pred=pos_bbox_preds_3d_angle_info,
                    target=pos_dir_offset_targets,
                    mask=pos_dir_cls_targets,
                    weight=bbox_weights[:, 6] * self.group_3d_info_weight,
                    avg_factor=equal_weights.sum(),
                )
                output.update(rot_mask_l1=rot_mask_l1)
            else:
                rot_3d_group_reg = OrderedDict(
                    pred=pos_bbox_preds_3d_angle_info[..., 0],
                    target=pos_dir_offset_targets,
                    weight=bbox_weights[:, 6] * self.group_3d_info_weight,
                    avg_factor=equal_weights.sum(),
                )
                output.update(rot_3d_group_reg=rot_3d_group_reg)
        else:
            rotsin_3d_group_reg = OrderedDict(
                pred=pos_bbox_preds_3d[:, 6],
                target=pos_bbox_targets_3d[:, 6],
                weight=bbox_weights[:, 6] * self.group_3d_info_weight,
                avg_factor=equal_weights.sum(),
            )
            output.update(rotsin_3d_group_reg=rotsin_3d_group_reg)

        if num_pos <= 0:
            # TODO(xudong.he) need valify
            dir_weight = equal_weights.new_full(equal_weights.shape, 0)
        else:
            dir_weight = equal_weights

        # TODO: add more check for use_direction_classifier
        if self.use_direction_classifier:
            if self.use_multibin and self.multibin_margin > 0:
                dir_reg = OrderedDict(
                    pred=pos_dir_cls_preds,
                    target=pos_dir_cls_targets,
                    weight=dir_weight[..., None] * self.group_3d_info_weight,
                    avg_factor=equal_weights.sum(),
                )
                output.update(dir_reg=dir_reg)
            else:
                dir_reg = OrderedDict(
                    pred=pos_dir_cls_preds,
                    target=pos_dir_cls_targets,
                    weight=dir_weight * self.group_3d_info_weight,
                    avg_factor=equal_weights.sum(),
                )
                output.update(dir_reg=dir_reg)

        # if name in head_channels should be assigned.
        output.update(
            offset_3d_group_reg=offset_3d_group_reg,
            depth_3d_group_reg=depth_3d_group_reg,
            dim_3d_group_reg=dim_3d_group_reg,
            ctrness_3d_reg=ctrness_3d_reg,
        )
        if self.disentangled_corner3D and not self.disentangled_IOU3D:
            output.update(
                corners_reg=corners,
            )
        return output
