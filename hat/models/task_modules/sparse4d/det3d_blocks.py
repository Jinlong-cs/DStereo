# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np
import torch
import torch.nn as nn

from hat.models.base_modules.activation import Scale
from hat.models.weight_init import xavier_init
from hat.registry import OBJECT_REGISTRY
from .blocks import linear_relu_ln
from .target import COS_YAW, SIN_YAW, VX, H, L, W, X, Y, Z

__all__ = [
    "SparseBox3DKeyPointsGenerator",
    "SparseBox3DRefinementModule",
    "SparseBox3DEncoder",
]


@OBJECT_REGISTRY.register
class SparseBox3DRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        output_dim=11,
        num_cls=10,
        normalize_yaw=False,
        refine_yaw=False,
        with_cls_branch=True,
        with_centerness_branch=False,
        with_yawness_branch=False,
    ):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw

        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                nn.Linear(self.embed_dims, self.num_cls),
            )
        self.with_centerness_branch = with_centerness_branch
        if with_centerness_branch:
            self.centerness_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                nn.Linear(self.embed_dims, 1),
            )
        self.with_yawness_branch = with_yawness_branch
        if with_yawness_branch:
            self.yawness_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                nn.Linear(self.embed_dims, 1),
            )

    def init_weight(self):
        if self.with_cls_branch:
            prior_prob = 0.01
            bias_init = float(-np.log((1 - prior_prob) / prior_prob))
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature,
        anchor,
        anchor_embed,
        time_interval=1.0,
        return_cls=True,
    ):
        feature = instance_feature + anchor_embed
        output = self.layers(feature)
        output[..., self.refine_state] = (
            output[..., self.refine_state] + anchor[..., self.refine_state]
        )
        if self.normalize_yaw:
            output[..., [SIN_YAW, COS_YAW]] = torch.nn.functional.normalize(
                output[..., [SIN_YAW, COS_YAW]], dim=-1
            )
        if self.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            translation = torch.transpose(output[..., VX:], 0, -1)
            velocity = torch.transpose(translation / time_interval, 0, -1)
            output[..., VX:] = velocity + anchor[..., VX:]

        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)
        else:
            cls = None
        if return_cls and self.with_centerness_branch:
            centerness = self.centerness_layers(feature).squeeze(dim=-1)
        else:
            centerness = None
        if return_cls and self.with_yawness_branch:
            yawness = self.yawness_layers(feature).squeeze(dim=-1)
        else:
            yawness = None
        return output, cls, centerness, yawness


@OBJECT_REGISTRY.register
class SparseBox3DKeyPointsGenerator(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
        rot_axis=2,
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = np.array(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(
                self.embed_dims, num_learnable_pts * 3
            )
        assert rot_axis in [0, 1, 2]
        self.rot_axis = rot_axis

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        bs, num_anchor = anchor.shape[:2]
        fix_scale = anchor.new_tensor(self.fix_scale)
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1])
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            scale = torch.cat([scale, learnable_scale], dim=-2)
        key_points = scale * anchor[..., None, [W, L, H]].exp()
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])
        # vcs or lidar coordinate
        if self.rot_axis == 2:
            rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
            rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
            rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
            rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
            rotation_mat[:, :, 2, 2] = 1
        # camera(opencv) coordinate
        elif self.rot_axis == 1:
            rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
            rotation_mat[:, :, 0, 2] = anchor[:, :, SIN_YAW]
            rotation_mat[:, :, 2, 0] = -anchor[:, :, SIN_YAW]
            rotation_mat[:, :, 2, 2] = anchor[:, :, COS_YAW]
            rotation_mat[:, :, 1, 1] = 1
        else:
            raise NotImplementedError

        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)
        key_points = key_points + anchor[..., None, [X, Y, Z]]

        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        temp_key_points_list = []
        velocity = anchor[..., VX : VX + 3]
        vel_dims = velocity.shape[-1]
        if vel_dims == 2:
            velocity = torch.cat(
                [
                    velocity,
                    torch.zeros_like(velocity[..., : 3 - vel_dims]),
                ],
                dim=-1,
            )
        for i in range(len(T_cur2temp_list)):
            if vel_dims > 0:
                time_interval = cur_timestamp - temp_timestamps[i]
                translation = (
                    velocity
                    * time_interval.to(dtype=velocity.dtype)[:, None, None]
                )
                temp_key_points = key_points - translation[:, :, None]
            else:
                temp_key_points = key_points
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    @staticmethod
    def anchor_projection(
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        enable_trans_with_vel=True,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            dst_anchor = anchor.clone()
            vel = anchor[..., VX:]
            vel_dims = vel.shape[-1]
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            center = dst_anchor[..., [X, Y, Z]]
            if (
                enable_trans_with_vel
                and src_timestamp is not None
                and dst_timestamps is not None
                and vel_dims > 0
            ):
                translation = vel.transpose(0, -1) * (
                    src_timestamp - dst_timestamps[i]
                ).to(dtype=vel.dtype)
                translation = translation.transpose(0, -1)
                if vel_dims == 2:
                    translation = torch.cat(
                        [
                            translation,
                            torch.zeros_like(
                                translation[..., : 3 - vel_dims]
                            ),  # noqa
                        ],
                        dim=-1,
                    )
                center = center - translation
            dst_anchor[..., [X, Y, Z]] = (
                torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :3, 3]
            )

            dst_anchor[..., [COS_YAW, SIN_YAW]] = torch.matmul(
                T_src2dst[..., :2, :2],
                dst_anchor[..., [COS_YAW, SIN_YAW], None],
            ).squeeze(-1)

            dst_anchor[..., VX:] = torch.matmul(
                T_src2dst[..., :vel_dims, :vel_dims], vel[..., None]
            ).squeeze(-1)

            dst_anchors.append(dst_anchor)
        return dst_anchors


@OBJECT_REGISTRY.register
class SparseBox3DEncoder(nn.Module):
    def __init__(self, embed_dims=256, vel_dims=3):
        super().__init__()
        self.embed_dims = embed_dims
        self.vel_dims = vel_dims

        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.pos_fc = embedding_layer(3)
        self.size_fc = embedding_layer(3)
        self.yaw_fc = embedding_layer(2)
        if vel_dims > 0:
            self.vel_fc = embedding_layer(self.vel_dims)
        self.output_fc = embedding_layer(self.embed_dims)

    def forward(self, box_3d):
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]])
        output = pos_feat + size_feat + yaw_feat
        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims])
            output = output + vel_feat
        output = self.output_fc(output)
        return output
