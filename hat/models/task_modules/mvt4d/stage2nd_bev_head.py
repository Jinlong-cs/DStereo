# mypy: ignore-errors

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.core.nus_box3d_utils import inverse_sigmoid
from hat.utils.package_helper import require_packages

try:
    from mmcv.version import __version__ as mm_version

    if mm_version >= "2.0.0":
        from mmengine.model.weight_init import xavier_init
    else:
        from mmcv.cnn import xavier_init
except ImportError:
    xavier_init = None

try:
    import torchvision
except ImportError:
    torchvision = None


class BevStage2nd(nn.Module):
    """Stage2nd head refine 3D boxes by extracting bev feat.

    Args:
        embed_dims: The embedding dimension of 3D box level instance.
        num_points: The number of points used for a 3D box.
            Only support 5 now.
        num_reg_fcs: Layers' number of regression fcs. Default: 2.
        num_pos_proj_fcs: Layers' number of positional encoding fcs.
            Default: 2.
        code_index: Sequence shows the meaning of per box
             code dims.
        roi_feat_output_size: Output size of roi align.
    """

    def __init__(
        self,
        embed_dims: int,
        num_points: int = 5,
        num_reg_fcs: int = 2,
        num_pos_proj_fcs: int = 2,
        code_index: Sequence[str] = (
            "CX",
            "CY",
            "W",
            "L",
            "CZ",
            "H",
            "SIN_YAW",
            "COS_YAW",
            "VX",
            "VY",
        ),
        roi_feat_output_size: tuple = (2, 4),
    ):
        super().__init__()
        self.roi_feat_output_size = roi_feat_output_size
        self.num_reg_fcs = num_reg_fcs
        self.num_pos_proj_fcs = num_pos_proj_fcs
        self.embed_dims = (
            embed_dims
            * self.roi_feat_output_size[0]
            * self.roi_feat_output_size[1]
        )
        self.num_points = num_points
        self.code_size = len(code_index)
        self.code_index = {v: i for i, v in enumerate(code_index)}

        if self.num_points == 5:
            self.corners_scale = np.array(
                [
                    [0, 0, 0],
                    [-0.5, 0.5, 0],
                    [0.5, 0.5, 0],
                    [0.5, -0.5, 0],
                    [-0.5, -0.5, 0],
                ]
            )
        else:
            raise ValueError("num_points must be 5 for BevStage2nd Head")

        assert np.all(self.corners_scale[0] == np.array([0, 0, 0]))
        self.init_weights()

    @require_packages("mmcv")
    def init_weights(self) -> None:
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        self.reg_branch = nn.Sequential(*reg_branch)
        xavier_init(self.reg_branch, distribution="uniform", bias=0.0)

        pos_proj = []
        pos_proj.append(nn.Linear(self.num_points * 3, self.embed_dims))
        for _ in range(self.num_pos_proj_fcs):
            pos_proj.append(nn.ReLU())
            pos_proj.append(nn.Linear(self.embed_dims, self.embed_dims))
        self.pos_proj = nn.Sequential(*pos_proj)
        xavier_init(self.pos_proj, distribution="uniform", bias=0.0)

    @require_packages("torchvision")
    @autocast(enabled=False)
    def forward(
        self,
        bev_feat: torch.Tensor,
        bbox3d: torch.Tensor,
        pc_range: List[float],
        bev_h: int,
        bev_w: int,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            mlvl_feats: Multi-level image feats from backbone and
                img_neck network.
            bbox3d: The bbox propoasls from DeformableHead.
                With shape (bs x query_num, 8) or (bs x query_num, 10).
                10 dims `cx, cy, w, l, cz, h, sin, cos, vx, vy`.
                optional: vx, vy.
            img_metas: image metas including calibs & image shape.
            pc_range: point cloud range, same as vcs_range.

        Returns:
            The refined 3D box results.
        """

        # bs x query_num x 10 -> (cx, cy, w, l, cz, h, sin, cos, vx, vy)
        B, N, _ = bbox3d.shape
        box_size = bbox3d[
            :,
            :,
            None,
            [self.code_index["W"], self.code_index["L"], self.code_index["H"]],
        ].exp()
        corners = box_size * bbox3d.new_tensor(self.corners_scale)
        rotate_mat = bbox3d.new_zeros([B, N, 3, 3])
        rotate_mat[:, :, 0, 0] = bbox3d[
            :, :, self.code_index["COS_YAW"]
        ]  # cos
        rotate_mat[:, :, 0, 1] = -bbox3d[
            :, :, self.code_index["SIN_YAW"]
        ]  # -sin
        rotate_mat[:, :, 1, 0] = bbox3d[
            :, :, self.code_index["SIN_YAW"]
        ]  # sin
        rotate_mat[:, :, 1, 1] = bbox3d[
            :, :, self.code_index["COS_YAW"]
        ]  # cos
        rotate_mat[:, :, 2, 2] = 1
        corners = (corners[..., None, :] * rotate_mat[:, :, None]).sum(
            dim=-1
        )  # B, N, points, 3
        corners = (
            corners
            + bbox3d[
                :,
                :,
                None,
                [
                    self.code_index["CX"],
                    self.code_index["CY"],
                    self.code_index["CZ"],
                ],
            ]
        )
        corners[..., 0:1] = (corners[..., 0:1] - pc_range[0]) / (
            pc_range[3] - pc_range[0]
        )  # cx
        corners[..., 1:2] = (corners[..., 1:2] - pc_range[1]) / (
            pc_range[4] - pc_range[1]
        )  # cy
        corners[..., 2:3] = (corners[..., 2:3] - pc_range[2]) / (
            pc_range[5] - pc_range[2]
        )  # cz

        bev_boxes = bbox3d.new_zeros([B, N, 5])

        # roi align box format on bev feat (bs_index, x1, y1, x2, y2).
        for bi in range(B):
            bev_boxes[bi : bi + 1, :, 0:1] = bi
            bev_boxes[bi : bi + 1, :, 1:2] = (
                corners[bi : bi + 1, :, 1:, 0].min(dim=-1, keepdim=True)[0]
                * bev_w
            )
            bev_boxes[bi : bi + 1, :, 2:3] = (
                corners[bi : bi + 1, :, 1:, 1].min(dim=-1, keepdim=True)[0]
                * bev_h
            )
            bev_boxes[bi : bi + 1, :, 3:4] = (
                corners[bi : bi + 1, :, 1:, 0].max(dim=-1, keepdim=True)[0]
                * bev_w
            )
            bev_boxes[bi : bi + 1, :, 4:5] = (
                corners[bi : bi + 1, :, 1:, 1].max(dim=-1, keepdim=True)[0]
                * bev_h
            )
        bev_boxes = bev_boxes.flatten(0, 1)

        hw, bs, c = bev_feat.shape
        bev_feat = bev_feat.permute(1, 2, 0).view(bs, c, bev_h, bev_w)
        roi_feats = torchvision.ops.roi_align(
            bev_feat, bev_boxes, self.roi_feat_output_size, spatial_scale=1.0
        )
        # roi_feats = roi_feats.flatten(1).unsqueeze(0)
        roi_feats = roi_feats.flatten(1)
        roi_feats = roi_feats.view(B, N, -1)

        pos_input = corners.reshape(B, N, self.num_points * 3)
        reference = inverse_sigmoid(corners[:, :, 0])  # index 0 must be center

        pos_feats = self.pos_proj(pos_input)  # torch.Size([2, 900, 2048])
        instance_feats = roi_feats + pos_feats

        refined_bbox3d = self.reg_branch(instance_feats)

        assert reference.shape[-1] == 3
        refined_bbox3d[
            ..., [self.code_index["CX"], self.code_index["CY"]]
        ] += reference[
            ..., 0:2
        ]  # cx, cy
        refined_bbox3d[
            ..., [self.code_index["CX"], self.code_index["CY"]]
        ] = refined_bbox3d[..., 0:2].sigmoid()
        refined_bbox3d[..., self.code_index["CZ"]] += reference[..., 2]  # cz
        refined_bbox3d[..., self.code_index["CZ"]] = refined_bbox3d[
            ..., self.code_index["CZ"]
        ].sigmoid()
        refined_bbox3d[..., self.code_index["CX"]] = (
            refined_bbox3d[..., self.code_index["CX"]]
            * (pc_range[3] - pc_range[0])
            + pc_range[0]
        )  # cx
        refined_bbox3d[..., self.code_index["CY"]] = (
            refined_bbox3d[..., self.code_index["CY"]]
            * (pc_range[4] - pc_range[1])
            + pc_range[1]
        )  # cy
        refined_bbox3d[..., self.code_index["CZ"]] = (
            refined_bbox3d[..., self.code_index["CZ"]]
            * (pc_range[5] - pc_range[2])
            + pc_range[2]
        )  # cz

        refined_bbox3d[
            ...,
            [self.code_index["W"], self.code_index["L"], self.code_index["H"]],
        ] = torch.log(
            refined_bbox3d[
                ...,
                [
                    self.code_index["W"],
                    self.code_index["L"],
                    self.code_index["H"],
                ],
            ].exp()
            * box_size[..., 0, :]
        )  # w, l, h

        if "VX" in self.code_index:
            refined_bbox3d[..., self.code_index["VX"] :] = (
                bbox3d[..., self.code_index["VX"] :]
                + refined_bbox3d[..., self.code_index["VX"] :]
            )  # vx, vy

        return refined_bbox3d
