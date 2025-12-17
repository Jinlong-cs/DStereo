# Copyright (c) Horizon Robotics. All rights reserved.

from types import MappingProxyType
from typing import List, OrderedDict

import numpy as np
import torch
from torch import nn

from hat.core.face3d.lbs import rot6d_to_rotmat
from hat.core.hand3d.hand3d_func import proj_func
from hat.registry import OBJECT_REGISTRY, build_from_registry

try:
    from kornia.geometry.subpix import spatial_soft_argmax2d
except Exception:
    spatial_soft_argmax2d = None

try:
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.structures import Meshes
except Exception:
    sample_points_from_meshes = None
    TexturesVertex = None
    Meshes = None

__all__ = ["H3DManoDecoder"]


@OBJECT_REGISTRY.register
class H3DManoDecoder(nn.Module):
    """Decoder network of hand pose estimation task.

    Args:
        intput_shape:
            Shape of input image. Defaults to (256, 256).
        heatmap_stride:
            Contraction factor of heatmap relative to input image.
            Defaults to 4.
        trans_coeff:
            Scaling factor for decoding translation vector \
            in camera extrinsic matrix. Defaults to 2.5.
        min_distance:
            Minimum predictable depth. Defaults to 0.001.
        hand_scale_coeff:
            Scaling factor for hand size. Defaults to 1.
        enable_grid_sample:
            Whether to enable grid sample. Defaults to True.
        enable_heatmap_head:
            Whether to use DecodeHeatmap to decode gaussian heatmap, \
            else use spatial_soft_argmax2d. Defaults to False.
        enable_render_head:
            Whether to use rendering to supervise model.
            Defaults to False.
        enable_handscale_head:
            Whether to use hand scale head. Defaults to False.
        enable_subdivide_points:
            Whether to subdivide points from mano to enhance image detail.
            Defaults to False.
        return_in_millimetres:
            Return 2d joints in pixels and 3d joints in millimetres.
            Defaults to True.
        num_verts_sampled:
            Number of vertices sampled from the mesh. Defaults to 50.
        mano_layer:
            Mano layer used for generating hand mesh. Defaults to None.
        point_nerf:
            PointNeRF layer used for rendering mesh. Defaults to None.
        infer_mode:
            If True, use predicted camera extrinsic matrix to render image, \
            else use ground truth camera extrinsic matrix. Defaults to False.
    """

    def __init__(
        self,
        intput_shape: List = (256, 256),
        heatmap_stride: int = 4,
        trans_coeff: float = 2.5,
        min_distance: float = 0.001,
        hand_scale_coeff: float = 1,
        enable_grid_sample: bool = True,
        enable_heatmap_head: bool = False,
        enable_render_head: bool = False,
        enable_handscale_head: bool = False,
        enable_subdivide_points: bool = False,
        return_in_millimetres: bool = True,
        num_verts_sampled: int = 50,
        mano_layer: nn.Module = None,
        point_nerf: nn.Module = None,
        infer_mode: bool = False,
    ):
        super(H3DManoDecoder, self).__init__()

        self.intput_shape = intput_shape
        self.heatmap_stride = heatmap_stride
        self.trans_coeff = trans_coeff
        self.min_distance = min_distance
        self.hand_scale_coeff = hand_scale_coeff
        self.enable_grid_sample = enable_grid_sample
        self.enable_heatmap_head = enable_heatmap_head
        self.enable_render_head = enable_render_head
        self.enable_handscale_head = enable_handscale_head
        self.enable_subdivide_points = enable_subdivide_points
        self.return_in_millimetres = return_in_millimetres
        self.num_verts_sampled = num_verts_sampled
        self.infer_mode = infer_mode

        self.mano_layer = mano_layer
        if self.mano_layer is not None:
            self.mano_mesh_faces = (
                torch.tensor(self.mano_layer.faces.astype(np.float32))
                .clone()
                .unsqueeze(0)
            )
            self.mano_layer.requires_grad_(False)

        self.heatmap_shape = [i // heatmap_stride for i in list(intput_shape)]
        if self.enable_heatmap_head:
            decode_heatmap = {
                "type": "DecodeHeatmap",
                "decoding_method": "taylor",
                "feat_stride": self.heatmap_stride,
            }
            self.decode_heatmap = build_from_registry(decode_heatmap)

        if self.enable_render_head:
            self.point_nerf = point_nerf

    def outputs_preprocess(self, outputs):
        outputs["trans"] = torch.sigmoid(outputs["trans"])

        if self.enable_handscale_head:
            outputs["scale_intr"] = torch.sigmoid(outputs["scale_intr"])

        self.device = outputs["trans"].device
        self.batch_len = outputs["trans"].shape[0]

        return MappingProxyType(outputs)

    def decode_heatmap(self, heatmap_latents):
        if self.enable_heatmap_head:
            pred_heatmap = torch.sigmoid(heatmap_latents)
            pred_ldmk = torch.zeros(
                (pred_heatmap.shape[0], self.joint_nb, 2),
                device=self.device,
            )
            for i in range(pred_heatmap.shape[0]):
                for j in range(self.joint_nb):
                    pred_ldmk[i, j] = self.decode_heatmap(pred_heatmap[i, j])
        else:
            pred_ldmk = spatial_soft_argmax2d(
                heatmap_latents,
                normalized_coordinates=False,
            )

        pred_ldmk[:, :, 0] *= self.heatmap_stride
        pred_ldmk[:, :, 1] *= self.heatmap_stride
        pred_ldmk = torch.cat(
            [
                pred_ldmk,
                torch.ones_like(pred_ldmk[..., :1]).to(self.device),
            ],
            dim=-1,
        )
        return pred_ldmk

    def decode_pred_mano(self, rot, pose, shape=None):
        """Decode pred mano params to verts and joints.

        Args:
            rot: Global pose of mano, B * 6.
            pose: Mano pose params, B * 90.
            shape: Shape pose params, B * 10.

        Returns:
            pred_verts: Hand vertices of shape (batch_size, 778, 3).
            pred_joints: Hand joints of shape (batch_size, 21, 3).
        """
        pose_6dof = torch.cat([rot, pose], dim=-1).view(-1, 6)
        pose_rotmat = (
            rot6d_to_rotmat(pose_6dof).view(-1, 16, 3, 3).contiguous()
        )

        if self.mano_layer is not None:
            pred_verts, pred_joints = self.mano_layer(
                betas=shape.contiguous() if shape is not None else None,
                global_orient=pose_rotmat[:, :1, :, :],
                hand_pose=pose_rotmat[:, 1:, :, :],
                pose2rot=False,
            )  # [B, 778, 3], [B, 21, 3]
        else:
            pred_verts = torch.zeros((self.batch_len, 778, 3)).to(self.device)
            pred_joints = torch.zeros((self.batch_len, 21, 3)).to(self.device)

        return pred_verts, pred_joints

    def decoder_gt_mano(self, gt_shape, gt_pose):
        """Decode ground truth shape and pose parameters.

        Args:
            gt_shape: Ground truth shape parameters
            of shape (batch_size, 10).
            gt_pose: Ground truth pose parameters
            of shape (batch_size, 48).

        Returns:
            gt_verts: Hand vertices of shape
            (batch_size, 778, 3).
            gt_joints: Hand joints of shape
            (batch_size, 21, 3).
        """
        gt_mano_shape = gt_shape
        gt_mano_pose = gt_pose.view(-1, 48)

        if self.mano_layer is not None:
            gt_verts, gt_joints = self.mano_layer(
                betas=gt_mano_shape,
                global_orient=gt_mano_pose[:, :3],
                hand_pose=gt_mano_pose[:, 3:],
                pose2rot=True,
            )
        else:
            gt_verts = torch.zeros((self.batch_len, 778, 3)).to(self.device)
            gt_joints = torch.zeros((self.batch_len, 21, 3)).to(self.device)

        return gt_verts, gt_joints

    def forward(self, outputs, data):
        outputs_decoder = OrderedDict()
        outputs_only_read = self.outputs_preprocess(outputs)

        # mano decoder
        verts_relat, ldmk3d_relat = self.decode_pred_mano(
            outputs_only_read["rot"],
            outputs_only_read["pose_mano"],
            outputs_only_read["shape_mano"]
            if "shape_mano" in outputs_only_read
            else None,
        )

        # trans decoder
        pred_trans = outputs_only_read["trans"].unsqueeze(1) * self.trans_coeff
        pred_trans[:, :, -1] = pred_trans[:, :, -1] + self.min_distance
        pred_trans[:, :, :2] = pred_trans[:, :, :2] - 0.5 * self.trans_coeff

        if self.return_in_millimetres:
            verts_relat *= 1000
            ldmk3d_relat *= 1000
            pred_trans *= 1000

        # reproj joints
        pred_verts = verts_relat + pred_trans
        pred_ldmk3d = ldmk3d_relat + pred_trans
        joints2d_proj = proj_func(
            pred_ldmk3d,
            data["virtual_intrinsic"]
            if self.enable_grid_sample
            else data["intrinsic"],
        )

        # pred 2d joints
        pred_ldmk = self.decode_heatmap(outputs["heatmap_latents"])
        if self.enable_grid_sample:
            pred_ldmk = pred_ldmk - torch.tensor(
                [[[self.intput_shape[1] / 2, self.intput_shape[0] / 2, 0]]]
            ).to(self.device)
            pred_ldmk = torch.matmul(pred_ldmk, data["trans_mat"].float())

        # hand mesh
        mano_hand_mesh = None
        if TexturesVertex and Meshes and self.mano_layer is not None:
            mano_hand_mesh = Meshes(
                pred_verts,
                self.mano_mesh_faces.to(self.device).repeat(
                    self.batch_len, 1, 1
                ),
            )

        # point render
        render_images = None
        if self.enable_render_head and mano_hand_mesh:
            textures = TexturesVertex(verts_features=outputs["text"])
            mano_hand_mesh.textures = textures
            if self.enable_subdivide_points:
                pred_verts_sub, text_feat_xyz = sample_points_from_meshes(
                    mano_hand_mesh,
                    num_samples=self.num_verts_sampled ** 3,
                    return_textures=True,
                )
            else:
                pred_verts_sub = pred_verts.clone()
                text_feat_xyz = outputs.get("text")

            render_images = self.point_nerf(
                pred_verts_sub, text_feat_xyz, data["virtual_intrinsic"]
            )

        outputs_decoder.update(
            {
                "pred_ldmk3d": pred_ldmk3d,
                "pred_ldmk3d_relat": ldmk3d_relat,
                "pred_verts": pred_verts,
                "pred_pose": outputs["pose_mano"],
                "pred_shape": outputs["shape_mano"],
                "pred_ldmk": pred_ldmk[..., :2],
                "pred_ldmk_proj": joints2d_proj,
                "pred_trans": pred_trans,
                "pred_mesh": mano_hand_mesh,
                "render_images": render_images,
            }
        )

        return outputs_decoder
