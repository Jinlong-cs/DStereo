# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.quantization import DeQuantStub

from hat.models.task_modules.bevformer.utils import (
    gen_coords,
    get_min_max_coords,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["BevFormerViewTransformer"]


@OBJECT_REGISTRY.register
class BevFormerViewTransformer(nn.Module):
    """The basic structure of BevFormerViewTransformer.

    Args:
        pc_range: VCS range or point cloud range.
        num_points_in_pillar: The num of points in pillar.
        bev_h: The height of bevfeat.
        bev_w: The width of bevfeat.
        embed_dims: The embedding dimension of Attention.
        queue_length: The length of data queue.
        encoder: The encoder module.
        positional_encoding: Positional Encoding.
        numcam: The num of camera.
        in_indices: Input indices for view transformer.
        is_compile: Wheather for compile.
    """

    def __init__(
        self,
        pc_range: List[float],
        num_points_in_pillar: int,
        bev_h: int,
        bev_w: int,
        embed_dims: int,
        queue_length: int,
        encoder: nn.Module,
        positional_encoding: nn.Module,
        numcam: int = 6,
        in_indices: List[int] = (2,),
        is_compile: bool = False,
    ):
        super(BevFormerViewTransformer, self).__init__()
        self.in_indices = in_indices
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_size = (self.bev_w, self.bev_h)
        self.grid_resolution = (
            self.real_w / self.bev_w,
            self.real_h / self.bev_h,
        )
        self.embed_dims = embed_dims
        self.is_compile = is_compile
        self.queue_length = queue_length
        self.numcam = numcam
        self.encoder = encoder
        self.positional_encoding = positional_encoding

        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims
        )

        self.quant_bev_pos = QuantStub()
        self.quant_bev_query = QuantStub()
        self.quant_prev_bev = QuantStub()
        self.quant_hybird_ref_2d = QuantStub()
        self.quant_norm_coords = QuantStub()
        self.quant_reference_points_cam = QuantStub()
        self.dequant = DeQuantStub()

        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "ego2global": None,
        }

    @fx_wrap()
    def gen_reference_points(
        self,
        H: int,
        W: int,
        Z: int = 8,
        num_points_in_pillar: int = 4,
        dim: str = "3d",
        bs: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Get the reference points used in SCA and TSA."""

        # reference points in 3D space.
        if dim == "3d":
            zs = (
                torch.linspace(
                    0.5,
                    Z - 0.5,
                    num_points_in_pillar,
                    dtype=dtype,
                    device=device,
                )
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane.
        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @autocast(enabled=False)
    @fx_wrap()
    def point_sampling(
        self,
        reference_points: Tensor,
        pc_range: List[float],
        img_metas: Dict,
        im_shape: Tuple[int],
    ) -> Tuple[Tensor, Tensor]:
        """Sample img points."""

        reference_points = reference_points.to(torch.float32)
        ego2img = img_metas["ego2img"]
        ego2img = np.asarray(ego2img)
        ego2img = reference_points.new_tensor(ego2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0])
            + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1])
            + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2])
            + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = ego2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        ego2img = ego2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )

        reference_points_cam = torch.matmul(
            ego2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(-1)

        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= im_shape[1]
        reference_points_cam[..., 1] /= im_shape[0]

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )

        bev_mask = torch.nan_to_num(bev_mask)

        reference_points_cam = torch.clamp(
            reference_points_cam, min=-2.1, max=2.1
        )
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    @fx_wrap()
    def export_reference_points(
        self, bs: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tuple[Tensor, Tensor]:
        """Export reference points."""
        ref_3d = self.gen_reference_points(
            self.bev_h,
            self.bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bs,
            device=device,
            dtype=dtype,
        )
        ref_2d = self.gen_reference_points(
            self.bev_h, self.bev_w, dim="2d", bs=bs, device=device, dtype=dtype
        )

        return ref_2d, ref_3d

    @fx_wrap()
    def set_eval(self, seq_feats: List[List[Tensor]]) -> List[List[Tensor]]:
        """Set model to eval mode."""
        self.eval()
        return seq_feats

    @fx_wrap()
    def set_train(self, seq_feats: List[List[Tensor]]) -> List[List[Tensor]]:
        """Set model to train mode."""
        self.train()
        return seq_feats

    def get_cur_scene_token(self, data: List[Dict], i: int) -> List[str]:
        """Get scene tokens."""
        return [each["scene"] for each in data[i]["meta"]]

    @fx_wrap()
    def update_pre_info(
        self,
        seq_meta: List[Dict],
        bev_embed: Tensor,
        i: int,
        seq_feats: List[List[Tensor]],
    ) -> List[List[Tensor]]:
        """Update prev bev info."""
        self.prev_frame_info["scene_token"] = self.get_cur_scene_token(
            seq_meta, i
        )
        self.prev_frame_info["ego2global"] = seq_meta[i]["ego2global"]
        self.prev_frame_info["prev_bev"] = self.dequant(bev_embed.detach())
        return seq_feats

    def get_fusion_bev(
        self, data: Dict, seq_feats: List[List[Tensor]]
    ) -> Tensor:
        """Get fusion bev feat for queue data."""
        seq_meta = data["seq_meta"]
        im_shape = data["img"].shape[2:]
        for i in range(self.queue_length - 1, 0, -1):
            seq_feats = self.set_eval(seq_feats)
            bev_emb = self.get_bev_embed(seq_meta, i, seq_feats, im_shape)
            seq_feats = self.update_pre_info(seq_meta, bev_emb, i, seq_feats)
            seq_feats = self.set_train(seq_feats)

        bev_emb = self.get_bev_embed(seq_meta, 0, seq_feats, im_shape)
        bev_emb = self.update_pre_info(seq_meta, bev_emb, 0, bev_emb)
        return bev_emb

    def get_bev_embed(
        self,
        seq_meta: List[Dict],
        i: int,
        seq_feats: List[List[Tensor]],
        im_shape: Tuple[int, int],
    ) -> Tensor:
        """Get bev feat."""
        feats = seq_feats[i]
        bs = feats[0].shape[0]
        device = feats[0].device

        prev_bev, prev_meta = self.get_prev_bev(i, bs, device, seq_meta[i])
        ref2d, ref3d = self.export_reference_points(bs, device=device)
        reference_points_cam, _ = self.point_sampling(
            ref3d, self.pc_range, seq_meta[i], im_shape
        )

        hybird_ref_2d = self.get_fusion_ref(ref2d)
        norm_coords = self.export_bev_transform_points(
            seq_meta[i], prev_meta
        ).to(device)

        bev_emb = self.bev_encoder(
            feats,
            prev_bev,
            hybird_ref_2d,
            reference_points_cam,
            norm_coords,
        )

        return bev_emb

    @fx_wrap()
    def get_fusion_ref(self, ref_2d: Tensor) -> Tensor:
        """Get refpoints."""
        shift_ref_2d = ref_2d.clone()
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
            bs * 2, len_bev, num_bev_level, 2
        )
        return hybird_ref_2d

    @fx_wrap()
    def get_prev_bev(
        self, idx: int, bs: int, device: torch.device, cur_meta: Dict
    ) -> Tuple[Tensor, Dict]:
        """Get prev bevfeat and metas."""
        if idx == self.queue_length - 1 and self.queue_length != 1:
            prev_bev = torch.zeros(
                (bs, self.bev_h * self.bev_w, self.embed_dims),
                dtype=torch.float32,
                device=device,
            )
            prev_meta = None
        else:
            prev_bev = self.prev_frame_info["prev_bev"]
            prev_meta = self.prev_frame_info
            if prev_bev is None:
                prev_bev = torch.zeros(
                    (bs, self.bev_h * self.bev_w, self.embed_dims),
                    dtype=torch.float32,
                    device=device,
                )
                prev_meta = None

        if prev_meta is not None:
            pre_scene = prev_meta["scene_token"]
            for i in range(bs):
                if pre_scene[i] != cur_meta["meta"][i]["scene"]:
                    prev_bev[i] = torch.zeros(
                        (self.bev_h * self.bev_w, self.embed_dims),
                        dtype=torch.float32,
                        device=device,
                    )
        return prev_bev, prev_meta

    @fx_wrap()
    def export_bev_transform_points(
        self, cur_meta: Dict, pre_meta: Dict
    ) -> Tensor:
        """Get normed coords for bevfeat transformer."""
        bs = len(cur_meta["meta"])
        real_range = (
            self.pc_range[0],
            self.pc_range[1],
            self.pc_range[3],
            self.pc_range[4],
        )

        coords = gen_coords(self.bev_size, self.pc_range)

        bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
            real_range,
            self.grid_resolution,
        )

        bs_new_coords = []
        if pre_meta is not None:
            pre_scene = pre_meta["scene_token"]
            for i in range(bs):
                if pre_scene[i] != cur_meta["meta"][i]["scene"]:
                    new_coords = coords.clone()
                else:
                    new_coords = self.get_refine_coords(
                        coords, cur_meta, pre_meta, i
                    )
                bs_new_coords.append(new_coords)
        else:
            new_coords = coords.clone().repeat(bs, 1, 1, 1)
            bs_new_coords.append(new_coords)

        bs_new_coords = torch.cat(bs_new_coords, dim=0)
        bs_new_coords[..., 0] = (bs_new_coords[..., 0] - bev_min_x) / (
            (bev_max_x - bev_min_x)
        )
        bs_new_coords[..., 1] = (bs_new_coords[..., 1] - bev_min_y) / (
            (bev_max_y - bev_min_y)
        )

        norm_coords = (bs_new_coords * 2 - 1).to(torch.float32)
        return norm_coords

    def forward(
        self,
        feats: List[Tensor],
        data: Dict,
    ) -> Tensor:
        """Forward bevformer viewtransformer."""
        if not self.is_compile:
            seq_feats = self.get_seq_feats(feats)
            bev_emb = self.get_fusion_bev(data, seq_feats)
        else:
            feats = self.get_deploy_feats(feats)
            hybird_ref_2d = data["hybird_ref_2d"]
            prev_bev = data["prev_bev"]
            reference_points_cam = data["reference_points_cam"]
            prev_bev_ref = data["prev_bev_ref"]
            bev_emb = self.bev_encoder(
                feats,
                prev_bev,
                hybird_ref_2d,
                reference_points_cam,
                prev_bev_ref,
            )
        return bev_emb

    @fx_wrap()
    def get_deploy_feats(self, feats: List[Tensor]) -> List[Tensor]:
        """Get deploy feats for view transformer inputs."""
        feats = [feats[i] for i in self.in_indices]
        for idx, feat in enumerate(feats):
            _, c, h, w = feat.shape
            feats[idx] = feat.view(-1, self.numcam, c, h, w)
        return feats

    @fx_wrap()
    def get_seq_feats(self, feats: List[Tensor]) -> List[List[Tensor]]:
        """Get seq feats for view transformer inputs."""
        feats = [feats[i] for i in self.in_indices]
        seq_feats = [[] for _ in range(self.queue_length)]
        for _, feat in enumerate(feats):
            _, c, h, w = feat.shape
            tmp_feat = feat.view(self.queue_length, -1, self.numcam, c, h, w)
            for i in range(self.queue_length):
                feat_i = tmp_feat[i]
                seq_feats[i].append(feat_i)
        return seq_feats

    @fx_wrap()
    def get_bev_pos(self, bs: int, device: torch.device) -> Tensor:
        """Get bev positional encoding."""
        bev_mask_pos = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=device
        ).to(torch.float32)
        bev_pos = self.positional_encoding(bev_mask_pos).to(torch.float32)
        return bev_pos

    def bev_encoder(
        self,
        mlvl_feats: List[Tensor],
        prev_bev: Tensor,
        hybird_ref_2d: Tensor,
        reference_points_cam: Tensor,
        norm_coords: Tensor,
    ) -> Tensor:
        """Encode imgfeat  to get bev feats."""
        bs = mlvl_feats[0].shape[0]
        device = mlvl_feats[0].device
        bev_query = self.bev_embedding.weight
        bev_query = bev_query.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = self.get_bev_pos(bs, device)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        bev_query = self.quant_bev_query(bev_query)
        bev_pos = self.quant_bev_pos(bev_pos)
        hybird_ref_2d = self.quant_hybird_ref_2d(hybird_ref_2d)
        prev_bev = self.quant_prev_bev(prev_bev)
        norm_coords = self.quant_norm_coords(norm_coords)

        tmp_prev_bev = prev_bev.reshape(
            bs, self.bev_h, self.bev_w, self.embed_dims
        ).permute(0, 3, 1, 2)
        prev_bev = F.grid_sample(
            tmp_prev_bev, norm_coords, "bilinear", "zeros", True
        )
        prev_bev = prev_bev.reshape(
            bs, self.embed_dims, self.bev_h * self.bev_w
        ).permute(0, 2, 1)

        reference_points_cam = self.quant_reference_points_cam(
            reference_points_cam
        )

        bev_embed = self.encoder(
            bev_query,
            mlvl_feats,
            bev_pos,
            prev_bev,
            hybird_ref_2d,
            reference_points_cam,
        )
        return bev_embed

    @fx_wrap()
    def get_refine_coords(
        self,
        coords: Tensor,
        img_metas: Dict,
        img_metas_pre: Dict,
        idx: int,
    ):
        """Get refine coords."""
        cur_e2g = img_metas["ego2global"][idx]
        cur_e2g = np.array(cur_e2g)
        prev_e2g = img_metas_pre["ego2global"][idx]
        prev_e2g = np.array(prev_e2g)
        prev_g2e = np.linalg.inv(prev_e2g)
        wrap_m = prev_g2e @ cur_e2g
        wrap_m = wrap_m[None, ...]
        wrap_r_t = wrap_m[:, :2, :3]
        # Extract the translation component
        trans = np.eye(3)[None, :, :]
        trans[:, :2, :3] = wrap_r_t
        output = trans.transpose((0, 2, 1))
        wrap_r1 = torch.tensor(output, dtype=torch.float32)
        ones = torch.ones((1, self.bev_h, self.bev_w, 1), dtype=torch.float32)
        coords_3d = torch.cat([coords, ones], dim=-1)
        new_coords_all = []
        batch = wrap_r1.shape[0]
        for i in range(batch):
            new_coord = torch.matmul(coords_3d, wrap_r1[i].double()).float()
            new_coords_all.append(new_coord)
        new_coords_all = torch.cat(new_coords_all)
        return new_coords_all[..., :2]

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        int16_models = [
            self.quant_hybird_ref_2d,
            self.quant_norm_coords,
            self.quant_reference_points_cam,
        ]

        for m in int16_models:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )

        self.positional_encoding.qconfig = None
        if hasattr(self.encoder, "set_qconfig"):
            self.encoder.set_qconfig()

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""
        if hasattr(self.encoder, "fuse_model"):
            self.encoder.fuse_model()
