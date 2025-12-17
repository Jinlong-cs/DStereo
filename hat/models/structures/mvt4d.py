import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["MVT4D"]


@OBJECT_REGISTRY.register
class MVT4D(nn.Module):
    """
    The basic structure of mvt4d.

    Args:
        backbone: Backbone module.
        img_neck: Neck module.
        bev_feat_encoder: Bev Feat Encoder module.
        pts_bbox_head: Detector Head for 3D Detection.
        losses: Loss module for MVT4D.
        postprocess: Postprocess module.
        bev_h: bev feat height.
        bev_w: bev feat weight.
        pc_range: point cloud range, same as vcs_range.
        enable_temporal_fusion: Whether to use temporal fusion.
    """

    def __init__(
        self,
        backbone: nn.Module,
        img_neck: nn.Module,
        bev_feat_encoder: nn.Module,
        pts_bbox_head: nn.Module,
        losses: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
        bev_h: int = 100,
        bev_w: int = 100,
        pc_range: List[float] = None,
        enable_temporal_fusion: bool = False,
    ):
        super(MVT4D, self).__init__()

        self.backbone = backbone
        self.img_neck = img_neck
        self.bev_feat_encoder = bev_feat_encoder
        self.pts_bbox_head = pts_bbox_head
        self.losses = losses
        self.postprocess = postprocess

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range

        self.embed_dims = self.bev_feat_encoder.embed_dims

        self.enable_temporal_fusion = enable_temporal_fusion
        self.his_bev_feat = None
        self.his_img_metas = None

    def extract_img_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(
        self, img: torch.Tensor, img_metas: Dict[str, Any]
    ) -> List[torch.Tensor]:
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)

        if ("img_horizontal_flip" in img_metas) and (
            img_metas["img_horizontal_flip"][0]
        ):
            # torch.Size([1, 6, 256, 10, 25])
            img_feats = [
                torch.flip(img_f, [len(img_f.shape) - 1])
                for img_f in img_feats
            ]

        return [x.float() for x in img_feats]

    def extract_bev_feat(
        self,
        img_feats: List[torch.Tensor],
        img_metas: Dict[str, Any],
        pre_bev_feat: Optional[torch.Tensor] = None,
        pre_img_metas: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if self.bev_feat_encoder is not None:
            bev_feats = self.bev_feat_encoder(
                img_feats, img_metas, pre_bev_feat, pre_img_metas
            )
        else:
            raise AttributeError("bev_feat_encoder must be not None")
        return bev_feats

    def forward_imgs_train(
        self,
        img_feats: List[torch.Tensor],
        gt_bboxes_3d: torch.Tensor,
        gt_labels_3d: torch.Tensor,
        img_metas: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:

        bev_feat = self.bev_feat_encoder(
            img_feats, img_metas, self.his_bev_feat, self.his_img_metas
        )
        outs = self.pts_bbox_head(
            bev_feat, img_metas, self.his_img_metas, img_feats
        )

        if self.enable_temporal_fusion:
            self.his_bev_feat = bev_feat.clone().detach()
            self.his_img_metas = copy.deepcopy(img_metas)

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.losses(*loss_inputs)  # type: ignore
        return losses

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: Dict[str, Any],
        gt_bboxes_3d: torch.Tensor,
        gt_labels_3d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = OrderedDict()
        losses_imgs = self.forward_imgs_train(
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
        )

        losses.update(losses_imgs)

        return losses

    def forward_test(
        self,
        img: torch.Tensor,
        img_metas: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:

        if self.enable_temporal_fusion:
            if self.his_img_metas is not None and len(
                self.his_img_metas["timestamp"]
            ) != len(img_metas["timestamp"]):
                self.his_bev_feat = None
                self.his_img_metas = None

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        cur_bev_feat = self.bev_feat_encoder(
            img_feats,
            img_metas,
            self.his_bev_feat,
            self.his_img_metas,
        )
        outs = self.pts_bbox_head(
            cur_bev_feat, img_metas, self.his_img_metas, img_feats
        )
        bbox_list = self.postprocess(outs)  # type: ignore
        bbox_results = [
            {
                "boxes_3d": bboxes.cpu(),
                "scores_3d": scores.cpu(),
                "labels_3d": labels.cpu(),
            }
            for bboxes, scores, labels in bbox_list
        ]

        if self.enable_temporal_fusion:
            self.his_bev_feat = cur_bev_feat.clone()
            self.his_img_metas = copy.deepcopy(img_metas)

        return bbox_results

    def forward(self, data: Dict[str, Any]) -> Any:

        if self.training and self.losses is not None:
            losses = self.forward_train(**data)
            return losses
        else:
            preds = self.forward_test(**data)
            return preds
