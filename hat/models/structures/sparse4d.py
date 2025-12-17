# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch import nn

from hat.registry import OBJECT_REGISTRY

try:
    from hat.models.task_modules.sparse4d.ops import (
        DeformableAggregationFunction as DAF,
    )
except ImportError:
    DAF = None

__all__ = ["Sparse4D"]


@OBJECT_REGISTRY.register
class Sparse4D(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        neck: Optional[nn.Module] = None,
        use_deformable_func: bool = False,
        depth_branch: Optional[nn.Module] = None,
        aux_head: Optional[nn.Module] = None,
    ):
        super(Sparse4D, self).__init__()
        if isinstance(backbone, Sequence):
            backbone = nn.ModuleList(backbone)
        if isinstance(neck, Sequence):
            neck = nn.ModuleList(neck)
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.use_deformable_func = use_deformable_func and DAF is not None
        self.depth_branch = depth_branch
        self.aux_head = aux_head

    def extract_feat(self, img, return_depth=False, metas=None, index=None):
        if isinstance(self.backbone, nn.ModuleList) and index is None:
            assert len(img) == len(self.backbone)
            feature_maps = []
            if return_depth:
                depths = []
            for idx in range(len(self.backbone)):
                out = self.extract_feat(
                    img[idx],
                    return_depth=return_depth,
                    metas=metas,
                    index=idx,
                )
                if return_depth:
                    feature_map, depth = out
                    depths.append(depth)
                else:
                    feature_map = out
                feature_maps.append(feature_map)
            if return_depth:
                return feature_maps, depths
            return feature_maps

        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1

        if isinstance(self.backbone, nn.ModuleList) and index is not None:
            feature_maps = self.backbone[index](img)
        else:
            feature_maps = self.backbone(img)

        if isinstance(self.neck, nn.ModuleList) and index is not None:
            feature_maps = list(self.neck[index](feature_maps))
        elif self.neck is not None:
            feature_maps = list(self.neck(feature_maps))

        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            ).float()

        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = DAF.feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    def forward(self, data: Dict[str, Any]) -> Union[Dict, List]:
        if self.training:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self, **data):
        img = data.pop("img")
        feature_maps, depths = self.extract_feat(img, True, data)

        if "data_queue" in data or "future_data_queue" in data:
            feature_queue = []
            meta_queue = []
            with torch.no_grad():
                for d in data.get("data_queue", []) + data.get(
                    "future_data_queue", []
                ):
                    img = d.pop("img")
                    feature_queue.append(self.extract_feat(img))
                    meta_queue.append(d)
        else:
            feature_queue = None
            meta_queue = None

        model_outs = self.head(feature_maps, data, feature_queue, meta_queue)
        output = self.head.loss(model_outs, data)

        if self.aux_head is not None:
            output.update(self.aux_head(feature_maps, data))

        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, **data):
        img = data.pop("img")
        feature_maps = self.extract_feat(img)
        if "future_data_queue" in data:
            feature_queue = []
            meta_queue = []
            with torch.no_grad():
                for d in data["future_data_queue"]:
                    img = d.pop("img")
                    feature_queue.append(self.extract_feat(img))
                    meta_queue.append(d)
        else:
            feature_queue = None
            meta_queue = None

        model_outs = self.head(feature_maps, data, feature_queue, meta_queue)
        results = self.head.post_process(model_outs)
        return results
