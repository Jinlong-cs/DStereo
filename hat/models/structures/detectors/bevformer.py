# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, List, Optional, Sequence

from torch import Tensor, nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["BevFormer"]


@OBJECT_REGISTRY.register
class BevFormer(nn.Module):
    """The basic structure of BevFormer.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        view_transformer:
            View transformer module for transforming from img view to bev view.
        out_indices: Out indices for backbone.
        bev_decoders: Decoder for bev feature.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: Optional[nn.Module] = None,
        view_transformer: nn.Module = None,
        out_indices: Sequence[int] = (5,),
        bev_decoders: List[nn.Module] = None,
    ):
        super(BevFormer, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.view_transformer = view_transformer
        self.out_indices = out_indices
        self.bev_decoders = nn.ModuleList(bev_decoders)

    def extract_feat(self, img: Tensor) -> List[Tensor]:
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        x = [x[i] for i in self.out_indices]
        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward(self, data: Dict) -> List:
        imgs = data["img"]
        feats = self.extract_feat(imgs)
        bev_feat = self.view_transformer(feats, data)

        results = []
        for bev_decoder in self.bev_decoders:
            result = bev_decoder(bev_feat, data)
            results = self._update_res(result, results)
        return results

    @fx_wrap()
    def _update_res(self, result: List, results: List) -> List:
        if result:
            results.extend(_as_list(result))
        return results

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""
        for module in [
            self.backbone,
            self.neck,
            self.view_transformer,
        ]:
            if module is not None:
                if hasattr(module, "fuse_model"):
                    module.fuse_model()
        for m in self.bev_decoders:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [
            self.backbone,
            self.neck,
            self.view_transformer,
        ]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
        for m in self.bev_decoders:
            if hasattr(m, "set_qconfig"):
                m.set_qconfig()
