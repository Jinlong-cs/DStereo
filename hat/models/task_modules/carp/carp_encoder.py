# Copyright (c) Horizon Robotics, All rights reserved.

from torch import nn

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "CocktailE2EStructureAEncoder",
]


@OBJECT_REGISTRY.register
class CocktailE2EStructureAEncoder(nn.Module):
    def __init__(
        self,
        vfea_module: nn.Module,
        afea_module: nn.Module,
        mix_module: nn.Module,
        shard_encoder: nn.Module,
    ):
        super().__init__()
        self.vfea_module = vfea_module
        self.afea_module = afea_module
        self.mix_module = mix_module
        self.shard_encoder = shard_encoder

    def forward(self, afea, vfea, a_lens, v_lens):
        if self.afea_module is not None:
            afea = self.afea_module(afea)
        if self.vfea_module is not None:
            vfea = self.vfea_module(vfea)
        fea = self.mix_module(afea, vfea)
        return self.shard_encoder(fea, v_lens)
