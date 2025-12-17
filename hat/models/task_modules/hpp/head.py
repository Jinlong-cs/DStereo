import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from .utils import HPPBlock

__all__ = ["HPPDecodeHead"]


@OBJECT_REGISTRY.register
class HPPDecodeHead(nn.Module):
    def __init__(
        self,
        block_num: int = 4,
        in_channels: int = None,
        out_channels: int = None,
    ):
        """HPP Decode Head.

        Args:
            block_num (int, optional): the number of hourglass block.
                Defaults to 4.
        """
        super(HPPDecodeHead, self).__init__()
        self.hourglass_block_list = nn.ModuleList()
        for _ in range(block_num):
            self.hourglass_block_list.append(
                HPPBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
            )

    def forward(self, x):
        res = {
            "offsets": [],
            "confidences": [],
            "instances": [],
            "att_feats": [],
        }  # noqa c408
        for hourglass_block in self.hourglass_block_list:
            (
                out_confidence,
                out_offset,
                out_instance,
                out,
                att_feat,
            ) = hourglass_block(x)
            res["offsets"].append(out_offset)
            res["confidences"].append(out_confidence)
            res["instances"].append(out_instance)
            res["att_feats"].append(att_feat)
            x = out
        return res

    def fuse_model(self):
        for module in self.hourglass_block_list:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
