from torch import nn
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY

# from .real3d_losses import hm_l1_loss
from .smooth_l1_loss import SmoothL1Loss

__all__ = ["TollGateLoss"]


@OBJECT_REGISTRY.register
class TollGateLoss(nn.Module):
    """TollGate Loss.

    Args:
        task: TasK name.
        loss_weight: Loss weight for hm loss and offset loss.
    """

    def __init__(
        self,
        task: str = "tollgate",
        loss_weights: dict = None,
    ):
        super(TollGateLoss, self).__init__()
        self.task = task
        self.loss_weights = loss_weights
        self.hm = SmoothL1Loss()
        self.off_smoothl1 = SmoothL1Loss()

    @autocast(enabled=False)
    def forward(self, pred, target):
        for k, v in pred.items():
            pred[k] = v.float()
        all_losses = {}
        hm_weight = self.loss_weights["hm"]
        off_weight = self.loss_weights["offset"]
        hm_loss = self.hm(
            pred["hm"], target["gt_heatmap"], target["gt_heatmap_weight"]
        )
        # off_loss = hm_l1_loss(
        #     pred["offset"],
        #     target["gt_offset"],
        #     target["gt_offset_weight"],
        #     None,
        #     heatmap_type="point",
        # )
        off_loss = self.off_smoothl1(
            pred["offset"],
            target["gt_offset"],
            target["gt_offset_weight"],
        )

        hm_loss = hm_loss * hm_weight
        off_loss = off_loss * off_weight
        all_losses[self.task + "_" + "hm_loss"] = hm_loss
        all_losses[self.task + "_" + "off_hm_l1_loss"] = off_loss
        return all_losses
