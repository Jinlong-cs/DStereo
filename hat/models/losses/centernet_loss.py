from torch import nn
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY
from .real3d_losses import hm_focal_loss, hm_l1_loss, sigmoid_and_clip

__all__ = ["CenterNetLoss"]


@OBJECT_REGISTRY.register
class CenterNetLoss(nn.Module):
    def __init__(
        self,
        task,
        loss_weights,
        heatmap_type,
        reg_keys,
    ):
        super(CenterNetLoss, self).__init__()
        self.task = task
        self.loss_weights = loss_weights
        self.heatmap_type = heatmap_type
        self.reg_keys = reg_keys

    @autocast(enabled=False)
    def forward(self, pred, target):
        # convert to float32 while using amp
        for k, v in pred.items():
            pred[k] = v.float()
        ignore_mask = target["ignore_mask"].float()
        all_losses = {}
        hm_weight = self.loss_weights["hm"]
        hm = sigmoid_and_clip(pred["hm"])
        hm_loss = hm_focal_loss(hm, target["hm"], ignore_mask)
        hm_loss = hm_loss * hm_weight
        all_losses[self.task + "_" + "hm_loss"] = hm_loss

        for key in self.reg_keys:
            if key not in pred:
                continue
            weight = self.loss_weights[key]
            loss = hm_l1_loss(
                pred[key],
                target[key],
                target["hm"],
                ignore_mask,
                heatmap_type=self.heatmap_type[key],
            )
            loss = loss * weight
            all_losses[self.task + "_" + "{}_loss".format(key)] = loss

        return all_losses
