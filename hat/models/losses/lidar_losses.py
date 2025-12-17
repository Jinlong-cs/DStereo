# Copyright (c) Horizon Robotics, all rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY
from .utils import transpose_and_gather_feat

__all__ = [
    "LidarRegLoss",
    "LidarFastFocalLoss",
    "LidarSmoothL1RegLoss",
]


@OBJECT_REGISTRY.register
class LidarRegLoss(nn.Module):
    """Regression loss for an output tensor."""

    def __init__(self):
        super(LidarRegLoss, self).__init__()

    def forward(
        self,
        output: torch.Tensor,
        mask: torch.Tensor,
        ind: torch.Tensor,
        target: torch.Tensor,
    ):
        """Forward calculation.

        Args:
            output (torch.Tensor): (batch x dim x h x w)
            mask (torch.Tensor): (batch x max_objects)
            ind (torch.Tensor): (batch x max_objects)
            target (torch.Tensor): (batch x max_objects x dim)
        """
        pred = transpose_and_gather_feat(output, ind)
        mask = mask.float().unsqueeze(2)

        loss = F.l1_loss(pred * mask, target * mask, reduction="none")
        loss = loss / (mask.sum() + 1e-4)
        loss = loss.transpose(2, 0).sum(dim=2).sum(dim=1)
        return loss


@OBJECT_REGISTRY.register
class LidarFastFocalLoss(nn.Module):
    """Focal loss, exactly the same as the CornerNet version.

    Faster and costs much less memory.
    """

    def __init__(self):
        super(LidarFastFocalLoss, self).__init__()

    def forward(
        self,
        out: torch.Tensor,
        target: torch.Tensor,
        ind: torch.Tensor,
        mask: torch.Tensor,
        cat: torch.Tensor,
    ) -> float:
        """Forward calculation.

        Args:
            out (torch.Tensor): B x C x H x W tensor for output of the model.
            target (torch.Tensor): B x C x H x W tensor for training target.
            ind (torch.Tensor): B x M index tensor for tensor gathering.
            mask (torch.Tensor): B x M mask tensor for tensor masking.
            cat (torch.Tensor): B x M tensor indicating category id for peaks.

        Returns:
            [float]: loss value.
        """

        mask = mask.float()
        gt = torch.pow(1 - target, 4)
        neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
        neg_loss = neg_loss.sum()

        pos_pred_pix = transpose_and_gather_feat(out, ind)  # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2))  # B x M
        num_pos = mask.sum()
        pos_loss = (
            torch.log(pos_pred)
            * torch.pow(1 - pos_pred, 2)
            * mask.unsqueeze(2)
        )
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos


@OBJECT_REGISTRY.register
class LidarSmoothL1RegLoss(nn.Module):
    """Regression loss for an output tensor."""

    def __init__(self):
        super(LidarSmoothL1RegLoss, self).__init__()

    def forward(
        self,
        output: torch.Tensor,
        mask: torch.Tensor,
        ind: torch.Tensor,
        target: torch.Tensor,
    ):
        """Forward calculation.

        Args:
            output (torch.Tensor): B x C x H x W tensor for output of model.
            mask (torch.Tensor): B x M mask tensor for tensor masking.
            ind (torch.Tensor): B x M index tensor for tensor gathering.
            target (torch.Tensor): B x C x H x W tensor for training target.

        Returns:
            [float]: loss value.
        """
        pred = transpose_and_gather_feat(output, ind)
        mask = mask.float().unsqueeze(2)

        loss = F.smooth_l1_loss(pred * mask, target * mask, reduction="none")
        loss = loss / (mask.sum() + 1e-4)
        loss = loss.transpose(2, 0).sum(dim=2).sum(dim=1)
        return loss


class AfdetFocalLoss(nn.Module):
    def __init__(self):
        super(AfdetFocalLoss, self).__init__()
        """Anchor free detection facal loss."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ignore_mask: torch.Tensor = None,
    ):
        if ignore_mask is None:
            pos_inds = target.eq(1).float()
            neg_inds = target.lt(1).float()
        else:
            pos_inds = (target.eq(1) & ~ignore_mask).float()
            neg_inds = (target.lt(1) & ~ignore_mask).float()
        neg_weights = torch.pow(1 - target, 4)
        loss = 0
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = (
            torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class BoundaryLoss(nn.Module):
    def __init__(self, theta0: float = 3, theta: float = 5):
        """Boundary Loss for Remote Sensing Imagery Semantic Segmentation.

        you can read paper https://arxiv.org/abs/1905.07852.

        Args:
            theta0: kernel size of boundary map.
            theta: kernel size of extended boundary map.

        """
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            pred: the output from model (before softmax)
                    shape (N, C, H, W).
            gt: ground truth map
                    shape (N, H, w).
        Returns:
            boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        # one_hot_gt = self.one_hot(gt, c)
        one_hot_gt = gt

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        )
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        )
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b,
            kernel_size=self.theta,
            stride=1,
            padding=(self.theta - 1) // 2,
        )

        pred_b_ext = F.max_pool2d(
            pred_b,
            kernel_size=self.theta,
            stride=1,
            padding=(self.theta - 1) // 2,
        )

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (
            torch.sum(pred_b, dim=2) + 1e-7
        )
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (
            torch.sum(gt_b, dim=2) + 1e-7
        )

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss

    def one_hot(
        self, label: torch.Tensor, n_classes: int, requires_grad=True
    ) -> torch.Tensor:
        """Return one Hot Label.

        Args:
            label: origin label with type class.
            n_classes: num classes.
            requires_grad: requires grad or not.
        Returns:
            one_hot_label: one hot label from ori label.
        """
        device = label.device
        one_hot_label = torch.eye(
            n_classes, device=device, requires_grad=requires_grad
        )[label]
        one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

        return one_hot_label


@OBJECT_REGISTRY.register
class WeightedSmoothL1Loss(nn.Module):
    """Smooth L1 localization loss function.

    The smooth L1_loss is defined elementwise as .5 x^2 if abs(x)<1
    and abs(x)-.5 otherwise, where x is the difference between
    predictions and target.
    See also Equation (3) in the Fast R-CNN paper by Ross Girshick
    """

    def __init__(
        self,
        sigma=3.0,
        reduction="mean",
        code_weights=None,
        codewise=True,
        loss_weight=1.0,
        raw_l1=False,
    ):
        super(WeightedSmoothL1Loss, self).__init__()

        # just l1 loss
        self.raw_l1 = raw_l1

        print("Raw L1 Loss:", raw_l1)

        self._sigma = sigma

        if code_weights is not None:
            self._code_weights = torch.tensor(
                code_weights, dtype=torch.float32
            )
        else:
            self._code_weights = None

        # self._code_weights = None

        self._codewise = codewise
        self._reduction = reduction
        self._loss_weight = loss_weight

    def forward(self, prediction_tensor, target_tensor, weights=None):
        """Compute loss function.

        Args:
            prediction_tensor: Float tensor of shape [batch_size, num_anchors,
                code_size] represent the predicted locations of objects.
            target_tensor: Float tensor of shape [batch_size, num_anchors,
                code_size] representing the regression targets
            weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
            loss: a float tensor of shape [batch_size, num_anchors] tensor
                representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor
        if self._code_weights is not None:
            diff = self._code_weights.view(1, 1, -1).to(diff.device) * diff
        abs_diff = torch.abs(diff)

        if not self.raw_l1:
            abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma ** 2)).type_as(
                abs_diff
            )
            loss = abs_diff_lt_1 * 0.5 * torch.pow(
                abs_diff * self._sigma, 2
            ) + (abs_diff - 0.5 / (self._sigma ** 2)) * (1.0 - abs_diff_lt_1)
        else:
            loss = abs_diff

        if self._codewise:
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm *= weights.unsqueeze(-1)
        else:
            anchorwise_smooth_l1norm = torch.sum(loss, 2)  # * weights
            if weights is not None:
                anchorwise_smooth_l1norm *= weights

        return anchorwise_smooth_l1norm
