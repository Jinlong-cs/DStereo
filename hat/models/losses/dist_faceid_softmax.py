# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info

__all__ = [
    "DistFCCrossEntropyLoss",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class DistFCCrossEntropyLoss(nn.Module):
    """Calculate dist cross entropy loss of faceid.

    Args:
        resume: Select whether to restore the weight of softmax.
        margin_loss: A function of margin loss,
            eg: cosface, arcface.
        num_classes: The number of class center storage in
            current rank(CPU/GPU), usually is total_classes // world_size.
        embedding_size: The feature dimension.
        loss_name: The key of loss in return dict. if None, return loss.
        prefix: Path for save checkpoint.
        seed: Random seed.
        gpu_per_device: Gpu num per device.
    """

    def __init__(
        self,
        resume: bool,
        margin_loss: nn.Module,
        num_classes: int,
        embedding_size: int = 512,
        loss_name: str = None,
        prefix: str = "./",
        seed: int = 1234,
        gpu_per_device: int = 8,
    ):
        super(DistFCCrossEntropyLoss, self).__init__()

        rank, world_size = get_dist_info()
        self.num_classes = num_classes
        self.rank = rank
        self.local_rank = rank % gpu_per_device
        self.world_size = world_size
        self.device: torch.device = torch.device(
            "cuda:{}".format(self.local_rank)
        )
        self.num_local = num_classes // world_size + int(
            rank < num_classes % world_size
        )
        self.class_start = num_classes // world_size * rank + min(
            rank, num_classes % world_size
        )
        self.loss_name = loss_name
        self.prefix = prefix
        self.embedding_size = embedding_size
        self.margin_loss = margin_loss

        self.stream: torch.cuda.Stream = torch.cuda.Stream(self.local_rank)

        self.weight_name = os.path.join(
            self.prefix, "rank_{}_softmax_weight.pt".format(self.rank)
        )

        generator = torch.Generator(device=self.device)
        self.generator = generator.manual_seed(seed + self.rank)

        if resume:
            assert os.path.exists(
                self.weight_name
            ), f"weight path {self.weight_name} not exists"
            logger.info(
                "softmax weight resume from {}".format(self.weight_name)
            )
            self.weight: torch.Tensor = torch.load(self.weight_name).cuda(
                self.local_rank
            )

            if self.weight.shape[0] != self.num_local:
                AssertionError(
                    f"shape not equal, {self.weight.shape[0]}"
                    f" != {self.num_local}"
                )
            logger.info("softmax weight resume successfully!")
        else:
            self.weight = torch.normal(
                0,
                0.01,
                (self.num_local, self.embedding_size),
                device=self.device,
                generator=self.generator,
            )
            logger.info("softmax weight init successfully!")
        self.weight = nn.Parameter(self.weight)

    def prepare(self, features, target):

        batch_size = features.size()[0]
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            total_target = torch.zeros(
                size=[batch_size * self.world_size],
                device=self.device,
                dtype=torch.long,
            )
            dist.all_gather(
                list(total_target.chunk(self.world_size, dim=0)), target
            )
            index_positive = (self.class_start <= total_target) & (
                total_target < self.class_start + self.num_local
            )
            total_target[~index_positive] = -1
            total_target[index_positive] -= self.class_start

        gathered_features = all_gather_feature(features)

        return total_target, gathered_features

    @autocast(enabled=False)
    def forward(self, features, target, optimizer=None):
        total_target, total_features = self.prepare(features, target)

        norm_weight = F.normalize(self.weight)
        norm_feat = F.normalize(total_features)
        logits = dist_matmul(norm_feat, norm_weight)
        torch.cuda.current_stream().wait_stream(self.stream)

        theta = self.margin_loss(logits, total_target)
        loss = dist_softmax(theta, total_target)

        if self.loss_name is None:
            return loss
        else:
            return {self.loss_name: loss}


class DistMatmulFunc(Function):
    @staticmethod
    def forward(ctx, features, w):
        outputs = F.linear(features, w)
        ctx.save_for_backward(features, w)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        world_size = dist.get_world_size()

        total_features, w = ctx.saved_tensors
        grad_logits = torch.mm(grad_output, w)
        dist.all_reduce(grad_logits, dist.ReduceOp.SUM)

        grad_w = torch.mm(torch.t(total_features), grad_output)

        return grad_logits * world_size, torch.t(grad_w)


class FeatureGatherFunc(Function):
    @staticmethod
    def forward(ctx, features):
        batch_size, embedding_size = features.size()
        world_size = dist.get_world_size()
        total_features = torch.zeros(
            size=[batch_size * world_size, embedding_size],
            device=features.device,
        )
        dist.all_gather(
            list(total_features.chunk(world_size, dim=0)),
            features.detach().clone(),
        )

        return total_features

    @staticmethod
    def backward(ctx, grad_output):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        batch_size = grad_output.size()[0] // world_size
        return grad_output[rank * batch_size : (rank + 1) * batch_size, :]


class SoftmaxFunc(Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, total_labels: torch.Tensor):
        max_fc = torch.max(logits, dim=1, keepdim=True)[0]
        dist.all_reduce(max_fc, dist.ReduceOp.MAX)
        logits_exp = torch.exp(logits - max_fc)
        logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)
        dist.all_reduce(logits_sum_exp, dist.ReduceOp.SUM)

        logits_exp.div_(logits_sum_exp)

        grad = logits_exp * 1.0
        ctx.save_for_backward(grad, total_labels)
        index = torch.where(total_labels != -1)[0]
        loss = logits_exp[index].gather(1, total_labels[index].view(-1, 1))
        loss = loss.clamp_min_(1e-30).log_().sum() * (-1.0)
        dist.all_reduce(loss, dist.ReduceOp.SUM)
        loss = loss / (total_labels.size()[0])

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad, total_labels = ctx.saved_tensors

        index = torch.where(total_labels != -1)[0]
        one_hot = torch.zeros(
            size=[index.size()[0], grad.size()[1]], device=grad.device
        )
        one_hot.scatter_(1, total_labels[index, None], 1)
        grad[index] -= one_hot
        grad.div_(total_labels.size()[0])

        return grad * grad_output, None


all_gather_feature = FeatureGatherFunc.apply
dist_matmul = DistMatmulFunc.apply
dist_softmax = SoftmaxFunc.apply
