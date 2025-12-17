import numpy as np
import torch
from torch import nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@OBJECT_REGISTRY.register
class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        max_queue_length=-1,
        confidence_decay=0.6,
        anchor_grad=True,
        max_time_interval=2,
        enable_trans_with_vel=True,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.max_queue_length = max_queue_length
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.anchor_handler = anchor_handler
        if self.anchor_handler is not None:
            assert hasattr(self.anchor_handler, "anchor_projection")
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple, np.ndarray)):
            anchor = np.array(anchor)
        elif anchor is None:
            anchor = np.zeros((num_anchor, 11))
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=False,
        )
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.feature_queue = [] if max_queue_length > 0 else None
        self.meta_queue = [] if max_queue_length > 0 else None
        self.enable_trans_with_vel = enable_trans_with_vel

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)

    def get(self, batch_size, metas=None):
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if (
            self.cached_anchor is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            if self.anchor_handler is not None:
                T_temp2cur = (
                    metas["img_metas"]["T_global_inv"]
                    @ self.metas["img_metas"]["T_global"]
                )
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    self.metas["timestamp"],
                    [metas["timestamp"]],
                    self.enable_trans_with_vel,
                )[0]
            self.mask = (
                torch.abs(metas["timestamp"] - self.metas["timestamp"])
                <= self.max_time_interval
            )
        else:
            self.cached_feature = None
            self.cached_anchor = None
            self.confidence = None

        if (
            self.metas is None
            or batch_size != self.metas["timestamp"].shape[0]
        ) and (
            self.meta_queue is None
            or len(self.meta_queue) == 0
            or batch_size != self.meta_queue[0]["timestamp"].shape
        ):
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )
        else:
            if self.metas is not None:
                history_time = self.metas["timestamp"]
            else:
                history_time = self.meta_queue[0]["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            time_interval = torch.where(
                torch.logical_or(
                    time_interval == 0,
                    torch.abs(time_interval) > self.max_time_interval,
                ),
                time_interval.new_tensor(self.default_time_interval),
                time_interval,
            )
        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat(
            [self.cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [self.cached_anchor, selected_anchor], dim=1
        )
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature], dim=1
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.feature_queue is not None:
            while len(self.feature_queue) > self.max_queue_length - 1:
                self.feature_queue.pop()
                self.meta_queue.pop()
            self.feature_queue.insert(0, feature_maps)
            self.meta_queue.insert(0, metas)

        if self.num_temp_instances > 0:
            instance_feature = instance_feature.detach()
            anchor = anchor.detach()
            confidence = confidence.detach()

            self.metas = metas
            confidence = confidence.max(dim=-1).values.sigmoid()
            if self.confidence is not None:
                confidence[:, : self.num_temp_instances] = torch.maximum(
                    self.confidence * self.confidence_decay,
                    confidence[:, : self.num_temp_instances],
                )

            (
                self.confidence,
                (self.cached_feature, self.cached_anchor),
            ) = topk(
                confidence, self.num_temp_instances, instance_feature, anchor
            )
