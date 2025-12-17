# Copyright (c) Horizon Robotics. All rights reserved.
#
# Implementation of aggregator of Multipath++ model.
# https://arxiv.org/abs/2111.14973
#
# For any implementation details please refer to the developer guide:
# https://horizonrobotics.feishu.cn/docs/doccnBr8XXeUQG3MeNHAr0tcZxb
from typing import List, Union

import torch
import torch.nn as nn


class MultiContextGating(nn.Module):
    """Multi-Context Gating described in the MultiPath++ paper."""

    def __init__(
        self,
        N: int,
        feat_dim: int,
        context_dim: int,
        hidden_dim: int,
        input_size: Union[List[int], None] = None,
        pool_method: str = "max",
    ):
        """Initialize module.

        Args:
            N: number of CG blocks to use.
            feat_dim: size of the feature dimension.
            context_dim: size of the context dimension.
            hidden_dim: size of the hidden dimension.
            input_size: Dimension of each input
            apart from the feat dimension. Changes the behavior of LayerNorm
            in the module. For example, if a sample has
            [dim1, dim2, ..., feat_size], then use [dim1, dim2, ...] as
            input_size.
            Defaults to None, which causes the LayerNorm to normalize only the
            feat dimension.
            pool_method: Pooling method. Can choose from ["max", "mean"].
                Defaults to "max".
        """
        super().__init__()
        assert N >= 1, "Number of stacked ContextGating blocks should be >=1."
        assert pool_method in (
            "max",
            "mean",
        ), "Pooling method must be one of (max, mean)."
        self.pool_method = pool_method
        self.N = N
        self.hidden_dim = hidden_dim
        norm_shape = (
            hidden_dim if input_size is None else input_size + [hidden_dim]
        )

        self.feat_mlp = nn.ModuleList()
        self.context_mlp = nn.ModuleList()
        self.feat_norms = nn.ModuleList()
        self.context_norms = nn.ModuleList()
        for i in range(N):
            if i == 0:
                self.feat_mlp.append(
                    nn.Linear(feat_dim, hidden_dim, bias=True)
                )
                self.context_mlp.append(
                    nn.Linear(context_dim, hidden_dim, bias=True)
                )
            else:
                self.feat_mlp.append(
                    nn.Linear(hidden_dim, hidden_dim, bias=True)
                )
                self.context_mlp.append(
                    nn.Linear(hidden_dim, hidden_dim, bias=True)
                )
            self.feat_norms.append(nn.LayerNorm(norm_shape))
            self.context_norms.append(nn.LayerNorm(hidden_dim))

        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(
        self,
        feats: torch.Tensor,
        context: Union[torch.Tensor, None],
        mask: Union[torch.Tensor, None] = None,
    ):
        """Forward module.

        Args:
            feats: [N, max_nodes, feat_size] tensor
            context: [N, context_size] tensor
            mask: [N, max_nodes, 1] float tensor for masking
                feats. 1 means valid and 0 invalid.
        """
        # Sum of feats and contexts, used to calculate running average
        sum_feats = 0
        sum_contexts = 0
        for i in range(self.N):
            # Compute running average of inputs feats and context
            # NOTE: the first input is not tracked in the running average since
            # the size might differ. You can remove the if condition if your
            # input size is the same as hidden_dim, and change i in the
            # denominator to i+1.
            if i > 0:
                sum_feats = sum_feats + feats
                feats = sum_feats / i
                sum_contexts = sum_contexts + context
                context = sum_contexts / i

            feats = self.leaky_relu(
                self.feat_norms[i](self.feat_mlp[i](feats))
            )
            if context is None:
                context = torch.ones(
                    (feats.shape[0], self.hidden_dim),
                    dtype=feats.dtype,
                    device=feats.device,
                )
            else:
                # NOTE: LayerNorm applied to ones vector results in all zeroes.
                # so we apply LayerNorm only when the input context is not
                # None.
                context = self.context_norms[i](self.context_mlp[i](context))
            context = self.leaky_relu(context)
            feats = feats * context.unsqueeze(1)

            if mask is not None:
                feats_masked = feats * mask
                if self.pool_method == "mean":
                    context = feats_masked.sum(dim=1) / mask.sum(dim=1)
                else:
                    feats_masked[feats_masked == 0] = -1e6  # for avoid Nan
                    context, _ = feats_masked.max(dim=1)
            else:
                if self.pool_method == "mean":
                    context = feats.mean(dim=1)
                else:
                    context, _ = feats.max(dim=1)

        return feats, context
