# Copyright (c) Horizon Robotics. All rights reserved.
#
# Implementation of aggregator of Multipath++ model.
# https://arxiv.org/abs/2111.14973
#
# For any implementation details please refer to developer guide:
# https://horizonrobotics.feishu.cn/docs/doccnBr8XXeUQG3MeNHAr0tcZxb
import torch
import torch.nn as nn

from hat.models.task_modules.traj_pred.structures.multi_context_gating import (
    MultiContextGating,
)
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class MTPPlusAggregator(nn.Module):
    """Multipath++ aggregator.

    Initialize latent anchor with
    orthogonal initialization and got prediction encoding
    using MultiContextGating.
    """

    def __init__(
        self,
        num_anchor: int,
        anchor_emb_size: int,
        pred_mcg_layers: int,
        target_agent_mcg_hidden_size: int,
        target_agent_emb_size: int,
        nbr_agent_mcg_hidden_size: int,
        road_agent_mcg_hidden_size: int,
        pred_mcg_hidden_size: int,
    ):
        """Initialize module.

        Args:
            num_anchor: number of output modalities.
            anchor_emb_size: anchor embedding size.
            pred_mcg_layers: number of prediction MCG layers.
            target_agent_mcg_hidden_size: hidden size of MCG layers for
                neighboring agents.
            target_agent_emb_size: embedding size for target agent.
            nbr_agent_mcg_hidden_size: hidden size of MCG layers for
                neighboring agents.
            road_agent_mcg_hidden_size: road agent MCG hidden layer size.
            pred_mcg_hidden_size: hidden size of prediction MCG layers.
        """
        super().__init__()

        self.anchor_embedding = nn.Embedding(num_anchor, anchor_emb_size)
        nn.init.orthogonal_(self.anchor_embedding.weight)
        self.predictor_MCG = MultiContextGating(
            pred_mcg_layers,
            anchor_emb_size,
            target_agent_mcg_hidden_size
            + target_agent_emb_size
            + nbr_agent_mcg_hidden_size
            + road_agent_mcg_hidden_size,
            pred_mcg_hidden_size,
        )

    def forward(self, context_encoding: torch.Tensor) -> torch.Tensor:
        """Forward module.

        Args:
            context_encoding: [N, A] sized tensor, where A is the sum of
                target_agent_mcg_hidden_size, target_agent_emb_size,
                nbr_agent_mcg_hidden_size and road_agent_mcg_hidden_size.
        """
        anchor_embedding = self.anchor_embedding.weight

        pred_enc, _ = self.predictor_MCG(anchor_embedding, context_encoding)

        return pred_enc


@OBJECT_REGISTRY.register
class MultiHeadMTPPlusAggregator(nn.Module):
    """Multi-headed version of the multipath++ aggregator."""

    def __init__(
        self,
        num_header: int,
        num_anchor: int,
        anchor_emb_size: int,
        pred_mcg_layers: int,
        target_agent_mcg_hidden_size: int,
        target_agent_emb_size: int,
        nbr_agent_mcg_hidden_size: int,
        road_agent_mcg_hidden_size: int,
        pred_mcg_hidden_size: int,
    ):
        """Initialize the module.

        Args:
            num_header: number of heads to use.
            num_anchor: number of output modalities.
            anchor_emb_size: anchor embedding size.
            pred_mcg_layers: number of prediction MCG layers.
            target_agent_mcg_hidden_size: hidden size of MCG layers for
                neighboring agents.
            target_agent_emb_size: embedding size for target agent.
            nbr_agent_mcg_hidden_size: hidden size of MCG layers for
                neighboring agents.
            road_agent_mcg_hidden_size: road agent MCG hidden layer size.
            pred_mcg_hidden_size: hidden size of prediction MCG layers.
        """
        super().__init__()

        self.num_header = num_header
        self.mtpplus_headers = nn.ModuleList()

        for _ in range(self.num_header):
            self.mtpplus_headers.append(
                MTPPlusAggregator(
                    num_anchor,
                    anchor_emb_size,
                    pred_mcg_layers,
                    target_agent_mcg_hidden_size,
                    target_agent_emb_size,
                    nbr_agent_mcg_hidden_size,
                    road_agent_mcg_hidden_size,
                    pred_mcg_hidden_size,
                )
            )

    def forward(self, context_encoding: torch.Tensor) -> torch.Tensor:
        """Forward module.

        Args:
            context_encoding: [N, A] sized tensor, where A is the sum of
                target_agent_mcg_hidden_size, target_agent_emb_size,
                nbr_agent_mcg_hidden_size and road_agent_mcg_hidden_size.
        """
        return [
            aggregator.forward(context_encoding)
            for aggregator in self.mtpplus_headers
        ]
