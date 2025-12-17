# Copyright (c) Horizon Robotics. All rights reserved.
#
# For further details about model implementation, inputs/outputs, etc.,
# please see this developer guide:
# https://horizonrobotics.feishu.cn/docs/doccnBr8XXeUQG3MeNHAr0tcZxb
from typing import Any, Dict

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    from torchvision.models import resnet18, resnet34, resnet50
except ImportError:
    resnet18, resnet34, resnet50 = None, None, None


RESNET_BACKBONES = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


@OBJECT_REGISTRY.register
class HomeEncoder(nn.Module):
    """HOME: Heatmap Output for future Motion Estimation.

    https://arxiv.org/pdf/2105.10968.pdf
    See section 3 of the developer guide for further details:
    https://horizonrobotics.feishu.cn/docs/doccnBr8XXeUQG3MeNHAr0tcZxb
    """

    @require_packages("torchvision")
    def __init__(
        self,
        context_encoding_size: int,
        backbone: str,
        input_channels: int,
        agent_feat_size: int,
        agent_emb_size: int,
        agent_enc_size: int,
        social_encoding_size: int,
    ) -> None:
        """Initialize the encoder.

        Args:
            context_encoding_size: size of the context encoding.
            backbone: backbone type. Choose from "resnet18", "resnet34"
                and "resnet50".
            input_channels: number of input channels.
            agent_feat_size: feature size of the agent.
            agent_emb_size: embedding size of the agent.
            agent_enc_size: encoding size of the agent.
            social_encoding_size: size of social interaction encoding.
        """
        super(HomeEncoder, self).__init__()

        self.context_encoding_size = context_encoding_size

        # Initialize backbone:
        resnet_model = RESNET_BACKBONES[backbone](pretrained=False)
        conv1_new = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        modules = list(resnet_model.children())[:-3]
        modules[0] = conv1_new
        modules.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(1024, 512, 1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU6(),
                torch.nn.AdaptiveAvgPool2d(self.context_encoding_size),
            )
        )
        self.raster_backbone = torch.nn.Sequential(*modules)

        self.conv1d_ego = torch.nn.Linear(
            in_features=agent_feat_size,
            out_features=agent_emb_size,
        )
        self.conv1d_surrounding = torch.nn.Linear(
            in_features=agent_feat_size,
            out_features=agent_emb_size,
        )
        self.relu = torch.nn.ReLU()

        self.gru_ego = torch.nn.GRU(
            input_size=agent_emb_size,
            hidden_size=agent_enc_size,
            num_layers=1,
            batch_first=True,
        )
        self.gru_surrounding = torch.nn.GRU(
            input_size=agent_emb_size,
            hidden_size=agent_enc_size,
            num_layers=1,
            batch_first=True,
        )

        self.atten = torch.nn.MultiheadAttention(
            embed_dim=agent_enc_size, num_heads=1, batch_first=True
        )
        self.layer_norm = torch.nn.LayerNorm(agent_enc_size)

        self.social_linear = torch.nn.Linear(
            in_features=agent_enc_size,
            out_features=social_encoding_size,
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward model.

        Args:
            inputs: dictionary containing the following keys:
                - raster: [B, input_channels, H, W] sized raster.
                - ego_dynamics: [B, 1, T, agent_feat_size] tensor.
                - surrounding_dynamics: [B, num_agents, T, agent_feat_size]
                    tensor.
        """
        rasterized_input = inputs["raster"]

        surrounding_dynamics = inputs["surrounding_dynamics"]
        ego_dynamics = inputs["ego_dynamics"]

        batch_size = surrounding_dynamics.shape[0]
        surrounding_number = surrounding_dynamics.shape[1]
        ego_number = ego_dynamics.shape[1]
        time_step = surrounding_dynamics.shape[2]
        feat_size = surrounding_dynamics.shape[3]

        context_encoding = self.raster_backbone(rasterized_input)

        _, temp_encoding_ego = self.gru_ego(
            self.conv1d_ego(ego_dynamics.view(-1, time_step, feat_size))
        )
        temp_encoding_ego = temp_encoding_ego.permute(1, 0, 2).view(
            batch_size, ego_number, -1
        )

        _, temp_encoding_surrounding = self.gru_ego(
            self.conv1d_ego(
                surrounding_dynamics.view(-1, time_step, feat_size)
            )
        )
        temp_encoding_surrounding = temp_encoding_surrounding.permute(
            1, 0, 2
        ).view(batch_size, surrounding_number, -1)

        atten_encoding, _ = self.atten(
            query=temp_encoding_ego,
            key=temp_encoding_surrounding,
            value=temp_encoding_surrounding,
        )
        atten_encoding = self.layer_norm(atten_encoding + temp_encoding_ego)
        social_encoding = self.social_linear(atten_encoding)
        social_encoding = (
            social_encoding.unsqueeze(1)
            .repeat(
                1, self.context_encoding_size, self.context_encoding_size, 1
            )
            .permute(0, 3, 1, 2)
        )

        return {
            "social_encoding": social_encoding,
            "context_encoding": context_encoding,
        }
