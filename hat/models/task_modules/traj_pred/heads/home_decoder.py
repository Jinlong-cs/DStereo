# Copyright (c) Horizon Robotics. All rights reserved.
#
# For further details about model implementation, inputs/outputs, etc.,
# please see this developer guide:
# https://horizonrobotics.feishu.cn/docs/doccnBr8XXeUQG3MeNHAr0tcZxb
from typing import Dict

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class HomeConcat(nn.Module):
    """HOME concat module. Acts as the bridge between encoder and decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the HOMEConcat module.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        super(HomeConcat, self).__init__()

        in_channels = in_channels
        heatmap_channels = out_channels

        self.conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU6(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU6(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU6(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU6(),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=heatmap_channels,
                kernel_size=1,
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, agg_encoding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward model.

        Args:
            agg_encoding: [N, C, H, W] sized tensor.
        """
        agg_encoding = self.conv1(agg_encoding)
        agg_encoding = self.conv2(agg_encoding)
        agg_encoding = self.conv3(agg_encoding)
        agg_encoding = self.conv4(agg_encoding)
        heatmap = self.conv5(agg_encoding)

        predictions = {"heatmap": heatmap}

        return predictions


@OBJECT_REGISTRY.register
class HomeDecoder(nn.Module):
    """
    HOME decoder module.

    HOME: Heatmap Output for future Motion Estimation
    https://arxiv.org/pdf/2105.10968.pdf
    """

    def __init__(
        self,
        in_channels: int,
        heatmap_channels: int,
    ) -> None:
        """Initialize HOME decoder module.

        Args:
            in_channels: number of input channels.
            heatmap_channels: number of heatmap channels.
        """
        super(HomeDecoder, self).__init__()

        in_channels = in_channels
        heatmap_channels = heatmap_channels

        self.conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU6(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU6(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU6(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU6(),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=heatmap_channels,
                kernel_size=1,
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, agg_encoding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward model.

        Args:
            agg_endoing: [N, C, H, W] sized tensor.
        """
        agg_encoding = self.conv1(agg_encoding)
        agg_encoding = self.conv2(agg_encoding)
        agg_encoding = self.conv3(agg_encoding)
        agg_encoding = self.conv4(agg_encoding)
        heatmap = self.conv5(agg_encoding)

        predictions = {"heatmap": heatmap}

        return predictions
