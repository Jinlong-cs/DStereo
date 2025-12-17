from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY

__all__ = ["GenISP"]


class Diagonalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.diag = torch.diag_embed

    def forward(self, x):
        return self.diag(x)


class Resize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode="bilinear")


@OBJECT_REGISTRY.register
class GenISP(nn.Module):
    """
    implementation of "GenISP: Neural ISP for Low-Light Machine Cognition".

    for more details, please refer to https://arxiv.org/abs/2205.03688.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels

    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        split_transform=False,
    ):
        super(GenISP, self).__init__()

        self.split_transform = split_transform

        self.image_to_parameter = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AvgPool2d(kernel_size=32),
            nn.Flatten(1),
        )

        self.conv_wb = nn.Sequential(
            Resize((256, 256)),
            deepcopy(self.image_to_parameter),
            nn.Linear(128, 3),
            Diagonalize(),
        )

        self.conv_cc = nn.Sequential(
            Resize((256, 256)),
            deepcopy(self.image_to_parameter),
            nn.Linear(128, 9),
            nn.Unflatten(1, (3, 3)),
        )

        # A non-linear local image enhancement by a shallow ConvNet
        self.shallow_conv_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(16),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(16),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, data):
        batch = data["img"]
        n, c, h, w = batch.shape
        if self.split_transform:
            me_in_img = data["me_in_img"]
            _, _, h_me, w_me = me_in_img.shape
            wb_matrix = self.conv_wb(me_in_img)
            me_in_img = me_in_img.permute(0, 2, 3, 1).reshape(
                n, h_me * w_me, c
            )
            me_in_img = (
                torch.bmm(me_in_img, wb_matrix)
                .reshape(n, h_me, w_me, c)
                .permute(0, 3, 1, 2)
            )
        else:
            wb_matrix = self.conv_wb(batch)
        batch = batch.permute(0, 2, 3, 1).reshape(n, h * w, c)
        batch = (
            torch.bmm(batch, wb_matrix).reshape(n, h, w, c).permute(0, 3, 1, 2)
        )
        if self.split_transform:
            cc_matrix = self.conv_cc(me_in_img)
        else:
            cc_matrix = self.conv_cc(batch)
        batch = batch.permute(0, 2, 3, 1).reshape(n, h * w, c)
        batch = (
            torch.bmm(batch, cc_matrix).reshape(n, h, w, c).permute(0, 3, 1, 2)
        )
        x = self.shallow_conv_net(batch)
        return x
